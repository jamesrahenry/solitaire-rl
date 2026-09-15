#!/usr/bin/env python3
"""Behavior-cloning pretraining: supervised warm-start for the DQN from the
heuristic policy's harvested games (all 5000 episodes by default, not just
the 86 wins - the heuristic's individual decisions are reasonably sound
throughout, win or lose; a loss usually comes from a bad deal or one early
irreversible mistake, not consistently bad play, so the non-winning games
still carry useful "what does sound play look like" signal).

Produces a checkpoint with the exact same shape as agent.train's DQN, so it
can be used directly as an RL fine-tuning starting point via:
    python -m agent.train --resume-from runs/bc_pretrained.pt ...
(--resume-step defaults to 0, so schedules start fresh - only the weights
are warm-started, not the step-based epsilon/beta/curriculum schedules.)

Usage:
    python -m agent.behavior_cloning --source runs/curriculum_wins.jsonl --out runs/bc_pretrained.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, preprocess_decomposed, NUM_FEATURES, NUM_DECOMPOSED_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, DecomposedDQN


def build_dataset(source: str, preprocess_fn, num_features: int, memmap_dir: str):
    """Replay every logged episode (deterministic from its seed) and record
    (features, mask, action) at every step. Returns arrays sized to the
    exact total step count - one pass to count, one pass to fill, so we
    never over-allocate or need to grow arrays.

    Backed by disk (np.memmap) rather than plain np.zeros: this dataset is
    ~5GB at raw encoding (more at decomposed), and a single up-front RAM
    allocation that size was triggering repeated OOM kills at the exact
    same point on this machine. memmap lets the OS page it in from disk as
    needed instead of reserving it all in RSS at once.

    Caches its output: a repeat call with the same source/memmap_dir/
    num_features (e.g. re-running train_bc with a different network size)
    reuses the existing replay instead of redoing it - the replay is by far
    the slowest part (tens of minutes) and doesn't depend on the network at
    all, only build_dataset's own inputs."""
    with open(source) as f:
        records = [json.loads(line) for line in f]
    total_steps = sum(r["num_steps"] for r in records)
    print(f"{len(records)} episodes, {total_steps} total state-action pairs")

    memmap_path = Path(memmap_dir)
    meta_path = memmap_path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("source") == str(source) and meta.get("num_features") == num_features and meta.get("total_steps") == total_steps:
            valid_steps = meta["valid_steps"]
            print(f"reusing cached replay at {memmap_dir} ({valid_steps} valid steps, skipping replay)")
            features = np.memmap(memmap_path / "features.dat", dtype=np.int32, mode="r", shape=(total_steps, num_features))
            masks = np.memmap(memmap_path / "masks.dat", dtype=bool, mode="r", shape=(total_steps, NUM_ACTIONS))
            actions = np.memmap(memmap_path / "actions.dat", dtype=np.int64, mode="r", shape=(total_steps,))
            return features[:valid_steps], masks[:valid_steps], actions[:valid_steps]

    memmap_path.mkdir(parents=True, exist_ok=True)
    features = np.memmap(memmap_path / "features.dat", dtype=np.int32, mode="w+", shape=(total_steps, num_features))
    masks = np.memmap(memmap_path / "masks.dat", dtype=bool, mode="w+", shape=(total_steps, NUM_ACTIONS))
    actions = np.memmap(memmap_path / "actions.dat", dtype=np.int64, mode="w+", shape=(total_steps,))

    env = gym.make("Solitaire-v0")
    idx = 0
    truncated_episodes = 0
    dropped_steps = 0
    start_time = time.time()
    for i, record in enumerate(records):
        obs, info = env.reset(seed=record["seed"])
        for step_num, action in enumerate(record["actions"]):
            f_, m_ = preprocess_fn(obs, info)
            if not m_[action]:
                # This action isn't actually legal in the state we just replayed
                # to - the recorded trajectory has diverged from what this
                # engine/config produces for this seed (e.g. a curriculum-started
                # episode: GameLogger only records the agent's tail actions, not
                # the fast-forwarded initial_moves prefix that preceded them, so
                # replaying "seed + tail-only actions" from a fresh deal starts
                # from a different board than the actions were chosen for).
                # env.step() would silently no-op on an illegal action rather
                # than raise, which would otherwise poison this and every
                # subsequent example in the episode with a target action pinned
                # at the illegal-action mask value in training. Drop the rest of
                # this episode instead of recording corrupted examples.
                truncated_episodes += 1
                dropped_steps += len(record["actions"]) - step_num
                break
            features[idx] = f_
            masks[idx] = m_
            actions[idx] = action
            idx += 1
            obs, reward, terminated, truncated, info = env.step(action)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            features.flush()
            masks.flush()
            actions.flush()
            print(f"replayed {i + 1}/{len(records)} episodes, {idx}/{total_steps} steps ({elapsed:.0f}s)")

    if truncated_episodes:
        print(
            f"WARNING: {truncated_episodes}/{len(records)} episodes diverged from their recorded actions "
            f"(most likely missing a curriculum fast-forward prefix) - dropped {dropped_steps} steps from "
            f"the point of divergence onward rather than record them as corrupted training examples"
        )
    features.flush()
    masks.flush()
    actions.flush()
    env.close()
    meta_path.write_text(json.dumps({"source": str(source), "num_features": num_features, "total_steps": total_steps, "valid_steps": idx}))
    return features[:idx], masks[:idx], actions[:idx]


def train_bc(net, features, masks, actions, epochs, batch_size, lr, device, val_frac, seed):
    n = len(actions)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"{len(train_idx)} train examples, {len(val_idx)} val examples")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def run_epoch(indices, training: bool):
        net.train(training)
        total_loss, correct = 0.0, 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x = torch.from_numpy(features[batch_idx]).to(device)
            m = torch.from_numpy(masks[batch_idx]).to(device)
            y = torch.from_numpy(actions[batch_idx]).to(device)

            if training:
                q = net(x, m)
                loss = F.cross_entropy(q, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    q = net(x, m)
                    loss = F.cross_entropy(q, y)

            total_loss += loss.item() * len(batch_idx)
            correct += (q.argmax(dim=1) == y).sum().item()
        return total_loss / len(indices), correct / len(indices)

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        t0 = time.time()
        train_loss, train_acc = run_epoch(train_idx, training=True)
        val_loss, val_acc = run_epoch(val_idx, training=False)
        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({time.time() - t0:.0f}s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone the heuristic policy as an RL warm-start")
    parser.add_argument("--source", default="runs/curriculum_wins.jsonl")
    parser.add_argument("--out", default="runs/bc_pretrained.pt")
    parser.add_argument("--card-encoding", choices=["raw", "decomposed"], default="raw")
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memmap-dir", default="runs/_bc_cache", help="disk-backed scratch directory for the (large) replayed dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.card_encoding == "decomposed":
        preprocess_fn, num_features = preprocess_decomposed, NUM_DECOMPOSED_FEATURES
        net = DecomposedDQN(
            num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim
        ).to(args.device)
    else:
        preprocess_fn, num_features = preprocess, NUM_FEATURES
        net = DQN(
            num_features=NUM_FEATURES,
            num_actions=NUM_ACTIONS,
            num_hidden_layers=args.hidden_layers,
            hidden_dim=args.hidden_dim,
        ).to(args.device)

    features, masks, actions = build_dataset(args.source, preprocess_fn, num_features, args.memmap_dir)
    train_bc(net, features, masks, actions, args.epochs, args.batch_size, args.lr, args.device, args.val_frac, args.seed)

    torch.save(net.state_dict(), args.out)
    print(f"done. saved behavior-cloned checkpoint to {args.out}")


if __name__ == "__main__":
    main()
