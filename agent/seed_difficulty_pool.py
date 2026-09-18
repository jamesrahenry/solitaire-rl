#!/usr/bin/env python3
"""Pool eval results across every run in the project (not just one run's
checkpoints) to ask a different question than seed_churn.py: is there a
seed that NOTHING we've ever built has won, across every architecture and
technique tried so far? That's a much stronger, more model-independent
signal than any single run's win/loss data, which is confounded by that
run's own training noise (see the argmax-tie-breaking churn already found
in run 021). Still not a formal proof of full-information unsolvability -
only a "nothing we've thrown at it has cracked it yet" empirical label -
but cheap to build from checkpoints that already exist, and directly
actionable (candidates for pruning from the fixed 100-seed eval set, since
a seed nothing can ever win just adds a flat, uninformative 0 to every
future measurement).

Always evaluates through LoopBreakerWrapper - it's a strict improvement
over raw greedy rollout with no downside, so using it removes a known,
already-diagnosed confound (getting stuck in a 2-action loop is not the
same as a deal being unsolvable) from every run's result, including ones
trained before the wrapper existed.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, preprocess_decomposed, NUM_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, DecomposedDQN
from agent.loop_breaker import LoopBreakerWrapper

EVAL_SEED_BASE = 900_000
NUM_SEEDS = 100
WINDOWS_ARCHIVE = Path("/mnt/c/Users/Public/solitaire_checkpoints")
RUNS_DIR = Path("runs")


_staging_dir = Path(tempfile.mkdtemp(prefix="ckpt_stage_"))


def find_checkpoint(run_dir: Path) -> Path | None:
    local = run_dir / "checkpoint_final.pt"
    if local.exists():
        return local
    archived = WINDOWS_ARCHIVE / run_dir.name / "checkpoint_final.pt"
    if archived.exists():
        # Stage with a plain sequential cp rather than letting torch.load do
        # its own (more random-access) reads directly over the WSL2 9p/drvfs
        # bridge to /mnt/c - suspected trigger of a system crash that
        # happened mid-way through this exact same cross-mount read pattern.
        staged = _staging_dir / run_dir.name / "checkpoint_final.pt"
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(archived, staged)
        return staged
    return None


def load_net(run_dir: Path, checkpoint_path: Path):
    config = json.loads((run_dir / "config.json").read_text())
    encoding = config.get("card_encoding", "raw")
    hidden_layers = config.get("hidden_layers", 2)
    hidden_dim = config.get("hidden_dim", 512)
    allow_undo = config.get("allow_undo", True)
    layer_norm = config.get("layer_norm", False)

    if encoding == "decomposed":
        preprocess_fn = preprocess_decomposed
        net = DecomposedDQN(num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim, layer_norm=layer_norm)
    else:
        preprocess_fn = preprocess
        net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim, layer_norm=layer_norm)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()
    return net, preprocess_fn, allow_undo


def eval_seeds(net, preprocess_fn, allow_undo: bool, max_steps: int = 1000) -> np.ndarray:
    won = np.zeros(NUM_SEEDS, dtype=bool)
    for i in range(NUM_SEEDS):
        env = LoopBreakerWrapper(gym.make("Solitaire-v0", allow_undo=allow_undo))
        obs, info = env.reset(seed=EVAL_SEED_BASE + i)
        features, mask = preprocess_fn(obs, info)
        terminated = truncated = False
        for _ in range(max_steps):
            with torch.no_grad():
                x = torch.from_numpy(features).unsqueeze(0)
                m = torch.from_numpy(mask).unsqueeze(0)
                q = net(x, m)
                a = int(q.argmax(dim=1).item())
            obs, r, terminated, truncated, info = env.step(a)
            features, mask = preprocess_fn(obs, info)
            if terminated or truncated:
                break
        won[i] = terminated and int(obs["foundations"].sum()) == 52
        env.close()
    return won


def main() -> None:
    run_dirs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name[0].isdigit())

    ever_won = np.zeros(NUM_SEEDS, dtype=bool)
    per_run_wins = {}

    for run_dir in run_dirs:
        ckpt = find_checkpoint(run_dir)
        if ckpt is None:
            print(f"{run_dir.name}: no checkpoint found, skipping")
            continue
        try:
            net, preprocess_fn, allow_undo = load_net(run_dir, ckpt)
            won = eval_seeds(net, preprocess_fn, allow_undo)
        except Exception as e:
            print(f"{run_dir.name}: eval failed ({e}), skipping")
            continue
        per_run_wins[run_dir.name] = won
        ever_won |= won
        print(f"{run_dir.name}: {won.sum()}/100 wins (final checkpoint, loop-breaker on)", flush=True)

    # fold in run 021's dense 40-checkpoint matrix too, for extra resolution
    dense_matrix_path = RUNS_DIR / "021_double_dqn_soft_nstep3_per_dqfd_moredemo_epsfloor_loopbreaker" / "seed_churn_matrix.npy"
    if dense_matrix_path.exists():
        dense = np.load(dense_matrix_path)
        ever_won |= dense.any(axis=0)
        print(f"folded in run 021's 40-checkpoint matrix ({dense.any(axis=0).sum()}/100 ever-won there alone)")

    never_won = ~ever_won
    print()
    print(f"Across {len(per_run_wins)} runs' final checkpoints (+ run 021's full 40-checkpoint history):")
    print(f"  seeds won by at least one checkpoint, anywhere, ever: {ever_won.sum()}/100")
    print(f"  seeds NEVER won by anything we've built: {never_won.sum()}/100")
    print(f"  never-won seed ids: {[EVAL_SEED_BASE + i for i in np.flatnonzero(never_won)]}")

    np.save(RUNS_DIR / "seed_pooled_ever_won.npy", ever_won)
    with open(RUNS_DIR / "seed_pooled_per_run.json", "w") as f:
        json.dump({name: won.tolist() for name, won in per_run_wins.items()}, f)
    print(f"\nsaved to {RUNS_DIR / 'seed_pooled_ever_won.npy'} and {RUNS_DIR / 'seed_pooled_per_run.json'}")


if __name__ == "__main__":
    main()
