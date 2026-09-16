#!/usr/bin/env python3
"""One-off sweep: re-evaluate every run's final checkpoint, with and without
LoopBreakerWrapper, to see how much eval win rate was being masked by
degenerate repeating-state loops rather than genuine incompetence. No
retraining involved - same weights, just a different action-selection
wrapper at evaluation time.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, preprocess_decomposed, NUM_FEATURES, NUM_DECOMPOSED_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, DecomposedDQN
from agent.loop_breaker import LoopBreakerWrapper

EVAL_SEED_BASE = 900_000
WINDOWS_ARCHIVE = Path("/mnt/c/Users/Public/solitaire_checkpoints")


def find_checkpoint(run_dir: Path) -> Path | None:
    local = run_dir / "checkpoint_final.pt"
    if local.exists():
        return local
    archived = WINDOWS_ARCHIVE / run_dir.name / "checkpoint_final.pt"
    if archived.exists():
        return archived
    return None


def load_net(run_dir: Path, checkpoint_path: Path):
    config = json.loads((run_dir / "config.json").read_text())
    encoding = config.get("card_encoding", "raw")
    hidden_layers = config.get("hidden_layers", 2)
    hidden_dim = config.get("hidden_dim", 512)
    allow_undo = config.get("allow_undo", True)

    if encoding == "decomposed":
        preprocess_fn = preprocess_decomposed
        net = DecomposedDQN(num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim)
    else:
        preprocess_fn = preprocess
        net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()
    return net, preprocess_fn, allow_undo


def run_eval(net, preprocess_fn, allow_undo: bool, use_loop_breaker: bool, episodes: int = 100, max_steps: int = 1000):
    wins = 0
    total_foundation = 0
    for i in range(episodes):
        env = gym.make("Solitaire-v0", allow_undo=allow_undo)
        if use_loop_breaker:
            env = LoopBreakerWrapper(env)
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
        won = terminated and int(obs["foundations"].sum()) == 52
        wins += won
        total_foundation += int(obs["foundations"].sum())
        env.close()
    return wins, total_foundation / episodes


def main() -> None:
    runs_root = Path("runs")
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir() and d.name[:3].isdigit())

    print(f"{'run':<55} {'no-LB wins':>10} {'no-LB fnd':>10} {'LB wins':>10} {'LB fnd':>10}")
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        if not config_path.exists():
            print(f"{run_dir.name:<55} (no config.json, skipped)")
            continue
        checkpoint_path = find_checkpoint(run_dir)
        if checkpoint_path is None:
            print(f"{run_dir.name:<55} (no checkpoint_final.pt found, skipped)")
            continue
        try:
            net, preprocess_fn, allow_undo = load_net(run_dir, checkpoint_path)
        except Exception as e:
            print(f"{run_dir.name:<55} (failed to load: {e})")
            continue
        wins0, fnd0 = run_eval(net, preprocess_fn, allow_undo, use_loop_breaker=False)
        wins1, fnd1 = run_eval(net, preprocess_fn, allow_undo, use_loop_breaker=True)
        print(f"{run_dir.name:<55} {wins0:>10} {fnd0:>10.1f} {wins1:>10} {fnd1:>10.1f}")


if __name__ == "__main__":
    main()
