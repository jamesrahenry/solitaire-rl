#!/usr/bin/env python3
"""Build a (checkpoint x eval-seed) win/loss matrix for a run, to check
whether the trained policy's per-seed competence is monotonic (once a seed
is solved, it stays solved) or churns (solves seed A at one checkpoint,
loses it later while gaining seed B, etc.) - a different question from the
aggregate win-rate trend, which can't distinguish "the same seeds, more
reliably" from "a constantly-shifting set of seeds."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, NUM_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN
from agent.loop_breaker import LoopBreakerWrapper

EVAL_SEED_BASE = 900_000


def eval_checkpoint(net, episodes=100, max_steps=1000):
    won = np.zeros(episodes, dtype=bool)
    for i in range(episodes):
        env = LoopBreakerWrapper(gym.make("Solitaire-v0"))
        obs, info = env.reset(seed=EVAL_SEED_BASE + i)
        features, mask = preprocess(obs, info)
        terminated = truncated = False
        for _ in range(max_steps):
            with torch.no_grad():
                x = torch.from_numpy(features).unsqueeze(0)
                m = torch.from_numpy(mask).unsqueeze(0)
                q = net(x, m)
                a = int(q.argmax(dim=1).item())
            obs, r, terminated, truncated, info = env.step(a)
            features, mask = preprocess(obs, info)
            if terminated or truncated:
                break
        won[i] = terminated and int(obs["foundations"].sum()) == 52
        env.close()
    return won


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/021_double_dqn_soft_nstep3_per_dqfd_moredemo_epsfloor_loopbreaker")
    config = json.loads((run_dir / "config.json").read_text())
    hidden_layers = config.get("hidden_layers", 2)
    hidden_dim = config.get("hidden_dim", 512)

    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 10**9)
    checkpoints = [p for p in checkpoints if p.stem.split("_")[1].isdigit()]
    steps = [int(p.stem.split("_")[1]) for p in checkpoints]

    matrix = np.zeros((len(checkpoints), 100), dtype=bool)
    for i, (step, ckpt) in enumerate(zip(steps, checkpoints)):
        net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim)
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        net.eval()
        matrix[i] = eval_checkpoint(net)
        print(f"step {step}: {matrix[i].sum()}/100 wins", flush=True)

    np.save(run_dir / "seed_churn_matrix.npy", matrix)
    with open(run_dir / "seed_churn_steps.json", "w") as f:
        json.dump(steps, f)
    print("saved matrix to", run_dir / "seed_churn_matrix.npy")


if __name__ == "__main__":
    main()
