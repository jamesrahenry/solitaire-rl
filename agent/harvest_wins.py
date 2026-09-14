#!/usr/bin/env python3
"""Harvest real, seeded winning games to bootstrap curriculum learning.

Runs episodes (guided by a trained checkpoint, an epsilon-greedy mix, or the
rule-based greedy Klondike heuristic - see agent.heuristic_policy) with
explicit per-episode seeds and GameLogger attached, so any wins found land
in a games.jsonl with a real, independently-reproducible {seed, actions}
record - unlike wins logged by older runs (before the per-episode seeding
fix), which have seed: None.

Dedicated harvesting via a trained checkpoint or epsilon-greedy exploration
found ZERO wins across ~2000 combined episodes (the foundation-undo move
creates an oscillation trap for undirected play). --policy heuristic uses a
rule-based greedy policy instead, which found real wins where those didn't.

Usage:
    python -m agent.harvest_wins --episodes 500 --policy heuristic --out runs/curriculum_wins.jsonl
    python -m agent.harvest_wins --episodes 500 --policy epsilon --epsilon 0.15 \
        --checkpoint runs/003_double_dqn_soft_nstep3_per/checkpoint_final.pt \
        --out runs/curriculum_wins.jsonl
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, NUM_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from solitaire_gym.logging_wrapper import GameLogger
from agent.qnetwork import DQN
from agent.heuristic_policy import GreedyKlondikePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest real seeded wins for curriculum learning")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--policy", choices=["epsilon", "heuristic"], default="heuristic")
    parser.add_argument("--epsilon", type=float, default=0.15, help="(--policy epsilon) exploration rate on top of the checkpoint (1.0 = pure random if no --checkpoint given)")
    parser.add_argument("--checkpoint", default=None, help="(--policy epsilon) path to a trained DQN checkpoint; omit for pure random legal-action play")
    parser.add_argument("--out", default="runs/curriculum_wins.jsonl")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    net = None
    if args.policy == "epsilon" and args.checkpoint:
        net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS).to(args.device)
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
        net.eval()
    heuristic = GreedyKlondikePolicy() if args.policy == "heuristic" else None

    env = GameLogger(gym.make("Solitaire-v0"), log_path=args.out)
    game = env.unwrapped.game

    wins = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        features, mask = preprocess(obs, info)
        if heuristic is not None:
            heuristic.reset()
        while True:
            if heuristic is not None:
                action = heuristic.select_action(game, np.flatnonzero(mask), rng)
                if action is None:
                    break  # only undo moves legal - a genuine dead end for this policy
            elif net is None or rng.random() < args.epsilon:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    x = torch.from_numpy(features).to(args.device)
                    m = torch.from_numpy(mask).to(args.device)
                    action = int(net(x, m).argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            features, mask = preprocess(obs, info)
            if terminated or truncated:
                if terminated and int(obs["foundations"].sum()) == 52:
                    wins += 1
                break
        if (ep + 1) % 50 == 0:
            print(f"{ep + 1}/{args.episodes} episodes, {wins} wins so far")

    env.close()
    print(f"done: {wins}/{args.episodes} wins, logged to {args.out}")


if __name__ == "__main__":
    main()
