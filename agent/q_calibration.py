#!/usr/bin/env python3
"""Value calibration for one checkpoint: Q(s, a_taken) vs the discounted
return G_t the episode then actually realized from that point, broken down
by game phase, plus the share of transitions carrying a reward above
Huber's beta=1 transition point.

Why this exists (SESSION_SUMMARY §33): run 029's checkpoints predicted Q ~36
at the deal against realized G_0 ~630 (Q/G ~ 0.03 in every phase). The
cause was the TD loss, not the network - smooth_l1 with beta=1 has unit
gradient on every reward-bearing transition (rewards are 50-5000), so Q
settles at a fixed point set by reward *frequency* p, roughly
p / ((1-p)(1-gamma)), independent of reward magnitude. agent.train's
evaluate() now logs eval_q0/eval_g0 at every checkpoint; this script is
the finer-grained, per-phase version for diagnosis.

Rollouts use softmax sampling at --temperature (default 1.0) rather than
argmax so a looping policy doesn't spend 1000 steps in one state and swamp
the per-phase medians; pass --temperature 0 for pure greedy.
Usage:
    python -m agent.q_calibration runs/<run>/checkpoint_400000.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import gymnasium as gym

import solitaire_gym  # noqa: F401
from solitaire_gym.preprocessing import preprocess, NUM_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, ILLEGAL_Q_VALUE
from agent.eval_action_selection import load_seeds

PHASES = [(0, 25), (25, 100), (100, 300), (300, 10_000)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--eval-seed-source", default="runs/eval_corpus.jsonl")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dev = args.device
    net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, hidden_dim=args.hidden_dim,
              num_hidden_layers=args.hidden_layers, layer_norm=args.layer_norm).to(dev)
    net.load_state_dict(torch.load(args.checkpoint, map_location=dev, weights_only=True))
    net.eval()
    gen = torch.Generator(device=dev).manual_seed(0)
    env = gym.make("Solitaire-v0")

    Q, G, T, R, q0s, g0s = [], [], [], [], [], []
    for seed in load_seeds(args.eval_seed_source or None, args.episodes):
        obs, info = env.reset(seed=seed)
        qs, rs = [], []
        while True:
            features, mask = preprocess(obs, info)
            with torch.no_grad():
                q = net(torch.from_numpy(features).to(dev), torch.from_numpy(mask).to(dev))[0]
            legal = torch.nonzero(q > ILLEGAL_Q_VALUE / 2).squeeze(1)
            if args.temperature <= 0:
                a = int(q.argmax())
            else:
                ql = q[legal]
                a = int(legal[torch.multinomial(torch.softmax((ql - ql.max()) / args.temperature, 0), 1, generator=gen)])
            qs.append(float(q[a]))
            obs, reward, terminated, truncated, info = env.step(a)
            rs.append(reward)
            if terminated or truncated:
                break
        g, Gs = 0.0, [0.0] * len(rs)
        for i in range(len(rs) - 1, -1, -1):
            g = rs[i] + args.gamma * g
            Gs[i] = g
        Q += qs; G += Gs; T += list(range(len(rs))); R += rs
        q0s.append(qs[0]); g0s.append(Gs[0])

    Q, G, T, R = map(np.asarray, (Q, G, T, R))
    print(f"{args.checkpoint}: {args.episodes} episodes, {len(Q)} decisions, gamma={args.gamma}, rollout T={args.temperature}")
    print(f"at the deal: Q median {np.median(q0s):7.1f}   realized G_0 median {np.median(g0s):7.1f}   ratio {np.median(q0s) / max(np.median(g0s), 1e-6):.2f}")
    print(f"{'phase (t)':>14s} {'n':>7s} {'Q med':>8s} {'G med':>8s} {'Q/G':>6s}  {'p(|r|>1)':>9s}  {'Huber-fixpt':>11s}")
    for lo, hi in PHASES:
        sel = (T >= lo) & (T < hi)
        if not sel.any():
            continue
        p = (np.abs(R[sel]) > 1).mean()
        fixpt = p / ((1 - p) * (1 - args.gamma)) if p < 1 else float("inf")
        ratio = np.median(Q[sel]) / np.median(G[sel]) if np.median(G[sel]) > 1 else float("nan")
        print(f"{f'[{lo},{hi})':>14s} {sel.sum():7d} {np.median(Q[sel]):8.1f} {np.median(G[sel]):8.1f} {ratio:6.2f}  {p:9.1%}  {fixpt:11.1f}")
    print("Huber-fixpt = p/((1-p)(1-gamma)): where Q should sit under saturated smooth_l1(beta=1); irrelevant under --td-loss mse")


if __name__ == "__main__":
    main()
