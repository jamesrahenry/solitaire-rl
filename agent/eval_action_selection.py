#!/usr/bin/env python3
"""Evaluate one checkpoint under several action-selection rules on the same
fixed deals: greedy argmax vs Boltzmann (softmax) sampling at a range of
temperatures, each with and without LoopBreakerWrapper, plus the heuristic
harvester as a reference. Also reports the top-1 vs top-2 legal Q gap
distribution, since that's what decides whether argmax is a decision or a
coin flip.

Motivation (2026-09-30 review, SESSION_SUMMARY §32): on run 029's 390k
checkpoint the logged argmax+loop-breaker number was 33.5%; softmax T=1
with NO loop-breaker scored 46%, and T=1 + loop-breaker 52.5% - same
weights, zero retraining. The median top-1/top-2 Q gap was 0.19, so argmax
was deterministically committing to noise and then repeating itself.
Usage:
    python -m agent.eval_action_selection runs/<run>/checkpoint_390000.pt
    python -m agent.eval_action_selection <ckpt> --temperatures 0 0.5 1 2 --heuristic
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import gymnasium as gym

import solitaire_gym  # noqa: F401  (registers Solitaire-v0)
from solitaire_gym.preprocessing import preprocess, NUM_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, ILLEGAL_Q_VALUE
from agent.loop_breaker import LoopBreakerWrapper
from agent.heuristic_policy import GreedyKlondikePolicy


def load_seeds(path: str | None, episodes: int) -> list[int]:
    if path:
        with open(path) as f:
            seeds = [json.loads(line)["seed"] for line in f]
        return seeds[:episodes]
    from agent.train import EVAL_SEED_BASE
    return [EVAL_SEED_BASE + i for i in range(episodes)]


def run_policy(env: gym.Env, seeds: list[int], choose, label: str, on_reset=None) -> dict:
    """choose(obs, info) -> (action, top1_top2_gap or None). Prints one summary line."""
    wins, lengths, foundations, gaps = 0, [], [], []
    t0 = time.time()
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        if on_reset:
            on_reset()
        n = 0
        while True:
            action, gap = choose(obs, info)
            if gap is not None:
                gaps.append(gap)
            obs, reward, terminated, truncated, info = env.step(action)
            n += 1
            if terminated or truncated:
                if terminated and not info.get("stalled") and int(obs["foundations"].sum()) == 52:
                    wins += 1
                break
        lengths.append(n)
        foundations.append(int(obs["foundations"].sum()))
    g = np.asarray(gaps) if gaps else None
    gap_txt = (
        f"  gap median {np.median(g):.2f}  frac<1: {(g < 1).mean():.0%}" if g is not None and len(g) else ""
    )
    print(
        f"{label:44s} win={wins / len(seeds):6.1%}  mean_len={np.mean(lengths):5.0f}  "
        f"mean_found={np.mean(foundations):4.1f}{gap_txt}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"label": label, "win_rate": wins / len(seeds), "mean_length": float(np.mean(lengths))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--eval-seed-source", default="runs/eval_corpus.jsonl", help="jsonl of {'seed': ...}; omit (pass '') for the fixed 900000+i range")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0], help="0 = pure argmax")
    parser.add_argument("--loop-breaker", choices=["both", "on", "off"], default="both")
    parser.add_argument("--heuristic", action="store_true", help="also score GreedyKlondikePolicy on the same deals")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()

    seeds = load_seeds(args.eval_seed_source or None, args.episodes)
    dev = args.device
    net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, hidden_dim=args.hidden_dim,
              num_hidden_layers=args.hidden_layers, layer_norm=args.layer_norm).to(dev)
    net.load_state_dict(torch.load(args.checkpoint, map_location=dev, weights_only=True))
    net.eval()
    gen = torch.Generator(device=dev).manual_seed(args.sample_seed)

    def make_chooser(temperature: float):
        def choose(obs, info):
            features, mask = preprocess(obs, info)
            with torch.no_grad():
                q = net(torch.from_numpy(features).to(dev), torch.from_numpy(mask).to(dev))[0]
            legal = torch.nonzero(q > ILLEGAL_Q_VALUE / 2).squeeze(1)
            ql = q[legal]
            gap = float(torch.topk(ql, 2).values.diff().abs()) if ql.numel() > 1 else None
            if temperature <= 0:
                return int(q.argmax()), gap
            probs = torch.softmax((ql - ql.max()) / temperature, 0)
            return int(legal[torch.multinomial(probs, 1, generator=gen)]), gap
        return choose

    wrappers = {"both": [True, False], "on": [True], "off": [False]}[args.loop_breaker]
    print(f"checkpoint {args.checkpoint}, {len(seeds)} deals")
    if args.heuristic:
        policy = GreedyKlondikePolicy()
        rng = np.random.default_rng(args.sample_seed)
        env = gym.make("Solitaire-v0")

        def choose_heuristic(obs, info):
            a = policy.select_action(env.unwrapped.game, np.flatnonzero(info["action_mask"]), rng)
            return (a if a is not None else 0), None

        run_policy(env, seeds, choose_heuristic, "heuristic (GreedyKlondikePolicy)", on_reset=policy.reset)
    for wrap in wrappers:
        env = LoopBreakerWrapper(gym.make("Solitaire-v0")) if wrap else gym.make("Solitaire-v0")
        for T in args.temperatures:
            rule = "argmax" if T <= 0 else f"softmax T={T:g}"
            run_policy(env, seeds, make_chooser(T), f"{rule}, loop-breaker {'ON ' if wrap else 'off'}")


if __name__ == "__main__":
    main()
