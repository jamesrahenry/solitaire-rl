#!/usr/bin/env python3
"""Re-evaluate every checkpoint of an already-finished run against a
different eval seed source than the one it was trained with - to separate
"the policy actually got better" from "the eval battery reads differently
by construction" when comparing two runs that used different eval pools.

Reuses agent.train.evaluate() directly (not a reimplementation) so the
scoring logic is guaranteed identical to whatever a live training run would
have computed with the same --eval-seed-source/--eval-episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, preprocess_decomposed, NUM_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, DecomposedDQN
from agent.loop_breaker import LoopBreakerWrapper
from agent.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--eval-seed-source", required=True)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--out", default=None, help="path to write per-checkpoint results json (default: <run_dir>/reeval_<basename of eval-seed-source>.json)")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    encoding = config.get("card_encoding", "raw")
    hidden_layers = config.get("hidden_layers", 2)
    hidden_dim = config.get("hidden_dim", 512)
    allow_undo = config.get("allow_undo", True)
    loop_breaker = config.get("loop_breaker", True)
    loop_breaker_threshold = config.get("loop_breaker_threshold", 2)
    layer_norm = config.get("layer_norm", False)

    if encoding == "decomposed":
        preprocess_fn = preprocess_decomposed
        net = DecomposedDQN(num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim, layer_norm=layer_norm)
    else:
        preprocess_fn = preprocess
        net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=hidden_layers, hidden_dim=hidden_dim, layer_norm=layer_norm)

    with open(args.eval_seed_source) as f:
        eval_seeds = [json.loads(line)["seed"] for line in f]
    print(f"loaded {len(eval_seeds)} eval seeds from {args.eval_seed_source}, using first {args.eval_episodes}")

    eval_env = gym.make("Solitaire-v0", allow_undo=allow_undo)
    if loop_breaker:
        eval_env = LoopBreakerWrapper(eval_env, threshold=loop_breaker_threshold)

    checkpoints = sorted(
        args.run_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 10**9,
    )
    checkpoints = [p for p in checkpoints if p.stem.split("_")[1].isdigit()]

    results = []
    for ckpt in checkpoints:
        step = int(ckpt.stem.split("_")[1])
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        net.eval()
        stats = evaluate(net, eval_env, args.eval_episodes, "cpu", preprocess_fn, eval_seeds=eval_seeds)
        stats["step"] = step
        results.append(stats)
        print(f"step {step:>8}: win_rate={stats['win_rate']*100:5.2f}%  mean_foundation={stats['mean_foundation']:5.1f}/52", flush=True)

    out_path = Path(args.out) if args.out else args.run_dir / f"reeval_{Path(args.eval_seed_source).stem}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
