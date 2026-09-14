#!/usr/bin/env python3
"""Append any won human games from logs/games.jsonl (played via play.py) into
the shared training corpus (runs/curriculum_wins.jsonl), used by both
curriculum learning (agent/curriculum.py) and behavior cloning
(agent/behavior_cloning.py). Both files use the same {seed, actions, reason,
...} schema (both are written by solitaire_gym.logging_wrapper.GameLogger),
so a human win drops in exactly like a harvested one.

Idempotent - skips wins already present (matched by exact seed+action
sequence), so it's safe to re-run any time after playing more games.

Usage:
    python -m agent.sync_human_wins
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="logs/games.jsonl", help="human play log (play.py's GameLogger output)")
    parser.add_argument("--corpus", default="runs/curriculum_wins.jsonl", help="shared curriculum/BC training corpus")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    existing = []
    if corpus_path.exists():
        with corpus_path.open() as f:
            existing = [json.loads(line) for line in f]
    seen = {(r["seed"], tuple(r["actions"])) for r in existing}

    with open(args.source) as f:
        candidates = [json.loads(line) for line in f]
    new_wins = [
        r
        for r in candidates
        if r.get("reason") == "won"
        and r.get("seed") is not None
        and (r["seed"], tuple(r["actions"])) not in seen
    ]

    if not new_wins:
        print(f"no new human wins to add ({len(existing)} already in {args.corpus})")
        return

    with corpus_path.open("a") as f:
        for r in new_wins:
            f.write(json.dumps(r) + "\n")
    print(f"added {len(new_wins)} new human win(s) to {args.corpus} (was {len(existing)}, now {len(existing) + len(new_wins)})")
    for r in new_wins:
        print(f"  seed={r['seed']} num_steps={r['num_steps']} total_reward={r['total_reward']:.2f}")


if __name__ == "__main__":
    main()
