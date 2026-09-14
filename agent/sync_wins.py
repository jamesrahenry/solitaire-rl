#!/usr/bin/env python3
"""Sweep any games.jsonl-schema log (human play via play.py, or a training
run's own per-episode log) for won episodes and fold the new ones into the
shared training corpus (runs/curriculum_wins.jsonl), used by both curriculum
learning (agent/curriculum.py) and behavior cloning
(agent/behavior_cloning.py). All of these logs share the same {seed,
actions, reason, ...} schema (all written by
solitaire_gym.logging_wrapper.GameLogger), so a win from any source drops in
exactly like a harvested one.

Idempotent - skips wins already present (matched by exact seed+action
sequence), so it's safe to re-run any time: after playing more games, after
a training run finishes (or periodically while one is still going), etc.
Note training runs don't do this automatically mid-run - agent/train.py
loads the curriculum corpus once before its loop starts and never refreshes
it, so a run's own organically-discovered wins only reach the shared corpus
(and only benefit *future* runs) if this script is run afterward.

Usage:
    python -m agent.sync_wins                          # human play log + every run's games.jsonl
    python -m agent.sync_wins --source runs/013_*/games.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

DEFAULT_SOURCES = ["logs/games.jsonl", "runs/*/games.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=f"glob pattern to sweep for wins (repeatable); default sweeps {DEFAULT_SOURCES}",
    )
    parser.add_argument("--corpus", default="runs/curriculum_wins.jsonl", help="shared curriculum/BC training corpus")
    args = parser.parse_args()
    sources = args.source or DEFAULT_SOURCES

    corpus_path = Path(args.corpus)
    existing = []
    if corpus_path.exists():
        with corpus_path.open() as f:
            existing = [json.loads(line) for line in f]
    seen = {(r["seed"], tuple(r["actions"])) for r in existing}

    candidates = []
    for pattern in sources:
        for path in glob.glob(pattern):
            if Path(path).resolve() == corpus_path.resolve():
                continue  # don't re-scan the corpus itself if a pattern happens to match it
            with open(path) as f:
                candidates.extend(json.loads(line) for line in f)

    new_wins = [
        r
        for r in candidates
        if r.get("reason") == "won"
        and r.get("seed") is not None
        and (r["seed"], tuple(r["actions"])) not in seen
    ]
    # a win could be logged identically in more than one matched file (e.g. a
    # resumed run appending to the same games.jsonl) - dedupe within this batch too
    deduped, batch_seen = [], set()
    for r in new_wins:
        key = (r["seed"], tuple(r["actions"]))
        if key not in batch_seen:
            batch_seen.add(key)
            deduped.append(r)
    new_wins = deduped

    if not new_wins:
        print(f"no new wins to add ({len(existing)} already in {args.corpus})")
        return

    with corpus_path.open("a") as f:
        for r in new_wins:
            f.write(json.dumps(r) + "\n")
    print(f"added {len(new_wins)} new win(s) to {args.corpus} (was {len(existing)}, now {len(existing) + len(new_wins)})")
    for r in new_wins:
        print(f"  seed={r['seed']} num_steps={r['num_steps']} total_reward={r['total_reward']:.2f}")


if __name__ == "__main__":
    main()
