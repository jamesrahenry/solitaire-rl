#!/usr/bin/env python3
"""Grow a persistent corpus of seeds proven solvable (by the ground-truth
ShootMe/Klondike-Solver, see agent/klondike_solver_check.py) in fewer than
--max-moves moves - a much stronger label than the heuristic-policy-won
curriculum corpus (runs/curriculum_wins.jsonl), since it's a proof, not a
lucky rollout, and a move-count ceiling well under our agent's 1000-step
episode budget leaves headroom for the agent's own inefficiency (illegal
attempts, backtracking, extra draws) on top of the solver's minimal line.

Candidates are drawn from a large, fixed-seed RNG stream distinct from both
the fixed 100-seed eval range (900000-900099) and the arbitrary large seeds
already used by the heuristic curriculum harvest - collision with either is
astronomically unlikely at this scale regardless, but excluding the eval
range explicitly keeps this corpus safely separate from anything used for
measurement.

For corpus-building we don't need to resolve every candidate to a definite
verdict the way the eval-set classification did - we only care about
accumulating good (solvable, short) seeds, so a modest state cap is used
and anything that doesn't resolve favorably within it is simply discarded,
no matter whether that's because it's actually unsolvable, needs a longer
solution, or just needs more search than we bothered to spend on a reject.

Resumable: candidate index position and accept/reject counts are persisted
in a sidecar state file so repeated invocations keep extending the same
corpus rather than re-testing seeds already tried.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from agent.klondike_solver_check import check_seed

EVAL_SEED_LO, EVAL_SEED_HI = 900_000, 900_100
MASTER_SEED = 20260916  # fixed, arbitrary - just needs to be stable across runs
CORPUS_PATH = Path("runs/solvable_corpus.jsonl")
STATE_PATH = Path("runs/solvable_corpus_state.json")

MOVE_RE = re.compile(r"(?:Minimal solution in|Solved in) (\d+) moves")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"next_index": 0, "accepted": 0, "rejected": 0}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def candidate_stream(start_index: int):
    """Deterministic seed sequence: draw start_index+1 values from a fixed
    RNG and yield from where we left off, skipping the reserved eval range."""
    rng = np.random.default_rng(MASTER_SEED)
    i = 0
    while True:
        seed = int(rng.integers(1, 2**31 - 1))
        if EVAL_SEED_LO <= seed < EVAL_SEED_HI:
            continue
        if i >= start_index:
            yield i, seed
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-candidates", type=int, default=200, help="how many new candidates to try this run")
    parser.add_argument("--target-corpus-size", type=int, default=None, help="stop early once the corpus reaches this size")
    parser.add_argument("--max-moves", type=int, default=250)
    parser.add_argument("--max-states", type=int, default=1_000_000, help="solver search cap per candidate - kept modest since rejects are just discarded, not proven unsolvable")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    state = load_state()
    existing_size = sum(1 for _ in CORPUS_PATH.open()) if CORPUS_PATH.exists() else 0
    print(f"resuming at candidate index {state['next_index']} ({existing_size} already in corpus, "
          f"{state['accepted']} accepted / {state['rejected']} rejected so far)")

    tried = 0
    t_start = time.time()
    with CORPUS_PATH.open("a") as corpus_f:
        for idx, seed in candidate_stream(state["next_index"]):
            if tried >= args.num_candidates:
                break
            if args.target_corpus_size is not None and existing_size >= args.target_corpus_size:
                print(f"reached target corpus size {args.target_corpus_size}, stopping")
                break

            verdict, out = check_seed(seed, max_states=args.max_states, timeout=args.timeout)
            tried += 1
            state["next_index"] = idx + 1

            if verdict == "SOLVABLE":
                m = MOVE_RE.search(out)
                moves = int(m.group(1)) if m else None
                if moves is not None and moves < args.max_moves:
                    record = {"seed": seed, "moves": moves}
                    corpus_f.write(json.dumps(record) + "\n")
                    corpus_f.flush()
                    existing_size += 1
                    state["accepted"] += 1
                    print(f"  [{tried}/{args.num_candidates}] seed {seed}: ACCEPTED ({moves} moves)", flush=True)
                    continue

            state["rejected"] += 1
            print(f"  [{tried}/{args.num_candidates}] seed {seed}: rejected ({verdict})", flush=True)

    save_state(state)
    elapsed = time.time() - t_start
    print(f"\ndone: tried {tried} candidates in {elapsed:.1f}s, corpus now has {existing_size} seeds "
          f"(lifetime {state['accepted']} accepted / {state['rejected']} rejected)")


if __name__ == "__main__":
    main()
