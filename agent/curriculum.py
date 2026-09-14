"""Curriculum learning from real logged wins: start some training episodes
a few moves from a known win (trivial - guarantees the network actually
experiences success) and gradually push the starting point earlier, back
toward full fresh deals, as training progresses.

Built on solitaire_gym.env.SolitaireEnv.reset(options={"initial_moves": [...]})
and GameLogger's {seed, actions} records - a win entry with a real
(non-None) seed can be replayed exactly, then truncated to "everything but
the last `tail_steps` moves" to produce a near-completion starting state.
"""
from __future__ import annotations

import glob
import json


def load_wins(paths: list[str]) -> list[dict]:
    """Scan games.jsonl files (glob patterns allowed) for won episodes with
    a real, reproducible seed (not the pre-fix `seed: None` records)."""
    wins = []
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path) as f:
                for line in f:
                    record = json.loads(line)
                    if record.get("reason") == "won" and record.get("seed") is not None:
                        wins.append(record)
    return wins


def tail_steps_by_step(step: int, start_tail: int, end_tail: int, anneal_steps: int) -> int:
    """Curriculum schedule: how many of the win's final moves are left for
    the agent to actually play (the rest get fast-forwarded through).
    Starts small (easy - right next to the win) and grows over training
    (harder - starting further back), capped at end_tail."""
    frac = min(1.0, step / anneal_steps)
    return int(start_tail + frac * (end_tail - start_tail))


def make_initial_moves(win_record: dict, tail_steps: int) -> tuple[int, list[int]]:
    """Returns (seed, initial_moves) to fast-forward through via
    env.reset(seed=seed, options={"initial_moves": initial_moves}), leaving
    the last `tail_steps` actions of the recorded win for the agent itself
    to play (0 initial_moves if tail_steps already covers the whole game)."""
    actions = win_record["actions"]
    prefix_len = max(0, len(actions) - tail_steps)
    return win_record["seed"], actions[:prefix_len]
