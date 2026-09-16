#!/usr/bin/env python3
"""Check ground-truth (full-information) solvability of specific deals using
the external ShootMe/Klondike-Solver binary (github.com/ShootMe/Klondike-Solver,
cloned+built locally, not part of this repo) - not reimplementing a solver
ourselves, just converting our own already-known deal into its documented
deck-string input format and reading back its verdict.

The binary's /DECK string is "the order a deck of cards is dealt to the
board" - i.e. the real physical round-robin deal (1 card to each of piles
1..7, then 1 card to piles 2..7, ..., then 1 card to pile 7, remainder to
stock), NOT the order our own engine happens to internally pop() cards in
(which deals one column fully before moving to the next - a different but
equally valid way to realize the same post-shuffle arrangement). We don't
need to replicate our engine's dealing loop at all here - we just read the
final tableau/stock arrangement straight off a reset env and re-express it
in the solver's dealing convention. tableau[col][r-1] is "whichever card
this column received on physical round r", independent of what order our
own code happened to assign it in.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gymnasium as gym

import solitaire_gym  # noqa: F401
from solitaire_gym.cards import NUM_RANKS

SOLVER_BIN = Path("/home/jhenry/Source/external/Klondike-Solver/KlondikeSolver")
SUIT_TO_SOLVER = {0: "4", 1: "3", 2: "1", 3: "2"}  # our SUITS="SHCD" -> solver's 1=C 2=D 3=H 4=S


def card_to_solver(card: int) -> str:
    rank = card % NUM_RANKS  # 0=A..12=K
    suit = card // NUM_RANKS  # 0=S,1=H,2=C,3=D
    return f"{rank + 1:02d}{SUIT_TO_SOLVER[suit]}"


def deal_to_deck_string(game) -> str:
    tokens = []
    for r in range(1, 8):  # physical round 1..7
        for col in range(r - 1, 7):
            tokens.append(card_to_solver(game.tableau[col][r - 1][0]))
    # stock[-1] is drawn first in our engine; the solver's remainder-of-string
    # convention (verified against TestDeals.txt's known deal) reads left to
    # right as next-to-be-drawn first, so reverse our stock (stored bottom to
    # top, i.e. stock[-1] is the top/next-drawn) into that order.
    for card in reversed(game.stock):
        tokens.append(card_to_solver(card))
    return "".join(tokens)


def check_seed(seed: int, max_states: int = 3_000_000, timeout: int = 120) -> tuple[str, str]:
    env = gym.make("Solitaire-v0")
    env.reset(seed=seed)
    game = env.unwrapped.game
    deck_str = deal_to_deck_string(game)
    env.close()
    assert len(deck_str) == 156, f"expected 156 chars, got {len(deck_str)}"

    try:
        result = subprocess.run(
            [str(SOLVER_BIN), "/D", deck_str, "/DC", "1", "/S", str(max_states)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    out = result.stdout
    if "Minimal solution" in out or "Solved in" in out:
        verdict = "SOLVABLE"
    elif "Impossible." in out:
        verdict = "UNSOLVABLE"
    elif "Unknown." in out:
        verdict = "INCONCLUSIVE (raise --S / max-states)"
    else:
        verdict = "UNKNOWN (unrecognized output)"
    return verdict, out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    verdict, out = check_seed(seed)
    print(f"seed {seed}: {verdict}")
    print(out)
