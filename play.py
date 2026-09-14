#!/usr/bin/env python3
"""Interactive terminal client for the Solitaire-v0 Gymnasium environment.

Drives the actual registered gym env (not a separate copy of the rules), so
what you play here is exactly what an agent trains against.

Usage:
    python play.py [--seed N]
"""
from __future__ import annotations

import argparse

import numpy as np
import gymnasium as gym

import solitaire_gym  # noqa: F401  (registers Solitaire-v0)
from solitaire_gym.cards import NUM_RANKS, card_str, color_of
from solitaire_gym.logging_wrapper import GameLogger
from solitaire_gym.game import (
    ACTION_DRAW,
    ACTION_TABLEAU_TO_FOUNDATION_START,
    ACTION_WASTE_TO_FOUNDATION,
    ACTION_WASTE_TO_TABLEAU_START,
    NUM_TABLEAU,
    foundation_move_action,
    tableau_move_action,
)

SUIT_LETTERS = "shcd"  # spades, hearts, clubs, diamonds

RED = "\033[91m"
RESET = "\033[0m"

HELP = """
Commands (all in <source> <destination> order):
  <enter> or d        draw a card from the stock
  w <col>              move the waste's top card onto tableau column <col> (1-7)
  w f                   move the waste's top card to its foundation
  <col> f               move tableau column <col>'s top card to its foundation
  <suit> <col>          move that suit's foundation top card onto tableau column <col>
                        (suits: s=spades h=hearts c=clubs d=diamonds)
  <src> <dst>           move tableau column <src>'s entire face-up run onto column <dst>
  <src> <n> <dst>       move just the top <n> face-up cards of column <src> onto column <dst>
  m                     show the legal-move mask (raw action ids)
  n / new               start a new game
  h / help / ?          show this help
  q / quit / exit       quit
""".strip(
    "\n"
)


def colorize(card_id: int) -> str:
    s = card_str(card_id)
    return f"{RED}{s}{RESET}" if color_of(card_id) == 1 else s


CELL_WIDTH = 4


def pad_cell(display: str, visible_len: int) -> str:
    """Pad `display` to CELL_WIDTH based on its *visible* length, since ANSI
    color escapes count toward str length but render as zero-width."""
    return display + " " * max(0, CELL_WIDTH - visible_len)


def render(game) -> str:
    rows = max((len(p) for p in game.tableau), default=0)
    lines = [" ".join(f"C{c + 1:<3}" for c in range(NUM_TABLEAU))]
    for r in range(rows):
        cells = []
        for pile in game.tableau:
            if r < len(pile):
                card, up = pile[r]
                plain = card_str(card) if up else "??"
                display = colorize(card) if up else plain
            else:
                plain = display = ""
            cells.append(pad_cell(display, len(plain)))
        lines.append(" ".join(cells))

    lines.append("")
    suits_order = "SHCD"
    found_cells = []
    for s in range(4):
        count = game.foundations[s]
        if count > 0:
            top_card = s * NUM_RANKS + (count - 1)
            plain = card_str(top_card)
            display = colorize(top_card)
        else:
            plain = display = "--"
        card_field = pad_cell(display, len(plain))
        found_cells.append(f"{suits_order[s]}:{card_field}({count:2d})")
    lines.append("Foundations  " + "  ".join(found_cells))
    waste = colorize(game.waste[-1]) if game.waste else "--"
    lines.append(
        f"Waste: {waste}    Stock remaining: {len(game.stock)}    Waste pile size: {len(game.waste)}"
    )
    return "\n".join(lines)


def _check_col(col: int, label: str) -> None:
    if not 0 <= col < NUM_TABLEAU:
        raise ValueError(f"{label} column must be between 1 and {NUM_TABLEAU}")


def parse_command(cmd: str, game) -> int | None:
    """Translate a typed command into an action id for env.step()."""
    parts = cmd.split()
    if not parts or (len(parts) == 1 and parts[0] == "d"):
        return ACTION_DRAW
    if parts[0] == "w" and len(parts) == 2:
        if parts[1] == "f":
            return ACTION_WASTE_TO_FOUNDATION
        col = int(parts[1]) - 1
        _check_col(col, "target")
        return ACTION_WASTE_TO_TABLEAU_START + col
    if len(parts) == 2 and parts[1] == "f":
        col = int(parts[0]) - 1
        _check_col(col, "source")
        return ACTION_TABLEAU_TO_FOUNDATION_START + col
    if len(parts) == 2 and parts[0] in SUIT_LETTERS:
        suit = SUIT_LETTERS.index(parts[0])
        try:
            dst = int(parts[1]) - 1
        except ValueError:
            raise ValueError(f"unrecognized command: {cmd!r} (commands are <source> <destination>)")
        _check_col(dst, "destination")
        return foundation_move_action(suit, dst)
    if len(parts) == 3:
        try:
            src, count, dst = int(parts[0]) - 1, int(parts[1]), int(parts[2]) - 1
        except ValueError:
            raise ValueError(f"unrecognized command: {cmd!r} (commands are <source> <count> <destination>)")
        _check_col(src, "source")
        _check_col(dst, "destination")
        if src == dst:
            raise ValueError("source and destination columns must differ")
        if count < 1:
            raise ValueError("card count must be at least 1")
        return tableau_move_action(src, dst, count)
    if len(parts) == 2:
        try:
            src, dst = int(parts[0]) - 1, int(parts[1]) - 1
        except ValueError:
            raise ValueError(f"unrecognized command: {cmd!r} (commands are <source> <destination>)")
        _check_col(src, "source")
        _check_col(dst, "destination")
        if src == dst:
            raise ValueError("source and destination columns must differ")
        count = game.face_up_run_length(src)
        if count == 0:
            raise ValueError(f"column {src + 1} has no face-up cards to move")
        return tableau_move_action(src, dst, count)
    raise ValueError(f"unrecognized command: {cmd!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive terminal client for Solitaire-v0")
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="deal seed, for reproducing a specific game"
    )
    args = parser.parse_args()
    seed = args.seed

    env = GameLogger(gym.make("Solitaire-v0"), log_path="logs/games.jsonl")
    obs, info = env.reset(seed=seed)
    game = env.unwrapped.game
    total_reward = 0.0

    print(HELP)
    print(f"(games are logged to {env.log_path})")
    while True:
        print()
        print(render(game))
        print(f"Score: {total_reward:.2f}")
        if not info["action_mask"].any():
            print("No legal moves remain — this deal is stuck. Type 'n' for a new game.")
        try:
            cmd = input("solitaire> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit", "exit"):
            break
        if cmd in ("h", "help", "?"):
            print(HELP)
            continue
        if cmd in ("n", "new"):
            obs, info = env.reset()
            total_reward = 0.0
            continue
        if cmd == "m":
            print("legal actions:", np.flatnonzero(info["action_mask"]).tolist())
            continue

        try:
            action = parse_command(cmd, game)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if not info["valid_action"]:
            print("Illegal move.")
        if terminated:
            print(render(game))
            if info["stalled"]:
                print(f"\nStuck — cycled the whole stock with no legal move. Final score: {total_reward:.2f}")
            else:
                print(f"\nYou win! Final score: {total_reward:.2f}")
            break
        if truncated:
            print(f"\nOut of moves (truncated). Final score: {total_reward:.2f}")
            break

    env.close()


if __name__ == "__main__":
    main()
