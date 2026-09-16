"""Breaks repeating-state loops in the action mask, on the agent's side -
not a change to the environment's own legal-move rules (a human playing
via play.py, or anything not wrapped in this, sees ordinary Klondike
legality unchanged).

Motivated by a concrete failure mode found by hand: several checkpoints
from run 020, evaluated greedily on the same deal that a nearby checkpoint
did solve, got stuck oscillating between two tableau moves (C1<->C2) for
the entire 1000-step budget, never making further progress. This is the
same structural failure as an earlier-diagnosed undo/redo loop (foundation
<-> tableau) - two actions with near-tied Q-values and nothing to break the
tie once the policy enters the cycle.

This is sound to intervene on, not just a heuristic guess: Klondike, once
dealt, has no further randomness (draws are deterministic given the fixed
stock order, and hidden cards' identities never change, only their
visibility) - so if the exact same full game state ever recurs within an
episode, revisiting it can provably never reach anywhere the earlier visit
couldn't already reach. That's the same reasoning solitaire_gym.game's own
is_stalled() already relies on for "cycled the whole stock with no
progress"; this generalizes it to any exact state repeat, not just the
stock-cycle case.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


def _state_key(game) -> tuple:
    return (
        tuple(tuple((card, face_up) for card, face_up in col) for col in game.tableau),
        tuple(game.foundations),
        tuple(game.stock),
        tuple(game.waste),
    )


class LoopBreakerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, threshold: int = 2):
        """threshold: how many prior actions must already be recorded from a
        state before its mask gets restricted - 2 means the 3rd visit to
        that exact state is the first one that gets intervened on."""
        super().__init__(env)
        self.threshold = threshold
        self._visited_actions: dict[tuple, list[int]] = {}

    def _adjust(self, info: dict) -> dict:
        key = _state_key(self.unwrapped.game)
        prior = self._visited_actions.get(key, [])
        if len(prior) >= self.threshold:
            restricted = info["action_mask"].copy()
            restricted[prior] = False
            if restricted.any():  # never hand back an all-False mask - a genuine dead end is is_stalled's job, not ours
                info["action_mask"] = restricted
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited_actions = {}
        info = self._adjust(info)
        return obs, info

    def step(self, action):
        pre_key = _state_key(self.unwrapped.game)
        self._visited_actions.setdefault(pre_key, []).append(int(action))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._adjust(info)
        return obs, reward, terminated, truncated, info
