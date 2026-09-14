"""Rule-based greedy Klondike policy for harvesting real wins.

Adapted from a heuristic sketched in discussion with Gemini (pseudocode
only - it assumed a different env API and had no implementation of the
actual decision logic). Reimplemented here against our real action-ID
encoding (solitaire_gym.game) and engine (SolitaireGame).

Priority order: reveal a hidden card > send a card to the foundation >
move a king onto an empty column > consolidate the tableau > draw as a
last resort. Never uses the foundation->tableau undo move, so it can't
oscillate the way undirected epsilon-greedy exploration does (which is
exactly what made harvesting via random/epsilon play find zero wins
across ~2000 episodes earlier in this project).

Not meant to be an optimal solver - some deals it could win with a bit of
lookahead (e.g. holding a card back from the foundation) it will fail, since
it never reconsiders a foundation move once made. For harvesting purposes
(get real wins to seed curriculum learning) that's an acceptable tradeoff.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from solitaire_gym.cards import rank_of
from solitaire_gym.game import (
    ACTION_DRAW,
    ACTION_FOUNDATION_TO_TABLEAU_START,
    ACTION_TABLEAU_TO_FOUNDATION_START,
    ACTION_TABLEAU_TO_TABLEAU_START,
    ACTION_WASTE_TO_FOUNDATION,
    ACTION_WASTE_TO_TABLEAU_START,
    MAX_RUN_LEN,
    NUM_RANKS,
    NUM_TABLEAU,
    SolitaireGame,
    TABLEAU_MOVES,
    tableau_move_action,
)

SCORE_REVEAL = 100
SCORE_FOUNDATION = 75
SCORE_KING_TO_EMPTY = 50
SCORE_CONSOLIDATE = 25  # tableau->tableau or waste->tableau moves that aren't a king-to-empty
SCORE_DRAW = 1


def is_undo_action(action: int) -> bool:
    return action >= ACTION_FOUNDATION_TO_TABLEAU_START


def is_draw_action(action: int) -> bool:
    return action == ACTION_DRAW


def is_waste_to_tableau(action: int) -> bool:
    return ACTION_WASTE_TO_TABLEAU_START <= action < ACTION_WASTE_TO_TABLEAU_START + NUM_TABLEAU


def is_foundation_move(action: int) -> bool:
    return action == ACTION_WASTE_TO_FOUNDATION or (
        ACTION_TABLEAU_TO_FOUNDATION_START <= action < ACTION_TABLEAU_TO_FOUNDATION_START + NUM_TABLEAU
    )


def is_tableau_to_tableau(action: int) -> bool:
    return ACTION_TABLEAU_TO_TABLEAU_START <= action < ACTION_FOUNDATION_TO_TABLEAU_START


def decode_tableau_move(action: int) -> tuple[int, int, int]:
    """Returns (src, dst, count) for a tableau->tableau action."""
    move_idx, count = divmod(action - ACTION_TABLEAU_TO_TABLEAU_START, MAX_RUN_LEN)
    src, dst = TABLEAU_MOVES[move_idx]
    return src, dst, count + 1


def is_king_to_empty_column(game: SolitaireGame, action: int) -> bool:
    """True for a waste->tableau or tableau->tableau move that places a king
    onto a currently-empty column."""
    if is_waste_to_tableau(action):
        dst = action - ACTION_WASTE_TO_TABLEAU_START
        if game.tableau[dst] or not game.waste:
            return False
        return rank_of(game.waste[-1]) == NUM_RANKS - 1
    if is_tableau_to_tableau(action):
        src, dst, count = decode_tableau_move(action)
        if game.tableau[dst]:
            return False
        split = len(game.tableau[src]) - count
        moving_card = game.tableau[src][split][0]
        return rank_of(moving_card) == NUM_RANKS - 1
    return False


def reveals_hidden_card(game: SolitaireGame, action: int) -> bool:
    """True if this action would flip a previously face-down tableau card
    face-up - i.e. it removes the *entire* current face-up run from a
    column that still has a face-down card underneath."""
    if ACTION_TABLEAU_TO_FOUNDATION_START <= action < ACTION_TABLEAU_TO_FOUNDATION_START + NUM_TABLEAU:
        col = action - ACTION_TABLEAU_TO_FOUNDATION_START
        return game.face_up_run_length(col) == 1 and len(game.tableau[col]) > 1
    if is_tableau_to_tableau(action):
        src, _, count = decode_tableau_move(action)
        return count == game.face_up_run_length(src) and len(game.tableau[src]) > count
    return False


def score_action(game: SolitaireGame, action: int) -> int:
    if reveals_hidden_card(game, action):
        return SCORE_REVEAL
    if is_foundation_move(action):
        return SCORE_FOUNDATION
    if is_king_to_empty_column(game, action):
        return SCORE_KING_TO_EMPTY
    if is_tableau_to_tableau(action) or is_waste_to_tableau(action):
        return SCORE_CONSOLIDATE
    if is_draw_action(action):
        return SCORE_DRAW
    return 0


def _reverse_of(action: int) -> int | None:
    """The action that would immediately undo a tableau->tableau move (same
    cards, opposite direction) - None for action types that have no such
    reversal (or aren't reversible in one step)."""
    if not is_tableau_to_tableau(action):
        return None
    src, dst, count = decode_tableau_move(action)
    return tableau_move_action(dst, src, count)


def greedy_klondike_policy(
    game: SolitaireGame, legal_actions: np.ndarray, rng: np.random.Generator, last_action: int | None = None
) -> int | None:
    """Stateless single-step scoring: highest-scoring legal action, ties
    broken randomly, excluding the exact reversal of `last_action` within
    the top tier if any other option ties for that score. Only breaks
    length-2 oscillation (A<->B) - use GreedyKlondikePolicy for anything
    that plays a full episode, since longer cycles (verified this happens:
    a length-4 loop the length-2 check can't see) need real state-history
    tracking. Returns None if the only legal actions are undo moves."""
    forward_actions = np.array([a for a in legal_actions if not is_undo_action(a)])
    if len(forward_actions) == 0:
        return None

    reverse_of_last = _reverse_of(last_action) if last_action is not None else None
    scores = np.array([score_action(game, a) for a in forward_actions])

    for s in sorted(set(scores.tolist()), reverse=True):
        tier = forward_actions[scores == s]
        non_reverse = tier[tier != reverse_of_last]
        if len(non_reverse) > 0:
            return int(rng.choice(non_reverse))

    # every legal forward action is exactly the reversal (shouldn't normally
    # happen) - allow it rather than deadlocking outright
    return int(rng.choice(forward_actions))


class GreedyKlondikePolicy:
    """Stateful version for playing a full episode: tracks recent board
    states and detects when it's stuck cycling through a repeating set of
    them (pure greedy scoring has no memory, so *any* loop of moves that
    are each locally "the best available option" traps it forever - not
    just the length-2 case greedy_klondike_policy() guards against).  When
    stuck, perturbs out with a few random legal moves instead of the
    top-scoring choice."""

    def __init__(self, history_len: int = 16, stuck_threshold: int = 2, perturb_steps: int = 3):
        self.history_len = history_len
        self.stuck_threshold = stuck_threshold
        self.perturb_steps = perturb_steps
        self.reset()

    def reset(self) -> None:
        self._state_history: deque = deque(maxlen=self.history_len)
        self._last_action: int | None = None
        self._repeat_count = 0
        self._perturb_remaining = 0

    @staticmethod
    def _state_key(game: SolitaireGame):
        tableau_key = tuple(tuple((c, u) for c, u in pile) for pile in game.tableau)
        return (tableau_key, tuple(game.foundations), tuple(game.waste), tuple(game.stock))

    def select_action(self, game: SolitaireGame, legal_actions: np.ndarray, rng: np.random.Generator) -> int | None:
        forward_actions = np.array([a for a in legal_actions if not is_undo_action(a)])
        if len(forward_actions) == 0:
            return None

        key = self._state_key(game)
        self._repeat_count = self._repeat_count + 1 if key in self._state_history else 0
        self._state_history.append(key)
        if self._repeat_count >= self.stuck_threshold:
            self._perturb_remaining = self.perturb_steps

        if self._perturb_remaining > 0:
            self._perturb_remaining -= 1
            action = int(rng.choice(forward_actions))
        else:
            reverse_of_last = _reverse_of(self._last_action) if self._last_action is not None else None
            scores = np.array([score_action(game, a) for a in forward_actions])
            action = None
            for s in sorted(set(scores.tolist()), reverse=True):
                tier = forward_actions[scores == s]
                non_reverse = tier[tier != reverse_of_last]
                if len(non_reverse) > 0:
                    action = int(rng.choice(non_reverse))
                    break
            if action is None:
                action = int(rng.choice(forward_actions))

        self._last_action = action
        return action
