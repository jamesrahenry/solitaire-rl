"""
Klondike solitaire rules engine.

Refactored out of vendor/tteeoo-solitaire/solitaire.py (MIT License, Theo
Henson) into a pure, stateful game object with no I/O and no globals, so it
can be driven programmatically by a Gymnasium environment. The core rules
(alternating-color descending tableau sequences, kings-only-to-empty-column,
ace-up foundations, draw-1 stock/waste with recycling) match the original;
the REPL/rendering/terminal-input code has been removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cards import NUM_CARDS, NUM_RANKS, NUM_SUITS, color_of, rank_of, suit_of

NUM_TABLEAU = 7
NUM_FOUNDATIONS = 4
MAX_TABLEAU_LEN = NUM_CARDS  # safe upper bound; a column could in principle hold all 52

# Action layout:
#   0          : draw from stock (recycling waste if stock is empty)
#   1..7       : waste top card -> tableau column (action - 1)
#   8          : waste top card -> foundation
#   9..15      : tableau column (action - 9) top card -> foundation
#   16..561    : tableau column i -> tableau column j (i != j; 42 ordered pairs),
#                taking the top `count` face-up cards (count = 1..MAX_RUN_LEN),
#                i.e. any valid sub-run, not just the entire face-up stack.
#   562..589   : foundation suit's top card -> tableau column (28 = 4 suits x 7 columns)
ACTION_DRAW = 0
ACTION_WASTE_TO_TABLEAU_START = 1
ACTION_WASTE_TO_FOUNDATION = 8
ACTION_TABLEAU_TO_FOUNDATION_START = 9
ACTION_TABLEAU_TO_TABLEAU_START = 16

MAX_RUN_LEN = NUM_RANKS  # a valid alternating-descending run is at most King..Ace long

TABLEAU_MOVES = [(i, j) for i in range(NUM_TABLEAU) for j in range(NUM_TABLEAU) if i != j]
ACTION_FOUNDATION_TO_TABLEAU_START = ACTION_TABLEAU_TO_TABLEAU_START + len(TABLEAU_MOVES) * MAX_RUN_LEN
NUM_ACTIONS = ACTION_FOUNDATION_TO_TABLEAU_START + NUM_SUITS * NUM_TABLEAU
assert ACTION_FOUNDATION_TO_TABLEAU_START == 16 + 42 * 13 == 562
assert NUM_ACTIONS == 562 + 4 * 7 == 590


def tableau_move_action(src: int, dst: int, count: int) -> int:
    """Action id for moving the top `count` face-up cards of column `src`
    onto column `dst`. `count` is 1-indexed (1 = just the top/exposed card)."""
    move_idx = TABLEAU_MOVES.index((src, dst))
    return ACTION_TABLEAU_TO_TABLEAU_START + move_idx * MAX_RUN_LEN + (count - 1)


def foundation_move_action(suit: int, dst: int) -> int:
    """Action id for moving suit `suit`'s foundation top card onto column `dst`."""
    return ACTION_FOUNDATION_TO_TABLEAU_START + suit * NUM_TABLEAU + dst


REWARD_NEW_FOUNDATION_HIGH = 10.0  # only for exceeding a suit's high-water mark, not any foundation move
REWARD_REVEAL = 50.0                # flipping a previously hidden tableau card face-up
REWARD_KING_ON_EMPTY = 40.0          # a king lands on an empty tableau column, first time for that column only
REWARD_COLUMN_UNCOVERED = 30.0       # a column's last hidden card is revealed, first time for that column only
REWARD_COLUMN_EMPTIED = 20.0          # a column reaches zero cards, first time for that column only (groundwork for a future king move there)
REWARD_ILLEGAL = -10.0
REWARD_STEP = -0.01
REWARD_WIN_BONUS = 5000.0  # deliberately dominant: a full clearing already banks ~2050 from the other milestone bonuses above, so a same-order-of-magnitude win bonus gave weak marginal incentive to actually finish vs. stopping just short


@dataclass
class SolitaireGame:
    tableau: list = field(default_factory=list)   # 7 lists of [card_id, face_up]
    foundations: list = field(default_factory=list)  # 4 ints, count placed per suit (0..13)
    foundation_high_water: list = field(default_factory=list)  # 4 ints, best-ever count per suit (never decreases)
    king_on_empty: list = field(default_factory=list)      # 4 bools, indexed by suit - whether *that king* has ever triggered its bonus (not per-column, so a single king can't farm it by touring distinct never-before-used empty columns)
    column_uncovered: list = field(default_factory=list)   # 7 bools, whether this column has ever gotten a fully-uncovered (0 hidden cards) bonus
    column_emptied: list = field(default_factory=list)      # 7 bools, whether this column has ever gotten a fully-emptied (0 total cards) bonus
    stock: list = field(default_factory=list)      # draw pile, stock[-1] is next to draw
    waste: list = field(default_factory=list)      # waste[-1] is the visible top card
    draws_since_progress: int = 0  # consecutive draws with no other move in between
    allow_undo: bool = True  # if False, foundation->tableau is permanently illegal (not reset per-episode)
    foundation_undo_penalty: float = 0.0  # subtracted on every foundation->tableau move; 0 = old behavior (free reversal)

    def reset(self, np_random: np.random.Generator) -> None:
        deck = list(range(NUM_CARDS))
        np_random.shuffle(deck)

        self.tableau = [[] for _ in range(NUM_TABLEAU)]
        for col in range(NUM_TABLEAU):
            for row in range(col + 1):
                card = deck.pop()
                self.tableau[col].append([card, row == col])

        self.foundations = [0] * NUM_FOUNDATIONS
        self.foundation_high_water = [0] * NUM_FOUNDATIONS
        self.king_on_empty = [False] * NUM_FOUNDATIONS
        # column 0 is dealt with 0 hidden cards (already fully uncovered) - pre-flag it
        # so its first reveal-adjacent move doesn't fire a spurious bonus. No column is
        # dealt empty, though, so column_emptied starts all-False with no exceptions.
        self.column_uncovered = [self._tableau_run_start(col) == 0 for col in range(NUM_TABLEAU)]
        self.column_emptied = [False] * NUM_TABLEAU
        self.stock = deck  # remaining cards, stock[-1] drawn first
        self.waste = []
        self.draws_since_progress = 0

    # -- helpers -------------------------------------------------------

    def _flip_top(self, col: int) -> bool:
        """Flip a newly exposed top card face-up if it was hidden. Returns
        True iff a previously face-down card was just revealed."""
        if self.tableau[col] and not self.tableau[col][-1][1]:
            self.tableau[col][-1][1] = True
            return True
        return False

    def _add_to_foundation(self, card: int) -> float:
        """Place `card` on its suit's foundation and return the reward for
        doing so: +1 only if this sets a new high-water mark for the suit,
        0 if it's just re-reaching a level already banked once before."""
        suit = suit_of(card)
        self.foundations[suit] += 1
        if self.foundations[suit] > self.foundation_high_water[suit]:
            self.foundation_high_water[suit] = self.foundations[suit]
            return REWARD_NEW_FOUNDATION_HIGH
        return 0.0

    def _reveal_bonus(self, col: int) -> float:
        """Call right after a successful _flip_top(col). Returns
        REWARD_COLUMN_UNCOVERED the first time this column has zero hidden
        cards left, 0.0 otherwise."""
        if not self.column_uncovered[col] and self._tableau_run_start(col) == 0:
            self.column_uncovered[col] = True
            return REWARD_COLUMN_UNCOVERED
        return 0.0

    def _king_on_empty_bonus(self, suit: int, was_empty: bool) -> float:
        """Call after placing a run onto a tableau column, passing the *suit
        of the card that landed on the (possibly empty) column*. `was_empty`
        must be captured before the move - only an empty column accepts a
        king (rule-enforced), so no rank check is needed here. Gated per-king
        (per suit), not per-column: each of the 4 kings pays out once ever,
        the first time it lands on an empty column, no matter which column -
        gating by column instead would let one king tour every never-before-
        used empty column and collect the bonus repeatedly."""
        if was_empty and not self.king_on_empty[suit]:
            self.king_on_empty[suit] = True
            return REWARD_KING_ON_EMPTY
        return 0.0

    def _column_emptied_bonus(self, col: int) -> float:
        """Call right after a move that may have removed a column's last
        card. Returns REWARD_COLUMN_EMPTIED the first time this column
        reaches zero total cards (distinct from column_uncovered, which
        tracks zero *hidden* cards - a column can be uncovered long before
        it's ever emptied, or emptied without ever having had a hidden card
        to begin with, like column 0)."""
        if not self.tableau[col] and not self.column_emptied[col]:
            self.column_emptied[col] = True
            return REWARD_COLUMN_EMPTIED
        return 0.0

    def _tableau_run_start(self, col: int) -> int:
        """Index of the first face-up card in the (contiguous) face-up run at
        the bottom of the column, or len(column) if the column is empty."""
        pile = self.tableau[col]
        i = len(pile)
        while i > 0 and pile[i - 1][1]:
            i -= 1
        return i

    def face_up_run_length(self, col: int) -> int:
        """How many face-up cards sit at the top of this column (0 if empty
        or its top card is face-down)."""
        return len(self.tableau[col]) - self._tableau_run_start(col)

    def is_won(self) -> bool:
        return sum(self.foundations) == NUM_CARDS

    def is_stalled(self) -> bool:
        """True once we've cycled all the way through the current stock+waste
        pool via pure draws with no intervening move *and* the resulting
        state - including whatever that final draw just revealed - genuinely
        has no legal move left besides drawing again. The draw count alone
        isn't sufficient: the very last draw of a cycle can reveal a waste-top
        card that's immediately playable (e.g. straight to a foundation), and
        that has to actually be checked, not just counted past - otherwise
        the episode can end on the exact step that hands the agent a legal
        move, before it ever gets a chance to take it."""
        pool = len(self.stock) + len(self.waste)
        if pool == 0 or self.draws_since_progress < pool:
            return False
        return not any(self.is_legal(a) for a in range(NUM_ACTIONS) if a != ACTION_DRAW)

    # -- legality --------------------------------------------------------

    def _legal_waste_to_tableau(self, col: int) -> bool:
        if not self.waste:
            return False
        card = self.waste[-1]
        target = self.tableau[col]
        if not target:
            return rank_of(card) == NUM_RANKS - 1  # king
        top_card, top_up = target[-1]
        return top_up and color_of(card) != color_of(top_card) and rank_of(card) == rank_of(top_card) - 1

    def _legal_waste_to_foundation(self) -> bool:
        if not self.waste:
            return False
        card = self.waste[-1]
        return rank_of(card) == self.foundations[suit_of(card)]

    def _legal_tableau_to_foundation(self, col: int) -> bool:
        if not self.tableau[col]:
            return False
        card, up = self.tableau[col][-1]
        if not up:
            return False
        return rank_of(card) == self.foundations[suit_of(card)]

    def _legal_foundation_to_tableau(self, suit: int, dst: int) -> bool:
        if self.foundations[suit] == 0:
            return False
        card = suit * NUM_RANKS + (self.foundations[suit] - 1)
        target = self.tableau[dst]
        if not target:
            return rank_of(card) == NUM_RANKS - 1  # king
        top_card, top_up = target[-1]
        return top_up and color_of(card) != color_of(top_card) and rank_of(card) == rank_of(top_card) - 1

    def _legal_tableau_to_tableau(self, src: int, dst: int, count: int) -> bool:
        if src == dst or count < 1:
            return False
        run_len = self.face_up_run_length(src)
        if count > run_len:
            return False
        split = len(self.tableau[src]) - count
        moving_card = self.tableau[src][split][0]
        target = self.tableau[dst]
        if not target:
            return rank_of(moving_card) == NUM_RANKS - 1  # king
        top_card, top_up = target[-1]
        return top_up and color_of(moving_card) != color_of(top_card) and rank_of(moving_card) == rank_of(top_card) - 1

    def is_legal(self, action: int) -> bool:
        if action == ACTION_DRAW:
            return bool(self.stock) or bool(self.waste)
        if ACTION_WASTE_TO_TABLEAU_START <= action < ACTION_WASTE_TO_TABLEAU_START + NUM_TABLEAU:
            return self._legal_waste_to_tableau(action - ACTION_WASTE_TO_TABLEAU_START)
        if action == ACTION_WASTE_TO_FOUNDATION:
            return self._legal_waste_to_foundation()
        if ACTION_TABLEAU_TO_FOUNDATION_START <= action < ACTION_TABLEAU_TO_FOUNDATION_START + NUM_TABLEAU:
            return self._legal_tableau_to_foundation(action - ACTION_TABLEAU_TO_FOUNDATION_START)
        if ACTION_TABLEAU_TO_TABLEAU_START <= action < ACTION_FOUNDATION_TO_TABLEAU_START:
            move_idx, count = divmod(action - ACTION_TABLEAU_TO_TABLEAU_START, MAX_RUN_LEN)
            src, dst = TABLEAU_MOVES[move_idx]
            return self._legal_tableau_to_tableau(src, dst, count + 1)
        if ACTION_FOUNDATION_TO_TABLEAU_START <= action < NUM_ACTIONS:
            if not self.allow_undo:
                return False
            suit, dst = divmod(action - ACTION_FOUNDATION_TO_TABLEAU_START, NUM_TABLEAU)
            return self._legal_foundation_to_tableau(suit, dst)
        raise ValueError(f"invalid action id {action}")

    def action_mask(self) -> np.ndarray:
        return np.array([self.is_legal(a) for a in range(NUM_ACTIONS)], dtype=bool)

    # -- apply -------------------------------------------------------------

    def apply(self, action: int) -> tuple[float, bool]:
        """Apply an action. Returns (reward, was_legal)."""
        if not self.is_legal(action):
            return REWARD_ILLEGAL, False

        reward = REWARD_STEP

        if action == ACTION_DRAW:
            self.draws_since_progress += 1
            if not self.stock:
                self.stock = self.waste[::-1]
                self.waste = []
            self.waste.append(self.stock.pop())

        elif ACTION_WASTE_TO_TABLEAU_START <= action < ACTION_WASTE_TO_TABLEAU_START + NUM_TABLEAU:
            self.draws_since_progress = 0
            col = action - ACTION_WASTE_TO_TABLEAU_START
            was_empty = not self.tableau[col]
            card = self.waste.pop()
            self.tableau[col].append([card, True])
            reward += self._king_on_empty_bonus(suit_of(card), was_empty)

        elif action == ACTION_WASTE_TO_FOUNDATION:
            self.draws_since_progress = 0
            card = self.waste.pop()
            reward += self._add_to_foundation(card)

        elif ACTION_TABLEAU_TO_FOUNDATION_START <= action < ACTION_TABLEAU_TO_FOUNDATION_START + NUM_TABLEAU:
            self.draws_since_progress = 0
            col = action - ACTION_TABLEAU_TO_FOUNDATION_START
            card, _ = self.tableau[col].pop()
            reward += self._add_to_foundation(card)
            reward += self._column_emptied_bonus(col)
            if self._flip_top(col):
                reward += REWARD_REVEAL
                reward += self._reveal_bonus(col)

        elif action < ACTION_FOUNDATION_TO_TABLEAU_START:  # tableau -> tableau
            self.draws_since_progress = 0
            move_idx, count = divmod(action - ACTION_TABLEAU_TO_TABLEAU_START, MAX_RUN_LEN)
            src, dst = TABLEAU_MOVES[move_idx]
            count += 1
            was_empty = not self.tableau[dst]
            split = len(self.tableau[src]) - count
            run = self.tableau[src][split:]
            del self.tableau[src][split:]
            self.tableau[dst].extend(run)
            reward += self._king_on_empty_bonus(suit_of(run[0][0]), was_empty)
            reward += self._column_emptied_bonus(src)
            if self._flip_top(src):
                reward += REWARD_REVEAL
                reward += self._reveal_bonus(src)

        else:  # foundation -> tableau (undo); does not lower the high-water mark
            self.draws_since_progress = 0
            suit, dst = divmod(action - ACTION_FOUNDATION_TO_TABLEAU_START, NUM_TABLEAU)
            was_empty = not self.tableau[dst]
            self.foundations[suit] -= 1
            card = suit * NUM_RANKS + self.foundations[suit]
            self.tableau[dst].append([card, True])
            reward += self._king_on_empty_bonus(suit, was_empty)
            reward -= self.foundation_undo_penalty

        if self.is_won():
            reward += REWARD_WIN_BONUS
        elif self.is_stalled():
            # provable dead end: neutral terminal reward, not a punishment -
            # the agent already banked whatever dense rewards it earned
            # getting here (reveals, foundation progress); we just stop
            # paying it to spin the stock pile forever.
            reward = 0.0

        return reward, True
