"""Suit-swap symmetry augmentation: swapping Spades<->Clubs and Hearts<->
Diamonds everywhere in a transition produces an equally valid, equally
likely game (rules only ever depend on rank+color for tableau/waste
sequencing - suit only distinguishes which foundation a card belongs on).
Storing both the real transition and its suit-swapped twin in the replay
buffer doubles usable data for free and directly teaches the network
suit-invariance, complementing agent.qnetwork.DecomposedDQN's structural
version of the same fix.

Operates on POST-preprocessing flat feature arrays (works with either
solitaire_gym.preprocessing encoding) plus the action id and legal-action
mask - not on the raw Dict observation, so it drops in after
extract_features()/extract_decomposed_features() regardless of which is active.
"""
from __future__ import annotations

import numpy as np

from solitaire_gym.game import ACTION_FOUNDATION_TO_TABLEAU_START, NUM_TABLEAU, foundation_move_action
from solitaire_gym.preprocessing import FEATURE_DTYPE, NUM_SLOTS, SLOT_FACE_UP, TABLEAU_FEATURES

# Spades<->Clubs, Hearts<->Diamonds (suit ids per solitaire_gym.cards: S=0,H=1,C=2,D=3)
SUIT_SWAP = np.array([2, 3, 0, 1])


def swap_action(action: int) -> int:
    """Only the foundation->tableau (undo) actions encode a specific suit in
    their id - every other action (draw, waste/tableau moves) is suit-blind
    by construction and passes through unchanged."""
    if action < ACTION_FOUNDATION_TO_TABLEAU_START:
        return action
    idx = action - ACTION_FOUNDATION_TO_TABLEAU_START
    suit, dst = divmod(idx, NUM_TABLEAU)
    return foundation_move_action(int(SUIT_SWAP[suit]), dst)


def swap_mask(mask: np.ndarray) -> np.ndarray:
    swapped = mask.copy()
    for suit in range(4):
        for dst in range(NUM_TABLEAU):
            orig = foundation_move_action(suit, dst)
            new = foundation_move_action(int(SUIT_SWAP[suit]), dst)
            swapped[new] = mask[orig]
    return swapped


def _swap_card_id(card_ids: np.ndarray) -> np.ndarray:
    """card_ids: raw card ids (0-51) or negative sentinels (left untouched)."""
    real = card_ids >= 0
    suit = card_ids // 13
    rank = card_ids % 13
    swapped_suit = SUIT_SWAP[suit % 4]  # %4 guard is harmless for real ids (0-3 already); avoids OOB on sentinels
    return np.where(real, swapped_suit * 13 + rank, card_ids).astype(card_ids.dtype)


def _swap_foundations(foundations: np.ndarray) -> np.ndarray:
    return foundations[..., [2, 3, 0, 1]]


def swap_features_raw(features: np.ndarray) -> np.ndarray:
    """Suit-swap a flat array from solitaire_gym.preprocessing.extract_features:
    [tableau(364), waste_top(1), foundations(4)]."""
    tableau = features[:TABLEAU_FEATURES]
    waste_top = features[TABLEAU_FEATURES : TABLEAU_FEATURES + 1]
    foundations = features[TABLEAU_FEATURES + 1 :]

    swapped_tableau = _swap_card_id(tableau)
    # waste_top: 0=empty, else card_id+1
    waste_card = waste_top - 1
    swapped_waste_card = _swap_card_id(waste_card)
    swapped_waste_top = np.where(waste_top > 0, swapped_waste_card + 1, waste_top)
    swapped_foundations = _swap_foundations(foundations)

    return np.concatenate([swapped_tableau, swapped_waste_top, swapped_foundations]).astype(FEATURE_DTYPE)


def swap_features_decomposed(features: np.ndarray) -> np.ndarray:
    """Suit-swap a flat array from
    solitaire_gym.preprocessing.extract_decomposed_features: [slot_type(365),
    rank(365), color(365), suit(365), foundations(4)]. slot_type/rank/color
    are all suit-invariant by construction - only the suit channel (and only
    where a real card sits, i.e. slot_type == face-up) and foundations change."""
    n = NUM_SLOTS
    slot_type = features[0:n]
    rank = features[n : 2 * n]
    color = features[2 * n : 3 * n]
    suit = features[3 * n : 4 * n]
    foundations = features[4 * n :]

    face_up = slot_type == SLOT_FACE_UP
    swapped_suit = np.where(face_up, SUIT_SWAP[suit % 4], suit).astype(FEATURE_DTYPE)
    swapped_foundations = _swap_foundations(foundations)

    return np.concatenate([slot_type, rank, color, swapped_suit, swapped_foundations]).astype(FEATURE_DTYPE)
