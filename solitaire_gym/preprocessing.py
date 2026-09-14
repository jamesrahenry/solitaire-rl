"""
Feature extraction for feeding Solitaire-v0 observations to a PyTorch network.

Gymnasium hands back a structured Dict observation; PyTorch wants flat
tensors. This module strips the Dict apart into flat integer arrays, and
separates out the action mask so it can be applied directly during action
selection. Two encodings are available:

- extract_features / NUM_FEATURES: the original, raw-card-ID encoding (each
  card is a single arbitrary integer 0-51). Simple, but forces the network
  to learn from scratch that e.g. 7-of-spades and 7-of-clubs behave almost
  identically (same rank, same color - only foundation-building actually
  distinguishes them by suit).
- extract_decomposed_features / NUM_DECOMPOSED_FEATURES: decomposes each
  card slot into (slot_type, rank, color, suit) channels instead of one
  joint ID, so rank/color-dependent behavior (which is most of the game)
  doesn't have to be re-learned separately per suit.

Deliberately excludes stock_size/waste_size - those are card *counts*, not
card identities, so they don't fit the "flat array of raw card IDs" here.
Add them back in (and bump the relevant NUM_* constant) if the policy turns
out to need that signal.
"""
from __future__ import annotations

import numpy as np

from .cards import NUM_RANKS

FEATURE_DTYPE = np.int32

TABLEAU_FEATURES = 7 * 52
WASTE_TOP_FEATURES = 1
FOUNDATIONS_FEATURES = 4
NUM_FEATURES = TABLEAU_FEATURES + WASTE_TOP_FEATURES + FOUNDATIONS_FEATURES  # 369

# --- decomposed encoding ---------------------------------------------------

NUM_SLOTS = TABLEAU_FEATURES + WASTE_TOP_FEATURES  # 365 card "slots" (tableau cells + waste top)
NUM_DECOMPOSED_FEATURES = NUM_SLOTS * 4 + FOUNDATIONS_FEATURES  # 1464: 4 channels/slot + raw foundations

SLOT_EMPTY = 0
SLOT_HIDDEN = 1
SLOT_FACE_UP = 2


def extract_features(obs: dict) -> np.ndarray:
    """Flatten the tableau grid, waste-top card, and foundation counts into a
    single 1D array of shape (NUM_FEATURES,), dtype FEATURE_DTYPE."""
    tableau_flat = np.asarray(obs["tableau"], dtype=FEATURE_DTYPE).reshape(-1)
    waste_top = np.asarray([obs["waste_top"]], dtype=FEATURE_DTYPE)
    foundations = np.asarray(obs["foundations"], dtype=FEATURE_DTYPE)
    return np.concatenate([tableau_flat, waste_top, foundations])


def _slot_channels(card_ids: np.ndarray, empty_value: int, hidden_value: int | None) -> tuple[np.ndarray, ...]:
    """card_ids: raw values as encoded in the observation (already translated
    into the -N=empty/-N=hidden/0-51=card_id domain). Returns
    (slot_type, rank, color, suit) arrays, same shape as card_ids - rank,
    color and suit are 0 (a fixed dummy) wherever slot_type isn't face-up."""
    slot_type = np.full(card_ids.shape, SLOT_FACE_UP, dtype=FEATURE_DTYPE)
    slot_type[card_ids == empty_value] = SLOT_EMPTY
    if hidden_value is not None:
        slot_type[card_ids == hidden_value] = SLOT_HIDDEN

    face_up = slot_type == SLOT_FACE_UP
    rank = np.where(face_up, card_ids % NUM_RANKS, 0).astype(FEATURE_DTYPE)
    suit = np.where(face_up, card_ids // NUM_RANKS, 0).astype(FEATURE_DTYPE)
    color = np.where(face_up, suit % 2, 0).astype(FEATURE_DTYPE)
    return slot_type, rank, color, suit


def extract_decomposed_features(obs: dict) -> np.ndarray:
    """Flat array of shape (NUM_DECOMPOSED_FEATURES,): [slot_type(365),
    rank(365), color(365), suit(365), foundations(4)]. Foundations are left
    as raw counts (0-13), not decomposed - they're already a clean ordinal
    "progress" value, not an arbitrary ID needing this treatment."""
    tableau_flat = np.asarray(obs["tableau"], dtype=FEATURE_DTYPE).reshape(-1)
    tab_type, tab_rank, tab_color, tab_suit = _slot_channels(tableau_flat, empty_value=-2, hidden_value=-1)

    # waste_top uses a different raw convention (0=empty, else card_id+1) -
    # translate to the same "-1=empty, 0-51=card_id" domain _slot_channels expects.
    waste_raw = int(obs["waste_top"])
    waste_domain = np.array([waste_raw - 1], dtype=FEATURE_DTYPE)
    waste_type, waste_rank, waste_color, waste_suit = _slot_channels(waste_domain, empty_value=-1, hidden_value=None)

    slot_type = np.concatenate([tab_type, waste_type])
    rank = np.concatenate([tab_rank, waste_rank])
    color = np.concatenate([tab_color, waste_color])
    suit = np.concatenate([tab_suit, waste_suit])
    foundations = np.asarray(obs["foundations"], dtype=FEATURE_DTYPE)

    return np.concatenate([slot_type, rank, color, suit, foundations])


def extract_action_mask(info: dict) -> np.ndarray:
    """The legal-action boolean mask, pulled out of `info` for a matching
    entry point to extract_features - pass this straight to the policy for
    action masking."""
    return info["action_mask"]


def preprocess(obs: dict, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: (flat_features, action_mask) in one call, raw-ID encoding."""
    return extract_features(obs), extract_action_mask(info)


def preprocess_decomposed(obs: dict, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: (flat_features, action_mask) in one call, decomposed encoding."""
    return extract_decomposed_features(obs), extract_action_mask(info)
