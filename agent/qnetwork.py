"""
Q-networks for DQN training on Solitaire-v0.

DQN: consumes the flat integer feature vector from
solitaire_gym.preprocessing.extract_features (raw card IDs / counts - one
integer per tableau cell, plus waste-top card, plus per-suit foundation
counts) through a single joint embedding table, and an action-legality
mask, outputting Q-values over the full action space with illegal actions
masked out.

DecomposedDQN: same idea, but consumes
solitaire_gym.preprocessing.extract_decomposed_features - each card slot
split into (slot_type, rank, color, suit) channels, each with its own
embedding, instead of one joint raw-ID embedding. Gives the network an
explicit structural hint that e.g. 7-of-spades and 7-of-clubs are identical
for every tableau/waste purpose (only foundation-building distinguishes
them by suit) - see also agent.symmetry for a complementary data-side fix
(suit-swap augmentation) for the same underlying redundancy.
"""
from __future__ import annotations

import torch
from torch import nn

from solitaire_gym.cards import NUM_RANKS, NUM_SUITS
from solitaire_gym.game import NUM_ACTIONS
from solitaire_gym.preprocessing import (
    FOUNDATIONS_FEATURES,
    NUM_DECOMPOSED_FEATURES,
    NUM_FEATURES,
    NUM_SLOTS,
)

# Raw feature values range from -2 (empty tableau slot) to 52 (a card on the
# waste, encoded as card_id + 1, max card_id 51). Embedding indices must be
# non-negative, so we shift every raw value by this offset before lookup.
VOCAB_OFFSET = 2
VOCAB_SIZE = 55  # raw values -2..52 inclusive -> 55 distinct values

ILLEGAL_Q_VALUE = -1e9


class DQN(nn.Module):
    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_actions: int = NUM_ACTIONS,
        embedding_dim: int = 16,
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        layers: list[nn.Module] = [nn.Linear(num_features * embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: raw feature values straight from solitaire_gym.preprocessing.extract_features
           (NOT pre-offset) - shape (num_features,) or (batch, num_features), any integer dtype.
        mask: legal-action boolean mask (True = legal) - shape (num_actions,) or
              (batch, num_actions), matching x's batching.
        Returns: Q-values, same batch shape as input, illegal actions set to -1e9.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        indices = x.long() + VOCAB_OFFSET
        embedded = self.embedding(indices)              # (batch, num_features, embedding_dim)
        flattened = embedded.flatten(start_dim=1)        # (batch, num_features * embedding_dim)
        q_values = self.net(flattened)                   # (batch, num_actions)
        q_values = q_values.masked_fill_(~mask.bool(), ILLEGAL_Q_VALUE)
        return q_values


NUM_SLOT_TYPES = 3  # empty, hidden, face-up


class DecomposedDQN(nn.Module):
    def __init__(
        self,
        num_slots: int = NUM_SLOTS,
        num_foundations: int = FOUNDATIONS_FEATURES,
        num_actions: int = NUM_ACTIONS,
        slot_type_dim: int = 4,
        rank_dim: int = 8,
        color_dim: int = 2,
        suit_dim: int = 4,
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        if num_slots == NUM_SLOTS and num_foundations == FOUNDATIONS_FEATURES:
            assert num_slots * 4 + num_foundations == NUM_DECOMPOSED_FEATURES
        self.num_slots = num_slots
        self.num_foundations = num_foundations
        self.slot_type_embed = nn.Embedding(NUM_SLOT_TYPES, slot_type_dim)
        self.rank_embed = nn.Embedding(NUM_RANKS, rank_dim)
        self.color_embed = nn.Embedding(2, color_dim)
        self.suit_embed = nn.Embedding(NUM_SUITS, suit_dim)

        per_slot_dim = slot_type_dim + rank_dim + color_dim + suit_dim
        input_dim = num_slots * per_slot_dim + num_foundations
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: raw feature values straight from
           solitaire_gym.preprocessing.extract_decomposed_features - layout
           [slot_type(num_slots), rank(num_slots), color(num_slots),
           suit(num_slots), foundations(num_foundations)] - shape
           (num_features,) or (batch, num_features), any integer dtype.
        mask: legal-action boolean mask, shape (num_actions,) or (batch, num_actions).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        n = self.num_slots
        slot_type = x[:, 0:n].long()
        rank = x[:, n : 2 * n].long()
        color = x[:, 2 * n : 3 * n].long()
        suit = x[:, 3 * n : 4 * n].long()
        foundations = x[:, 4 * n : 4 * n + self.num_foundations].float()

        per_slot = torch.cat(
            [
                self.slot_type_embed(slot_type),
                self.rank_embed(rank),
                self.color_embed(color),
                self.suit_embed(suit),
            ],
            dim=-1,
        )  # (batch, num_slots, per_slot_dim)
        flattened = per_slot.flatten(start_dim=1)  # (batch, num_slots * per_slot_dim)
        combined = torch.cat([flattened, foundations], dim=1)
        q_values = self.net(combined)
        q_values = q_values.masked_fill_(~mask.bool(), ILLEGAL_Q_VALUE)
        return q_values
