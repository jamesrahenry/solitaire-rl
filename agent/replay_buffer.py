"""Fixed-capacity, preallocated-numpy replay buffer for DQN training.

Transitions store an n-step return (not necessarily a single-step reward)
plus a per-transition `discount` - the effective gamma^k to apply to the
bootstrap term - since an n-step window can end up shorter than n when it
runs into a real episode end (see agent.nstep_buffer.NStepAccumulator).
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, num_features: int, num_actions: int):
        self.capacity = capacity
        self.features = np.zeros((capacity, num_features), dtype=np.int32)
        self.masks = np.zeros((capacity, num_actions), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)  # n-step (or shorter) discounted return
        self.next_features = np.zeros((capacity, num_features), dtype=np.int32)
        self.next_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=bool)
        self.discounts = np.zeros(capacity, dtype=np.float32)  # gamma^k for the bootstrap term
        self.pos = 0
        self.size = 0

    def add(self, features, mask, action, reward, next_features, next_mask, done, discount) -> None:
        idx = self.pos
        self.features[idx] = features
        self.masks[idx] = mask
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_features[idx] = next_features
        self.next_masks[idx] = next_mask
        self.dones[idx] = done
        self.discounts[idx] = discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple:
        idxs = rng.integers(0, self.size, size=batch_size)
        return (
            self.features[idxs],
            self.masks[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_features[idxs],
            self.next_masks[idxs],
            self.dones[idxs],
            self.discounts[idxs],
        )

    def __len__(self) -> int:
        return self.size
