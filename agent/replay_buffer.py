"""Fixed-capacity, preallocated-numpy replay buffer for DQN training.

Transitions store an n-step return (not necessarily a single-step reward)
plus a per-transition `discount` - the effective gamma^k to apply to the
bootstrap term - since an n-step window can end up shorter than n when it
runs into a real episode end (see agent.nstep_buffer.NStepAccumulator).
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, num_features: int, num_actions: int, num_demo: int = 0):
        """num_demo reserves the first `num_demo` slots as a permanent region
        for demonstration transitions (DQfD, Hester et al. 2017): populated
        once via load_demos() and never overwritten by self-play add() calls,
        which write into the remaining [num_demo, capacity) region instead."""
        self.capacity = capacity
        self.num_demo = num_demo
        self.features = np.zeros((capacity, num_features), dtype=np.int32)
        self.masks = np.zeros((capacity, num_actions), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)  # n-step (or shorter) discounted return
        self.next_features = np.zeros((capacity, num_features), dtype=np.int32)
        self.next_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=bool)
        self.discounts = np.zeros(capacity, dtype=np.float32)  # gamma^k for the bootstrap term
        self.is_demo = np.zeros(capacity, dtype=bool)
        self.pos = 0  # self-play write pointer, relative to (offset by) num_demo
        self.size = 0

    def load_demos(self, transitions: list[tuple]) -> None:
        """Populate the permanent demo region [0, num_demo) directly. Call
        once, before any add() calls, with exactly num_demo transitions."""
        assert len(transitions) == self.num_demo, f"expected exactly {self.num_demo} demo transitions, got {len(transitions)}"
        for idx, (features, mask, action, reward, next_features, next_mask, done, discount) in enumerate(transitions):
            self.features[idx] = features
            self.masks[idx] = mask
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_features[idx] = next_features
            self.next_masks[idx] = next_mask
            self.dones[idx] = done
            self.discounts[idx] = discount
            self.is_demo[idx] = True
        self.size = max(self.size, self.num_demo)

    def add(self, features, mask, action, reward, next_features, next_mask, done, discount) -> None:
        idx = self.num_demo + self.pos
        self.features[idx] = features
        self.masks[idx] = mask
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_features[idx] = next_features
        self.next_masks[idx] = next_mask
        self.dones[idx] = done
        self.discounts[idx] = discount
        self.pos = (self.pos + 1) % (self.capacity - self.num_demo)
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
            self.is_demo[idxs],
        )

    def __len__(self) -> int:
        return self.size
