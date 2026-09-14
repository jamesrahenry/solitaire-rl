"""Prioritized experience replay (Schaul et al., 2015).

Samples transitions proportional to their TD-error magnitude instead of
uniformly, so the rare transitions that actually matter (e.g. ones near a
win) get revisited far more than their natural 1-in-many frequency in the
buffer - which is exactly what uniform replay was failing to do (see run
002's flat 0% eval win rate despite stable loss). Non-uniform sampling
biases the gradient, so importance-sampling (IS) weights correct for it.

Uses a sum-tree for O(log n) priority-weighted sampling and O(log n)
priority updates, instead of an O(n) linear scan - necessary at this
buffer's scale (100k) to not dominate training time.

New transitions are inserted at the current max priority seen so far, so
they get sampled (and their real priority established from an actual
TD-error) soon after being added, instead of defaulting to ~0 priority and
being ignored until sampled by chance.
"""
from __future__ import annotations

import numpy as np


class SumTree:
    """Binary tree where each leaf holds one priority and each internal
    node holds the sum of its children's values - supports O(log n)
    "find the leaf whose cumulative range contains this value" queries and
    O(log n) priority updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def total(self) -> float:
        return self.tree[0]

    def get(self, cumulative_value: float) -> tuple[int, float]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumulative_value <= self.tree[left]:
                idx = left
            else:
                cumulative_value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, num_features: int, num_actions: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.features = np.zeros((capacity, num_features), dtype=np.int32)
        self.masks = np.zeros((capacity, num_actions), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_features = np.zeros((capacity, num_features), dtype=np.int32)
        self.next_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=bool)
        self.discounts = np.zeros(capacity, dtype=np.float32)

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

        self.tree.update(idx, self.max_priority**self.alpha)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, beta: float) -> tuple:
        """Stratified proportional sampling (as in the paper): split
        [0, total_priority) into `batch_size` equal segments and draw one
        sample from each, for lower-variance coverage than pure weighted
        random sampling. Returns (batch, indices, importance_sampling_weights)."""
        total = self.tree.total()
        segment = total / batch_size
        idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            v = rng.uniform(low, high)
            data_idx, priority = self.tree.get(v)
            idxs[i] = data_idx
            priorities[i] = priority

        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()  # normalize so the max weight is 1 (stability, not bias)

        batch = (
            self.features[idxs],
            self.masks[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_features[idxs],
            self.next_masks[idxs],
            self.dones[idxs],
            self.discounts[idxs],
        )
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(idxs, priorities):
            self.tree.update(int(idx), float(p))
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def __len__(self) -> int:
        return self.size
