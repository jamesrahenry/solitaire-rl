"""N-step return accumulation for DQN.

Sits between the environment loop and the replay buffer: takes raw 1-step
transitions one at a time, and emits n-step transitions once enough reward
has accumulated (or the episode ends early, in which case it flushes
whatever's left as shorter-than-n returns - there's no bootstrap needed past
a real episode end anyway).

Speeds up credit assignment versus pure 1-step TD: a completed n-step
transition already carries n real, unbootstrapped rewards summed together,
so value doesn't have to slowly diffuse backward one step at a time through
repeated bootstrapped updates - it's the standard fix for exactly the kind
of long-horizon credit assignment problem Solitaire has.
"""
from __future__ import annotations

from collections import deque


class NStepAccumulator:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer: deque = deque()

    def add(self, features, mask, action, reward, next_features, next_mask, done) -> list[tuple]:
        """Add one raw 1-step transition. Returns a list of completed n-step
        transitions (features, mask, action, n_step_return, next_features,
        next_mask, done, discount) ready for the replay buffer - normally 0
        or 1 of them, but several in a row when an episode just ended and
        the tail of the buffer gets flushed."""
        self.buffer.append((features, mask, action, reward, next_features, next_mask, done))
        completed = []

        if done:
            while self.buffer:
                completed.append(self._make_transition())
                self.buffer.popleft()
        elif len(self.buffer) >= self.n:
            completed.append(self._make_transition())
            self.buffer.popleft()

        return completed

    def _make_transition(self) -> tuple:
        features0, mask0, action0 = self.buffer[0][0], self.buffer[0][1], self.buffer[0][2]
        n_step_return = 0.0
        discount = 1.0
        final_next_features = final_next_mask = None
        final_done = False
        for _, _, _, reward, next_features, next_mask, done in self.buffer:
            n_step_return += discount * reward
            final_next_features, final_next_mask, final_done = next_features, next_mask, done
            discount *= self.gamma
            if done:
                break
        return features0, mask0, action0, n_step_return, final_next_features, final_next_mask, final_done, discount
