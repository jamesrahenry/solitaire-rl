"""Episode logging for Solitaire-v0.

Appends one JSON line per finished episode to a log file. Since the engine
is fully deterministic given a seed, logging just the seed and the action
sequence is enough to exactly replay any logged game later - no need to
store full observations at every step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym


class GameLogger(gym.Wrapper):
    def __init__(self, env: gym.Env, log_path: str = "logs/games.jsonl"):
        super().__init__(env)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._seed = None
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        self._flush("abandoned")
        obs, info = self.env.reset(seed=seed, options=options)
        self._seed = seed
        self._actions = []
        self._rewards = []
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._last_obs = obs
        if terminated:
            if info.get("stalled"):
                reason = "stalled"
            elif int(obs["foundations"].sum()) == 52:
                reason = "won"
            else:
                reason = "terminated"
            self._flush(reason)
        elif truncated:
            self._flush("truncated")
        return obs, reward, terminated, truncated, info

    def close(self):
        self._flush("abandoned")
        return self.env.close()

    def _flush(self, reason: str) -> None:
        if not self._actions:
            return
        record = {
            "time": time.time(),
            "seed": self._seed,
            "num_steps": len(self._actions),
            "actions": self._actions,
            "total_reward": sum(self._rewards),
            "final_foundation_total": (
                int(self._last_obs["foundations"].sum()) if self._last_obs is not None else None
            ),
            "reason": reason,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._actions = []
        self._rewards = []


def replay(env: gym.Env, record: dict):
    """Reset `env` to a logged episode's seed and replay its exact action
    sequence. Returns the final (obs, info). Useful for reproducing a
    specific logged game to debug it step by step."""
    obs, info = env.reset(seed=record["seed"])
    for action in record["actions"]:
        obs, reward, terminated, truncated, info = env.step(action)
    return obs, info
