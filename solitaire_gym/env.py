from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .cards import card_str
from .game import (
    MAX_TABLEAU_LEN,
    NUM_ACTIONS,
    NUM_FOUNDATIONS,
    NUM_TABLEAU,
    SolitaireGame,
)

EMPTY_SLOT = -2
FACE_DOWN = -1


class SolitaireEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, allow_undo: bool = True):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.game = SolitaireGame(allow_undo=allow_undo)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "tableau": spaces.Box(
                    low=EMPTY_SLOT, high=51, shape=(NUM_TABLEAU, MAX_TABLEAU_LEN), dtype=np.int8
                ),
                "foundations": spaces.MultiDiscrete([14] * NUM_FOUNDATIONS),
                "waste_top": spaces.Discrete(53),  # 0 = empty, card_id + 1 otherwise
                "stock_size": spaces.Discrete(53),
                "waste_size": spaces.Discrete(53),
            }
        )

    def _get_obs(self) -> dict:
        tableau = np.full((NUM_TABLEAU, MAX_TABLEAU_LEN), EMPTY_SLOT, dtype=np.int8)
        for col, pile in enumerate(self.game.tableau):
            for row, (card, face_up) in enumerate(pile):
                tableau[col, row] = card if face_up else FACE_DOWN

        waste_top = (self.game.waste[-1] + 1) if self.game.waste else 0

        return {
            "tableau": tableau,
            "foundations": np.array(self.game.foundations, dtype=np.int64),
            "waste_top": waste_top,
            "stock_size": len(self.game.stock),
            "waste_size": len(self.game.waste),
        }

    def _get_info(self) -> dict:
        return {"action_mask": self.game.action_mask()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options={"initial_moves": [...]}: after dealing, fast-forward
        through this action sequence before returning control - e.g. to
        resume mid-way through a previously logged game (same seed + a
        prefix of its actions), for curriculum learning from real logged
        wins. Raises if a move turns out illegal, since that means the
        seed/actions don't actually match this deal rather than silently
        handing back a corrupted state."""
        super().reset(seed=seed)
        self.game.reset(self.np_random)
        initial_moves = (options or {}).get("initial_moves")
        if initial_moves:
            for action in initial_moves:
                if not self.game.is_legal(action):
                    raise ValueError(
                        f"initial_moves replay diverged: action {action} is illegal in this state "
                        "(wrong seed, or moves don't match this deal?)"
                    )
                self.game.apply(action)
        obs, info = self._get_obs(), self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        reward, valid = self.game.apply(int(action))
        stalled = self.game.is_stalled()
        terminated = self.game.is_won() or stalled
        truncated = False
        obs = self._get_obs()
        info = self._get_info()
        info["valid_action"] = valid
        info["stalled"] = stalled
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """SB3-contrib MaskablePPO convention: boolean mask over action_space."""
        return self.game.action_mask()

    # -- rendering -----------------------------------------------------

    def _render_text(self) -> str:
        lines = []
        rows = max((len(p) for p in self.game.tableau), default=0)
        header = " ".join(f"C{c + 1:<2}" for c in range(NUM_TABLEAU))
        lines.append(header)
        for r in range(rows):
            cells = []
            for pile in self.game.tableau:
                if r < len(pile):
                    card, up = pile[r]
                    cells.append(card_str(card) if up else "??")
                else:
                    cells.append("  ")
            lines.append(" ".join(f"{c:<3}" for c in cells))

        suits = "SHCD"
        found = " ".join(
            f"{suits[s]}:{self.game.foundations[s]:<2}" for s in range(NUM_FOUNDATIONS)
        )
        lines.append("")
        lines.append(f"Foundations  {found}")
        waste = card_str(self.game.waste[-1]) if self.game.waste else "--"
        lines.append(f"Waste: {waste}   Stock: {len(self.game.stock)}   Waste pile: {len(self.game.waste)}")
        return "\n".join(lines)

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "human":
            print(self._render_text())
            print()
            return None
        return None
