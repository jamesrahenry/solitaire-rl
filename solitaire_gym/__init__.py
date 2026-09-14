from gymnasium.envs.registration import register

from .env import SolitaireEnv

register(
    id="Solitaire-v0",
    entry_point="solitaire_gym.env:SolitaireEnv",
    max_episode_steps=500,
)

__all__ = ["SolitaireEnv"]
