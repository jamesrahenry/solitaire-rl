from gymnasium.envs.registration import register

from .env import SolitaireEnv

register(
    id="Solitaire-v0",
    entry_point="solitaire_gym.env:SolitaireEnv",
    # 500 was too tight: 19 of the 136 wins in runs/curriculum_wins.jsonl (14%)
    # already took >=480 steps, one hit exactly 500 - real, non-negligible
    # truncation risk for harder deals even under decent play, let alone a
    # still-learning or noisier policy.
    max_episode_steps=1000,
)

__all__ = ["SolitaireEnv"]
