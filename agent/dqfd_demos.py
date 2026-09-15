"""Build permanent demonstration transitions for DQfD (Hester et al., 2017)
from real logged wins - not the full heuristic-harvest corpus used for
behavior cloning, but wins specifically. DQfD's margin loss pushes Q(s,
a_expert) to dominate every other action's Q-value at that state by a
fixed margin, which only makes sense if a_expert is genuinely good; a
losing game's moves are exactly the kind of demonstration that loss
shouldn't be anchoring the network toward.

Replays each win through the real registered Solitaire-v0 env (so reward
comes from the actual, current reward function, not whatever it was when
the game was originally logged - reward shaping has changed several times
this project), validating each action's legality before applying it and
truncating an episode at the first point of divergence - the same
data-corruption fix behavior_cloning.py needed (see its module docstring)
applies here too, for the same reason: some logged wins came from
curriculum-enabled runs whose logged action list is missing its
fast-forwarded prefix.

Transitions are accumulated into n-step returns via
agent.nstep_buffer.NStepAccumulator, in exactly the same format
agent.train.py's own training loop produces, so they can be loaded directly
into a replay buffer's permanent demo region alongside self-generated
experience.
"""
from __future__ import annotations

import json
import random

import gymnasium as gym

import solitaire_gym  # noqa: F401  (registers Solitaire-v0)
from agent.nstep_buffer import NStepAccumulator


def build_demo_transitions(
    source: str,
    preprocess_fn,
    n_step: int,
    gamma: float,
    target_transitions: int,
    seed: int = 0,
) -> list[tuple]:
    """Reads win episodes from `source` (a games.jsonl-schema file), shuffles
    them, and replays enough to produce at least `target_transitions` n-step
    transitions (trimmed to exactly that many). Raises if the source doesn't
    contain enough winning episodes to reach the target."""
    with open(source) as f:
        records = [json.loads(line) for line in f]
    wins = [r for r in records if r.get("reason") == "won" and r.get("seed") is not None]
    random.Random(seed).shuffle(wins)

    env = gym.make("Solitaire-v0")
    transitions: list[tuple] = []
    truncated_episodes = 0
    used_episodes = 0

    for record in wins:
        if len(transitions) >= target_transitions:
            break
        used_episodes += 1
        # a fresh accumulator per episode: if this episode diverges partway
        # through (see below) and we break out without a real `done`, any
        # transition still pending in the accumulator's internal window must
        # be discarded, not carried over and finalized against the next
        # episode's unrelated trajectory - starting clean each episode is the
        # simplest way to guarantee that
        nstep = NStepAccumulator(n_step, gamma)
        obs, info = env.reset(seed=record["seed"])
        features, mask = preprocess_fn(obs, info)
        for action in record["actions"]:
            if not mask[action]:
                truncated_episodes += 1
                break
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_features, next_mask = preprocess_fn(next_obs, next_info)
            done = terminated or truncated
            for t in nstep.add(features, mask, action, reward, next_features, next_mask, done):
                transitions.append(t)
            features, mask, obs, info = next_features, next_mask, next_obs, next_info
            if done:
                break

    env.close()
    if len(transitions) < target_transitions:
        raise ValueError(
            f"only produced {len(transitions)} demo transitions from {len(wins)} available wins "
            f"({used_episodes} used, {truncated_episodes} diverged) - need {target_transitions}; "
            "provide a source with more wins or lower --num-demo-transitions"
        )
    print(
        f"built {target_transitions} demo transitions from {used_episodes} winning episodes "
        f"({truncated_episodes} diverged and were skipped)"
    )
    return transitions[:target_transitions]
