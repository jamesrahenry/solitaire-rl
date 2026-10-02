#!/usr/bin/env python3
"""DQN training loop for Solitaire-v0.

Double DQN: epsilon-greedy behavior policy (choosing only among legal
actions), a target network with periodic hard updates, experience replay,
and Huber loss on the Bellman error. Action masking is handled by the
network itself (agent.qnetwork.DQN masks illegal actions to -1e9), so
argmax over its output is always a legal action.

Double DQN specifically: the online network SELECTS the best next action
(argmax), the target network EVALUATES it. Plain DQN uses the target
network for both, which systematically overestimates Q-values (max over
noisy estimates is itself a biased estimate of the max) - this is what
caused run 001's loss to diverge. Decoupling selection from evaluation is
the standard fix.

N-step returns (agent.nstep_buffer.NStepAccumulator): transitions pushed
into the replay buffer carry n real, summed-and-discounted rewards before
any bootstrapping happens, instead of just 1 - this speeds up credit
assignment across Solitaire's long horizons instead of relying purely on
value slowly diffusing backward through repeated 1-step bootstrapped
updates. Set --n-step 1 to recover plain 1-step TD.

Usage:
    python -m agent.train --tag double_dqn [--steps N] [--device cuda|cpu] ...
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

import solitaire_gym
from solitaire_gym.preprocessing import preprocess, preprocess_decomposed, NUM_FEATURES, NUM_DECOMPOSED_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from solitaire_gym.logging_wrapper import GameLogger
from agent.qnetwork import DQN, DecomposedDQN
from agent.symmetry import swap_action, swap_mask, swap_features_raw, swap_features_decomposed
from agent.replay_buffer import ReplayBuffer
from agent.prioritized_replay_buffer import PrioritizedReplayBuffer
from agent.nstep_buffer import NStepAccumulator
from agent.metrics import MetricsLogger, EVAL_CSV_FIELDS, plot_eval
from agent.structure_metrics import structure_fields, compute_structure_row
from agent.run_utils import next_run_dir, save_config, write_readme_start, finalize_readme, TeeLogger
from agent.curriculum import load_wins, tail_steps_by_step, make_initial_moves
from agent.dqfd_demos import build_demo_transitions
from agent.loop_breaker import LoopBreakerWrapper


def epsilon_by_step(
    step: int,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    reheats: list[tuple[int, float, int]] | None = None,
) -> float:
    """Linear decay from eps_start to eps_end over decay_steps. `reheats` is
    an optional list of (reheat_step, reheat_eps, reheat_decay_steps)
    triples - at each reheat_step, epsilon jumps to that reheat_eps (however
    low it had already decayed) and linearly decays back down to eps_end
    over the following reheat_decay_steps, re-triggering exploration-driven
    discovery after the normal decay has settled at its floor. Multiple
    reheats are supported (e.g. a second, smaller "echo" reheat later in
    training) - the most recent one whose step has been reached applies."""
    active = None
    for reheat_step, reheat_eps, reheat_decay_steps in reheats or []:
        if step >= reheat_step:
            active = (reheat_step, reheat_eps, reheat_decay_steps)
    if active is not None:
        reheat_step, reheat_eps, reheat_decay_steps = active
        frac = min(1.0, (step - reheat_step) / reheat_decay_steps)
        return reheat_eps + frac * (eps_end - reheat_eps)
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def beta_by_step(step: int, beta_start: float, beta_end: float, total_steps: int) -> float:
    """Anneal the prioritized-replay importance-sampling correction from
    beta_start toward beta_end (usually 1.0) over the course of training -
    the bias correction matters most once the policy is closer to done."""
    frac = min(1.0, step / total_steps)
    return beta_start + frac * (beta_end - beta_start)


def soft_update(target_net: DQN, online_net: DQN, tau: float) -> None:
    """Polyak averaging: target <- tau * online + (1 - tau) * target, applied
    every training step instead of periodically overwriting the target
    wholesale. Smooths out the bootstrap target instead of yanking it to
    match the online net's latest (possibly inflated) estimates all at once."""
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)


EVAL_SEED_BASE = 900_000  # far outside any plausible training seed, and fixed across calls for comparability


def evaluate(net: DQN, env: gym.Env, episodes: int, device: str, preprocess_fn=preprocess, eval_seeds: list[int] | None = None, gamma: float = 0.99) -> dict:
    """Run `episodes` full episodes with the policy acting purely greedily
    (epsilon=0, no exploration) against a fixed set of held-out deals, so
    results are directly comparable across evaluation calls at different
    points in training. This is the clean read on whether the *learned
    policy* can win on its own - separate from whatever exploration noise
    is mixed into the training-time win rate.

    eval_seeds, when given, replaces the default fixed EVAL_SEED_BASE+i
    range with an explicit list (e.g. a much larger ground-truth-solvable
    battery) - the first `episodes` of them are used, cycling if `episodes`
    exceeds the list length.

    Also reports a value-calibration pair: q0 = the greedy action's Q-value
    at the deal, and g0 = the discounted (gamma) return the episode then
    actually realized from that same point. A well-calibrated Q-network has
    q0 ~ g0; run 029's final checkpoints read q0 ~ 36 against g0 ~ 630,
    which is what led to the TD-loss change (--td-loss) - win rate alone
    can't surface a 20x value-scale error."""
    net.eval()
    returns, lengths, foundations, q0s, g0s = [], [], [], [], []
    wins = 0
    with torch.no_grad():
        for i in range(episodes):
            seed = eval_seeds[i % len(eval_seeds)] if eval_seeds else EVAL_SEED_BASE + i
            obs, info = env.reset(seed=seed)
            features, mask = preprocess_fn(obs, info)
            ep_return = 0.0
            ep_len = 0
            disc_return = 0.0
            disc = 1.0
            while True:
                x = torch.from_numpy(features).to(device)
                m = torch.from_numpy(mask).to(device)
                q = net(x, m)
                action = int(q.argmax(dim=1).item())
                if ep_len == 0:
                    q0s.append(float(q[0, action].item()))
                obs, reward, terminated, truncated, info = env.step(action)
                features, mask = preprocess_fn(obs, info)
                ep_return += reward
                disc_return += disc * reward
                disc *= gamma
                ep_len += 1
                if terminated or truncated:
                    if terminated and not info.get("stalled") and int(obs["foundations"].sum()) == 52:
                        wins += 1
                    break
            returns.append(ep_return)
            lengths.append(ep_len)
            foundations.append(int(obs["foundations"].sum()))
            g0s.append(disc_return)
    return {
        "win_rate": wins / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_foundation": float(np.mean(foundations)),
        "mean_length": float(np.mean(lengths)),
        "q0": float(np.mean(q0s)),
        "g0": float(np.mean(g0s)),
    }


def select_action(net: DQN, features: np.ndarray, mask: np.ndarray, epsilon: float, rng: np.random.Generator, device: str) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(np.flatnonzero(mask)))
    with torch.no_grad():
        x = torch.from_numpy(features).to(device)
        m = torch.from_numpy(mask).to(device)
        q = net(x, m)
        return int(q.argmax(dim=1).item())


def compute_loss(
    net: DQN,
    target_net: DQN,
    batch: tuple,
    device: str,
    double_dqn: bool = True,
    is_weights: torch.Tensor | None = None,
    margin: float = 0.8,
    margin_loss_weight: float = 1.0,
    td_loss: str = "huber",
    huber_beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, per_sample_td_error). td_error is detached and only
    used to refresh priorities in the prioritized replay buffer; it's
    ignored entirely when using plain uniform replay.

    td_loss: "huber" (smooth L1 with transition point huber_beta - runs
    001-031 used beta=1.0) or "mse". With beta=1 and rewards of +50/+5000,
    every reward-bearing transition sits in Huber's linear region, where the
    per-sample gradient is exactly +-1 regardless of the error's size - so
    the loss behaves as median regression and the reward *magnitude* never
    reaches the network, only its frequency. The resulting fixed point is
    roughly Q ~ p / ((1-p)(1-gamma)) for p = fraction of transitions with
    |r| > 1, which predicts the Q ~ 7 (vs realized returns ~600) measured on
    run 029. MSE (or a beta at the reward scale) restores mean regression,
    so a 50-point reveal and a 10-point foundation move pull Q by different
    amounts. The gradient-norm clip in the training loop still bounds the
    per-batch update size either way.

    If any transitions in the batch are marked is_demo (DQfD, Hester et al.
    2017 - permanent demonstration transitions in the replay buffer), adds
    a supervised large-margin classification loss for those rows only:
    max_a[Q(s,a) + margin(a, a_E)] - Q(s, a_E), where margin(a_E, a_E) = 0
    and margin(a, a_E) = `margin` otherwise. This pushes the demonstrated
    action's Q-value to beat every other action's by at least that margin,
    unless TD-learning has independently established a genuinely higher
    value for a different action - unlike behavior-cloning pretraining,
    this loss is applied continuously throughout RL training (every batch
    that includes a demo transition), not just once at initialization, so
    the network can't drift away from the demonstrations over time the way
    a one-shot BC warm-start can."""
    features, masks, actions, rewards, next_features, next_masks, dones, discounts, is_demo = batch

    features_t = torch.from_numpy(features).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_features_t = torch.from_numpy(next_features).to(device)
    next_masks_t = torch.from_numpy(next_masks).to(device)
    dones_t = torch.from_numpy(dones).to(device)
    discounts_t = torch.from_numpy(discounts).to(device)
    is_demo_t = torch.from_numpy(is_demo).to(device)

    q_values = net(features_t, masks_t)
    q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            # Double DQN: online net picks the action, target net evaluates it.
            # Decouples selection from evaluation to avoid the maximization-bias
            # overestimation that plain DQN's target_net.max() is prone to.
            next_q_online = net(next_features_t, next_masks_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = target_net(next_features_t, next_masks_t)
            next_q_selected = next_q_target.gather(1, next_actions).squeeze(1)
        else:
            # Plain DQN: target net both picks and evaluates the next action.
            next_q_target = target_net(next_features_t, next_masks_t)
            next_q_selected = next_q_target.max(dim=1).values
        # rewards_t is already an n-step (or shorter, at episode end) discounted
        # return; discounts_t is the matching gamma^k for the bootstrap term.
        targets = rewards_t + discounts_t * next_q_selected * (~dones_t).float()

    td_errors = q_selected - targets
    if td_loss == "mse":
        per_sample_loss = F.mse_loss(q_selected, targets, reduction="none")
    else:
        per_sample_loss = F.smooth_l1_loss(q_selected, targets, reduction="none", beta=huber_beta)
    if is_weights is not None:
        loss = (is_weights * per_sample_loss).mean()
    else:
        loss = per_sample_loss.mean()

    if is_demo_t.any():
        # margin(a, a_E): 0 at the demonstrated action, `margin` everywhere else.
        # Illegal actions are already masked to ~-1e9 by the network itself, so
        # adding `margin` (at most 0.8) leaves them enormously negative and they
        # never win the max() below - no separate masking needed here.
        margin_matrix = torch.full_like(q_values, margin)
        margin_matrix.scatter_(1, actions_t.unsqueeze(1), 0.0)
        supervised_gap = (q_values + margin_matrix).max(dim=1).values - q_selected
        margin_loss = (supervised_gap * is_demo_t.float()).sum() / is_demo_t.float().sum().clamp(min=1.0)
        loss = loss + margin_loss_weight * margin_loss

    return loss, td_errors.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Solitaire-v0")
    parser.add_argument("--steps", type=int, default=20_000, help="total environment steps")
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-step", type=int, default=3, help="n-step return length (1 = plain 1-step TD)")
    parser.add_argument("--hidden-layers", type=int, default=2, help="number of hidden layers in the Q-network (default 2, matching all prior runs)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="units per hidden layer (default 512, matching all prior runs) - must match a --resume-from checkpoint's own architecture")
    parser.add_argument("--layer-norm", action="store_true", help="insert LayerNorm before each hidden layer's ReLU - a standard mitigation for the effective-rank collapse structure_metrics.py has been tracking (off by default, matching all prior runs)")
    parser.add_argument(
        "--card-encoding",
        choices=["raw", "decomposed"],
        default="raw",
        help="raw = single joint card-id embedding (all prior runs); decomposed = separate slot_type/rank/color/suit embeddings, giving the network an explicit hint that e.g. 7S and 7C behave identically outside foundation-building",
    )
    parser.add_argument(
        "--symmetry-augment",
        action="store_true",
        default=False,
        help="also store each transition's Spades<->Clubs, Hearts<->Diamonds suit-swapped twin in the replay buffer (a free 2x data augmentation - works with either --card-encoding)",
    )
    parser.add_argument(
        "--no-undo",
        dest="allow_undo",
        action="store_false",
        default=True,
        help="make foundation->tableau (undo) permanently illegal, shrinking the *effective* action space from 590 to 562 (the Discrete(590) dimension itself is unchanged, for checkpoint/network compatibility - those 28 actions are just always masked out)",
    )
    parser.add_argument(
        "--foundation-undo-penalty", type=float, default=0.0,
        help="reward subtracted on every foundation->tableau (undo) move; 0 (default, matching all prior runs) means reversing a foundation move is free. "
             "Motivated by measured behavior: run 023's game logs show ~75%% of foundation-sends eventually get undone, and undoing costs nothing while a "
             "genuine new high-water-mark send earns +10 (REWARD_NEW_FOUNDATION_HIGH) - a send/undo/resend round trip nets nearly the full +10 for free.",
    )
    parser.add_argument(
        "--foundation-reward-ramp", type=float, default=0.0,
        help="max extra multiplier on REWARD_NEW_FOUNDATION_HIGH, phased in continuously (not a hard switch) as hidden tableau cards get revealed - "
             "1.0 at the deal, 1+this value once fully uncovered. 0 (default, matching all prior runs) keeps the reward flat throughout. Motivated by "
             "the current reward function already being tableau-heavy (REWARD_REVEAL=50/COLUMN_UNCOVERED=30/KING_ON_EMPTY=40 vs. foundation's 10), "
             "leaving the post-uncover endgame reward-sparse - exactly the phase where fully-uncovered-but-truncated games were found to stall.",
    )
    parser.add_argument(
        "--no-prioritized-replay",
        dest="prioritized_replay",
        action="store_false",
        default=True,
        help="use plain uniform replay instead of prioritized experience replay",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6, help="prioritization exponent (0=uniform, 1=fully proportional to |TD error|)")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="initial importance-sampling correction exponent")
    parser.add_argument("--per-beta-end", type=float, default=1.0, help="final IS-correction exponent, annealed to linearly over --steps")
    parser.add_argument(
        "--demo-source",
        default=None,
        help="DQfD (Hester et al. 2017): games.jsonl-schema file of real wins to permanently seed into the replay buffer "
        "alongside self-generated experience, plus a supervised large-margin loss anchoring the network toward the "
        "demonstrated actions throughout training (not just at initialization, unlike behavior-cloning pretraining). "
        "Omit to disable entirely (default) - a plain Double DQN + PER run otherwise identical to every prior run.",
    )
    parser.add_argument("--num-demo-transitions", type=int, default=30_000, help="how many n-step demo transitions to build from --demo-source's wins (must be < --buffer-capacity)")
    parser.add_argument("--margin", type=float, default=0.8, help="DQfD supervised margin loss constant l(a_E, a): 0 when a is the demonstrated action, else this value")
    parser.add_argument("--margin-loss-weight", type=float, default=1.0, help="weight on the margin loss term relative to the TD loss")
    parser.add_argument("--demo-priority-eps", type=float, default=1.0, help="priority floor added to demo transitions' |TD error| before exponentiating (PER only) - keeps them sampled at a meaningful rate even once their TD-error shrinks, unlike self-play's much tinier built-in floor (1e-6)")
    parser.add_argument(
        "--no-loop-breaker",
        dest="loop_breaker",
        action="store_false",
        default=True,
        help="disable LoopBreakerWrapper (agent.loop_breaker) on both the training and eval envs. Found via runs 001-020: every one of them showed a flat 0%% greedy-eval win rate that turned out to be substantially a naive-argmax artifact, not incompetence - the policy would get stuck oscillating between 2-3 actions with near-tied Q-values in an exact repeated game state and burn the whole step budget there. On the 3rd exact repeat of a state, blocks whichever action(s) were taken from it before, forcing a genuinely new choice. Sound (not just heuristic) for this game specifically: Klondike has no randomness once dealt, so an exact state repeat can provably never reach anywhere the earlier visit couldn't already reach. On by default; disable to reproduce the exact behavior of runs 001-020.",
    )
    parser.add_argument("--loop-breaker-threshold", type=int, default=2, help="how many prior visits to an exact state before its previously-taken action(s) get masked out (2 = intervene starting on the 3rd visit)")
    parser.add_argument("--revisit-penalty", type=float, default=0.0, help="small reward penalty (subtracted, so pass a positive number) applied on the training env only whenever an action lands back in a state already visited this episode - a training signal on top of (not instead of) the loop-breaker mask, which only prevents the current episode's waste, not learning. 0 (default) disables it. Not applied to the eval env, which should reflect unshaped task performance.")
    parser.add_argument(
        "--td-loss", choices=["huber", "mse"], default="huber",
        help="TD loss on the Bellman error. huber (default, matching runs 001-031) = smooth L1 with transition point --huber-beta; "
             "mse = plain squared error. See compute_loss(): with beta=1 against rewards of 50-5000, Huber saturates on every "
             "reward-bearing transition and the network learns reward frequency, not magnitude (run 029: Q at the deal ~36 vs "
             "realized discounted return ~630).",
    )
    parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber transition point (|error| below this is quadratic, above is linear). Only used with --td-loss huber. 1.0 matches all prior runs; a value at the reward scale (e.g. 50+) behaves like MSE for ordinary transitions.")
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument(
        "--no-double-dqn",
        dest="double_dqn",
        action="store_false",
        default=True,
        help="use plain DQN (target net both selects and evaluates the next action) instead of Double DQN",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=200_000)
    parser.add_argument(
        "--reheat-step",
        type=int,
        action="append",
        default=None,
        help="step at which epsilon jumps back up to a --reheat-eps value, then decays back to --eps-end over a --reheat-decay-steps window; repeatable for multiple reheats (e.g. a second, smaller 'echo' reheat later in training). Omit to disable entirely.",
    )
    parser.add_argument(
        "--reheat-eps",
        type=float,
        action="append",
        default=None,
        help="repeatable, paired positionally with --reheat-step; if given once but --reheat-step given multiple times, that single value is reused for all of them",
    )
    parser.add_argument(
        "--reheat-decay-steps",
        type=int,
        action="append",
        default=None,
        help="repeatable, paired positionally with --reheat-step; if given once but --reheat-step given multiple times, that single value is reused for all of them",
    )
    parser.add_argument("--learning-starts", type=int, default=5_000, help="steps before training begins")
    parser.add_argument("--train-freq", type=int, default=4, help="train every N environment steps")
    parser.add_argument(
        "--target-update-mode",
        choices=["soft", "hard"],
        default="soft",
        help="soft = Polyak-average the target net every training step (tau); hard = periodically overwrite it wholesale",
    )
    parser.add_argument("--tau", type=float, default=0.005, help="soft target-update rate (only used when --target-update-mode=soft)")
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=1_000,
        help="hard-update the target net every N steps (only used when --target-update-mode=hard)",
    )
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="run", help="short label for this run, e.g. 'double_dqn'")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--notes", default="", help="free-text description of this run's purpose, written into its README.md"
    )
    parser.add_argument("--no-log-games", action="store_true", help="disable per-episode game logging")
    parser.add_argument("--print-every", type=int, default=1_000, help="print progress / log metrics every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=50_000, help="save a numbered checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=20_000, help="run a frozen-greedy evaluation pass every N steps")
    parser.add_argument("--eval-episodes", type=int, default=20, help="number of fixed held-out deals per evaluation pass")
    parser.add_argument(
        "--curriculum-source",
        default=None,
        help="glob pattern for games.jsonl file(s) with real seeded wins to build a curriculum from (e.g. 'runs/curriculum_wins.jsonl'); omit to disable curriculum learning entirely",
    )
    parser.add_argument("--curriculum-fraction", type=float, default=0.5, help="fraction of episode resets that start from a curriculum state rather than a fresh deal")
    parser.add_argument("--curriculum-start-tail", type=int, default=10, help="initial number of a win's final moves left for the agent to play (easy - right next to the win)")
    parser.add_argument("--curriculum-end-tail", type=int, default=10_000, help="final tail length (effectively 'whole game' once it exceeds any real game length)")
    parser.add_argument("--curriculum-anneal-steps", type=int, default=300_000, help="steps over which tail length grows from start to end")
    parser.add_argument(
        "--solvable-seed-source",
        default=None,
        help="jsonl of {'seed': int, ...} records, ground-truth proven solvable (agent/build_solvable_corpus.py); "
             "when set, fresh-deal episode resets (i.e. not a curriculum-tail start) draw their seed from this pool "
             "instead of a fully random one, so self-play never burns budget on a deal that can't be won",
    )
    parser.add_argument(
        "--solvable-seed-steps", type=int, default=None,
        help="only draw fresh-deal seeds from --solvable-seed-source for this many steps, then revert to fully "
             "random; omit to use the pool for the entire run",
    )
    parser.add_argument(
        "--eval-seed-source",
        default=None,
        help="jsonl of {'seed': int, ...} records to evaluate against instead of the fixed 900000+i range "
             "(e.g. runs/eval_corpus.jsonl, 1000 ground-truth-solvable seeds - a larger battery than the "
             "original 100, to shrink eval win-rate sampling noise); uses the first --eval-episodes of them",
    )
    parser.add_argument("--resume-from", default=None, help="checkpoint path to resume from (loads weights into both online and target nets)")
    parser.add_argument("--resume-step", type=int, default=0, help="the step count that checkpoint was saved at - training continues from resume_step+1, and schedules (epsilon/beta/curriculum) pick up from there instead of restarting")
    parser.add_argument("--run-dir", default=None, help="reuse this exact run directory instead of creating a new numbered one (for resuming: appends to the same stdout.log/metrics.csv/games.jsonl rather than starting fresh)")
    args = parser.parse_args()

    reheat_steps = args.reheat_step or []
    reheat_epss = args.reheat_eps or []
    reheat_decays = args.reheat_decay_steps or []
    if reheat_steps:
        if len(reheat_epss) == 1:
            reheat_epss = reheat_epss * len(reheat_steps)
        if len(reheat_decays) == 1:
            reheat_decays = reheat_decays * len(reheat_steps)
        assert len(reheat_steps) == len(reheat_epss) == len(reheat_decays), (
            "--reheat-step, --reheat-eps, --reheat-decay-steps must be given the same number of times "
            "(or --reheat-eps/--reheat-decay-steps given once to reuse for every --reheat-step)"
        )
    reheats = sorted(zip(reheat_steps, reheat_epss, reheat_decays))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = next_run_dir(Path(args.runs_dir), args.tag)
    config = save_config(run_dir, args)
    write_readme_start(run_dir, args.tag, args.notes, config)
    log = TeeLogger(run_dir / "stdout.log", append=bool(args.resume_from))
    log(f"run directory: {run_dir}" + (f" (resuming from step {args.resume_step})" if args.resume_from else ""))

    env = gym.make("Solitaire-v0", allow_undo=args.allow_undo, foundation_undo_penalty=args.foundation_undo_penalty, foundation_reward_ramp=args.foundation_reward_ramp)
    if not args.no_log_games:
        env = GameLogger(env, log_path=str(run_dir / "games.jsonl"))
    eval_env = gym.make("Solitaire-v0", allow_undo=args.allow_undo, foundation_undo_penalty=args.foundation_undo_penalty, foundation_reward_ramp=args.foundation_reward_ramp)  # separate instance so evaluation never disturbs the training episode in progress
    if args.loop_breaker:
        env = LoopBreakerWrapper(env, threshold=args.loop_breaker_threshold, revisit_penalty=args.revisit_penalty)
        eval_env = LoopBreakerWrapper(eval_env, threshold=args.loop_breaker_threshold)

    eval_seeds = None
    if args.eval_seed_source:
        with open(args.eval_seed_source) as f:
            eval_seeds = [json.loads(line)["seed"] for line in f]
        log(f"eval: using {len(eval_seeds)}-seed battery from {args.eval_seed_source} "
            f"(first {args.eval_episodes} of them per eval pass) instead of the fixed {EVAL_SEED_BASE}+i range")

    if args.card_encoding == "decomposed":
        preprocess_fn = preprocess_decomposed
        swap_features_fn = swap_features_decomposed
        num_features = NUM_DECOMPOSED_FEATURES
        online_net = DecomposedDQN(
            num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim, layer_norm=args.layer_norm
        ).to(args.device)
        target_net = DecomposedDQN(
            num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim, layer_norm=args.layer_norm
        ).to(args.device)
    else:
        preprocess_fn = preprocess
        swap_features_fn = swap_features_raw
        num_features = NUM_FEATURES
        online_net = DQN(
            num_features=NUM_FEATURES,
            num_actions=NUM_ACTIONS,
            num_hidden_layers=args.hidden_layers,
            hidden_dim=args.hidden_dim,
            layer_norm=args.layer_norm,
        ).to(args.device)
        target_net = DQN(
            num_features=NUM_FEATURES,
            num_actions=NUM_ACTIONS,
            num_hidden_layers=args.hidden_layers,
            hidden_dim=args.hidden_dim,
            layer_norm=args.layer_norm,
        ).to(args.device)
    # Snapshot the true random init (this exact seed's, per torch.manual_seed
    # above) before any --resume-from load overwrites it -- the reference
    # point structure_metrics diffs every later snapshot against.
    init_state = {k: v.detach().clone().cpu() for k, v in online_net.state_dict().items()}
    if args.resume_from:
        online_net.load_state_dict(torch.load(args.resume_from, map_location=args.device, weights_only=True))
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    structure_metrics = MetricsLogger(str(run_dir / "structure_metrics.csv"), fields=structure_fields(init_state))

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    num_demo = args.num_demo_transitions if args.demo_source else 0
    if num_demo:
        assert num_demo < args.buffer_capacity, "--num-demo-transitions must be smaller than --buffer-capacity"
    if args.prioritized_replay:
        buffer = PrioritizedReplayBuffer(
            args.buffer_capacity, num_features, NUM_ACTIONS, alpha=args.per_alpha, num_demo=num_demo, demo_eps=args.demo_priority_eps
        )
    else:
        buffer = ReplayBuffer(args.buffer_capacity, num_features, NUM_ACTIONS, num_demo=num_demo)
    if args.demo_source:
        log(f"DQfD: building {num_demo} permanent demo transitions from {args.demo_source}")
        demo_transitions = build_demo_transitions(
            source=args.demo_source,
            preprocess_fn=preprocess_fn,
            n_step=args.n_step,
            gamma=args.gamma,
            target_transitions=num_demo,
            seed=args.seed,
        )
        buffer.load_demos(demo_transitions)
        log(f"DQfD: loaded {num_demo} demo transitions into the permanent replay-buffer region")
    nstep = NStepAccumulator(args.n_step, args.gamma)

    curriculum_wins = load_wins([args.curriculum_source]) if args.curriculum_source else []
    if args.curriculum_source:
        log(f"curriculum: loaded {len(curriculum_wins)} real seeded wins from {args.curriculum_source}")
    curriculum_completed_count = 0  # incremented when a curriculum-started episode actually finishes, to stay in lockstep with episode_count

    solvable_seeds = []
    if args.solvable_seed_source:
        with open(args.solvable_seed_source) as f:
            solvable_seeds = [json.loads(line)["seed"] for line in f]
        scope = f"first {args.solvable_seed_steps} steps" if args.solvable_seed_steps else "the entire run"
        log(f"solvable-seed pool: loaded {len(solvable_seeds)} ground-truth-solvable seeds from "
            f"{args.solvable_seed_source}, used for fresh-deal resets over {scope}")

    def reset_episode(current_step: int) -> tuple:
        """Returns (obs, info, used_curriculum) - the caller is responsible
        for counting used_curriculum against completed episodes, not
        episodes merely started (there's always exactly one episode still
        in progress when training ends, which would otherwise throw off
        the curriculum-episode ratio)."""
        if curriculum_wins and rng.random() < args.curriculum_fraction:
            win = curriculum_wins[rng.integers(0, len(curriculum_wins))]
            tail = tail_steps_by_step(current_step, args.curriculum_start_tail, args.curriculum_end_tail, args.curriculum_anneal_steps)
            seed, initial_moves = make_initial_moves(win, tail)
            obs, info = env.reset(seed=seed, options={"initial_moves": initial_moves})
            return obs, info, True
        use_solvable_pool = solvable_seeds and (
            args.solvable_seed_steps is None or current_step < args.solvable_seed_steps
        )
        if use_solvable_pool:
            fresh_seed = int(solvable_seeds[rng.integers(0, len(solvable_seeds))])
        else:
            fresh_seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=fresh_seed)
        return obs, info, False

    metrics = MetricsLogger(str(run_dir / "metrics.csv"))
    plot_path = run_dir / "progress.png"
    eval_metrics = MetricsLogger(str(run_dir / "eval_metrics.csv"), fields=EVAL_CSV_FIELDS)
    eval_plot_path = run_dir / "eval_progress.png"
    last_eval_stats: dict = {}

    # Every episode gets its own explicit, recorded seed (not left to the env's
    # internal RNG continuing forward) so GameLogger's {seed, actions} records
    # are independently reproducible - a prerequisite for curriculum learning
    # from real logged wins, not just a nicety.
    obs, info, episode_is_curriculum = reset_episode(0)
    features, mask = preprocess_fn(obs, info)

    episode_return = 0.0
    episode_len = 0
    episode_count = 0
    lifetime_outcomes = {"won": 0, "stalled": 0, "truncated": 0, "other": 0}
    # seed from this run's own games.jsonl (append-mode, so it already holds
    # the true full history across any resume/restart) rather than always
    # starting at 0 - otherwise a --resume-from continuation (or an OOM
    # auto-restart) looks like it "forgot" every win from before that point,
    # in both the printed log line and the lifetime_wins plot column
    games_log_path = run_dir / "games.jsonl"
    if games_log_path.exists():
        with games_log_path.open() as f:
            for line in f:
                reason = json.loads(line).get("reason")
                if reason in lifetime_outcomes:
                    lifetime_outcomes[reason] += 1
                elif reason is not None:
                    lifetime_outcomes["other"] += 1
    # reset every print/plot interval, to show recent trends rather than lifetime averages
    window_outcomes = {"won": 0, "stalled": 0, "truncated": 0, "other": 0}
    window_episodes = 0
    recent_returns: list[float] = []
    recent_losses: list[float] = []
    recent_episode_lens: list[int] = []
    recent_foundation_totals: list[int] = []
    mean_return = mean_loss = mean_foundation = float("nan")
    start_time = time.time()

    for step in range(args.resume_step + 1, args.steps + 1):
        epsilon = epsilon_by_step(
            step,
            args.eps_start,
            args.eps_end,
            args.eps_decay_steps,
            reheats=reheats,
        )
        online_net.eval()
        action = select_action(online_net, features, mask, epsilon, rng, args.device)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_features, next_mask = preprocess_fn(next_obs, next_info)
        done = terminated or truncated

        for transition in nstep.add(features, mask, action, reward, next_features, next_mask, done):
            buffer.add(*transition)
            if args.symmetry_augment:
                t_features, t_mask, t_action, t_reward, t_next_features, t_next_mask, t_done, t_discount = transition
                buffer.add(
                    swap_features_fn(t_features),
                    swap_mask(t_mask),
                    swap_action(t_action),
                    t_reward,
                    swap_features_fn(t_next_features),
                    swap_mask(t_next_mask),
                    t_done,
                    t_discount,
                )

        features, mask = next_features, next_mask
        episode_return += reward
        episode_len += 1

        if done:
            episode_count += 1
            if episode_is_curriculum:
                curriculum_completed_count += 1
            window_episodes += 1
            recent_returns.append(episode_return)
            recent_episode_lens.append(episode_len)
            final_foundation_total = int(next_obs["foundations"].sum())
            recent_foundation_totals.append(final_foundation_total)
            if terminated and next_info.get("stalled"):
                outcome = "stalled"
            elif terminated and final_foundation_total == 52:
                outcome = "won"
            elif truncated:
                outcome = "truncated"
            else:
                outcome = "other"
            lifetime_outcomes[outcome] += 1
            window_outcomes[outcome] += 1

            obs, info, episode_is_curriculum = reset_episode(step)
            features, mask = preprocess_fn(obs, info)
            episode_return = 0.0
            episode_len = 0

        if step >= args.learning_starts and step % args.train_freq == 0 and len(buffer) >= args.batch_size:
            online_net.train()
            if args.prioritized_replay:
                beta = beta_by_step(step, args.per_beta_start, args.per_beta_end, args.steps)
                batch, sample_idxs, is_weights = buffer.sample(args.batch_size, rng, beta)
                is_weights_t = torch.from_numpy(is_weights).to(args.device)
                loss, td_errors = compute_loss(
                    online_net,
                    target_net,
                    batch,
                    args.device,
                    args.double_dqn,
                    is_weights_t,
                    margin=args.margin,
                    margin_loss_weight=args.margin_loss_weight,
                    td_loss=args.td_loss,
                    huber_beta=args.huber_beta,
                )
            else:
                batch = buffer.sample(args.batch_size, rng)
                loss, td_errors = compute_loss(
                    online_net,
                    target_net,
                    batch,
                    args.device,
                    args.double_dqn,
                    margin=args.margin,
                    margin_loss_weight=args.margin_loss_weight,
                    td_loss=args.td_loss,
                    huber_beta=args.huber_beta,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), args.grad_clip)
            optimizer.step()
            recent_losses.append(loss.item())

            if args.prioritized_replay:
                buffer.update_priorities(sample_idxs, td_errors.cpu().numpy())

            if args.target_update_mode == "soft":
                soft_update(target_net, online_net, args.tau)

        if args.target_update_mode == "hard" and step % args.target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        if step % args.print_every == 0:
            elapsed = time.time() - start_time
            mean_return = np.mean(recent_returns) if recent_returns else float("nan")
            mean_loss = np.mean(recent_losses) if recent_losses else float("nan")
            mean_episode_len = np.mean(recent_episode_lens) if recent_episode_lens else float("nan")
            mean_foundation = np.mean(recent_foundation_totals) if recent_foundation_totals else float("nan")
            win_rate = window_outcomes["won"] / window_episodes if window_episodes else 0.0
            stalled_rate = window_outcomes["stalled"] / window_episodes if window_episodes else 0.0
            truncated_rate = window_outcomes["truncated"] / window_episodes if window_episodes else 0.0

            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # peak RSS so far, in MB (Linux)
            log(
                f"step {step:>8}/{args.steps}  eps={epsilon:.3f}  episodes={episode_count:>5}  "
                f"mean_return={mean_return:7.3f}  mean_loss={mean_loss:8.5f}  "
                f"mean_foundation={mean_foundation:5.1f}/52  lifetime_outcomes={lifetime_outcomes}  "
                f"rss={rss_mb:7.1f}MB  ({elapsed:.0f}s)"
            )

            metrics.log(
                step=step,
                episodes=episode_count,
                epsilon=epsilon,
                mean_return=mean_return,
                mean_loss=mean_loss,
                mean_episode_len=mean_episode_len,
                win_rate=win_rate,
                stalled_rate=stalled_rate,
                truncated_rate=truncated_rate,
                mean_final_foundation=mean_foundation,
                lifetime_wins=lifetime_outcomes["won"],
            )
            metrics.plot(str(plot_path), eval_csv_path=str(run_dir / "eval_metrics.csv"))

            recent_returns.clear()
            recent_losses.clear()
            recent_episode_lens.clear()
            recent_foundation_totals.clear()
            window_outcomes = {"won": 0, "stalled": 0, "truncated": 0, "other": 0}
            window_episodes = 0

        if step % args.checkpoint_every == 0:
            torch.save(online_net.state_dict(), run_dir / f"checkpoint_{step}.pt")
            structure_metrics.log(**compute_structure_row(step, init_state, online_net.state_dict()))

        if step % args.eval_every == 0:
            last_eval_stats = evaluate(online_net, eval_env, args.eval_episodes, args.device, preprocess_fn, eval_seeds=eval_seeds, gamma=args.gamma)
            log(
                f"  EVAL  step {step:>8}  win_rate={last_eval_stats['win_rate']:.2%}  "
                f"mean_return={last_eval_stats['mean_return']:7.3f}  "
                f"mean_foundation={last_eval_stats['mean_foundation']:5.1f}/52  "
                f"mean_length={last_eval_stats['mean_length']:5.1f}  "
                f"q0={last_eval_stats['q0']:7.1f} vs g0={last_eval_stats['g0']:7.1f}  (greedy, {args.eval_episodes} fixed deals)"
            )
            eval_metrics.log(
                step=step,
                eval_episodes=args.eval_episodes,
                eval_win_rate=last_eval_stats["win_rate"],
                eval_mean_return=last_eval_stats["mean_return"],
                eval_mean_foundation=last_eval_stats["mean_foundation"],
                eval_mean_length=last_eval_stats["mean_length"],
                eval_q0=last_eval_stats["q0"],
                eval_g0=last_eval_stats["g0"],
            )
            plot_eval(str(run_dir / "eval_metrics.csv"), str(eval_plot_path))

    torch.save(online_net.state_dict(), run_dir / "checkpoint_final.pt")
    if args.steps % args.checkpoint_every != 0:  # avoid a duplicate row when it divides evenly
        structure_metrics.log(**compute_structure_row(args.steps, init_state, online_net.state_dict()))
    log(f"done. saved final checkpoint to {run_dir / 'checkpoint_final.pt'}")

    finalize_readme(
        run_dir,
        {
            "total_steps": step,
            "total_episodes": episode_count,
            "curriculum_episodes": f"{curriculum_completed_count}/{episode_count}",
            "final_mean_loss": f"{mean_loss:.4f}",
            "final_mean_return": f"{mean_return:.2f}",
            "final_mean_foundation": f"{mean_foundation:.2f}/52",
            "lifetime_outcomes": lifetime_outcomes,
            "final_eval": last_eval_stats or "(no evaluation pass completed - run shorter than --eval-every)",
            "elapsed_seconds": round(time.time() - start_time, 1),
        },
    )

    log.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
