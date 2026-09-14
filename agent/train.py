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
from agent.run_utils import next_run_dir, save_config, write_readme_start, finalize_readme, TeeLogger
from agent.curriculum import load_wins, tail_steps_by_step, make_initial_moves


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


def evaluate(net: DQN, env: gym.Env, episodes: int, device: str, preprocess_fn=preprocess) -> dict:
    """Run `episodes` full episodes with the policy acting purely greedily
    (epsilon=0, no exploration) against a fixed set of held-out deals, so
    results are directly comparable across evaluation calls at different
    points in training. This is the clean read on whether the *learned
    policy* can win on its own - separate from whatever exploration noise
    is mixed into the training-time win rate."""
    net.eval()
    returns, lengths, foundations = [], [], []
    wins = 0
    with torch.no_grad():
        for i in range(episodes):
            obs, info = env.reset(seed=EVAL_SEED_BASE + i)
            features, mask = preprocess_fn(obs, info)
            ep_return = 0.0
            ep_len = 0
            while True:
                x = torch.from_numpy(features).to(device)
                m = torch.from_numpy(mask).to(device)
                q = net(x, m)
                action = int(q.argmax(dim=1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                features, mask = preprocess_fn(obs, info)
                ep_return += reward
                ep_len += 1
                if terminated or truncated:
                    if terminated and not info.get("stalled") and int(obs["foundations"].sum()) == 52:
                        wins += 1
                    break
            returns.append(ep_return)
            lengths.append(ep_len)
            foundations.append(int(obs["foundations"].sum()))
    return {
        "win_rate": wins / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_foundation": float(np.mean(foundations)),
        "mean_length": float(np.mean(lengths)),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, per_sample_td_error). td_error is detached and only
    used to refresh priorities in the prioritized replay buffer; it's
    ignored entirely when using plain uniform replay."""
    features, masks, actions, rewards, next_features, next_masks, dones, discounts = batch

    features_t = torch.from_numpy(features).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_features_t = torch.from_numpy(next_features).to(device)
    next_masks_t = torch.from_numpy(next_masks).to(device)
    dones_t = torch.from_numpy(dones).to(device)
    discounts_t = torch.from_numpy(discounts).to(device)

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
    per_sample_loss = F.smooth_l1_loss(q_selected, targets, reduction="none")
    if is_weights is not None:
        loss = (is_weights * per_sample_loss).mean()
    else:
        loss = per_sample_loss.mean()
    return loss, td_errors.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Solitaire-v0")
    parser.add_argument("--steps", type=int, default=20_000, help="total environment steps")
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-step", type=int, default=3, help="n-step return length (1 = plain 1-step TD)")
    parser.add_argument("--hidden-layers", type=int, default=2, help="number of 512-unit hidden layers in the Q-network (default 2, matching all prior runs)")
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
        "--no-prioritized-replay",
        dest="prioritized_replay",
        action="store_false",
        default=True,
        help="use plain uniform replay instead of prioritized experience replay",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6, help="prioritization exponent (0=uniform, 1=fully proportional to |TD error|)")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="initial importance-sampling correction exponent")
    parser.add_argument("--per-beta-end", type=float, default=1.0, help="final IS-correction exponent, annealed to linearly over --steps")
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

    env = gym.make("Solitaire-v0", allow_undo=args.allow_undo)
    if not args.no_log_games:
        env = GameLogger(env, log_path=str(run_dir / "games.jsonl"))
    eval_env = gym.make("Solitaire-v0", allow_undo=args.allow_undo)  # separate instance so evaluation never disturbs the training episode in progress

    if args.card_encoding == "decomposed":
        preprocess_fn = preprocess_decomposed
        swap_features_fn = swap_features_decomposed
        num_features = NUM_DECOMPOSED_FEATURES
        online_net = DecomposedDQN(num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers).to(args.device)
        target_net = DecomposedDQN(num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers).to(args.device)
    else:
        preprocess_fn = preprocess
        swap_features_fn = swap_features_raw
        num_features = NUM_FEATURES
        online_net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers).to(args.device)
        target_net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=args.hidden_layers).to(args.device)
    if args.resume_from:
        online_net.load_state_dict(torch.load(args.resume_from, map_location=args.device, weights_only=True))
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    if args.prioritized_replay:
        buffer = PrioritizedReplayBuffer(args.buffer_capacity, num_features, NUM_ACTIONS, alpha=args.per_alpha)
    else:
        buffer = ReplayBuffer(args.buffer_capacity, num_features, NUM_ACTIONS)
    nstep = NStepAccumulator(args.n_step, args.gamma)

    curriculum_wins = load_wins([args.curriculum_source]) if args.curriculum_source else []
    if args.curriculum_source:
        log(f"curriculum: loaded {len(curriculum_wins)} real seeded wins from {args.curriculum_source}")
    curriculum_completed_count = 0  # incremented when a curriculum-started episode actually finishes, to stay in lockstep with episode_count

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
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
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
                    online_net, target_net, batch, args.device, args.double_dqn, is_weights_t
                )
            else:
                batch = buffer.sample(args.batch_size, rng)
                loss, td_errors = compute_loss(online_net, target_net, batch, args.device, args.double_dqn)

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

        if step % args.eval_every == 0:
            last_eval_stats = evaluate(online_net, eval_env, args.eval_episodes, args.device, preprocess_fn)
            log(
                f"  EVAL  step {step:>8}  win_rate={last_eval_stats['win_rate']:.2%}  "
                f"mean_return={last_eval_stats['mean_return']:7.3f}  "
                f"mean_foundation={last_eval_stats['mean_foundation']:5.1f}/52  "
                f"mean_length={last_eval_stats['mean_length']:5.1f}  (greedy, {args.eval_episodes} fixed deals)"
            )
            eval_metrics.log(
                step=step,
                eval_episodes=args.eval_episodes,
                eval_win_rate=last_eval_stats["win_rate"],
                eval_mean_return=last_eval_stats["mean_return"],
                eval_mean_foundation=last_eval_stats["mean_foundation"],
                eval_mean_length=last_eval_stats["mean_length"],
            )
            plot_eval(str(run_dir / "eval_metrics.csv"), str(eval_plot_path))

    torch.save(online_net.state_dict(), run_dir / "checkpoint_final.pt")
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
