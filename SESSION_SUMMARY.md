# Session Summary — Solitaire RL Project (2026-09-11 → 2026-09-13)

An end-to-end session building a Klondike Solitaire Gymnasium environment from scratch and training a DQN agent against it, for an RL class. Below is what was built, what was learned, and where things stand. (This supersedes the original 2026-09-12 version of this file, which stopped after run 004.)

## 1. Project setup & licensing

- Checked `suryadutta/solitaire` (the originally proposed base) — **no license** (confirmed via GitHub API: `license: null`, no `LICENSE` file, no license text anywhere in the repo). Rejected it.
- Switched to `tteeoo/solitaire` — confirmed **MIT licensed** (LICENSE file, GitHub API, README). Vendored under `vendor/tteeoo-solitaire/` with `NOTICE.md` for attribution.
- Set up `.venv` with `gymnasium`, `numpy`; later added `torch` (2.6.0+cu124, GPU-enabled — this machine has an RTX 500 Ada, 4GB VRAM) and `matplotlib`.

## 2. The environment (`solitaire_gym/`)

A from-scratch Klondike rules engine (referencing tteeoo's logic, not literally editing that file) wrapped as a `gymnasium.Env`, registered as `Solitaire-v0`.

**Action space** — `Discrete(590)`:
- `0`: draw from stock (recycles waste when stock empties)
- `1–7`: waste top card → tableau column
- `8`: waste top card → foundation
- `9–15`: tableau column top card → foundation
- `16–561`: tableau column *i* → column *j* (42 ordered pairs), moving any partial sub-run (1 to 13 cards) — not just the whole face-up stack, which was a real early bug (couldn't dig into a column without moving its entire run at once)
- `562–589`: foundation → tableau (the "undo" move, legal in real Klondike) — 28 = 4 suits × 7 columns; gated by `allow_undo` (see run 008)

**Observation space** — `Dict`: tableau grid (7×52, face-down cards genuinely masked as hidden — a real POMDP), per-suit foundation counts, waste-top card, stock/waste sizes. Supports `reset(options={"initial_moves": [...]})` for deterministic fast-forward replay (used by curriculum learning and behavior cloning).

**Reward function** (2026-09-14 revision — rescaled 10x from the original and given three new milestone bonuses, to make room in the reward space for meaningful intermediate values without everything crowding into 0-10):
- `+10` for a foundation move **only if it exceeds that suit's high-water mark** (tracked separately per suit, never decreases) — re-placing a card you already banked and undid gets `0`, not another `+10`
- `+50` for revealing a previously face-down tableau card
- `+40` **first time a given king** (tracked per-suit, not per-column — there are only 4 kings) lands on a column that was empty — rule-enforced: only a king can ever legally occupy an empty column, so this needs no separate rank check, just "was this column empty before the move." Gating per-king rather than per-column matters: per-column gating would let a single king tour every never-before-used empty column and collect the bonus repeatedly (up to 7 times with only 4 physical kings); per-king gating caps it at 4 total, ever, no matter how the kings get shuffled around
- `+30` **first time** a column's *last* hidden card is revealed (i.e. the column now has zero face-down cards left, though it may still hold face-up cards) — stacks with the `+50` reveal bonus for that same flip, since finishing a column is worth more than a routine reveal; per-column one-time flag, pre-satisfied at reset for column 0 (dealt with 0 hidden cards, so it has nothing to ever reveal)
- `+20` **first time** a column reaches zero *total* cards (fully emptied, not just uncovered) — distinct from the uncovered bonus above (a column can be uncovered long before it's ever emptied, e.g. it may still hold a face-up run) and rewarded independently of whether a king is ever subsequently moved there, since clearing a column is useful groundwork on its own (most commonly: clearing column 0's single starting card, which starts with 0 hidden cards and can't ever trigger the uncovered bonus, but very much can trigger this one)
- `+5000` win bonus (deliberately dominant — a full clearing already banks ~2050 from the other milestone bonuses above along the way, so a same-order-of-magnitude win bonus, like the original `+100` or even `+1000`, gave weak marginal incentive to actually finish the last few cards versus stopping just short with almost all the shaped reward already banked), `-0.01` per-step cost, `-10` for an illegal action (only reachable by bypassing the action mask)
- **Stall/cycle-check termination**: if the agent draws all the way through the current stock+waste pool with no other move in between, that's a provable dead end (every possible waste-top exposure has been tried against a frozen board) — episode ends with a **neutral 0 reward**, not a penalty, since the agent keeps whatever it earned getting there

All three new bonuses target the same reward-shaping principle as the foundation high-water mark: reward genuine progress once, not the physically-repeatable act of achieving it again. Gradient-norm clipping (`--grad-clip`, default 10.0) was already active before this change, which caps update size independent of reward scale — the main practical effect of the 10x rescale is that loss values now read ~100x larger (squared TD error) than in runs 001-011, which is expected, not a new divergence.

**Known deliberate tradeoff**: the foundation→tableau undo move is real-rules-faithful but collapses naive/random exploration's win rate to ~0% (verified: uniform-random-among-legal-actions play went from ~25% wins in an early, simpler version of the engine to 0/200 after undo was added, even at 3000-step budgets). Kept anyway — full rules fidelity was prioritized over an easier baseline, with the understanding that this makes the RL problem itself harder. (Run 008 later tested disabling it via `--no-undo`; see §6.)

## 3. Human play (`play.py`)

Drives the actual registered `Solitaire-v0` env (not a separate rules copy). Command grammar fully regularized to `<source> <destination>`: `d`/Enter draw, `w <col>`, `w f`, `<col> f`, `<suit> <col>` (undo), `<src> <dst>` (full run), `<src> <n> <dst>` (partial run), plus `m`/`n`/`h`/`q` utilities. `--seed`/`-s` CLI flag for reproducing specific deals.

Bugs found and fixed along the way: draw marked legal with empty stock+waste, an ANSI-color-escape string-padding bug that broke column alignment, `f 4`/`w f` argument-order inconsistencies, a `d`/`d 3` parsing collision (bare draw vs. diamonds-foundation-move).

## 4. Game logging

`GameLogger` (a `gym.Wrapper`) appends one JSON line per episode — `{seed, action sequence, total_reward, outcome}` — to a `.jsonl` file. Cheap because the engine is deterministic from a seed, so a logged game can be exactly replayed later (`replay()` helper). Wired into `play.py` by default; opt-in for training scripts. Every episode reset (not just the first) is independently seeded, so every logged game — win or loss — is reproducible.

## 5. The RL training pipeline (`agent/`)

- **`preprocessing.py`**: two encodings, both selectable via `--card-encoding`:
  - `raw` (`extract_features`, 369 features): flattens the Dict observation into a single int32 array of raw joint card IDs.
  - `decomposed` (`extract_decomposed_features`, 1,464 features): each tableau/waste slot split into separate slot-type / rank / color / suit channels, so the network sees rank+color (which drive tableau legality) and suit (which only matters for foundations/undo) as structurally distinct inputs instead of one opaque joint card ID.
- **`qnetwork.py`**: `DQN` (embedding+MLP for raw encoding) and `DecomposedDQN` (separate small embeddings per channel for decomposed encoding) — both mask illegal actions to `-1e9` before returning Q-values. `--hidden-layers` configurable (default 2).
- **`replay_buffer.py`** / **`prioritized_replay_buffer.py`**: uniform and prioritized (sum-tree based, O(log n) sampling/updates) replay buffers — both kept, selectable via `--no-prioritized-replay`.
- **`nstep_buffer.py`**: n-step return accumulation (default n=3; `--n-step 1` recovers plain 1-step TD), correctly handling the episode-end tail (shorter windows with correctly shrinking discounts).
- **`symmetry.py`**: suit-swap (Spades↔Clubs, Hearts↔Diamonds) data augmentation — every transition's suit-swapped "twin" is an equally valid game, so storing both in the replay buffer doubles usable data and teaches suit-invariance directly (`--symmetry-augment`), complementing `DecomposedDQN`'s structural version of the same idea. Verified via ground-truth engine-level comparison (600/600 matches) and an involution check over all 590 actions.
- **`heuristic_policy.py`**: `GreedyKlondikePolicy` — a hand-written, non-learned scoring policy (foundation moves > reveals > king-to-empty > waste-to-tableau > tableau-to-tableau > undo, with stateful cycle detection to escape oscillation) used purely to *harvest* real, reproducible wins for curriculum/behavior-cloning, not as an RL baseline.
- **`curriculum.py`**: loads real logged wins, fast-forwards episodes to a tail-length-scheduled point near the end of a real win and anneals the starting point earlier over training (`--curriculum-source/-fraction/-start-tail/-end-tail/-anneal-steps`).
- **`harvest_wins.py`**: runs many episodes under either an epsilon-greedy policy or the heuristic policy and logs the results to a `.jsonl` file for curriculum/BC use.
- **`behavior_cloning.py`**: supervised pretraining of the DQN on the heuristic policy's harvested episodes (cross-entropy over legal actions), producing a checkpoint usable as an RL warm-start via `--resume-from`.
- **`metrics.py`**: CSV logging + auto-regenerated PNG progress plots (training metrics and a separate greedy-eval plot).
- **`run_utils.py`**: run versioning (`runs/<NNN>_<tag>/`, auto-incrementing, each with its own config/checkpoints/metrics/plot/logs/README), git-commit capture, per-run human-readable `README.md` (notes + full config at start, real results appended at the end).
- **`train.py`**: the full training loop — epsilon-greedy behavior policy (with an optional slower/longer decay schedule), Double DQN target computation (selectable via `--no-double-dqn`), soft (Polyak) or hard target updates, periodic frozen-greedy evaluation against fixed held-out deals, periodic checkpointing, curriculum wiring, `--resume-from`/`--resume-step`/`--run-dir` for resuming a killed run mid-flight or warm-starting from a BC checkpoint.
- **`train_resilient.sh`**: a wrapper that auto-detects a dead training process (no `checkpoint_final.pt`) and relaunches from the latest saved checkpoint instead of losing all progress. Fixed a subtle bug during run 011: on a restart, the wrapper's own `--resume-from <latest checkpoint>` and any caller-supplied `--resume-from` (e.g. a BC checkpoint, meant only for attempt 1) would both be on the command line — argparse keeps the *last* one, so every restart would have silently reverted to the original BC checkpoint and discarded all progress since. Fixed to only apply the caller's `--resume-from`/`--resume-step` on attempt 1.

## 6. Algorithm iteration — what we tried, in order

| Run | Change | Eval win rate (greedy) |
|---|---|---|
| (pre-versioning) vanilla DQN | baseline | diverged, not measured |
| `001_double_dqn_soft_nstep3` | Double DQN + soft target updates + lower LR + n-step=3 | not yet measured |
| `002_..._eval` | + periodic frozen-greedy evaluation | flat 0.0%, entire run |
| `003_..._per` | + prioritized experience replay | flat 0.0%, entire run |
| `004_..._per_seeded` | + per-episode seeding fix (curriculum infra built, unseeded) | flat 0.0%, entire run |
| `005_..._curriculum` | + curriculum seeded from harvested RL-policy wins | flat 0.0%, entire run |
| `006_..._curriculum_v2` | curriculum bugfix (off-by-one episode counting) | flat 0.0%, entire run |
| `007_..._curriculum_hidden3` | + deeper network (3 hidden layers) | flat 0.0%, entire run |
| `008_..._no_undo` | foundation→tableau undo disabled (`--no-undo`) | flat 0.0%, entire run |
| `009_..._decomposed_symaug` | decomposed card features + suit-swap symmetry augmentation | flat 0.0%, entire run |
| `010_..._slow_eps` | slower/longer epsilon decay (600k steps, floor at step ~300k) | flat 0.0%, entire run |
| `011_..._bc_finetune` | RL fine-tune warm-started from behavior cloning (see §8) | **flat 0.0%**, entire run (but see §8 for what did change) |

**Why each change was made:**
- **Double DQN**: vanilla DQN's `target_net.max()` systematically overestimates Q-values (max over noisy estimates is itself biased) — this is what caused the original divergence.
- **Soft target updates + lower LR**: Double DQN alone only *delayed* divergence — needed both together to actually stabilize.
- **N-step returns**: speeds up credit assignment across Solitaire's long horizons instead of relying on value slowly diffusing backward through repeated 1-step bootstraps.
- **Periodic frozen-greedy evaluation**: training-time "wins" were confounded with epsilon-driven exploration luck — this turned out to be the single most important addition, because it revealed the real story across ten full runs: **every configuration trains stably (loss converges low), and occasionally finds wins during high-epsilon exploration, but zero of them show any win competence once exploration is removed.**
- **Prioritized replay**: hypothesis was that rare successful trajectories were diluted by uniform sampling. Result: much better loss stability, no movement on eval win rate.
- **Curriculum learning**: hypothesis was that starting near a known win gives the agent a short, learnable "final stretch" to master before pushing the start earlier. Needed real seed data first (see below) — even once seeded, didn't move eval win rate off zero by itself.
- **Deeper network**: ruled out "the function is too complex for a 2-layer MLP to represent" as the bottleneck — no change.
- **Disabling undo**: shrinks the action space and removes a source of directly-reversible reward-farming, to isolate whether the undo move itself was the obstacle — no change; the sparse-reward/hard-exploration problem persisted even in the simpler variant.
- **Decomposed features + symmetry augmentation**: addressed a real, independently-verified inefficiency (the network was being asked to learn that ~everything except the specific suit is identical across ~13× more raw card-ID values than necessary) — no change to win rate, though this is a legitimate representational improvement worth keeping regardless.
- **Slower epsilon decay**: motivated by noticing wins clustering suspiciously tightly in one epsilon range (~0.09–0.62) across several runs, right where epsilon was bottoming out — tested whether more time spent in that regime helps. No change (though the RNG trajectory differs across runs with different epsilon schedules, which is itself a confound on any single win-count comparison).

**Bottom line after 10 runs**: stability is fully solved, and every representational/exploration lever we pulled works exactly as advertised on its own terms (loss, sample efficiency, action-space size) — but none of it, alone, produced a policy that wins independently of exploration noise. This is characteristic of a **sparse-reward, hard-exploration RL problem** (the same family as Montezuma's Revenge in Atari): the network optimizes just fine on the experience it gets, but near-greedy rollouts essentially never contain a full winning trajectory to learn from in the first place.

## 7. Harvesting real wins & curriculum learning

- Fixed the seeding bug (only episode 1 of each run ever got an explicit seed) so every logged episode is independently reproducible.
- Dedicated harvesting attempts using an **epsilon-greedy policy** (frozen checkpoints at ε=0.2, ε=0.5, and pure random) across thousands of episodes found **essentially zero wins** — consistent with the hard-exploration diagnosis above.
- Built `agent/heuristic_policy.py`, a hand-written scoring policy (not learned) prioritizing foundation moves, hidden-card reveals, king-to-empty-column moves, then routine tableau/waste moves, with anti-oscillation cycle detection. Used **only** to harvest wins for curriculum/BC data — never used as the RL baseline itself.
- `harvest_wins.py --policy heuristic` over 5,000 episodes found **86 real wins (1.72%)**, lengths 156–497 steps, logged to `runs/curriculum_wins.jsonl`. This is what finally unblocked curriculum learning (runs 005–006) and behavior cloning (run 011).
- Curriculum learning, once properly seeded, ran without errors but did not move eval win rate off zero by itself.

## 8. Behavior cloning (imitation-learning warm-start)

Rationale: if the RL agent essentially never experiences a full winning trajectory under near-greedy play, no amount of TD learning on that experience can teach it to win. AlphaGo-style pretrain-then-finetune uses supervised imitation of expert data to *start* the network somewhere already competent, before RL fine-tunes it.

- `agent/behavior_cloning.py`: replays all 5,000 heuristic-harvested episodes (not just the 86 wins — the heuristic's individual decisions are reasonably sound throughout, win or lose, so even losing games carry useful "what does sound play look like" signal) and trains the DQN with cross-entropy loss to predict the heuristic's chosen action at each state.
- **OOM bug and fix**: the initial `np.zeros`-based dataset build (~5GB single in-RAM allocation) was OOM-killed twice, at the *exact same point* (500/5000 episodes, 61s) both times — that precise reproducibility pointed at the single large allocation itself, not random system contention. Fixed by switching to `np.memmap`-backed disk arrays with periodic `.flush()`; the fix was verified by watching the run survive past the old kill point and complete the full 612s replay.
- **BC pretraining results** (8 epochs, 2.35M train / 48k val examples): train accuracy climbed from 34.3% → 35.7%, val accuracy flat around 34.7–35.0%, most of the gain in epoch 1. Honest read: ~35% top-1 exact-action match is a soft imitation signal, not a tight clone — many states have several roughly-equivalent legal moves, so the ceiling for "exactly match the heuristic's tie-breaking" is well under 100% even for a perfect model. Checkpoint saved to `runs/bc_pretrained.pt`.
- **Fine-tune run 011** (`runs/011_double_dqn_soft_nstep3_per_bc_finetune`): 400k steps, warm-started from `bc_pretrained.pt`, otherwise our cleanest baseline config. Result:
  - **Greedy eval win rate stayed at 0.00% across all 40 checkpoints (10k–400k steps), 4,000 total greedy games, zero wins.**
  - It *did* move some numbers: mean foundation depth (greedy eval) rose from ~2.4/52 early to a peak of ~7.0/52 around step 310k–360k — faster and higher than cold-start runs typically reached — and episodes stopped hitting the full 500-step truncation, ending via the stall-detector instead.
  - But by step 400k, mean foundation depth had **regressed back down to ~4.0/52**, and only 9 total organic training-time wins accumulated across 1,876 episodes (barely more than the 8 already logged by step 300k) — almost no further organic wins in the back half of the run.
  - **Working theory**: this looks like catastrophic forgetting of the imitation prior. Plain BC-then-finetune only uses demonstrations to *initialize* weights; ordinary TD learning is then free to drift away from that prior over 400k steps, and with wins this rare there's essentially no reward signal to hold it in place. The properly-named technique for avoiding exactly this is **DQfD (Deep Q-learning from Demonstrations, Hester et al. 2017)**, which keeps demonstration transitions *permanently* in the replay buffer alongside self-generated experience and adds a supervised large-margin loss term throughout training (not just at pretraining time) — not something we've implemented yet.

## 9. Infrastructure lessons (the expensive ones)

- **Twice deleted real run data via `rm -rf runs`** while trying to clean up smoke-test directories. Fixed structurally: smoke tests now always use `--runs-dir runs/_smoketest`, a completely separate tree from real runs.
- **Training-loop OOM kills**: run 004 died to a system-level OOM kill 7 times in a row. Root-caused via isolated component testing to a real, reproducible ~9–10MB/2000-step combined-pipeline growth (present on both CPU and GPU), but no definitive single root cause was found (no cgroup/ulimit constraint; a concurrent unrelated user task was a likely contributing factor but kills persisted after). Resolved pragmatically with the auto-resume wrapper (`train_resilient.sh`) rather than continuing to chase the exact mechanism.
- **Behavior-cloning dataset OOM**: separately diagnosed and *definitively* fixed (see §8) — the "identical kill point twice" signature made this one much less ambiguous than the training-loop OOM above.
- **`train_resilient.sh` resume-arg collision**: see §5 — caught before it could silently discard fine-tuning progress on a restart.

## 10. Current state / open questions for next time

- Environment, TUI, logging, the full DQN+Double-DQN+n-step+PER+curriculum+eval training pipeline, decomposed features + symmetry augmentation, a heuristic win-harvesting policy, and a behavior-cloning pretraining pipeline are all built, tested, and working end-to-end.
- **No policy has yet demonstrated a non-zero win rate under frozen-greedy evaluation, across 11 runs and every lever tried** (Double DQN, soft updates, n-step, PER, curriculum, a deeper network, undo removal, decomposed encoding + symmetry augmentation, slower epsilon decay, and BC warm-start). This is the central honest negative result of the project so far.
- The BC warm-start run (011) is the first to show a real, if temporary and ultimately regressing, qualitative shift (faster/deeper foundation progress, organic training-time wins under exploration) — suggesting the imitation signal is doing *something*, just not enough to survive 400k steps of fine-tuning on a still-extremely-sparse reward signal.
- Natural next directions discussed but not yet started:
  - **DQfD-style permanent demonstration replay + margin loss**, to directly address the apparent forgetting of the BC prior during fine-tuning.
  - **BC warm-start + curriculum combined** (start fine-tuning near real wins while the network is still close to its imitation-trained prior).
  - **Qualitative inspection**: replay a mid-training checkpoint (e.g. `runs/011_.../checkpoint_350000.pt`, near peak foundation depth) by hand to see *what specifically* the greedy policy does when it gets stuck, rather than only tracking summary statistics.
  - Widening the eval set size/composition, in case a handful of unusually hard deals in the fixed eval set are masking real partial progress.
