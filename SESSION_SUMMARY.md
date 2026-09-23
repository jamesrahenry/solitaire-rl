# Session Summary — Solitaire RL Project (2026-09-11 → 2026-09-15)

An end-to-end session building a Klondike Solitaire Gymnasium environment from scratch and training a DQN agent against it, for an RL class. Below is what was built, what was learned, and where things stand. (This supersedes the 2026-09-14 version of this file, which stopped after run 014 and the epsilon-reheat experiments.)

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

## 9. Infrastructure lessons (the expensive ones, part 1)

- **Twice deleted real run data via `rm -rf runs`** while trying to clean up smoke-test directories. Fixed structurally: smoke tests now always use `--runs-dir runs/_smoketest`, a completely separate tree from real runs.
- **Training-loop OOM kills**: run 004 died to a system-level OOM kill 7 times in a row. Root-caused via isolated component testing to a real, reproducible ~9–10MB/2000-step combined-pipeline growth (present on both CPU and GPU), but no definitive single root cause was found (no cgroup/ulimit constraint; a concurrent unrelated user task was a likely contributing factor but kills persisted after). Resolved pragmatically with the auto-resume wrapper (`train_resilient.sh`) rather than continuing to chase the exact mechanism.
- **Behavior-cloning dataset OOM**: separately diagnosed and *definitively* fixed (see §8) — the "identical kill point twice" signature made this one much less ambiguous than the training-loop OOM above.
- **`train_resilient.sh` resume-arg collision**: see §5 — caught before it could silently discard fine-tuning progress on a restart.

## 10. A real engine bug found by hand: premature stall detection (2026-09-14)

Playing `play.py` by hand, the user hit a case where a draw revealed a card that was immediately legal to play to a foundation — but the game declared "stuck" instead of offering the move. Root cause in `SolitaireGame.is_stalled()`: it declared a stall purely by counting consecutive draws against the stock+waste pool size, without ever checking whether the draw that hit that count had *also* just revealed a playable card. Since the tableau/foundation state is frozen for a whole no-progress draw streak, the very last card of an exhausted cycle could be exactly the one that unlocks a move — and the old code ended the episode in that same step, before the move could be taken.

**This affected every training run to date (001–012), not just the TUI** — `SolitaireEnv.step()` calls `is_stalled()` directly. Fixed by also requiring that no legal non-draw action exists once the draw-count threshold is reached, rather than trusting the count alone. Run 012 (already in flight when this was found) was killed and replaced by run 013 under the fix, rather than let a confounded run finish.

Separately, the fixed engine exposed a second, unrelated, structural obstacle: replaying a real truncated training episode's action log directly revealed a literal 2-action loop — `foundation→tableau (undo)` immediately followed by `tableau→foundation` on the same card, repeated for hundreds of steps until truncation. Both actions net zero reward (undoing costs nothing; replaying the same card doesn't exceed the foundation high-water mark), meaning the network's Q-values were tied or near-tied between them in that state, and greedy argmax just alternated forever with nothing pushing it off that ledge. This ties directly back to [`solitaire_undo_move_decision.md`]'s original framing (undo kept for rules-fidelity, "treat it as an RL training problem later") — it looks like this *is* that problem showing up concretely: the undo action hands an imperfect Q-network an easy, zero-cost treadmill to get stuck on, a candidate structural cause alongside the sparse-reward/hard-exploration diagnosis, not a replacement for it.

## 11. Reward function reshaped (2026-09-14)

Two motivations converged: the sparse-reward diagnosis suggested milestones should be spaced out more meaningfully, and — independently — the user noticed the win bonus was proportionally tiny next to what a full clearing already banks along the way (a complete game nets roughly 2,050 from milestone bonuses alone under the new scale below), giving weak marginal incentive to actually finish versus stopping just short.

- **10x rescale** of every existing reward (foundation high-water +10, reveal +50, step -0.01, illegal -10).
- **Three new one-time milestone bonuses**, all following the same "reward genuine progress once, not the repeatable act of achieving it again" principle as the existing foundation high-water mark:
  - `+40` first time **a given king** (tracked per-suit — there are only 4) lands on an empty column. Gated per-king rather than per-column deliberately: per-column gating would let a single king tour every never-before-used empty column and farm the bonus repeatedly (up to 7x with only 4 real kings); per-king gating caps it at 4 total, ever.
  - `+30` first time a column's *last* hidden card is revealed (zero face-down cards left) — stacks with the existing `+50` reveal bonus for that flip.
  - `+20` first time a column reaches zero *total* cards — distinct from "uncovered" (a column can be uncovered long before it's ever emptied), and specifically covers clearing column 0's single starting card, which has no hidden cards to ever trigger the uncovered bonus.
- **Win bonus raised from +100 to +5000** — deliberately dominant over the ~2,050 available from everything else combined.

All verified via targeted random-play tests (correct one-time/per-king firing, correct additive stacking) and a full training-loop smoke test. Gradient-norm clipping (already active, norm 10) bounds update size regardless of the larger reward scale; the practical effect was loss values reading ~10-25x larger than before, which is expected, not new instability.

## 12. Episode length ceiling raised 500 → 1000

Prompted by a direct question ("is 500 too low?") and confirmed with real data before acting: 19 of the (then) 136 known wins in `runs/curriculum_wins.jsonl` already took ≥480 of the old 500-step cap, one hit exactly 500 — real, non-negligible truncation risk for harder deals even under already-decent play. Raised to 1000 (`solitaire_gym/__init__.py`). Roughly doubles per-episode compute and how far reward has to diffuse backward through n-step TD bootstrapping, but removes a confirmed truncation risk.

## 13. Runs 012–014: reshaped reward + bug fixes + episode budget + epsilon reheat

| Run | Change | Eval win rate |
|---|---|---|
| `012_..._reward_shaping` | Reshaped reward function alone, cold start — **but run under the still-buggy `is_stalled()`** | Killed mid-flight once the bug was found (confounded, not a clean result) |
| `013_..._reward_shaping_stallfix` | Same, re-run clean under the fixed engine, 400k steps | flat 0.00%, all 40 checkpoints |
| `013_..._reheat` (continuation) | + single epsilon reheat at step 400k (→0.4, decay to floor by 450k), 50k more steps | flat 0.00%; **zero new wins** during the reheat (10 total, unchanged from the base run) |
| `014_..._reward_shaping_reheat_echo` | 1000-step ceiling + **dual** epsilon reheat (step 400k→0.4, step 500k→0.2 "echo"), 600k steps, cold start | flat 0.00%, all 60 checkpoints |

**Run 013's honest result**: reward reshaping + the stall fix, alone, did not break the plateau. The reheat found zero new organic wins on top of the base run's 10.

**Run 014 told a more nuanced story.** With the doubled episode ceiling, this run found training-time wins far faster and in far greater quantity than any prior run — **4 wins by step 70,000 at the identical epsilon value where run 013 had zero** (before either reheat had even fired), directly implicating the longer ceiling, not epsilon, for that early difference. Wins climbed to 40 by step 150k, and kept climbing before flattening at 59 once epsilon hit its floor around step 200k (same clustering-in-high-epsilon pattern seen throughout the project, just at a much larger scale: **59 organic wins in one run vs. 10 in run 013's entire base run**). The first reheat (step 400k) reignited win discovery clearly — 59 → 75 by step ~506k. The echo reheat (step 500k, lower peak of 0.2) found **zero** additional wins. **Across all 60 eval checkpoints, greedy win rate never once moved off 0.00%** — including every single checkpoint during and after both reheats. Loss ran hotter than any prior run (peaked ~12.7 around step 150k, fully recovered to ~1.1 by the end) but never diverged.

**Bottom line**: the engine bug fix and longer episode ceiling are real, load-bearing improvements — they dramatically increase how often *exploration* stumbles into a win. Neither reward reshaping nor either flavor of epsilon reheat has yet caused the *trained/greedy* policy to win even once. The user's own framing of this mid-session: "I don't want random wins, I want trained wins" — exploration volume and policy competence are turning out to be almost entirely decoupled variables in this problem so far.

## 14. Is epsilon-reheating even a real RL technique?

Asked directly and answered honestly: no, not as we implemented it. Standard epsilon-greedy decay is textbook (Mnih et al. 2015). Manually re-spiking epsilon partway through training is not a published, benchmarked technique with theory behind it the way Double DQN/PER/n-step are — it's a heuristic loosely inspired by *simulated annealing's* "reheating" (real term, different field) and *Cyclical Learning Rates* (real, peer-reviewed, but for the learning rate, not epsilon). The properly-validated techniques the field actually uses for this exact symptom (hard exploration, sparse reward) are NoisyNets (Fortunato et al. 2017 — learned, self-annealing per-weight noise), Random Network Distillation (Burda et al. 2018 — intrinsic novelty reward), Go-Explore (Ecoffet et al. 2019 — explicitly "remember and return to promising states, then explore further," much closer in spirit to our own curriculum learning than to epsilon reheat), and Agent57/Never Give Up (Badia et al. 2020 — a population of policies with a *spread* of exploration rates run in parallel, rather than one schedule over time).

## 15. Growing the win corpus (2026-09-14)

Two gaps found and closed, both around the same theme: wins were being discovered but not making it back into the shared training corpus (`runs/curriculum_wins.jsonl`).
- **Human wins**: `play.py` already logged games via the same `GameLogger` schema as the corpus, but nothing folded them in. Added `agent/sync_wins.py` (idempotent, dedupes by exact seed+action sequence) — sweeps both the human play log and every run directory's own `games.jsonl` for wins and appends new ones to the corpus. First run picked up 49 previously-uncounted wins spanning runs 001–012 (the "won" classification is reward-scale-independent, so old- and new-reward-era wins both count) plus the user's own hand-played win, taking the corpus from 86 to 136 wins.
- **Scaling up harvesting itself**: reasoning directly from the numbers — 9-10 organic DQN-exploration wins is nowhere near enough density for a network to generalize "what does winning look like" across a huge state space, even with a 5000-point win bonus rewarding the specific transitions involved. The fix is more harvested wins, cheaply, via the heuristic policy (pure CPU rollout, no GPU/training involved) rather than waiting on sparse DQN exploration. Launched a 50,000-episode heuristic harvest (`--seed 456`, distinct from the original run's seed to avoid re-exploring the same deals) in parallel with run 014's GPU training. Its hit rate came in far higher than the original 5,000-episode harvest's 1.72% — steady at **~13%**, an ~8x jump. Best explanation: that original harvest predated both the `is_stalled` fix and the 500→1000 ceiling increase, so a meaningful fraction of the heuristic policy's actual wins were likely being suppressed by exactly those two bugs the whole time — independent, convergent confirmation that both fixes mattered substantively.

## 16. Visualization fixes and additions (2026-09-14)

- **Trendline** (rolling mean overlay) added to both "mean foundation total" panels (training-window and eval), since per-point noise was hiding the real trend by eye.
- **"Trained wins vs random wins per checkpoint" panel** added to fill the previously-empty plot slot — directly compares, at each eval checkpoint, how many of the 100 fixed eval deals the greedy ("trained") policy solved against how many wins epsilon-exploration ("random") found in training since the last checkpoint. Answers the user's core question ("I don't want random wins, I want trained wins") visually: on run 014's data, trained stays flat at 0 the entire time while random visibly rises and falls.
- **`lifetime_wins` tracking bug across resumes**: the counter (and a historical-backfill reconstruction of it) both assumed per-process counters were globally monotonic, which breaks at every `--resume-from` boundary (a fresh process's own `episode_count` restarts at 0). Fixed by seeding from the run's own `games.jsonl` at startup (ground truth, append-only, unaffected by restarts) and by making the historical backfill segment-aware (detects the counter reset and carries the *win* total across it, since wins have no reason to reset just because the process did).
- **`MetricsLogger` schema migration**: appending to an existing `metrics.csv` under code that logs a newly-added column used to crash outright (`csv.DictReader` stuffs the extra value into a `None`-keyed list). Fixed to migrate the file in place (rewrite with the new header, backfill old rows) instead of crashing or corrupting column alignment.
- **Atomic PNG saves**: `fig.savefig()` writes the destination path directly and non-atomically; a viewer polling the file (reported: over a WSL/Windows mount) could catch it mid-write during a live run and see a torn image (top rendered, bottom cut off — PNGs render top-to-bottom). Fixed by writing to a temp file and atomically renaming over the destination.

## 17. Epsilon floor experiment (run 015) and disk-management interlude

**Run 015**: same clean config as run 013, but with a permanently higher epsilon floor (`--eps-end 0.15` instead of 0.05) — the cheapest untested lever, since run 010 ("slow eps") only ever stretched the *decay rate*, never the floor itself. Result: **it worked exactly as hypothesized on the exploration side** — comparing the 50k-step windows immediately before/after epsilon settles at its floor (step 200,000), win discovery continued at essentially the same rate (17 new wins pre-floor vs. 16 post-floor), instead of the hard flatline every prior run showed at the 0.05 floor (run 013 found zero new wins post-floor; run 014 found only a trickle until its reheat). Final: 89 organic wins, the richest single-run total yet - but **greedy eval was still flat 0.00% across all 40 checkpoints.** Closes out the epsilon-tuning family of experiments (reheat, dual reheat, higher floor) with a consistent conclusion: every one of them worked as intended on exploration volume, none transferred to the trained policy.

Also this session: disk usage crept up to 90% (26GB free) from the accumulating checkpoints (606 files, 8.3GB, across 15 runs) plus a large BC memmap cache. Rather than set up external cloud storage, found and used a much simpler option already available: this machine is WSL, with a Windows partition mounted at `/mnt/c` (230GB free). Moved all historical checkpoints there (`/mnt/c/Users/Public/solitaire_checkpoints/<run_name>/`) without touching the actively-running BC job's own scratch cache. No data lost, nothing in git was affected (checkpoints were already gitignored).

## 18. Corpus growth and a second data-corruption bug found via BC

- **`agent/sync_wins.py`** generalized (from a human-play-only script) to sweep every run's own `games.jsonl` too, since `agent/train.py` never feeds a run's organically-discovered wins back into the shared corpus on its own. First sweep recovered 49 previously-uncounted wins spanning runs 001–012, taking the corpus from 86 to 136 wins.
- **A 50,000-episode heuristic harvest** (`--seed 456`, distinct from the original harvest's seed) was run in parallel with training. Hit rate came in at a steady ~13% — about 8x the original 5,000-episode harvest's 1.72%, strong independent confirmation that the `is_stalled` bug and the 500-step ceiling had been suppressing a large fraction of the heuristic policy's true win rate the whole time. Final corpus: **55,050 episodes, 6,821 wins (12.39%)** — roughly 50x more win data than at the start of this session.
- **Re-running BC against this much larger corpus surfaced a second real data-corruption bug**, this one in `agent/behavior_cloning.py` itself. First attempt (a disk-budget-safe 14,954-episode subset: all 6,821 wins + a random sample of non-wins) produced a training loss of **~570,000** — nonsense for a 590-class cross-entropy problem. Root-caused to 6,889 examples (0.057%) whose recorded "correct" action was marked illegal in its own freshly-computed mask; cross-entropy pins an illegal action's logit at -1e9, so those examples' loss was individually astronomical, dominating the batch mean at roughly one bad example per batch. Cause: some of the recovered wins came from curriculum-enabled training runs, whose episodes start via `env.reset(options={"initial_moves": [...]})` - a fast-forward that happens inside the wrapped env's `reset()`, bypassing `GameLogger.step()` entirely, so the logged action list is missing its prefix. Replaying "seed + tail-only actions" from a fresh deal diverges from the state those actions were actually chosen for. `env.step()` on an illegal action silently no-ops rather than raising, so the existing `assert idx == total_steps` sanity check could never catch this.
- Fixed by validating each action's legality against its freshly-computed mask before recording it, truncating the rest of an episode at the first point of divergence rather than recording corrupted examples. Verified on a sample (correctly found and dropped 4/3000 diverged episodes, 0 illegal-target examples remaining) before re-running the full dataset.
- **Corrected result**: train_acc=30.9%, val_acc=30.7% (hidden_dim=512) — flat after epoch 1, and genuinely *lower* than the original small run's 35.7%/34.7%, despite ~5x more data and ~79x more wins. Tested whether this was a network-capacity ceiling by re-running at hidden_dim=1024 (4x the parameters, reusing the cached replay via a new `build_dataset()` caching mechanism rather than repeating the ~50-minute replay): **train_acc=31.1%, val_acc=30.9% - no meaningful change.** Rules out capacity; the likely explanation is that a larger, more diverse corpus of winning trajectories has more genuinely-different-but-equally-valid "correct" actions per state than the original corpus's narrow, single-heuristic-style homogeneity, which caps top-1 imitation accuracy independent of model size.
- **Run 016**: RL fine-tune from the wider BC checkpoint (hidden_dim=1024, the larger/richer corpus), otherwise identical to run 011's setup. Result: same story as everywhere else - 63 organic training-time wins, **greedy eval flat 0.00% across all 40 checkpoints.**

## 19. DQfD implementation and first run (017)

Implemented DQfD (Hester et al. 2017) properly rather than another BC variant: permanent demonstration transitions in the replay buffer (a reserved region self-play's insertion pointer never overwrites, populated once from real wins) plus a supervised large-margin loss applied to demo rows every batch throughout training, not just at initialization. Built on top of the existing Double DQN + PER + n-step infrastructure rather than replacing it. Key pieces:
- `ReplayBuffer`/`PrioritizedReplayBuffer` gained `num_demo`/`load_demos()`; `PrioritizedReplayBuffer` also gives demo transitions a much larger priority floor (`--demo-priority-eps`) so PER doesn't let them fade out as their TD-error shrinks.
- `agent/dqfd_demos.py` builds real n-step transitions from **wins only** (not the full BC corpus) - the margin loss only makes sense anchored to genuinely good actions, unlike BC's "imitate everything, win or lose" objective. Reuses the same divergence-detection/truncation fix from `behavior_cloning.py` for the same underlying reason (curriculum-episode replay gaps).
- `compute_loss()` adds `max_a[Q(s,a) + margin(a,a_E)] - Q(s,a_E)` for demo rows only, on top of the existing TD loss.
- All new args default to fully disabled, so every prior run stays exactly reproducible; verified via targeted unit tests (buffer demo-region preservation, priority floor, hand-calculated margin loss values) before running anything real.

**Run 017** (cold start, 30,000 permanent demo transitions from the win corpus, 150,000-capacity buffer, margin=0.8, margin-loss-weight=1.0, otherwise the usual clean baseline): loss declined smoothly and monotonically the entire run (87.7 → 1.1, no spike-then-recover pattern, unlike every reward-shaping-driven run) - the cleanest, most stable optimization trace of any run so far. 64 organic training-time wins. **Eval win rate 0.00% across all 40 checkpoints.** First attempt at the most mechanistically-differentiated remaining lever, and it landed in the same place as everything else.

## 20. DQfD hyperparameter sweep (018-020) and the first trained win

Three variants tried against run 017's baseline (30,000 demos, margin=0.8, margin-loss-weight=1.0, default 0.05 epsilon floor):
- **Run 018** (demo transitions 30,000 → 100,000, buffer capacity 150,000 → 300,000): 70 organic wins, loss ~1.78 final, **eval 0.00% across all 40 checkpoints.** Demo density wasn't the limiting factor.
- **Run 019** (margin-loss-weight 1.0 → 5.0): 75 organic wins, loss ~2.23 final, **eval 0.00% across all 40 checkpoints.** Stronger margin enforcement wasn't either.
- **Run 020** (combined DQfD with the higher epsilon floor from run 015, `--eps-end 0.15` instead of 0.05, everything else matching run 017): 89 organic wins, loss ~1.4 final. **Eval win rate broke 0% for the first time all session** - `EVAL step 250000 win_rate=1.00%` (1/100 fixed deals). Every other one of its 40 checkpoints, before and after, was flat 0.00% (the very next checkpoint, step 260000, reverted to 0%) - a single isolated data point, not a sustained capability, but the first reproducible one anywhere in the project.

Verified directly rather than trusting the number blind: reloaded `checkpoint_250000.pt` and replayed all 100 fixed eval deals under pure greedy (argmax, no exploration) policy outside the training loop. **Confirmed**: eval seed 900096 solved deterministically in 132 steps, total reward 7048.68 (matching a genuine full clearing under the current reward scale). Fully reproducible - greedy argmax has no randomness, so this isn't a fluke of stochastic evaluation.

Honest framing: this is real evidence that the combination (permanent demonstration anchoring + sustained, undiminished exploration providing a steady stream of fresh near-win experience) can occasionally produce a genuinely competent greedy policy for a specific deal - not evidence that the approach reliably produces one. 89 organic training wins and 400,000 steps produced exactly one eval-deal solve, once, non-persistently. Still enormously sparse, but categorically different from a flat 0.00% that never moved once across 19 prior runs.

## 21. The real breakthrough: the 0% wall was substantially a measurement artifact

Investigating run 020's single win by hand (comparing the human's and the trained model's solutions to the same deal, then diffing neighboring checkpoints' behavior on it) found something bigger than the win itself: checkpoints right next to the winning one weren't strategically incompetent - they were getting stuck oscillating between 2-3 actions with near-tied Q-values in an exact repeated game state, burning the *entire* step budget with zero further progress. Structurally identical to the undo-loop failure mode found earlier (§10), just a different pair of actions.

This is sound to fix, not just a heuristic patch: Klondike has no randomness once dealt (draws are deterministic given the fixed stock order, hidden cards' identities never change), so an exact state repeat within an episode can provably never reach anywhere the earlier visit couldn't already reach - the same logic `is_stalled()` already relies on for "cycled the whole stock with no progress," generalized to any exact state repeat. Implemented `agent/loop_breaker.py` (`LoopBreakerWrapper`, a `gym.Wrapper` - agent-side, not a change to the environment's own legality rules, so human play via `play.py` is unaffected): tracks exact states visited this episode; on the 3rd visit to any given state, masks out whichever action(s) were taken from it before, forcing a genuinely new choice (or falling back to normal behavior if nothing legal remains).

**Swept every prior run's final checkpoint with and without this wrapper - no retraining, same exact weights, just a different action-selection wrapper at evaluation time** (`agent/sweep_loop_breaker.py`, full results in `runs/loop_breaker_sweep.log`):

| Run | Without loop-breaker | With loop-breaker |
|---|---|---|
| 001 (original baseline, before PER/curriculum/reward-shaping/DQfD) | 0/100 | **26/100** |
| 002 (+ eval) | 0/100 | 26/100 (identical net to 001) |
| 003 (+ PER) | 0/100 | 0/100 |
| 004 (+ seeded) | 0/100 | 0/100 |
| 005 (+ curriculum) | 0/100 | 1/100 |
| 006 (curriculum v2) | 0/100 | 0/100 |
| 007 (+ hidden3) | 0/100 | 0/100 |
| 008 (no undo) | 0/100 | 0/100 |
| 009 (decomposed+symaug) | 0/100 | 19/100 |
| 010 (slow eps) | 0/100 | 0/100 |
| 011 (BC finetune) | 0/100 | 18/100 |
| 013 (reward shaping+stallfix) | 0/100 | 5/100 |
| 014 (dual reheat) | 0/100 | 8/100 |
| 015 (epsilon floor 0.15) | 0/100 | 20/100 |
| 016 (BC wide finetune) | 0/100 | 15/100 |
| 017 (DQfD baseline) | 0/100 | 27/100 |
| 018 (DQfD, 100k demos) | 0/100 | **38/100 (best of all 19)** |
| 019 (DQfD, margin=5.0) | 0/100 | 22/100 |
| 020 (DQfD + eps floor) | 0/100 | 27/100 |

(012 skipped - no final checkpoint, killed mid-flight for the `is_stalled` bug.)

**Every single one of the 19 evaluated runs showed exactly 0/100 without the loop-breaker** - the "flat 0% eval win rate" that held across the entire project wasn't primarily a training failure, it was this same argmax-degeneracy artifact, universally. With it removed: all four DQfD runs (017-020) land in the strongest tier (22-38%), and **run 018's "demo count doesn't matter" conclusion needs revising** - it's actually the single best run of the project (38%) once the artifact is removed, meaningfully ahead of run 017 (27%) with 3x fewer demos. Some runs (003, 004, 006, 007, 008, 010) show genuinely ~0 even with the fix - real incompetence, not masked competence, so this doesn't retroactively validate everything either. And run 001, the very first and simplest baseline, already had 26% real capability the whole time.

Wired into `agent/train.py` on both the training and eval envs, on by default (`--no-loop-breaker` to disable and exactly reproduce runs 001-020's behavior), following the same default-on/opt-out pattern as `--double-dqn`/`--prioritized-replay`.

## 22. Run 021: the loop-breaker wired in from the start of training

Combined the two strongest post-hoc configurations - DQfD with 100,000 demo transitions (run 018) and the higher epsilon floor (run 020) - with `LoopBreakerWrapper` active during training itself, not just eval, testing whether self-play also benefits from not wasting experience on oscillation loops. Otherwise the usual clean baseline, cold start, 400,000 steps.

**Result: eval win rate sustained 22-39% across effectively the entire run** (39 checkpoints, low outlier 11% at step 100k, peak 39% at steps 360k-370k). This is categorically different from every one of the previous 20 runs. Specifics:
- First eval checkpoint (step 10,000) was already at 29% - instantly, not after a long warmup.
- No degradation through the epsilon-decay-to-floor transition (~step 200,000) - if anything the opposite: first-half average (10k-190k) ~26%, second-half average (200k-400k) ~30.6%, with the run's two best checkpoints (39%) coming late, near the end.
- Final: 33% eval win rate, 167 organic training-time wins out of 482 episodes (~35%, consistent with eval), loss settled at ~2.3, no OOM/restart.

This validates the hypothesis directly: training-time exploration really was wasting a substantial fraction of its budget on the same oscillation loops that were masking eval results, and fixing that during training (not just measuring around it at eval time) compounds with DQfD + the higher epsilon floor into a policy that reliably wins roughly a third of held-out deals - not once, not fragile, sustained for the entire second half of a 400k-step run.

## 23. Run 022: tripling run 021 to 1.2M steps

Same recipe as 021, steps 400k -> 1.2M, to see whether the win rate kept climbing or had already plateaued. Quarterly means 24.3% -> 28.5% -> 30.1% -> 30.8% (021's own 4 quarters) continuing into 022's own trajectory of 26.1% -> 29.2% -> 30.6% -> 32.9% (overall 29.7%, peak 46%, min 15%) - still climbing, decelerating but not flat, while loss kept falling steeply the whole time with no sign of convergence. Conclusion at the time: real headroom left, worth a much longer hold - which is exactly what motivated tripling again later (run 024, §28).

## 24. Ground-truth solvability: is a seed nothing has ever won actually unsolvable?

Pooled every run's final checkpoint (`agent/seed_difficulty_pool.py`) plus run 021's full 40-checkpoint history against the fixed 100-seed eval range: **72/100 seeds won by something, ever; 28/100 never won by anything across the whole project.** That raised the obvious question - are those 28 dead deals, or just hard?

Sourced (not written) a real answer: cloned and built `ShootMe/Klondike-Solver` (github.com/ShootMe/Klondike-Solver, C++, lives outside this repo), a full-information exhaustive/branch-and-bound solver. `agent/klondike_solver_check.py` converts one of our env's already-dealt boards into its deck-string input format - validated by an exact character-for-character match between the solver's echoed board and our own engine's board for a known seed, not just a coincidentally-correct verdict. (Also found and worked around a real bug in the tool: its self-reported "Took N ms" is off by ~1000x on Linux, a classic `clock()`/`CLOCKS_PER_SEC` portability bug from Windows-authored code - cosmetic only, doesn't affect verdicts.)

Result on the 28 never-won seeds: **20 proven solvable** (real, currently-unmet capability gaps, not dead seeds - minimal solutions ranging 117-206 moves), **1 proven unsolvable** (seed 900033), **7 unresolved** within the solver's search cap (proving impossibility costs far more than finding a solution). Recorded in `runs/seed_classification.json`; the unsolvable one plus the 7 unresolved are treated as excluded from clean win-rate accounting given our 1000-step episode budget.

Separately, `agent/seed_churn.py` built a (40 checkpoint x 100 seed) win/loss matrix for run 021: only 1/66 ever-won seeds was solved monotonically (won once, stayed won); the other 65 flip win/loss repeatedly across checkpoints (one seed flipped 14 times). But first-half vs. second-half per-seed solve-rate correlation was 0.889 - so there's a real, fairly stable underlying difficulty ranking under all that checkpoint-to-checkpoint noise, consistent with the argmax-tie-breaking instability already diagnosed in §21, not the network forgetting and relearning whole strategies.

## 25. Two new solver-proven seed corpora

`agent/build_solvable_corpus.py` (generalized, reused for both): draws fresh candidate seeds (a range distinct from the fixed 100-seed eval pool), checks each against the ground-truth solver, keeps the ones that resolve favorably within a modest search cap - no need to prove *why* a reject failed, just discard and move on.

- **`runs/solvable_corpus.jsonl`: 1,578 seeds, each proven solvable in under 250 moves** - for curriculum training. A stronger label than the existing heuristic-policy-won corpus (`curriculum_wins.jsonl`, actually only 6,821 real wins out of 55,050 harvested episodes, not "50,000 known solvable" as it first looked from the line count alone), which only proves solvable-at-all with no efficiency guarantee.
- **`runs/eval_corpus.jsonl`: 1,000 seeds, no move-count cap** - a much larger eval battery than the original fixed 100. Deliberately *not* filtered toward easy deals (that would bias measured win rate away from the real task distribution); sized to shrink eval sampling noise (binomial stderr at n=100 is ~4.6 points at a 30% win rate, a real chunk of the run-to-run swings being puzzled over up to this point).

## 26. Wiring the corpora into training: run 023

`agent/train.py` gained two independent, opt-in features: `--solvable-seed-source`/`--solvable-seed-steps` (fresh-deal episode resets draw from the solvable corpus for a configurable number of steps, then revert to fully random) and `--eval-seed-source` (evaluate against an explicit seed list instead of the fixed range). Run 023: run 021's exact recipe (DQfD 100k demos, eps floor 0.15, loop-breaker in training, 400k steps) plus a 100k-step easy-seed curriculum phase and eval against 200-of-1000 seeds from the new battery.

**Quarterly win rate 26.9% -> 30.6% -> 32.3% -> 34.8% (overall 31.2%, peak 42.5%), every quarter above run 021's corresponding one.** Caveat flagged at the time and never fully resolved: the eval battery changed too, so part of the apparent gain could be the new battery reading higher by construction (no dead seeds) rather than better training - a same-battery cross-eval of 021's checkpoints would separate the two effects (never actually run).

## 27. The rank-collapse discovery and run 024's genuine plateau

A side project (`Model_Zoo_Cartography`) contributed `agent/structure_metrics.py`: tracks `effective_dim` (participation ratio of a weight matrix's eigenspectrum - low means collapsed onto one dominant direction) and `frobenius_drift` (distance from init) per weight matrix, live at every checkpoint. Backfilled against run 023 (`agent/reconstruct_structure_metrics.py`, since it predated this instrumentation): **net.0 (input layer) collapsed from full rank to ~2 effective dimensions within the first ~100-150k steps and stayed there**; net.2 (hidden) dipped to a minimum of 36.9 around step 260-290k then partially recovered to 39.8 by 400k; net.4 (readout) declined smoothly and was **still falling at 62.8 at step 400,000, no plateau**. This is the "implicit rank collapse" pathology from the DRL literature (Kumar et al. 2020; Nikishin et al., primacy bias; Lyle et al., plasticity loss).

**Run 024** (023's exact recipe, tripled to 1.2M steps) was built to test whether net.4's decline and the win-rate climb would eventually level off together. Both did, but not in lockstep the way §21's story might have predicted: **quarterly win rate 28.7% -> 33.2% -> 32.6% -> 33.3% (regression slope collapsed from 023's 2.40 pts/100k to 0.39 pts/100k)** - a real, non-noise plateau, essentially all the improvement happening in the first ~300-400k steps. Structurally: net.2 bottomed at 3.80 (step 720k) then genuinely recovered to 4.53 by the end (confirming 023's U-shape wasn't a fluke, just later/deeper); net.4 was **still declining at 21.92 at step 1,200,000, across the entire run** - continuing to change substantially with zero corresponding win-rate improvement for the second two-thirds of training. Correction to an earlier hypothesis: net.2's recovery this time happened *after* the win-rate plateau had already set in, not coincident with an accelerating quarter as in 023 - that earlier correlation was most likely coincidental noise from one short run, not a real signal.

(Also recorded here since it happened mid-run-024: a ~3.9-hour single-window gap in the training log turned out to be the host machine sleeping, not a training bug - confirmed by the process's OS-level age not matching its internal elapsed-time counter, with no crash/restart markers anywhere in the log.)

## 28. Two hypothesis-driven interventions against the plateau, both informative negatives

**LayerNorm** (`agent/qnetwork.py`, `--layer-norm`, opt-in): inserted before each hidden layer's ReLU, targeting the exact collapse measured in §27. Run 025 (023's recipe + `--layer-norm`) **made things worse across the board**: quarterly 24.9%/24.2%/27.4%/23.8% (overall 25.1%, peak 31.5%, vs. 023's 31.2%/42.5%). Mechanism, confirmed structurally: it worked exactly as designed at net.0 (eff_dim 304.5 at step 400k, essentially no collapse) - but the collapse pressure didn't disappear, it relocated to the one layer that *can't* be normalized without corrupting real Q-values: net.6 (the readout) collapsed to **3.33 effective dimensions**, far worse than 023's uncollapsed-equivalent (net.4 at 62.8). Preventing collapse where measured just moved the bottleneck somewhere worse.

**Foundation-undo penalty** (`solitaire_gym/game.py`, `--foundation-undo-penalty`, opt-in): motivated by a direct behavioral measurement, not just intuition - run 023's game logs showed foundation-to-tableau (undo) moves are 8.42% of all moves, and **~75% of every foundation-send eventually gets undone**, for free (the code comment literally said "no penalty, and does not lower the high-water mark" - a send/undo/resend round trip nets nearly the full +10 milestone reward for nothing). Tested at two magnitudes against the same 023 baseline:
- Run 026 (penalty=5, half the milestone reward): undo recall dropped 75.3% -> 70.3%, win rate essentially unchanged (31.5% vs. 31.2% overall).
- Run 027 (penalty=20, double the milestone reward): undo recall barely moved further, 70.3% -> 70.2% - **statistically identical to the 5x weaker penalty.** Win rate ticked up slightly (32.7%) but within each run's own noise band.

Clean, decisive negative: this isn't a "penalty wasn't big enough" story. Quadrupling it produced no further behavioral change, meaning the remaining ~70% of undos are load-bearing, strategically necessary moves the network correctly judges worth almost any cost - a real, legitimate Klondike skill (temporarily banking then reclaiming a card to enable tableau sequencing), not a fixable mistake. Reward-shaping this specific behavior isn't the lever that breaks the ceiling.

A parallel structural investigation (never turned into its own run) found real - if inconclusive - texture underneath these results: net.0's input columns reading foundation/waste-related features collapse to a *more* extreme effective dimension (~1.16) than tableau-reading columns (~4.4), via far larger drift (14x vs. 2x) and *growing* weight magnitude, not neglect - and about a quarter of net.0's hidden units (128/512) have already organically specialized toward that tiny 80-column feature group (41-49% of their weight-norm, vs. ~1.4% if unspecialized), entirely without architectural encouragement. The network partially self-organizes task specialization on its own; nothing downstream (net.2, net.4) respects that split, though. Raised as a candidate multi-head/MoE architecture idea, not yet tested.

## 29. Supporting infrastructure and UX picked up along the way

- `play.py` gained an end-of-game summary (win/stall/truncated/quit): seed (now always concrete, even for `new` games that never had one before), moves made, illegal-attempt count, a move-type breakdown, final foundation total, whether/when the tableau was ever fully uncovered, final score.
- `GameLogger` now records `fully_uncovered_step` per episode - checked against real run-021 wins: the tableau is fully uncovered at ~69% of a win's total move count on average, confirming (with real numbers, not just intuition) that uncovering dominates game length, though "pretty much over" after that oversold it - a third of moves are still real work.
- `agent/metrics.py` fixed two real plotting bugs found by eye, not by instrument: `mean_return`/`mean_episode_len` (training) and `eval_win_rate`/`eval_mean_return` (eval) were missing the trend line `mean_final_foundation` already had, making them unreadable noise clouds and visually overestimating how many episodes score low; and `_rolling_mean`'s edge-padding scheme could let one noisy first data point read as a false ceiling (confirmed on run 022's `eval_mean_foundation`, where the very first, near-random-policy eval point happened to score anomalously high) - replaced with a proper expanding-then-rolling window. Also gave `stalled_rate` (0.6-0.8% of episodes, confirmed still real and happening throughout training, not eliminated) its own dedicated auto-scaled panel, since it was an invisible sliver inside the outcome stackplot.

## 30. Current state / open questions for next time

- **Where we actually stand:** the project's central result (§21-22) still holds - DQfD + epsilon floor + loop-breaker-in-training reliably produces a policy winning ~30-35% of held-out deals, sustained, not fragile. Runs 023-027 layered a solver-proven curriculum/eval battery on top (real, if not fully isolated from measurement-battery effects) and then hit a genuine plateau (§27) that two targeted interventions (§28) both failed to break - one made things worse (LayerNorm), one confirmed the target behavior is a feature, not a bug (undo-penalty).
- **Unresolved from §26:** never ran the same-battery cross-eval (021's checkpoints on the 1000-seed battery, or 023's on the old 100-seed range) that would cleanly separate "the new stuff made training better" from "the new eval battery reads higher by construction."
- **The one untested idea with a real structural case behind it now:** a multi-head/MoE split giving foundation-related decisions their own pathway, motivated by §28's finding that the network already partially self-organizes this specialization on its own (a quarter of net.0's units), but nothing downstream preserves it, and by LayerNorm's failure mode showing that collapse pressure has to go *somewhere* if not given a dedicated outlet.
- Still open from the original §23 (never revisited): why runs 003/004/006/007/008/010 showed genuinely ~0 win rate even with the post-hoc loop-breaker fix applied - real incompetence, not just a masked artifact.
  - Longer training and/or a broader eval set (more than 100 fixed deals) to get a tighter estimate of run 021's true win rate and see if it climbs further or has plateaued.
