# Run: dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_foundramp

- **Started:** 2026-09-27T13:46:21
- **Git commit:** 6d88490f015b25db06d19243d17b1042f9dd1363 (dirty working tree)

## Notes

Direct A/B against run 023 (identical recipe and duration otherwise) - only change is --foundation-reward-ramp 2.0, phasing REWARD_NEW_FOUNDATION_HIGH from 1x up to 3x continuously as hidden tableau cards get revealed (1.0x at the deal, 3.0x once fully uncovered). Motivated by the reward function already being tableau-heavy (REWARD_REVEAL=50/COLUMN_UNCOVERED=30/KING_ON_EMPTY=40 vs. foundation's flat 10) leaving the post-uncover endgame reward-sparse, exactly the phase where the session's structural investigation found fully-uncovered-but-truncated games stalling (deeper/more concentrated piles, less foundation progress banked than games that go on to win). Demo corpus for this run was regenerated fresh (271 wins from a new heuristic harvest) after the original 55k-episode corpus was accidentally destroyed by a git-filter-repo history rewrite earlier in the session - not the same specific demonstrations as runs 023-027 used, worth keeping in mind if this comparison looks unusual.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 300000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| hidden_dim | 512 |
| layer_norm | False |
| card_encoding | raw |
| symmetry_augment | False |
| allow_undo | True |
| foundation_undo_penalty | 0.0 |
| foundation_reward_ramp | 2.0 |
| prioritized_replay | True |
| per_alpha | 0.6 |
| per_beta_start | 0.4 |
| per_beta_end | 1.0 |
| demo_source | runs/curriculum_wins.jsonl |
| num_demo_transitions | 100000 |
| margin | 0.8 |
| margin_loss_weight | 1.0 |
| demo_priority_eps | 1.0 |
| loop_breaker | True |
| loop_breaker_threshold | 2 |
| revisit_penalty | 0.0 |
| lr | 2.5e-05 |
| double_dqn | True |
| gamma | 0.99 |
| eps_start | 1.0 |
| eps_end | 0.15 |
| eps_decay_steps | 200000 |
| reheat_step | None |
| reheat_eps | None |
| reheat_decay_steps | None |
| learning_starts | 5000 |
| train_freq | 4 |
| target_update_mode | soft |
| tau | 0.005 |
| target_update_freq | 1000 |
| grad_clip | 10.0 |
| device | cuda |
| seed | 0 |
| tag | dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_foundramp |
| runs_dir | runs |
| notes | Direct A/B against run 023 (identical recipe and duration otherwise) - only change is --foundation-reward-ramp 2.0, phasing REWARD_NEW_FOUNDATION_HIGH from 1x up to 3x continuously as hidden tableau cards get revealed (1.0x at the deal, 3.0x once fully uncovered). Motivated by the reward function already being tableau-heavy (REWARD_REVEAL=50/COLUMN_UNCOVERED=30/KING_ON_EMPTY=40 vs. foundation's flat 10) leaving the post-uncover endgame reward-sparse, exactly the phase where the session's structural investigation found fully-uncovered-but-truncated games stalling (deeper/more concentrated piles, less foundation progress banked than games that go on to win). Demo corpus for this run was regenerated fresh (271 wins from a new heuristic harvest) after the original 55k-episode corpus was accidentally destroyed by a git-filter-repo history rewrite earlier in the session - not the same specific demonstrations as runs 023-027 used, worth keeping in mind if this comparison looks unusual. |
| no_log_games | False |
| print_every | 1000 |
| checkpoint_every | 10000 |
| eval_every | 10000 |
| eval_episodes | 200 |
| curriculum_source | None |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 10000 |
| curriculum_anneal_steps | 300000 |
| solvable_seed_source | runs/solvable_corpus.jsonl |
| solvable_seed_steps | 100000 |
| eval_seed_source | runs/eval_corpus.jsonl |
| resume_from | None |
| resume_step | 0 |
| git_commit | 6d88490f015b25db06d19243d17b1042f9dd1363 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-27T16:04:08
- **total_steps:** 400000
- **total_episodes:** 489
- **curriculum_episodes:** 0/489
- **final_mean_loss:** 2.6132
- **final_mean_return:** 966.19
- **final_mean_foundation:** 9.00/52
- **lifetime_outcomes:** {'won': 189, 'stalled': 5, 'truncated': 295, 'other': 0}
- **final_eval:** {'win_rate': 0.185, 'mean_return': 2494.6276928571365, 'mean_foundation': 22.425, 'mean_length': 937.945}
- **elapsed_seconds:** 8239.7
