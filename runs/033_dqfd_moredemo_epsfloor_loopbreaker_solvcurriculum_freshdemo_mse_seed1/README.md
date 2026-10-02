# Run: 033_dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_freshdemo_mse_seed1

- **Started:** 2026-10-02T10:17:25
- **Git commit:** 3be637dadb12f87891500b0665955ceb5581be22 (dirty working tree)

## Notes

Seed-1 replicate of run 032 (029 recipe + --td-loss mse). Pairs with 030 (seed-1 Huber control). Tests whether 032's +9-pt second-half gain and 0.02 checkpoint SD hold across seeds.

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
| foundation_reward_ramp | 0.0 |
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
| td_loss | mse |
| huber_beta | 1.0 |
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
| seed | 1 |
| tag | 033_dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_freshdemo_mse_seed1 |
| runs_dir | runs |
| notes | Seed-1 replicate of run 032 (029 recipe + --td-loss mse). Pairs with 030 (seed-1 Huber control). Tests whether 032's +9-pt second-half gain and 0.02 checkpoint SD hold across seeds. |
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
| git_commit | 3be637dadb12f87891500b0665955ceb5581be22 |
| git_dirty | True |

## Results

- **Finished:** 2026-10-02T12:58:54
- **total_steps:** 400000
- **total_episodes:** 505
- **curriculum_episodes:** 0/505
- **final_mean_loss:** 1789.4534
- **final_mean_return:** 4131.29
- **final_mean_foundation:** 39.00/52
- **lifetime_outcomes:** {'won': 209, 'stalled': 4, 'truncated': 292, 'other': 0}
- **final_eval:** {'win_rate': 0.425, 'mean_return': 3509.746200000006, 'mean_foundation': 29.125, 'mean_length': 770.38, 'q0': 623.1179209899902, 'g0': 551.8238274818291}
- **elapsed_seconds:** 9661.4
