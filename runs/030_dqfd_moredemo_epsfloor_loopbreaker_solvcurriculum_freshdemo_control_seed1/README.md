# Run: 030_dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_freshdemo_control_seed1

- **Started:** 2026-09-30T14:46:43
- **Git commit:** 8a0c7dd26431dfe5686d2dc863d5686f13d001bd (dirty working tree)

## Notes

Seed-1 replicate of control (ramp=0), fresh corpus, to estimate run-to-run variance.

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
| tag | 030_dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_freshdemo_control_seed1 |
| runs_dir | runs |
| notes | Seed-1 replicate of control (ramp=0), fresh corpus, to estimate run-to-run variance. |
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
| git_commit | 8a0c7dd26431dfe5686d2dc863d5686f13d001bd |
| git_dirty | True |

## Results

- **Finished:** 2026-09-30T17:30:47
- **total_steps:** 400000
- **total_episodes:** 480
- **curriculum_episodes:** 0/480
- **final_mean_loss:** 2.4668
- **final_mean_return:** 3757.62
- **final_mean_foundation:** 30.50/52
- **lifetime_outcomes:** {'won': 169, 'stalled': 2, 'truncated': 309, 'other': 0}
- **final_eval:** {'win_rate': 0.335, 'mean_return': 3011.6432500000064, 'mean_foundation': 26.59, 'mean_length': 830.68}
- **elapsed_seconds:** 9816.0
