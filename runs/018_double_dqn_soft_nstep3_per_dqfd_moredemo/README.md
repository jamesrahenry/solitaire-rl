# Run: dqfd_moredemo

- **Started:** 2026-09-15T09:18:05
- **Git commit:** ca9e41d6bc5d5502a272543254add674404a80d7 (dirty working tree)

## Notes

DQfD hyperparameter sweep, variable 1: demo transition count 30000->100000 (still well under half of the ~4.3M available win-steps from 6821 wins), buffer capacity scaled up correspondingly (150000->300000, keeping 200000 self-play slots vs run 017's 120000). Margin (0.8) and margin-loss-weight (1.0) held constant to isolate demo density as the single changed variable vs run 017, which found smooth/stable loss but 0.00% eval across all 40 checkpoints. Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 300000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| hidden_dim | 512 |
| card_encoding | raw |
| symmetry_augment | False |
| allow_undo | True |
| prioritized_replay | True |
| per_alpha | 0.6 |
| per_beta_start | 0.4 |
| per_beta_end | 1.0 |
| demo_source | runs/curriculum_wins.jsonl |
| num_demo_transitions | 100000 |
| margin | 0.8 |
| margin_loss_weight | 1.0 |
| demo_priority_eps | 1.0 |
| lr | 2.5e-05 |
| double_dqn | True |
| gamma | 0.99 |
| eps_start | 1.0 |
| eps_end | 0.05 |
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
| tag | dqfd_moredemo |
| runs_dir | runs |
| notes | DQfD hyperparameter sweep, variable 1: demo transition count 30000->100000 (still well under half of the ~4.3M available win-steps from 6821 wins), buffer capacity scaled up correspondingly (150000->300000, keeping 200000 self-play slots vs run 017's 120000). Margin (0.8) and margin-loss-weight (1.0) held constant to isolate demo density as the single changed variable vs run 017, which found smooth/stable loss but 0.00% eval across all 40 checkpoints. Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start. |
| no_log_games | False |
| print_every | 1000 |
| checkpoint_every | 10000 |
| eval_every | 10000 |
| eval_episodes | 100 |
| curriculum_source | None |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 10000 |
| curriculum_anneal_steps | 300000 |
| resume_from | None |
| resume_step | 0 |
| git_commit | ca9e41d6bc5d5502a272543254add674404a80d7 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-15T10:37:59
- **total_steps:** 400000
- **total_episodes:** 429
- **curriculum_episodes:** 0/429
- **final_mean_loss:** 1.7815
- **final_mean_return:** 340.00
- **final_mean_foundation:** 4.00/52
- **lifetime_outcomes:** {'won': 70, 'stalled': 3, 'truncated': 356, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 357.8000000000076, 'mean_foundation': 2.6, 'mean_length': 1000.0}
- **elapsed_seconds:** 4762.7
