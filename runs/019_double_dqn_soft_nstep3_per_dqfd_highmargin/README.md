# Run: dqfd_highmargin

- **Started:** 2026-09-15T10:42:45
- **Git commit:** 091e2d7dc1e91d7118da12acb8cc0272c377ceb5 (dirty working tree)

## Notes

DQfD hyperparameter sweep, variable 2: margin-loss-weight 1.0->5.0 (5x stronger enforcement of the margin constraint relative to TD loss), demo count back to 30000 matching run 017 exactly (run 018 showed demo count 30000->100000 made no difference, isolating that variable). Testing whether the margin loss simply wasn't weighted heavily enough to meaningfully shape the greedy policy. Otherwise identical to run 017: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 150000 |
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
| num_demo_transitions | 30000 |
| margin | 0.8 |
| margin_loss_weight | 5.0 |
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
| tag | dqfd_highmargin |
| runs_dir | runs |
| notes | DQfD hyperparameter sweep, variable 2: margin-loss-weight 1.0->5.0 (5x stronger enforcement of the margin constraint relative to TD loss), demo count back to 30000 matching run 017 exactly (run 018 showed demo count 30000->100000 made no difference, isolating that variable). Testing whether the margin loss simply wasn't weighted heavily enough to meaningfully shape the greedy policy. Otherwise identical to run 017: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start. |
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
| git_commit | 091e2d7dc1e91d7118da12acb8cc0272c377ceb5 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-15T12:13:00
- **total_steps:** 400000
- **total_episodes:** 429
- **curriculum_episodes:** 0/429
- **final_mean_loss:** 2.2287
- **final_mean_return:** 1070.00
- **final_mean_foundation:** 11.00/52
- **lifetime_outcomes:** {'won': 75, 'stalled': 3, 'truncated': 351, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 359.1000000000077, 'mean_foundation': 2.83, 'mean_length': 1000.0}
- **elapsed_seconds:** 5402.6
