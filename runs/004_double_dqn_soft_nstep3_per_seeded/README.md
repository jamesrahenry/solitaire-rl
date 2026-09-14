# Run: double_dqn_soft_nstep3_per_seeded

- **Started:** 2026-09-12T20:31:18
- **Git commit:** (no commits yet)

## Notes

Same config as run 003 (Double DQN, soft target updates, n-step=3, PER). Curriculum disabled (no real seed data available yet - dedicated harvesting failed to find wins). Running via train_resilient.sh with --checkpoint-every 10000 for auto-resume, since this environment has been killing the plain training process via OOM every ~20-40k steps across 7 consecutive attempts.

## Config

| key | value |
|---|---|
| steps | 500000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
| prioritized_replay | True |
| per_alpha | 0.6 |
| per_beta_start | 0.4 |
| per_beta_end | 1.0 |
| lr | 2.5e-05 |
| double_dqn | True |
| gamma | 0.99 |
| eps_start | 1.0 |
| eps_end | 0.05 |
| eps_decay_steps | 200000 |
| learning_starts | 5000 |
| train_freq | 4 |
| target_update_mode | soft |
| tau | 0.005 |
| target_update_freq | 1000 |
| grad_clip | 10.0 |
| device | cuda |
| seed | 0 |
| tag | double_dqn_soft_nstep3_per_seeded |
| runs_dir | runs |
| notes | Same config as run 003 (Double DQN, soft target updates, n-step=3, PER). Curriculum disabled (no real seed data available yet - dedicated harvesting failed to find wins). Running via train_resilient.sh with --checkpoint-every 10000 for auto-resume, since this environment has been killing the plain training process via OOM every ~20-40k steps across 7 consecutive attempts. |
| no_log_games | False |
| print_every | 2000 |
| checkpoint_every | 10000 |
| eval_every | 20000 |
| eval_episodes | 20 |
| curriculum_source | None |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 10000 |
| curriculum_anneal_steps | 300000 |
| resume_from | None |
| resume_step | 0 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-12T20:57:55
- **total_steps:** 500000
- **total_episodes:** 1167
- **curriculum_episodes:** 0/1167
- **final_mean_loss:** 0.1135
- **final_mean_return:** 60.75
- **final_mean_foundation:** 6.50/52
- **lifetime_outcomes:** {'won': 0, 'stalled': 326, 'truncated': 841, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 30.572600000000232, 'mean_foundation': 2.3, 'mean_length': 477.45}
- **elapsed_seconds:** 1596.4
