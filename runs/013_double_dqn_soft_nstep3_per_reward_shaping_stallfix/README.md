# Run: reward_shaping_stallfix_reheat

- **Started:** 2026-09-14T11:39:56
- **Git commit:** 6b0e82ce4f3a01aab8a904bb903fb3374827b53a (dirty working tree)

## Notes

Epsilon reheat continuation: every win in the base 400k-step run landed during the initial high-epsilon decay, zero new wins in 100k+ steps after epsilon hit its 0.05 floor (final eval still 0.00% win rate). Bumping epsilon back to 0.4 at step 400000, decaying back to 0.05 by step 450000, to see if it re-triggers exploration-driven win discovery.

## Config

| key | value |
|---|---|
| steps | 450000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| card_encoding | raw |
| symmetry_augment | False |
| allow_undo | True |
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
| reheat_step | 400000 |
| reheat_eps | 0.4 |
| reheat_decay_steps | 50000 |
| learning_starts | 5000 |
| train_freq | 4 |
| target_update_mode | soft |
| tau | 0.005 |
| target_update_freq | 1000 |
| grad_clip | 10.0 |
| device | cuda |
| seed | 0 |
| tag | reward_shaping_stallfix_reheat |
| runs_dir | runs |
| notes | Epsilon reheat continuation: every win in the base 400k-step run landed during the initial high-epsilon decay, zero new wins in 100k+ steps after epsilon hit its 0.05 floor (final eval still 0.00% win rate). Bumping epsilon back to 0.4 at step 400000, decaying back to 0.05 by step 450000, to see if it re-triggers exploration-driven win discovery. |
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
| resume_from | runs/013_double_dqn_soft_nstep3_per_reward_shaping_stallfix/checkpoint_final.pt |
| resume_step | 400000 |
| git_commit | 6b0e82ce4f3a01aab8a904bb903fb3374827b53a |
| git_dirty | True |

## Results

- **Finished:** 2026-09-14T11:50:16
- **total_steps:** 450000
- **total_episodes:** 50
- **curriculum_episodes:** 0/50
- **final_mean_loss:** 2.2297
- **final_mean_return:** 710.00
- **final_mean_foundation:** 12.00/52
- **lifetime_outcomes:** {'won': 0, 'stalled': 1, 'truncated': 49, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 244.70000000000545, 'mean_foundation': 2.31, 'mean_length': 1000.0}
- **elapsed_seconds:** 618.5
