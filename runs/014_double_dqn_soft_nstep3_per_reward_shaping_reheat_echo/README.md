# Run: reward_shaping_reheat_echo

- **Started:** 2026-09-14T12:06:45
- **Git commit:** 6a5bcf3d4f160f98ef6c55dbe22c469ccf66bd81 (dirty working tree)

## Notes

Cold-start run with the 1000-step episode ceiling (up from 500) and two epsilon reheats: back to 0.4 at step 400000 (decaying to the 0.05 floor by 450000), then a smaller 'echo' reheat to 0.2 at step 500000 (decaying back to the floor by 550000), 600000 steps total. Follows directly from run 013, whose single reheat to 0.4 found zero new wins and zero eval-win-rate movement - testing whether a second, later, smaller pulse behaves differently, on top of the now-doubled episode budget (14%/26% of known wins previously needed >=480/450 of the old 500-step cap). Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed.

## Config

| key | value |
|---|---|
| steps | 600000 |
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
| reheat_step | [400000, 500000] |
| reheat_eps | [0.4, 0.2] |
| reheat_decay_steps | [50000] |
| learning_starts | 5000 |
| train_freq | 4 |
| target_update_mode | soft |
| tau | 0.005 |
| target_update_freq | 1000 |
| grad_clip | 10.0 |
| device | cuda |
| seed | 0 |
| tag | reward_shaping_reheat_echo |
| runs_dir | runs |
| notes | Cold-start run with the 1000-step episode ceiling (up from 500) and two epsilon reheats: back to 0.4 at step 400000 (decaying to the 0.05 floor by 450000), then a smaller 'echo' reheat to 0.2 at step 500000 (decaying back to the floor by 550000), 600000 steps total. Follows directly from run 013, whose single reheat to 0.4 found zero new wins and zero eval-win-rate movement - testing whether a second, later, smaller pulse behaves differently, on top of the now-doubled episode budget (14%/26% of known wins previously needed >=480/450 of the old 500-step cap). Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed. |
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
| git_commit | 6a5bcf3d4f160f98ef6c55dbe22c469ccf66bd81 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-14T14:30:03
- **total_steps:** 600000
- **total_episodes:** 629
- **curriculum_episodes:** 0/629
- **final_mean_loss:** 1.1115
- **final_mean_return:** 610.00
- **final_mean_foundation:** 4.00/52
- **lifetime_outcomes:** {'won': 75, 'stalled': 8, 'truncated': 546, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 245.0000000000056, 'mean_foundation': 2.29, 'mean_length': 1000.0}
- **elapsed_seconds:** 8597.4
