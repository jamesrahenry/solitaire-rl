# Run: double_dqn_soft_nstep3_eval

- **Started:** 2026-09-12T16:20:56
- **Git commit:** (no commits yet)

## Notes

Same config as double_dqn_soft_nstep3 (Double DQN, soft target updates tau=0.005, lr=2.5e-5, n-step=3), now with periodic frozen-greedy evaluation (every 20k steps, 20 fixed held-out deals) to get a clean read on whether the learned policy can win without exploration help. Previous run: final loss 1.71, 5 wins total but all during exploration (steps 80k-148k), zero wins once epsilon settled at its floor.

## Config

| key | value |
|---|---|
| steps | 500000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
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
| tag | double_dqn_soft_nstep3_eval |
| runs_dir | runs |
| notes | Same config as double_dqn_soft_nstep3 (Double DQN, soft target updates tau=0.005, lr=2.5e-5, n-step=3), now with periodic frozen-greedy evaluation (every 20k steps, 20 fixed held-out deals) to get a clean read on whether the learned policy can win without exploration help. Previous run: final loss 1.71, 5 wins total but all during exploration (steps 80k-148k), zero wins once epsilon settled at its floor. |
| no_log_games | False |
| print_every | 2000 |
| checkpoint_every | 50000 |
| eval_every | 20000 |
| eval_episodes | 20 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-12T16:50:59
- **total_steps:** 500000
- **total_episodes:** 1208
- **final_mean_loss:** 1.7108
- **final_mean_return:** 75.98
- **final_mean_foundation:** 7.20/52
- **lifetime_outcomes:** {'won': 5, 'stalled': 388, 'truncated': 815, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 34.78615000000006, 'mean_foundation': 3.8, 'mean_length': 414.05}
- **elapsed_seconds:** 1802.1
