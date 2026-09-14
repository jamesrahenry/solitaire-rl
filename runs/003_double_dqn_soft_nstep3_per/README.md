# Run: double_dqn_soft_nstep3_per

- **Started:** 2026-09-12T17:02:39
- **Git commit:** (no commits yet)

## Notes

Same config as run 002 (double_dqn_soft_nstep3_eval: Double DQN, soft target updates tau=0.005, lr=2.5e-5, n-step=3, periodic eval every 20k steps/20 fixed deals), now with prioritized experience replay (alpha=0.6, beta 0.4->1.0) added on top. Motivation: run 002's eval_win_rate was flat 0.0% for the entire 500k steps despite stable training loss - the learned policy showed no evidence of generalizing at all. Hypothesis: uniform replay sampling dilutes the ~5 rare successful trajectories among ~1200 episodes' worth of routine experience, so the network never gets to actually learn from them. PER should make wins (and their preceding TD-error-heavy transitions) get resampled far more often.

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
| tag | double_dqn_soft_nstep3_per |
| runs_dir | runs |
| notes | Same config as run 002 (double_dqn_soft_nstep3_eval: Double DQN, soft target updates tau=0.005, lr=2.5e-5, n-step=3, periodic eval every 20k steps/20 fixed deals), now with prioritized experience replay (alpha=0.6, beta 0.4->1.0) added on top. Motivation: run 002's eval_win_rate was flat 0.0% for the entire 500k steps despite stable training loss - the learned policy showed no evidence of generalizing at all. Hypothesis: uniform replay sampling dilutes the ~5 rare successful trajectories among ~1200 episodes' worth of routine experience, so the network never gets to actually learn from them. PER should make wins (and their preceding TD-error-heavy transitions) get resampled far more often. |
| no_log_games | False |
| print_every | 2000 |
| checkpoint_every | 50000 |
| eval_every | 20000 |
| eval_episodes | 20 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-12T17:42:05
- **total_steps:** 500000
- **total_episodes:** 1164
- **final_mean_loss:** 0.0756
- **final_mean_return:** 44.28
- **final_mean_foundation:** 4.75/52
- **lifetime_outcomes:** {'won': 1, 'stalled': 338, 'truncated': 825, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 37.54324999999987, 'mean_foundation': 4.4, 'mean_length': 456.85}
- **elapsed_seconds:** 2364.4
