# Run: double_dqn_soft_nstep3_per_curriculum_hidden3

- **Started:** 2026-09-13T01:28:27
- **Git commit:** (no commits yet)

## Notes

Same config as run 006 (Double DQN, soft target updates, n-step=3, PER, curriculum learning with corrected schedule), now with a 3-hidden-layer Q-network (was 2) - exploratory test of whether more MLP depth/capacity helps, per user's request. Prior 6 runs all showed 0.00% eval win rate regardless of algorithm changes; loss has consistently converged low and stable, suggesting the network isn't underfitting, so this is a lower-confidence experiment than the others, done for learning/exploration purposes.

## Config

| key | value |
|---|---|
| steps | 500000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 3 |
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
| tag | double_dqn_soft_nstep3_per_curriculum_hidden3 |
| runs_dir | runs |
| notes | Same config as run 006 (Double DQN, soft target updates, n-step=3, PER, curriculum learning with corrected schedule), now with a 3-hidden-layer Q-network (was 2) - exploratory test of whether more MLP depth/capacity helps, per user's request. Prior 6 runs all showed 0.00% eval win rate regardless of algorithm changes; loss has consistently converged low and stable, suggesting the network isn't underfitting, so this is a lower-confidence experiment than the others, done for learning/exploration purposes. |
| no_log_games | False |
| print_every | 2000 |
| checkpoint_every | 10000 |
| eval_every | 20000 |
| eval_episodes | 20 |
| curriculum_source | runs/curriculum_wins.jsonl |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 500 |
| curriculum_anneal_steps | 300000 |
| resume_from | None |
| resume_step | 0 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-13T01:57:23
- **total_steps:** 500000
- **total_episodes:** 1385
- **curriculum_episodes:** 665/1385
- **final_mean_loss:** 4.6422
- **final_mean_return:** 39.08
- **final_mean_foundation:** 4.20/52
- **lifetime_outcomes:** {'won': 17, 'stalled': 645, 'truncated': 723, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 21.72960000000022, 'mean_foundation': 2.1, 'mean_length': 370.7}
- **elapsed_seconds:** 1735.0
