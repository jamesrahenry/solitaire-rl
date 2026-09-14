# Run: double_dqn_soft_nstep3_per_slow_eps

- **Started:** 2026-09-13T15:00:40
- **Git commit:** (no commits yet)

## Notes

Same config as run 004 (Double DQN, soft target updates, n-step=3, PER, 2-layer network, undo allowed, raw encoding, no curriculum/augmentation) - our cleanest stable baseline (loss 0.113, 0 wins, 0% eval). Testing user's hypothesis after checking win-timing across 6 prior runs (001,003,006,007,009): every single training-time win across every run happened while epsilon was between ~0.09 and ~0.62, and ZERO wins ever occurred once epsilon settled at its 0.05 floor - despite 300k+ steps of every run happening in that 'settled' regime. Stretched --eps-decay-steps from 200000 to 300000 (giving the Q-function more training time before the useful-exploration window closes) and --steps from 500000 to 600000 (to preserve a meaningful post-decay exploitation-only phase). Only variables changed from run 004: eps-decay-steps and steps.

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
| eps_decay_steps | 300000 |
| learning_starts | 5000 |
| train_freq | 4 |
| target_update_mode | soft |
| tau | 0.005 |
| target_update_freq | 1000 |
| grad_clip | 10.0 |
| device | cuda |
| seed | 0 |
| tag | double_dqn_soft_nstep3_per_slow_eps |
| runs_dir | runs |
| notes | Same config as run 004 (Double DQN, soft target updates, n-step=3, PER, 2-layer network, undo allowed, raw encoding, no curriculum/augmentation) - our cleanest stable baseline (loss 0.113, 0 wins, 0% eval). Testing user's hypothesis after checking win-timing across 6 prior runs (001,003,006,007,009): every single training-time win across every run happened while epsilon was between ~0.09 and ~0.62, and ZERO wins ever occurred once epsilon settled at its 0.05 floor - despite 300k+ steps of every run happening in that 'settled' regime. Stretched --eps-decay-steps from 200000 to 300000 (giving the Q-function more training time before the useful-exploration window closes) and --steps from 500000 to 600000 (to preserve a meaningful post-decay exploitation-only phase). Only variables changed from run 004: eps-decay-steps and steps. |
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

- **Finished:** 2026-09-13T15:39:24
- **total_steps:** 600000
- **total_episodes:** 1429
- **curriculum_episodes:** 0/1429
- **final_mean_loss:** 0.1623
- **final_mean_return:** 48.23
- **final_mean_foundation:** 5.33/52
- **lifetime_outcomes:** {'won': 1, 'stalled': 448, 'truncated': 980, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 34.94999999999993, 'mean_foundation': 3.6, 'mean_length': 500.0}
- **elapsed_seconds:** 2322.5
