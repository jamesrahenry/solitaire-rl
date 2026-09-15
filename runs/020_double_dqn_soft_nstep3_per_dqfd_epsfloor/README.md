# Run: dqfd_epsfloor

- **Started:** 2026-09-15T12:15:42
- **Git commit:** df459301ab86e44dab29cbbc8762fa256953c54e (dirty working tree)

## Notes

DQfD hyperparameter sweep, variable 3: combine DQfD (30000 demos, margin=0.8, weight=1.0 - matching run 017 exactly) with the permanently higher epsilon floor from run 015 (0.15 instead of 0.05). Runs 017-019 isolated demo count and margin weight, neither moved eval off 0%. Testing whether DQfD's continuous demonstration anchoring plus sustained exploration (which run 015 showed keeps finding organic wins past the floor, unlike the 0.05 default) compound usefully. Otherwise identical to run 017: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start.

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
| margin_loss_weight | 1.0 |
| demo_priority_eps | 1.0 |
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
| seed | 0 |
| tag | dqfd_epsfloor |
| runs_dir | runs |
| notes | DQfD hyperparameter sweep, variable 3: combine DQfD (30000 demos, margin=0.8, weight=1.0 - matching run 017 exactly) with the permanently higher epsilon floor from run 015 (0.15 instead of 0.05). Runs 017-019 isolated demo count and margin weight, neither moved eval off 0%. Testing whether DQfD's continuous demonstration anchoring plus sustained exploration (which run 015 showed keeps finding organic wins past the floor, unlike the 0.05 default) compound usefully. Otherwise identical to run 017: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start. |
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
| git_commit | df459301ab86e44dab29cbbc8762fa256953c54e |
| git_dirty | True |

## Results

- **Finished:** 2026-09-15T14:06:51
- **total_steps:** 400000
- **total_episodes:** 430
- **curriculum_episodes:** 0/430
- **final_mean_loss:** 1.4068
- **final_mean_return:** 670.00
- **final_mean_foundation:** 5.00/52
- **lifetime_outcomes:** {'won': 89, 'stalled': 4, 'truncated': 337, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 332.9000000000076, 'mean_foundation': 2.55, 'mean_length': 1000.0}
- **elapsed_seconds:** 6641.1
