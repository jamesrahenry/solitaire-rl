# Run: bc_finetune

- **Started:** 2026-09-13T22:45:01
- **Git commit:** (no commits yet)

## Notes

RL fine-tune warm-started from behavior-cloning pretrain (agent/behavior_cloning.py, all 5000 harvested heuristic episodes, 8 epochs, final train_acc=0.357/val_acc=0.347). Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed.

## Config

| key | value |
|---|---|
| steps | 400000 |
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
| learning_starts | 5000 |
| train_freq | 4 |
| target_update_mode | soft |
| tau | 0.005 |
| target_update_freq | 1000 |
| grad_clip | 10.0 |
| device | cuda |
| seed | 0 |
| tag | bc_finetune |
| runs_dir | runs |
| notes | RL fine-tune warm-started from behavior-cloning pretrain (agent/behavior_cloning.py, all 5000 harvested heuristic episodes, 8 epochs, final train_acc=0.357/val_acc=0.347). Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed. |
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
| resume_from | runs/bc_pretrained.pt |
| resume_step | 0 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-13T23:23:38
- **total_steps:** 400000
- **total_episodes:** 1876
- **curriculum_episodes:** 0/1876
- **final_mean_loss:** 0.1122
- **final_mean_return:** 26.13
- **final_mean_foundation:** 4.67/52
- **lifetime_outcomes:** {'won': 9, 'stalled': 1374, 'truncated': 493, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 22.305110000000045, 'mean_foundation': 3.99, 'mean_length': 335.26}
- **elapsed_seconds:** 2316.3
