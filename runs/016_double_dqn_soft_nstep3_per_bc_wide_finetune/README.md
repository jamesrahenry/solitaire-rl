# Run: bc_wide_finetune

- **Started:** 2026-09-15T01:40:30
- **Git commit:** 99630034972ee81a86787df0d4ec2e40c8fe3ab9 (dirty working tree)

## Notes

RL fine-tune warm-started from the wider BC checkpoint (bc_pretrained_v2_wide.pt: hidden_dim=1024, trained on 14954 episodes incl. all 6821 wins from the 55k-episode harvest, final train_acc=31.1%/val_acc=30.9% - lower than the original 5000-episode BC run's 35.7%/34.7%, but widening the network (512->1024) made no difference, ruling out capacity as the explanation; likely the richer/more diverse corpus just has genuinely more valid distinct actions per state, capping top-1 imitation accuracy without necessarily being a worse RL starting point). Direct comparison point: run 011 was the original BC-warm-start fine-tune (from the narrower 86-win corpus, hidden_dim=512) and also never broke 0% eval win rate. Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, undo allowed, 1000-step episode ceiling.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| hidden_dim | 1024 |
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
| tag | bc_wide_finetune |
| runs_dir | runs |
| notes | RL fine-tune warm-started from the wider BC checkpoint (bc_pretrained_v2_wide.pt: hidden_dim=1024, trained on 14954 episodes incl. all 6821 wins from the 55k-episode harvest, final train_acc=31.1%/val_acc=30.9% - lower than the original 5000-episode BC run's 35.7%/34.7%, but widening the network (512->1024) made no difference, ruling out capacity as the explanation; likely the richer/more diverse corpus just has genuinely more valid distinct actions per state, capping top-1 imitation accuracy without necessarily being a worse RL starting point). Direct comparison point: run 011 was the original BC-warm-start fine-tune (from the narrower 86-win corpus, hidden_dim=512) and also never broke 0% eval win rate. Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, undo allowed, 1000-step episode ceiling. |
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
| resume_from | runs/bc_pretrained_v2_wide.pt |
| resume_step | 0 |
| git_commit | 99630034972ee81a86787df0d4ec2e40c8fe3ab9 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-15T03:09:16
- **total_steps:** 400000
- **total_episodes:** 428
- **curriculum_episodes:** 0/428
- **final_mean_loss:** 2.1592
- **final_mean_return:** 710.00
- **final_mean_foundation:** 7.00/52
- **lifetime_outcomes:** {'won': 63, 'stalled': 3, 'truncated': 362, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 269.4000000000062, 'mean_foundation': 3.11, 'mean_length': 1000.0}
- **elapsed_seconds:** 5324.4
