# Run: double_dqn_soft_nstep3_per_decomposed_symaug

- **Started:** 2026-09-13T10:15:31
- **Git commit:** (no commits yet)

## Notes

Same config as run 004 (Double DQN, soft target updates, n-step=3, PER, 2-layer network, undo allowed, no curriculum) - our cleanest stable baseline. Adding two new, related features from a Gemini-discussion (implemented properly against our real codebase, not run as-is): (1) --card-encoding decomposed - each card slot split into separate slot_type/rank/color/suit embeddings instead of one joint raw-card-id embedding, giving the network an explicit structural hint that e.g. 7S and 7C are identical outside foundation-building; (2) --symmetry-augment - every transition's Spades<->Clubs, Hearts<->Diamonds suit-swapped twin is also stored in the replay buffer, a free 2x data augmentation teaching the same suit-invariance from the data side. Both verified via ground-truth testing (600 real-engine suit-swapped states matched our function-level swaps exactly). Motivation: prior 8 runs all showed 0.00% eval win rate regardless of algorithm/architecture/curriculum/undo changes - this tests whether reducing the artificially-large effective state/action space via suit-symmetry helps, a different kind of lever than anything tried so far.

## Config

| key | value |
|---|---|
| steps | 500000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| card_encoding | decomposed |
| symmetry_augment | True |
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
| tag | double_dqn_soft_nstep3_per_decomposed_symaug |
| runs_dir | runs |
| notes | Same config as run 004 (Double DQN, soft target updates, n-step=3, PER, 2-layer network, undo allowed, no curriculum) - our cleanest stable baseline. Adding two new, related features from a Gemini-discussion (implemented properly against our real codebase, not run as-is): (1) --card-encoding decomposed - each card slot split into separate slot_type/rank/color/suit embeddings instead of one joint raw-card-id embedding, giving the network an explicit structural hint that e.g. 7S and 7C are identical outside foundation-building; (2) --symmetry-augment - every transition's Spades<->Clubs, Hearts<->Diamonds suit-swapped twin is also stored in the replay buffer, a free 2x data augmentation teaching the same suit-invariance from the data side. Both verified via ground-truth testing (600 real-engine suit-swapped states matched our function-level swaps exactly). Motivation: prior 8 runs all showed 0.00% eval win rate regardless of algorithm/architecture/curriculum/undo changes - this tests whether reducing the artificially-large effective state/action space via suit-symmetry helps, a different kind of lever than anything tried so far. |
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

- **Finished:** 2026-09-13T11:03:11
- **total_steps:** 500000
- **total_episodes:** 1246
- **curriculum_episodes:** 0/1246
- **final_mean_loss:** 0.8234
- **final_mean_return:** 65.42
- **final_mean_foundation:** 6.40/52
- **lifetime_outcomes:** {'won': 7, 'stalled': 473, 'truncated': 766, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 33.91590000000001, 'mean_foundation': 3.7, 'mean_length': 434.25}
- **elapsed_seconds:** 2857.4
