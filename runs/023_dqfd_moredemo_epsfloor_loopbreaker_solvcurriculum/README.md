# Run: dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum

- **Started:** 2026-09-17T02:29:14
- **Git commit:** d0f6b04b0296dc64bbeef0031b7156d85db9d122 (dirty working tree)

## Notes

Same recipe as run 021 (DQfD 100k demos + eps floor 0.15 + loop-breaker in training, 400k steps, cold start) but adds the two new ground-truth-solver-backed pieces: (1) --solvable-seed-source/--solvable-seed-steps biases fresh-deal resets toward the 1,578-seed <250-move-solvable corpus for the first 100k steps (25%) before reverting to fully random, testing whether an easy-first curriculum phase speeds up early learning without hurting eventual generalization; (2) --eval-seed-source swaps the fixed 100-seed eval range for the 1,000-seed solver-proven battery (200 of them sampled per eval pass, up from 100), to shrink eval win-rate sampling noise (stderr ~3.2% vs ~4.6% at a 30% win rate) - directly relevant since run 021/022's quarter-to-quarter win-rate swings were partly attributed to this. Direct A/B against run 021 on everything else held constant.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 300000 |
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
| num_demo_transitions | 100000 |
| margin | 0.8 |
| margin_loss_weight | 1.0 |
| demo_priority_eps | 1.0 |
| loop_breaker | True |
| loop_breaker_threshold | 2 |
| revisit_penalty | 0.0 |
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
| tag | dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum |
| runs_dir | runs |
| notes | Same recipe as run 021 (DQfD 100k demos + eps floor 0.15 + loop-breaker in training, 400k steps, cold start) but adds the two new ground-truth-solver-backed pieces: (1) --solvable-seed-source/--solvable-seed-steps biases fresh-deal resets toward the 1,578-seed <250-move-solvable corpus for the first 100k steps (25%) before reverting to fully random, testing whether an easy-first curriculum phase speeds up early learning without hurting eventual generalization; (2) --eval-seed-source swaps the fixed 100-seed eval range for the 1,000-seed solver-proven battery (200 of them sampled per eval pass, up from 100), to shrink eval win-rate sampling noise (stderr ~3.2% vs ~4.6% at a 30% win rate) - directly relevant since run 021/022's quarter-to-quarter win-rate swings were partly attributed to this. Direct A/B against run 021 on everything else held constant. |
| no_log_games | False |
| print_every | 1000 |
| checkpoint_every | 10000 |
| eval_every | 10000 |
| eval_episodes | 200 |
| curriculum_source | None |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 10000 |
| curriculum_anneal_steps | 300000 |
| solvable_seed_source | runs/solvable_corpus.jsonl |
| solvable_seed_steps | 100000 |
| eval_seed_source | runs/eval_corpus.jsonl |
| resume_from | None |
| resume_step | 0 |
| git_commit | d0f6b04b0296dc64bbeef0031b7156d85db9d122 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-17T04:40:00
- **total_steps:** 400000
- **total_episodes:** 464
- **curriculum_episodes:** 0/464
- **final_mean_loss:** 2.2969
- **final_mean_return:** 640.00
- **final_mean_foundation:** 2.00/52
- **lifetime_outcomes:** {'won': 139, 'stalled': 3, 'truncated': 322, 'other': 0}
- **final_eval:** {'win_rate': 0.385, 'mean_return': 3306.675200000007, 'mean_foundation': 27.99, 'mean_length': 832.485}
- **elapsed_seconds:** 7814.2
