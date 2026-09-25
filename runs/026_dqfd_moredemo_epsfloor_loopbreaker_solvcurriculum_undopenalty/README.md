# Run: dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_undopenalty

- **Started:** 2026-09-18T12:12:31
- **Git commit:** 78c7cdf37add1c799a387ed4d3362914ad918698 (dirty working tree)

## Notes

Direct A/B against run 023 (identical recipe and duration otherwise) - only change is --foundation-undo-penalty 5.0. Motivated by measured behavior: run 023's game logs show ~75% of foundation-sends eventually get undone, and undoing was completely free (doesn't claw back REWARD_NEW_FOUNDATION_HIGH=+10, doesn't reset the high-water mark), so a send/undo/resend round trip netted nearly +10 for free. Penalty magnitude (5.0) chosen as half the milestone reward it's meant to counteract: large enough that a frivolous round trip (most resends earn 0, since the high-water mark isn't reset) is a clear net loss, small enough that a genuinely valuable strategic undo (typically followed by a reveal/uncover bonus of 20-50) stays clearly worth it. Tests the cheaper, reward-shaping alternative to a multi-head/MoE architecture change for the same underlying foundation-timing concern.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 300000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| hidden_dim | 512 |
| layer_norm | False |
| card_encoding | raw |
| symmetry_augment | False |
| allow_undo | True |
| foundation_undo_penalty | 5.0 |
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
| tag | dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_undopenalty |
| runs_dir | runs |
| notes | Direct A/B against run 023 (identical recipe and duration otherwise) - only change is --foundation-undo-penalty 5.0. Motivated by measured behavior: run 023's game logs show ~75% of foundation-sends eventually get undone, and undoing was completely free (doesn't claw back REWARD_NEW_FOUNDATION_HIGH=+10, doesn't reset the high-water mark), so a send/undo/resend round trip netted nearly +10 for free. Penalty magnitude (5.0) chosen as half the milestone reward it's meant to counteract: large enough that a frivolous round trip (most resends earn 0, since the high-water mark isn't reset) is a clear net loss, small enough that a genuinely valuable strategic undo (typically followed by a reveal/uncover bonus of 20-50) stays clearly worth it. Tests the cheaper, reward-shaping alternative to a multi-head/MoE architecture change for the same underlying foundation-timing concern. |
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
| git_commit | 78c7cdf37add1c799a387ed4d3362914ad918698 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-18T15:18:05
- **total_steps:** 400000
- **total_episodes:** 485
- **curriculum_episodes:** 0/485
- **final_mean_loss:** 2.2104
- **final_mean_return:** 685.00
- **final_mean_foundation:** 11.00/52
- **lifetime_outcomes:** {'won': 178, 'stalled': 3, 'truncated': 304, 'other': 0}
- **final_eval:** {'win_rate': 0.33, 'mean_return': 2964.2945500000073, 'mean_foundation': 25.4, 'mean_length': 863.05}
- **elapsed_seconds:** 11094.7
