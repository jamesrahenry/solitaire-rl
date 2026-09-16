# Run: dqfd_combined_lb

- **Started:** 2026-09-16T00:01:16
- **Git commit:** bbe45f3acf22dee8e09be5faadc83674eeaef620 (dirty working tree)

## Notes

First run with LoopBreakerWrapper active from the start of training (not just post-hoc at eval), combining the two strongest configs from the post-hoc sweep: DQfD with 100000 demo transitions (run 018, best post-hoc result at 38/100) + the higher epsilon floor (run 020, 27/100 post-hoc, and the only run whose RAW pre-loop-breaker eval ever broke 0%). Hypothesis: if training-time exploration was also getting stuck in the same oscillation loops that masked eval results, fixing it during training too (not just eval) should let self-play collect more complete, useful trajectories throughout - potentially compounding beyond what either config alone showed under post-hoc correction. Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start. loop-breaker enabled by default (threshold=2, i.e. intervenes on the 3rd exact-state revisit).

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
| tag | dqfd_combined_lb |
| runs_dir | runs |
| notes | First run with LoopBreakerWrapper active from the start of training (not just post-hoc at eval), combining the two strongest configs from the post-hoc sweep: DQfD with 100000 demo transitions (run 018, best post-hoc result at 38/100) + the higher epsilon floor (run 020, 27/100 post-hoc, and the only run whose RAW pre-loop-breaker eval ever broke 0%). Hypothesis: if training-time exploration was also getting stuck in the same oscillation loops that masked eval results, fixing it during training too (not just eval) should let self-play collect more complete, useful trajectories throughout - potentially compounding beyond what either config alone showed under post-hoc correction. Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling, cold start. loop-breaker enabled by default (threshold=2, i.e. intervenes on the 3rd exact-state revisit). |
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
| git_commit | bbe45f3acf22dee8e09be5faadc83674eeaef620 |
| git_dirty | True |

## Results

- **Finished:** 2026-09-16T01:20:58
- **total_steps:** 400000
- **total_episodes:** 482
- **curriculum_episodes:** 0/482
- **final_mean_loss:** 2.3364
- **final_mean_return:** 7005.29
- **final_mean_foundation:** 52.00/52
- **lifetime_outcomes:** {'won': 167, 'stalled': 4, 'truncated': 311, 'other': 0}
- **final_eval:** {'win_rate': 0.33, 'mean_return': 2940.8732000000064, 'mean_foundation': 25.26, 'mean_length': 852.68}
- **elapsed_seconds:** 4750.5
