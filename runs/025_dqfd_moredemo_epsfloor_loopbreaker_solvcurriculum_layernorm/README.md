# Run: dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_layernorm

- **Started:** 2026-09-18T10:31:08
- **Git commit:** 5f29fe1d11df31ac415918f70c1faec22422497d (dirty working tree)

## Notes

Direct A/B against run 023 (identical recipe and duration: DQfD 100k demos + eps floor 0.15 + loop-breaker + 100k-step solvable-seed curriculum + 1000-seed eval battery, 400k steps, cold start) - only change is --layer-norm, testing whether it prevents/slows the effective-rank collapse structure_metrics.py found in net.0 (collapsed to ~2 eff dims by ~110k steps in run 023) and whether that changes the achievable win-rate ceiling. Run 024 (tripled-length rerun of 023) showed a genuine plateau (regression slope dropped from 2.40 to 0.39 pts/100k) with net.4's rank still declining with no further win-rate gain - motivating this as the natural next intervention rather than just running longer again.

## Config

| key | value |
|---|---|
| steps | 400000 |
| buffer_capacity | 300000 |
| batch_size | 128 |
| n_step | 3 |
| hidden_layers | 2 |
| hidden_dim | 512 |
| layer_norm | True |
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
| tag | dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_layernorm |
| runs_dir | runs |
| notes | Direct A/B against run 023 (identical recipe and duration: DQfD 100k demos + eps floor 0.15 + loop-breaker + 100k-step solvable-seed curriculum + 1000-seed eval battery, 400k steps, cold start) - only change is --layer-norm, testing whether it prevents/slows the effective-rank collapse structure_metrics.py found in net.0 (collapsed to ~2 eff dims by ~110k steps in run 023) and whether that changes the achievable win-rate ceiling. Run 024 (tripled-length rerun of 023) showed a genuine plateau (regression slope dropped from 2.40 to 0.39 pts/100k) with net.4's rank still declining with no further win-rate gain - motivating this as the natural next intervention rather than just running longer again. |
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
| git_commit | 5f29fe1d11df31ac415918f70c1faec22422497d |
| git_dirty | True |

## Results

- **Finished:** 2026-09-18T13:55:44
- **total_steps:** 400000
- **total_episodes:** 487
- **curriculum_episodes:** 0/487
- **final_mean_loss:** 2.1292
- **final_mean_return:** 1610.00
- **final_mean_foundation:** 24.00/52
- **lifetime_outcomes:** {'won': 173, 'stalled': 3, 'truncated': 311, 'other': 0}
- **final_eval:** {'win_rate': 0.225, 'mean_return': 2313.4159000000072, 'mean_foundation': 21.82, 'mean_length': 888.415}
- **elapsed_seconds:** 12244.3
