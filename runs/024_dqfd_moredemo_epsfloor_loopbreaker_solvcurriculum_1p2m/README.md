# Run: dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_1p2m

- **Started:** 2026-09-17T16:22:23
- **Git commit:** 1a5024082c45e007779fce5d25f55390cca2907d (dirty working tree)

## Notes

Tripled-length re-run of 023 (400000 -> 1200000 steps), identical config otherwise, matching the 021->022 escalation pattern. Purpose: get a full LIVE structure_metrics.csv trace (not backfilled) over a much longer run, to see whether net.4's still-declining effective rank (62.8 at 400k in 023's backfilled trace, never plateaued) eventually stabilizes, and whether that coincides with a real win-rate plateau - net.0 was already fully collapsed (~2 eff dims) by ~110k steps in run 023 yet win rate kept climbing the whole run, so collapse in that layer alone clearly isn't hard-capping performance; net.2 curiously bottomed out (~37 eff dim) around step 260-290k and partially recovered afterward, loosely coincident with run 023's fastest-improving quarter. This run is designed to see if that pattern holds/resolves over 3x the duration before considering an actual architectural intervention (e.g. LayerNorm) to directly test the collapse hypothesis.

## Config

| key | value |
|---|---|
| steps | 1200000 |
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
| tag | dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_1p2m |
| runs_dir | runs |
| notes | Tripled-length re-run of 023 (400000 -> 1200000 steps), identical config otherwise, matching the 021->022 escalation pattern. Purpose: get a full LIVE structure_metrics.csv trace (not backfilled) over a much longer run, to see whether net.4's still-declining effective rank (62.8 at 400k in 023's backfilled trace, never plateaued) eventually stabilizes, and whether that coincides with a real win-rate plateau - net.0 was already fully collapsed (~2 eff dims) by ~110k steps in run 023 yet win rate kept climbing the whole run, so collapse in that layer alone clearly isn't hard-capping performance; net.2 curiously bottomed out (~37 eff dim) around step 260-290k and partially recovered afterward, loosely coincident with run 023's fastest-improving quarter. This run is designed to see if that pattern holds/resolves over 3x the duration before considering an actual architectural intervention (e.g. LayerNorm) to directly test the collapse hypothesis. |
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
| git_commit | 1a5024082c45e007779fce5d25f55390cca2907d |
| git_dirty | True |

## Results

- **Finished:** 2026-09-18T03:26:07
- **total_steps:** 1200000
- **total_episodes:** 1462
- **curriculum_episodes:** 0/1462
- **final_mean_loss:** 2.4075
- **final_mean_return:** 7040.77
- **final_mean_foundation:** 52.00/52
- **lifetime_outcomes:** {'won': 534, 'stalled': 6, 'truncated': 922, 'other': 0}
- **final_eval:** {'win_rate': 0.3, 'mean_return': 2799.4417500000072, 'mean_foundation': 24.43, 'mean_length': 880.83}
- **elapsed_seconds:** 39759.4
