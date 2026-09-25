# Run: dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_undopenalty20

- **Started:** 2026-09-22T14:00:35
- **Git commit:** 78c7cdf37add1c799a387ed4d3362914ad918698 (dirty working tree)

## Notes

Follow-up to run 026 (foundation-undo-penalty=5.0, which only modestly moved the behavioral metric - undo recall rate 75.3% -> 70.3% - and left win rate statistically unchanged from baseline run 023: 31.5% vs 31.2% overall). Bumping to 20.0 (double the milestone reward REWARD_NEW_FOUNDATION_HIGH=10 it's meant to counteract, vs run 026's 5.0=half) to test decisively whether penalty magnitude was the limiting factor, or whether most of these undos are genuinely strategically necessary regardless of cost (in which case an even larger penalty should still fail to move win rate much, while further suppressing the undo rate).

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
| foundation_undo_penalty | 20.0 |
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
| tag | dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_undopenalty20 |
| runs_dir | runs |
| notes | Follow-up to run 026 (foundation-undo-penalty=5.0, which only modestly moved the behavioral metric - undo recall rate 75.3% -> 70.3% - and left win rate statistically unchanged from baseline run 023: 31.5% vs 31.2% overall). Bumping to 20.0 (double the milestone reward REWARD_NEW_FOUNDATION_HIGH=10 it's meant to counteract, vs run 026's 5.0=half) to test decisively whether penalty magnitude was the limiting factor, or whether most of these undos are genuinely strategically necessary regardless of cost (in which case an even larger penalty should still fail to move win rate much, while further suppressing the undo rate). |
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

- **Finished:** 2026-09-22T16:19:15
- **total_steps:** 400000
- **total_episodes:** 484
- **curriculum_episodes:** 0/484
- **final_mean_loss:** 2.2414
- **final_mean_return:** 6724.24
- **final_mean_foundation:** 52.00/52
- **lifetime_outcomes:** {'won': 184, 'stalled': 3, 'truncated': 297, 'other': 0}
- **final_eval:** {'win_rate': 0.375, 'mean_return': 3144.498100000006, 'mean_foundation': 27.76, 'mean_length': 835.195}
- **elapsed_seconds:** 8287.3
