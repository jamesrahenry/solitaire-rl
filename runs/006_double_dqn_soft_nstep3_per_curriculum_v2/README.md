# Run: double_dqn_soft_nstep3_per_curriculum_v2

- **Started:** 2026-09-12T23:15:23
- **Git commit:** (no commits yet)

## Notes

Retry of run 005 with corrected curriculum schedule. Bug found in 005: --curriculum-end-tail defaulted to 10000, but all 86 harvested wins are under 500 steps long, so tail length already exceeded every real win's length by step ~15000 (3% into the 300k-step anneal) - curriculum episodes became indistinguishable from fresh deals almost immediately, and 005 showed 0 training-time wins and flat 0.0% eval the entire run. Fixed: --curriculum-end-tail 500 (matches actual max harvested win length of 497 steps), same 300k-step anneal, so the schedule now spends meaningful time in the genuinely-assisted range (e.g. tail~43 at step 20k, ~255 at step 150k) instead of blowing past it instantly.

## Config

| key | value |
|---|---|
| steps | 500000 |
| buffer_capacity | 100000 |
| batch_size | 128 |
| n_step | 3 |
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
| tag | double_dqn_soft_nstep3_per_curriculum_v2 |
| runs_dir | runs |
| notes | Retry of run 005 with corrected curriculum schedule. Bug found in 005: --curriculum-end-tail defaulted to 10000, but all 86 harvested wins are under 500 steps long, so tail length already exceeded every real win's length by step ~15000 (3% into the 300k-step anneal) - curriculum episodes became indistinguishable from fresh deals almost immediately, and 005 showed 0 training-time wins and flat 0.0% eval the entire run. Fixed: --curriculum-end-tail 500 (matches actual max harvested win length of 497 steps), same 300k-step anneal, so the schedule now spends meaningful time in the genuinely-assisted range (e.g. tail~43 at step 20k, ~255 at step 150k) instead of blowing past it instantly. |
| no_log_games | False |
| print_every | 2000 |
| checkpoint_every | 10000 |
| eval_every | 20000 |
| eval_episodes | 20 |
| curriculum_source | runs/curriculum_wins.jsonl |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 500 |
| curriculum_anneal_steps | 300000 |
| resume_from | None |
| resume_step | 0 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-12T23:41:49
- **total_steps:** 500000
- **total_episodes:** 1350
- **curriculum_episodes:** 658/1350
- **final_mean_loss:** 0.1076
- **final_mean_return:** 50.66
- **final_mean_foundation:** 6.00/52
- **lifetime_outcomes:** {'won': 5, 'stalled': 573, 'truncated': 772, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 42.54249999999977, 'mean_foundation': 5.9, 'mean_length': 457.6}
- **elapsed_seconds:** 1585.0
