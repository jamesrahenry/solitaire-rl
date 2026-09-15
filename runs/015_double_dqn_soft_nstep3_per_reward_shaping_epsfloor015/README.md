# Run: reward_shaping_epsfloor015

- **Started:** 2026-09-14T14:53:08
- **Git commit:** 3428678a3c02cbace6a9c0fc0d0024d9b7a47d0f (dirty working tree)

## Notes

Test of a permanently higher epsilon floor (0.15 instead of the default 0.05) - untested despite 14 prior runs (run 010's 'slow eps' only stretched the decay rate, never changed the floor itself). Motivated by wins clustering in the ~0.09-0.62 epsilon range across many runs, suggesting the 0.05 floor sits just below where win-discovery actually happens - meaning roughly 2/3 of every run so far has been spent in a regime that structurally can't find new wins. Cheapest possible next lever: zero new code, direct single-variable comparison against run 013 (identical config, 400k steps, cold start, no reheat, floor 0.05). Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling.

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
| tag | reward_shaping_epsfloor015 |
| runs_dir | runs |
| notes | Test of a permanently higher epsilon floor (0.15 instead of the default 0.05) - untested despite 14 prior runs (run 010's 'slow eps' only stretched the decay rate, never changed the floor itself). Motivated by wins clustering in the ~0.09-0.62 epsilon range across many runs, suggesting the 0.05 floor sits just below where win-discovery actually happens - meaning roughly 2/3 of every run so far has been spent in a regime that structurally can't find new wins. Cheapest possible next lever: zero new code, direct single-variable comparison against run 013 (identical config, 400k steps, cold start, no reheat, floor 0.05). Otherwise our cleanest baseline: Double DQN, soft target updates, n-step=3, PER, raw card encoding, hidden-layers=2, undo allowed, 1000-step episode ceiling. |
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
| git_commit | 3428678a3c02cbace6a9c0fc0d0024d9b7a47d0f |
| git_dirty | True |

## Results

- **Finished:** 2026-09-14T16:26:24
- **total_steps:** 400000
- **total_episodes:** 440
- **curriculum_episodes:** 0/440
- **final_mean_loss:** 1.5815
- **final_mean_return:** 740.00
- **final_mean_foundation:** 5.00/52
- **lifetime_outcomes:** {'won': 89, 'stalled': 14, 'truncated': 337, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 237.68890000000556, 'mean_foundation': 1.4, 'mean_length': 981.13}
- **elapsed_seconds:** 5595.4
