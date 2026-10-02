# Run: 032_dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_freshdemo_mse

- **Started:** 2026-09-30T22:49:54
- **Git commit:** 3be637dadb12f87891500b0665955ceb5581be22 (dirty working tree)

## Notes

Run 029 recipe (fresh corpus, ramp 0, seed 0) with ONE change: --td-loss mse instead of Huber(beta=1). Hypothesis: Huber saturation made Q learn reward frequency not magnitude (029: q0~36 vs g0~630). Prediction: eval_q0 rises toward eval_g0; loss values incomparable to prior runs.

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
| foundation_undo_penalty | 0.0 |
| foundation_reward_ramp | 0.0 |
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
| td_loss | mse |
| huber_beta | 1.0 |
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
| tag | 032_dqfd_moredemo_epsfloor_loopbreaker_solvcurriculum_freshdemo_mse |
| runs_dir | runs |
| notes | Run 029 recipe (fresh corpus, ramp 0, seed 0) with ONE change: --td-loss mse instead of Huber(beta=1). Hypothesis: Huber saturation made Q learn reward frequency not magnitude (029: q0~36 vs g0~630). Prediction: eval_q0 rises toward eval_g0; loss values incomparable to prior runs. |
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
| git_commit | 3be637dadb12f87891500b0665955ceb5581be22 |
| git_dirty | True |

## Results

- **Finished:** 2026-10-01T00:58:00
- **total_steps:** 400000
- **total_episodes:** 503
- **curriculum_episodes:** 0/503
- **final_mean_loss:** 2029.6517
- **final_mean_return:** 7045.56
- **final_mean_foundation:** 52.00/52
- **lifetime_outcomes:** {'won': 211, 'stalled': 2, 'truncated': 290, 'other': 0}
- **final_eval:** {'win_rate': 0.365, 'mean_return': 3115.1092500000063, 'mean_foundation': 26.83, 'mean_length': 769.075, 'q0': 612.5135966491699, 'g0': 578.3370696582688}
- **elapsed_seconds:** 7658.6
