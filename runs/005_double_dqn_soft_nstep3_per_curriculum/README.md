# Run: double_dqn_soft_nstep3_per_curriculum

- **Started:** 2026-09-12T22:32:55
- **Git commit:** (no commits yet)

## Notes

Same config as run 004 (Double DQN, soft target updates, n-step=3, PER), now with curriculum learning enabled using 86 real seeded wins harvested via a rule-based greedy heuristic policy (agent/heuristic_policy.py - prioritizes revealing hidden cards > foundation progress > king-to-empty-column > tableau consolidation > draw, with state-history cycle detection since pure greedy scoring got stuck in oscillating loops otherwise). 50% of episodes start from a near-completion state (tail length grows from 10 moves-before-win to full games over the first 300k steps). This is the first real attempt at curriculum learning after every prior run showed 0% eval win rate despite stable training.

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
| tag | double_dqn_soft_nstep3_per_curriculum |
| runs_dir | runs |
| notes | Same config as run 004 (Double DQN, soft target updates, n-step=3, PER), now with curriculum learning enabled using 86 real seeded wins harvested via a rule-based greedy heuristic policy (agent/heuristic_policy.py - prioritizes revealing hidden cards > foundation progress > king-to-empty-column > tableau consolidation > draw, with state-history cycle detection since pure greedy scoring got stuck in oscillating loops otherwise). 50% of episodes start from a near-completion state (tail length grows from 10 moves-before-win to full games over the first 300k steps). This is the first real attempt at curriculum learning after every prior run showed 0% eval win rate despite stable training. |
| no_log_games | False |
| print_every | 2000 |
| checkpoint_every | 10000 |
| eval_every | 20000 |
| eval_episodes | 20 |
| curriculum_source | runs/curriculum_wins.jsonl |
| curriculum_fraction | 0.5 |
| curriculum_start_tail | 10 |
| curriculum_end_tail | 10000 |
| curriculum_anneal_steps | 300000 |
| resume_from | None |
| resume_step | 0 |
| git_commit | None |
| git_dirty | None |

## Results

- **Finished:** 2026-09-12T22:59:32
- **total_steps:** 500000
- **total_episodes:** 1222
- **curriculum_episodes:** 602/1222
- **final_mean_loss:** 0.0833
- **final_mean_return:** 73.09
- **final_mean_foundation:** 8.75/52
- **lifetime_outcomes:** {'won': 0, 'stalled': 439, 'truncated': 783, 'other': 0}
- **final_eval:** {'win_rate': 0.0, 'mean_return': 32.67205, 'mean_foundation': 2.95, 'mean_length': 478.0}
- **elapsed_seconds:** 1596.2
