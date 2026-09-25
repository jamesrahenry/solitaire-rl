# Solitaire RL

A from-scratch Klondike Solitaire [Gymnasium](https://gymnasium.farama.org/) environment, and a Double DQN family agent trained against it — a learning project in reinforcement learning, not a from-nowhere-to-perfect Solitaire bot. The point was to actually feel the RL tuning space: what breaks, what a measurement artifact looks like versus a real learning failure, and how much of "it doesn't work" turns out to be something else entirely.

The full, warts-and-all lab notebook is in [`SESSION_SUMMARY.md`](SESSION_SUMMARY.md) — every run, every hypothesis, every dead end, in the order it actually happened. This README is the short version.

## Play it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install gymnasium numpy torch matplotlib
python play.py --seed 12345
```

Draw-1 Klondike, undo allowed (`foundation -> tableau` is a legal move, same as real rules). Every game gets a real seed and an end-of-game summary (moves made, illegal attempts, when the tableau got fully uncovered).

## Train an agent

```bash
python -m agent.train --tag my_run --steps 400000 \
  --demo-source runs/curriculum_wins.jsonl --num-demo-transitions 100000 \
  --eps-end 0.15
```

See `python -m agent.train --help` for the full knob list (Double DQN, PER, n-step returns, DQfD demonstrations, epsilon-floor exploration, curriculum learning, LayerNorm, reward-shaping penalties, ...). Each run gets its own numbered, self-contained directory under `runs/` — config, checkpoints, metrics, plots, game logs, and a README.

## Headline result

For 20 straight runs, greedy-policy evaluation sat at a flat 0% win rate. It turned out to be substantially a **measurement artifact**: checkpoints were getting stuck oscillating between two tied-Q-value actions in an exact repeated game state, burning their entire step budget with zero further progress — masking real, learned competence underneath. A `LoopBreakerWrapper` that masks a previously-taken action out of a state visited a 3rd time (agent-side only, doesn't touch the environment's actual rules) fixed it: swept against all 19 prior runs' checkpoints with **zero retraining**, several jumped from 0/100 to 20-38/100 wins on the spot.

Wiring the fix into training itself (not just eval), combined with DQfD (permanent demonstration transitions + a large-margin loss) and a raised epsilon floor, produced the project's first genuinely reliable policy: **sustained 30-40% win rate on held-out deals**, not a fragile one-off.

## A few things that turned out to be true

- **Klondike's own solvability is the real ceiling, not just training.** Sourced (not written) a ground-truth solver ([ShootMe/Klondike-Solver](https://github.com/ShootMe/Klondike-Solver)) to check: of 28 eval seeds nothing had ever won across 22 runs, 20 were provably solvable (real, unmet capability gaps) and only 1 was provably unsolvable — the rest were genuine gaps, not dead deals.
- **The network's effective rank collapses hard during training** — one hidden layer measured at ~2 effective dimensions (out of ~500 possible) within the first 10-15% of a run, and stayed there. A LayerNorm fix that stopped that collapse made performance *worse*, not better — the collapse pressure just relocated to the one layer that can't be normalized (the Q-value readout).
- **A behavioral finding that reward-shaping couldn't touch:** ~75% of every foundation-send eventually gets undone. Quadrupling the penalty for undoing (5 -> 20, against a milestone reward of 10) barely moved that number — strong evidence it's a load-bearing strategic move, not a fixable mistake.
- **"Fully uncovered" doesn't mean "about to win."** Games that reveal every hidden card and then still lose build measurably deeper, more concentrated tableau piles and bank less foundation progress than games that go on to win — confirmed against real game logs, not just a hunch.

## Project layout

```
solitaire_gym/    the Gymnasium environment (Klondike rules engine, registered as Solitaire-v0)
agent/            the training pipeline (DQN/DQfD, replay buffers, preprocessing, metrics, analysis scripts)
play.py           interactive terminal client, drives the actual registered env
runs/             one directory per training run - config, checkpoints, metrics, plots, logs
SESSION_SUMMARY.md   the full run-by-run narrative
```

## Licensing

The Klondike rules engine references [`tteeoo/solitaire`](https://github.com/tteeoo/solitaire) (MIT), vendored under `vendor/tteeoo-solitaire/` — see [`NOTICE.md`](NOTICE.md). Everything else in this repo is original work from this project.
