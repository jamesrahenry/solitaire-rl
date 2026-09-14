"""CSV metrics logging + a periodically-regenerated PNG progress plot for
DQN training. The CSV is the source of truth (append-only, one row per
report interval); the PNG is just a snapshot re-rendered from it, so it's
safe to re-run/inspect independently of the training process."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV_FIELDS = [
    "step",
    "episodes",
    "epsilon",
    "mean_return",
    "mean_loss",
    "mean_episode_len",
    "win_rate",
    "stalled_rate",
    "truncated_rate",
    "mean_final_foundation",
    "lifetime_wins",
]

EVAL_CSV_FIELDS = [
    "step",
    "eval_episodes",
    "eval_win_rate",
    "eval_mean_return",
    "eval_mean_foundation",
    "eval_mean_length",
]


class MetricsLogger:
    def __init__(self, csv_path: str, fields: list[str] | None = None):
        self.csv_path = Path(csv_path)
        self.fields = fields or CSV_FIELDS
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(self.fields)
            return

        with self.csv_path.open() as f:
            existing_header = next(csv.reader(f), None)
        if existing_header is not None and existing_header != self.fields:
            # Schema changed since this file was created - e.g. resuming a run
            # (--resume-from/--run-dir reuse) under code that now logs an extra
            # column. Migrate in place rather than crash or silently corrupt
            # column alignment: keep every historical row, backfilling any new
            # column with "" (read back as NaN), instead of losing the run's
            # accumulated history.
            with self.csv_path.open() as f:
                old_rows = list(csv.DictReader(f))
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.fields)
                for row in old_rows:
                    writer.writerow([row.get(k, "") for k in self.fields])

    def log(self, **kwargs) -> None:
        row = [kwargs.get(k, "") for k in self.fields]
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

    def plot(self, png_path: str) -> None:
        rows = _read_rows(self.csv_path)
        if not rows:
            return

        steps = [r["step"] for r in rows]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f"Solitaire DQN training progress (step {int(steps[-1]):,})")

        _line(axes[0, 0], steps, [r["mean_return"] for r in rows], "Mean episode return", "return")
        _line(axes[0, 1], steps, [r["mean_episode_len"] for r in rows], "Mean episode length", "steps")
        _line(axes[0, 2], steps, [r["mean_loss"] for r in rows], "Mean training loss", "loss")
        _line(axes[0, 3], steps, [r["lifetime_wins"] for r in rows], "Cumulative training-time wins", "wins (count)")
        axes[0, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
        _line(axes[1, 0], steps, [r["epsilon"] for r in rows], "Epsilon", "epsilon")
        _line(
            axes[1, 1],
            steps,
            [r["mean_final_foundation"] for r in rows],
            "Mean final foundation total",
            "cards (of 52)",
        )

        ax = axes[1, 2]
        ax.stackplot(
            steps,
            [r["win_rate"] for r in rows],
            [r["stalled_rate"] for r in rows],
            [r["truncated_rate"] for r in rows],
            labels=["won", "stalled", "truncated"],
        )
        ax.set_title("Episode outcome breakdown (this window)")
        ax.set_xlabel("step")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)

        axes[1, 3].axis("off")

        fig.tight_layout()
        fig.savefig(png_path, dpi=100)
        plt.close(fig)


def plot_eval(csv_path: str, png_path: str) -> None:
    """Plot the frozen-greedy evaluation trend: unlike the main training
    metrics (which reflect whatever epsilon was active), this is a clean
    read on whether the learned policy itself - no exploration noise - is
    actually getting better at winning over the course of training."""
    path = Path(csv_path)
    if not path.exists():
        return
    rows = _read_rows(path)
    if not rows:
        return

    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Greedy-policy evaluation (no exploration) - step {int(steps[-1]):,}")

    _line(axes[0], steps, [r["eval_win_rate"] for r in rows], "Eval win rate", "win rate")
    axes[0].set_ylim(0, 1)
    _line(axes[1], steps, [r["eval_mean_return"] for r in rows], "Eval mean return", "return")
    _line(axes[2], steps, [r["eval_mean_foundation"] for r in rows], "Eval mean foundation total", "cards (of 52)")

    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    plt.close(fig)


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
        return rows


def _line(ax, x, y, title, ylabel) -> None:
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
