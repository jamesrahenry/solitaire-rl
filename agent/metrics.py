"""CSV metrics logging + a periodically-regenerated PNG progress plot for
DQN training. The CSV is the source of truth (append-only, one row per
report interval); the PNG is just a snapshot re-rendered from it, so it's
safe to re-run/inspect independently of the training process."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
import numpy as np

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


def _backfill_lifetime_wins(old_rows: list[dict]) -> list[int] | None:
    """Reconstruct an exact historical lifetime_wins series for rows logged
    before that column existed, from two columns that already existed:
    win_rate (this window's win fraction) and episodes (a *per-process*
    cumulative episode count so far). window win-count = round(win_rate *
    window_episodes) is exact, not an estimate, since win_rate was itself
    computed as an exact win_count / window_episodes ratio when it was
    logged. Returns None if the source columns aren't present.

    episodes resets to a small number every time training resumes into the
    same run directory (--resume-from/--run-dir reuse spawns a fresh
    process with its own episode_count starting at 0), so a plain
    row-to-row diff goes badly negative right at that boundary. Detect it
    (episodes decreasing) and treat it as the start of a new segment: that
    row's own window is just its own episode count, and the running
    cumulative total carries over from the end of the previous segment
    rather than resetting - it's the count of *wins*, which - unlike
    per-process episode/step counters - has no reason to reset just because
    the process did."""
    if not old_rows or "win_rate" not in old_rows[0] or "episodes" not in old_rows[0]:
        return None
    cumulative = 0
    result = []
    prev_episodes = None
    for row in old_rows:
        try:
            episodes = int(float(row["episodes"]))
            win_rate = float(row["win_rate"])
        except (KeyError, ValueError):
            return None
        window_episodes = episodes if prev_episodes is None or episodes < prev_episodes else episodes - prev_episodes
        cumulative += round(win_rate * window_episodes)
        result.append(cumulative)
        prev_episodes = episodes
    return result


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
            # column alignment: keep every historical row, instead of losing
            # the run's accumulated history.
            with self.csv_path.open() as f:
                old_rows = list(csv.DictReader(f))
            # lifetime_wins specifically is exactly reconstructible from two
            # columns that already existed before it did (win_rate, this
            # window's win fraction, and episodes, the cumulative count) -
            # backfill it properly rather than leaving old rows blank, which
            # would otherwise make the whole pre-migration history plot as a
            # flat/empty line.
            backfilled_wins = _backfill_lifetime_wins(old_rows) if "lifetime_wins" in self.fields else None
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.fields)
                for i, row in enumerate(old_rows):
                    values = []
                    for k in self.fields:
                        if k == "lifetime_wins" and backfilled_wins is not None and k not in row:
                            values.append(backfilled_wins[i])
                        else:
                            values.append(row.get(k, ""))
                    writer.writerow(values)

    def log(self, **kwargs) -> None:
        row = [kwargs.get(k, "") for k in self.fields]
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

    def plot(self, png_path: str, eval_csv_path: str | None = None) -> None:
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
        axes[0, 3].set_ylim(bottom=0)  # a cumulative count, never negative
        _line(axes[1, 0], steps, [r["epsilon"] for r in rows], "Epsilon", "epsilon")
        _line(
            axes[1, 1],
            steps,
            [r["mean_final_foundation"] for r in rows],
            "Mean final foundation total",
            "cards (of 52)",
            trend=True,
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

        eval_rows = _read_rows(Path(eval_csv_path)) if eval_csv_path and Path(eval_csv_path).exists() else []
        if eval_rows and "lifetime_wins" in rows[0]:
            _plot_trained_vs_random_wins(axes[1, 3], rows, eval_rows)
        else:
            axes[1, 3].axis("off")

        fig.tight_layout()
        _atomic_savefig(fig, png_path)
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
    _line(axes[2], steps, [r["eval_mean_foundation"] for r in rows], "Eval mean foundation total", "cards (of 52)", trend=True)

    fig.tight_layout()
    _atomic_savefig(fig, png_path)
    plt.close(fig)


def _atomic_savefig(fig, png_path: str) -> None:
    """Write to a temp file in the same directory, then atomically rename
    over the destination. fig.savefig() writes the destination path
    directly and non-atomically, so a concurrent reader (an image viewer
    polling the file - especially over a WSL/Windows mount) can catch it
    mid-write and see a torn, partially-rendered PNG (rendered top-to-
    bottom, so a partial read looks like "the top is fine, the bottom is
    cut off"). os.replace() is atomic on POSIX filesystems: a reader always
    sees either the complete old file or the complete new one."""
    path = Path(png_path)
    tmp_path = path.with_name(path.name + ".tmp")
    fig.savefig(tmp_path, dpi=100, format="png")  # explicit format: matplotlib infers it from the extension otherwise, and ".tmp" isn't one
    os.replace(tmp_path, path)


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
        return rows


def _plot_trained_vs_random_wins(ax, train_rows: list[dict], eval_rows: list[dict]) -> None:
    """Compare, at each eval checkpoint: how many of the fixed eval deals the
    greedy ("trained") policy actually solved, against how many wins
    epsilon-driven exploration ("random") turned up in training during the
    steps since the previous checkpoint. Both plotted as plain win counts
    over a matching interval - not a cumulative total against a rate -
    so they're directly comparable rather than needing to be reconciled by
    eye. This is the plot for "am I making progress toward a policy that
    wins on its own, or just accumulating lucky exploration wins.\""""
    eval_steps = [r["step"] for r in eval_rows]
    eval_win_counts = [round(r["eval_win_rate"] * r["eval_episodes"]) for r in eval_rows]

    train_by_step = {r["step"]: r["lifetime_wins"] for r in train_rows}
    train_steps_sorted = sorted(train_by_step)

    def lifetime_wins_at(step: float) -> float:
        best = 0.0
        for s in train_steps_sorted:
            if s > step:
                break
            best = train_by_step[s]
        return best

    random_win_counts = []
    prev_wins = 0.0
    for s in eval_steps:
        wins_now = lifetime_wins_at(s)
        random_win_counts.append(wins_now - prev_wins)
        prev_wins = wins_now

    ax.plot(eval_steps, eval_win_counts, marker="o", markersize=3, label="trained (greedy eval, /100 fixed deals)")
    ax.plot(eval_steps, random_win_counts, marker="o", markersize=3, label="random (training wins since last checkpoint)")
    ax.set_title("Trained wins vs random wins per checkpoint")
    ax.set_xlabel("step")
    ax.set_ylabel("win count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Edge-padded rolling mean, same length as the input, so it overlays
    directly on the raw series (including its first few points) rather than
    starting partway through or shifting the x-alignment."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return arr
    window = max(2, min(window, len(arr)))
    padded = np.concatenate([np.full(window - 1, arr[0]), arr])
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _line(ax, x, y, title, ylabel, trend: bool = False) -> None:
    if trend:
        ax.plot(x, y, alpha=0.35, color="tab:blue", label="raw")
        window = max(3, len(y) // 10)
        ax.plot(x, _rolling_mean(y, window), linewidth=2, color="tab:blue", label=f"trend ({window}-pt rolling mean)")
        ax.legend(fontsize=7, loc="best")
    else:
        ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
