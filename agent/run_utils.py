"""Run versioning/tagging: every training run gets its own numbered,
tagged directory under runs/ (e.g. runs/002_double_dqn/) holding everything
that run produced - config, a human-readable README, checkpoints, metrics,
plot, game log, stdout - so iterating on hyperparameters/algorithm changes
doesn't clobber previous runs and each one stays reproducible and available
for later study.

IMPORTANT: real runs live under runs/. Smoke tests / correctness checks
should always pass --runs-dir runs/_smoketest (a separate tree) so cleaning
them up (rm -rf runs/_smoketest) can never touch real run data. Never run
a bare `rm -rf runs` - that has already destroyed two completed real runs
in this project's history.
"""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path


def next_run_dir(runs_dir: Path, tag: str) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        prefix = d.name.split("_", 1)[0]
        if prefix.isdigit():
            max_idx = max(max_idx, int(prefix))
    run_dir = runs_dir / f"{max_idx + 1:03d}_{tag}"
    run_dir.mkdir(parents=True)
    return run_dir


def _git_info() -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def save_config(run_dir: Path, args_namespace) -> dict:
    config = vars(args_namespace).copy()
    config["run_dir"] = str(run_dir)
    config.update(_git_info())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return config


def write_readme_start(run_dir: Path, tag: str, notes: str, config: dict) -> None:
    """Write a human-readable README.md at the start of a run: what this run
    is (tag + free-text notes) and its full config, so a run is
    understandable at a glance without cross-referencing config.json. A
    placeholder Results section gets filled in by finalize_readme once
    training completes (or stays a placeholder if the run gets interrupted -
    which is itself useful, honest information)."""
    lines = [
        f"# Run: {tag}",
        "",
        f"- **Started:** {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- **Git commit:** {config.get('git_commit') or '(no commits yet)'}"
        + (" (dirty working tree)" if config.get("git_dirty") else ""),
        "",
        "## Notes",
        "",
        notes.strip() if notes and notes.strip() else "_(none provided)_",
        "",
        "## Config",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for k, v in config.items():
        if k == "run_dir":
            continue
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Results", "", "_(pending - run still in progress, or was interrupted before finishing)_", ""]
    (run_dir / "README.md").write_text("\n".join(lines))


def finalize_readme(run_dir: Path, summary: dict) -> None:
    """Replace the placeholder Results section with the real outcome."""
    path = run_dir / "README.md"
    text = path.read_text()
    results_lines = ["## Results", "", f"- **Finished:** {datetime.datetime.now().isoformat(timespec='seconds')}"]
    for k, v in summary.items():
        results_lines.append(f"- **{k}:** {v}")
    results_block = "\n".join(results_lines) + "\n"
    head = text.split("## Results")[0] if "## Results" in text else text + "\n"
    path.write_text(head + results_block)


class TeeLogger:
    """Writes progress lines to both stdout and a run-local file, so a
    background run's log lives with the rest of that run's artifacts
    regardless of how (or whether) the launching shell redirected stdout."""

    def __init__(self, path: Path, append: bool = False):
        self._file = path.open("a" if append else "w")

    def __call__(self, msg: str) -> None:
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
