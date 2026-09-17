#!/usr/bin/env python3
"""Backfill structure_metrics.csv for a run that finished before
agent/structure_metrics.py existed, by reconstructing its true random init
(same --seed, same architecture, via torch.manual_seed) and diffing every
saved checkpoint against it - exactly what train.py now does live, just
after the fact using checkpoints that are already on disk.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import time
from pathlib import Path

import torch

from solitaire_gym.preprocessing import NUM_FEATURES, NUM_DECOMPOSED_FEATURES
from solitaire_gym.game import NUM_ACTIONS
from agent.qnetwork import DQN, DecomposedDQN
from agent.structure_metrics import structure_fields, compute_structure_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    torch.manual_seed(config["seed"])
    if config.get("card_encoding") == "decomposed":
        net = DecomposedDQN(num_actions=NUM_ACTIONS, num_hidden_layers=config["hidden_layers"], hidden_dim=config["hidden_dim"])
    else:
        net = DQN(num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_hidden_layers=config["hidden_layers"], hidden_dim=config["hidden_dim"])
    init_state = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}

    ckpts = [p for p in glob.glob(str(args.run_dir / "checkpoint_*.pt")) if re.search(r"checkpoint_(\d+)\.pt$", p)]
    ckpts.sort(key=lambda p: int(re.search(r"checkpoint_(\d+)\.pt$", p).group(1)))

    out_path = args.run_dir / "structure_metrics_offline.csv"
    rows = []
    for p in ckpts:
        step = int(re.search(r"checkpoint_(\d+)\.pt$", p).group(1))
        t0 = time.time()
        sd = torch.load(p, map_location="cpu", weights_only=True)
        row = compute_structure_row(step, init_state, sd)
        rows.append(row)
        print(f"step {step}: " + "  ".join(f"{k.replace('eff_dim__', '')}={v:.1f}" for k, v in row.items() if k.startswith("eff_dim__")) + f"  ({time.time()-t0:.1f}s)", flush=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=structure_fields(init_state))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
