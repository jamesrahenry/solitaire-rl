#!/bin/bash
# Auto-resuming wrapper for agent.train: if the training process dies before
# reaching checkpoint_final.pt (e.g. an OOM kill), automatically relaunches
# from the latest saved checkpoint instead of losing all progress.
#
# Usage:
#   agent/train_resilient.sh <run-dir> <tag> [extra agent.train args...]
#
# Example:
#   agent/train_resilient.sh runs/004_foo double_dqn_soft --steps 500000 --checkpoint-every 10000
#
# If [extra args] includes --resume-from (e.g. to warm-start from a
# behavior-cloning checkpoint), that only applies on attempt 1. On any
# later attempt (after an OOM kill etc.) the wrapper's own --resume-from
# pointing at the latest in-run checkpoint is used instead, and the
# caller-supplied --resume-from/--resume-step pair is stripped out - argparse
# takes the *last* --resume-from on the command line, so without this the
# original checkpoint would silently win over the further-trained one on
# every restart, discarding all progress since.
set -uo pipefail

RUN_DIR="$1"; shift
TAG="$1"; shift

cd "$(dirname "$0")/.."
source .venv/bin/activate

INITIAL_ARGS=("$@")
RESUME_ARGS=()
ATTEMPT=1
while true; do
    echo "=== train_resilient: attempt $ATTEMPT ==="
    if [ "$ATTEMPT" -eq 1 ]; then
        EXTRA_ARGS=("${INITIAL_ARGS[@]}")
    else
        # strip any --resume-from/--resume-step (and their values) the caller passed,
        # so only the wrapper's own RESUME_ARGS (the latest in-run checkpoint) apply
        EXTRA_ARGS=()
        skip_next=0
        for arg in "${INITIAL_ARGS[@]}"; do
            if [ "$skip_next" -eq 1 ]; then skip_next=0; continue; fi
            case "$arg" in
                --resume-from|--resume-step) skip_next=1; continue ;;
            esac
            EXTRA_ARGS+=("$arg")
        done
    fi
    python -u -m agent.train --run-dir "$RUN_DIR" --tag "$TAG" "${RESUME_ARGS[@]}" "${EXTRA_ARGS[@]}"

    if [ -f "$RUN_DIR/checkpoint_final.pt" ]; then
        echo "=== train_resilient: training completed successfully ==="
        break
    fi

    LATEST_STEP=$(ls -1 "$RUN_DIR"/checkpoint_*.pt 2>/dev/null | grep -v checkpoint_final | sed -E 's/.*checkpoint_([0-9]+)\.pt/\1/' | sort -n | tail -1)
    if [ -z "$LATEST_STEP" ]; then
        echo "=== train_resilient: died before any checkpoint was saved; restarting from scratch ==="
        RESUME_ARGS=()
    else
        echo "=== train_resilient: died - resuming from checkpoint at step $LATEST_STEP ==="
        RESUME_ARGS=(--resume-from "$RUN_DIR/checkpoint_${LATEST_STEP}.pt" --resume-step "$LATEST_STEP")
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done
