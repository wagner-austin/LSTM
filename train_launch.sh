#!/usr/bin/env bash
# Detached training launcher: owns its own lifetime, independent of any
# tool session. Appends to the log the session's monitor is tailing.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== detached training start $(date) ===" >> train_2026-08.log
make train-bases >> train_2026-08.log 2>&1
echo "=== detached training end $(date) rc=$? ===" >> train_2026-08.log
