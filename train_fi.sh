#!/usr/bin/env bash
# Finnish only: the make target's PowerShell shell needs a console for its
# window title and a detached window has none, so the trainer is invoked
# directly.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== fi training start $(date) ===" >> train_2026-08.log
poetry run python -m char_lstm.train --lang fi >> train_2026-08.log 2>&1
echo "=== fi training end $(date) rc=$? ===" >> train_2026-08.log
