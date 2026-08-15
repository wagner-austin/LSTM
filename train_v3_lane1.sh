#!/usr/bin/env bash
# Rebuilt-corpora full training, lane 1 of 2.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== v3 lane1 start $(date) ===" >> train_v3.log
for lang in tr az kk ky; do
  echo "--- v3/$lang start $(date) ---" >> train_v3.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir "rebuild_2026-08/corpora_clean_v3" \
    --checkpoint-dir "checkpoints_v3" >> train_v3.log 2>&1
  echo "--- v3/$lang end rc=$? $(date) ---" >> train_v3.log
done
echo "=== v3 lane1 end $(date) ===" >> train_v3.log
