#!/usr/bin/env bash
# Resume lane 1: the models the orphaned first run never reached.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== v3 lane1 resume start $(date) ===" >> train_v3.log
for lang in az kk ky; do
  echo "--- v3/$lang start $(date) ---" >> train_v3.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir "rebuild_2026-08/corpora_clean_v3" \
    --checkpoint-dir "checkpoints_v3" >> train_v3.log 2>&1
  echo "--- v3/$lang end rc=$? $(date) ---" >> train_v3.log
done
echo "=== v3 lane1 end $(date) ===" >> train_v3.log
