#!/usr/bin/env bash
# Resume lane 2: the models the orphaned first run never reached.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== v3 lane2 resume start $(date) ===" >> train_v3_lane2.log
for lang in ug fi; do
  echo "--- v3/$lang start $(date) ---" >> train_v3_lane2.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir "rebuild_2026-08/corpora_clean_v3" \
    --checkpoint-dir "checkpoints_v3" >> train_v3_lane2.log 2>&1
  echo "--- v3/$lang end rc=$? $(date) ---" >> train_v3_lane2.log
done
echo "=== v3 lane2 end $(date) ===" >> train_v3_lane2.log
