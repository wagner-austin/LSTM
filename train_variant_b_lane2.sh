#!/usr/bin/env bash
# Variant-B training, lane 2 of 2.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== lane2 start $(date) ===" >> train_variant_b_lane2.log
for lang in uz ug fi; do
  echo "--- $lang start $(date) ---" >> train_variant_b_lane2.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir corpora_variant_b \
    --checkpoint-dir checkpoints_variant_b >> train_variant_b_lane2.log 2>&1
  echo "--- $lang end rc=$? $(date) ---" >> train_variant_b_lane2.log
done
echo "=== lane2 end $(date) ===" >> train_variant_b_lane2.log
