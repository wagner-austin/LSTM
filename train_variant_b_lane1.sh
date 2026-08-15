#!/usr/bin/env bash
# Variant-B training, lane 1 of 2.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== lane1 start $(date) ===" >> train_variant_b.log
for lang in tr az kk ky; do
  echo "--- $lang start $(date) ---" >> train_variant_b.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir corpora_variant_b \
    --checkpoint-dir checkpoints_variant_b >> train_variant_b.log 2>&1
  echo "--- $lang end rc=$? $(date) ---" >> train_variant_b.log
done
echo "=== lane1 end $(date) ===" >> train_variant_b.log
