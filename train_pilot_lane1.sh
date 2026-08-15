#!/usr/bin/env bash
# Punctuation-method pilot, lane 1 of 2: variants a and b, all languages.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== pilot lane1 start $(date) ===" >> train_pilot.log
for variant in a b; do
  for lang in tr az kk ky uz ug fi; do
    echo "--- $variant/$lang start $(date) ---" >> train_pilot.log
    poetry run python -m char_lstm.train --lang "$lang" \
      --corpus-dir "corpora_pilot_$variant" \
      --checkpoint-dir "checkpoints_pilot_$variant" >> train_pilot.log 2>&1
    echo "--- $variant/$lang end rc=$? $(date) ---" >> train_pilot.log
  done
done
echo "=== pilot lane1 end $(date) ===" >> train_pilot.log
