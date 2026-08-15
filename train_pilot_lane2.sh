#!/usr/bin/env bash
# Punctuation-method pilot, lane 2 of 2: variant c, all languages.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== pilot lane2 start $(date) ===" >> train_pilot_lane2.log
for lang in tr az kk ky uz ug fi; do
  echo "--- c/$lang start $(date) ---" >> train_pilot_lane2.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir corpora_pilot_c \
    --checkpoint-dir checkpoints_pilot_c >> train_pilot_lane2.log 2>&1
  echo "--- c/$lang end rc=$? $(date) ---" >> train_pilot_lane2.log
done
echo "=== pilot lane2 end $(date) ===" >> train_pilot_lane2.log
