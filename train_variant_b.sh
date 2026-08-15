#!/usr/bin/env bash
# Detached variant-B training: punctuation collapsed to one symbol, digits
# to another. Isolated corpus and checkpoint directories; the released
# artifacts are never touched.
cd /c/Users/Test/PROJECTS/lstm || exit 1
export WANDB_MODE=disabled
echo "=== variant B training start $(date) ===" >> train_variant_b.log
for lang in tr az kk ky uz ug fi; do
  echo "--- $lang start $(date) ---" >> train_variant_b.log
  poetry run python -m char_lstm.train --lang "$lang" \
    --corpus-dir corpora_variant_b \
    --checkpoint-dir checkpoints_variant_b >> train_variant_b.log 2>&1
  echo "--- $lang end rc=$? $(date) ---" >> train_variant_b.log
done
echo "=== variant B training end $(date) ===" >> train_variant_b.log
