# Turkic Language Mutual Intelligibility Experiments

Character-level LSTM experiments for measuring mutual intelligibility (MI) gradients across Turkic languages via cross-entropy loss and transfer learning.

## Paper

**Title:** Language Model Loss Captures Mutual Intelligibility Gradients in Turkic Languages

**Authors:** Moldir Baidildinova, Shiva Upadhye, Austin Wagner (UC Irvine)

## Setup

```bash
cd C:\Users\austi\PROJECTS\LSTM
poetry install
```

## Available Languages

| Code | Language    | Branch  | Corpus File                          |
|------|-------------|---------|--------------------------------------|
| `tr` | Turkish     | Oghuz   | `09_Downloaded_Corpora/oscar_tr_ipa.txt` |
| `az` | Azerbaijani | Oghuz   | `09_Downloaded_Corpora/oscar_az_ipa.txt` |
| `kk` | Kazakh      | Kipchak | `09_Downloaded_Corpora/oscar_kk_ipa.txt` |
| `ky` | Kyrgyz      | Kipchak | `09_Downloaded_Corpora/oscar_ky_ipa.txt` |
| `uz` | Uzbek       | Karluk  | `09_Downloaded_Corpora/oscar_uz_ipa.txt` |
| `ug` | Uyghur      | Karluk  | `09_Downloaded_Corpora/oscar_ug_ipa.txt` |

## Experimental Design

### Experiment Set 1: Turkish (Oghuz) Base

Tests whether Turkish transfers better to Azerbaijani (same Oghuz branch) than to Kazakh (Kipchak branch).

```bash
# Step 1: Train Turkish from scratch
poetry run python -m char_lstm.train --lang tr

# Step 2: Fine-tune on Azerbaijani (same branch - expect good transfer)
poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/tr_best.pt --freeze-embed

# Step 3: Fine-tune on Kazakh (different branch - expect worse transfer)
poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/tr_best.pt --freeze-embed
```

**Expected result:** Lower CE loss / faster convergence for Azerbaijani than Kazakh.

### Experiment Set 2: Uzbek (Karluk) Base

Tests transfer from Uzbek to Uyghur vs Kazakh.

```bash
# Step 1: Train Uzbek from scratch
poetry run python -m char_lstm.train --lang uz

# Step 2: Fine-tune on Uyghur
poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/uz_best.pt --freeze-embed

# Step 3: Fine-tune on Kazakh
poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/uz_best.pt --freeze-embed
```

## CLI Reference

```bash
poetry run python -m char_lstm.train --help
```

| Flag | Description | Default |
|------|-------------|---------|
| `--lang` | Language code (tr, az, kk, ky, uz, ug) | Required |
| `--from-checkpoint` | Path to checkpoint for fine-tuning | None (train from scratch) |
| `--freeze-embed` | Freeze embedding layer during fine-tuning | False |
| `--epochs` | Number of training epochs | 3 |
| `--lr` | Learning rate | 1e-4 |

## Output Checkpoints

Training from scratch:
- `checkpoints/tr_best.pt` - Turkish base model
- `checkpoints/uz_best.pt` - Uzbek base model

Fine-tuning:
- `checkpoints/tr_to_az.pt` - Turkish fine-tuned on Azerbaijani
- `checkpoints/tr_to_kk.pt` - Turkish fine-tuned on Kazakh
- `checkpoints/uz_to_ug.pt` - Uzbek fine-tuned on Uyghur
- `checkpoints/uz_to_kk.pt` - Uzbek fine-tuned on Kazakh

## Model Architecture

- **Type:** 2-layer character-level LSTM
- **Embedding dim:** 128
- **Hidden dim:** 256
- **Vocab:** IPA characters + `<unk>`

## Data

- **Source:** OSCAR corpus, filtered with FastText language ID
- **Preprocessing:** Deterministic transliteration to broad IPA
- **Max chars:** 10M per language
- **Split:** 75% train / 25% val

## Monitoring

Training logs to Weights & Biases:
- Project: `char-level-lstm`
- Dashboard: https://wandb.ai/austinwagner-uci/char-level-lstm

## Key Metrics

- **Cross-Entropy Loss:** Lower = model finds language more "plausible"
- **Perplexity (PPL):** exp(CE loss) - interpretable as "effective alphabet size"
- **AUC of loss curve:** Lower = faster convergence during fine-tuning

## Full Command Sequence

```bash
# === EXPERIMENT SET 1: Turkish base ===
poetry run python -m char_lstm.train --lang tr
poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/tr_best.pt --freeze-embed
poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/tr_best.pt --freeze-embed

# === EXPERIMENT SET 2: Uzbek base ===
poetry run python -m char_lstm.train --lang uz
poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/uz_best.pt --freeze-embed
poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/uz_best.pt --freeze-embed
```
