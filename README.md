# Turkic Language Mutual Intelligibility Experiments

Character-level LSTM experiments for measuring mutual intelligibility (MI)
gradients across Turkic languages via cross-entropy loss. The headline
experiment is **zero-shot**: train one model per language on only its own
language, then measure how surprised each model is by every other language's
IPA-transcribed perception text. Less surprise = more mutually predictable.
A transfer-learning track (pretrain, then fine-tune) is also supported.

## Paper

**Title:** Language Model Loss Captures Mutual Intelligibility Gradients in Turkic Languages

**Authors:** Moldir Baidildinova, Shiva Upadhye, Austin Wagner (UC Irvine)

## Setup

```bash
poetry install --with dev          # dev extras add pytest, mypy, ruff
```

Run all checks (guards, lint, type-check, tests at 100% statement+branch coverage):

```bash
make check
make lint    # guard.py + Ruff + Mypy over src, tests, scripts
make test    # pytest -n auto with branch coverage over src + scripts
```

## Languages

| Code | Language    | Role / Branch       |
|------|-------------|---------------------|
| `tr` | Turkish     | Oghuz               |
| `az` | Azerbaijani | Oghuz               |
| `kk` | Kazakh      | Kipchak             |
| `ky` | Kyrgyz      | Kipchak             |
| `uz` | Uzbek       | Karluk              |
| `ug` | Uyghur      | Karluk              |
| `fi` | Finnish     | non-Turkic control  |

## Data Layout

| Path | Contents |
|------|----------|
| `corpora_raw/oscar_{lang}_ipa.txt` | Raw OSCAR IPA corpora (inputs to cleaning; never modified) |
| `corpora_clean/oscar_{lang}_ipa.txt` | Cleaned corpora — what training reads |
| `data/perception/perception_{lang}.txt` | Original perception snippets (from Moldir; not modified) |
| `data/perception_clean/perception_{lang}.txt` | Snippets with the symbol map applied — what eval reads |
| `data/symbol_map.csv` | Per-language IPA harmonization decisions, each cited |
| `data/assimilation.csv` | Generated OOV substitution table (nearest-sound per listener) |
| `results/` | Eval CSVs + validity report |

The corpora are OSCAR text, language-ID filtered, deterministically
transliterated to broad IPA, equalized to the smallest corpus (~10.2M chars),
split 70/15/15 (train/val/test) by position. Cleaning collapses the raw
vocabularies (554–1,313 symbols) to ~67–79 real IPA characters.

Note: `perception_uz.txt` has 19 sections, not 20 — TEXT 2 passage 5 was
excluded at source (audio removed for speaker disfluency). See
`data/perception/manifest.json`. Do not reconstruct it.

## Pipeline

```bash
# 1. Clean raw corpora -> corpora_clean/ and harmonize snippets
#    -> data/perception_clean/ (applies data/symbol_map.csv, dedups,
#    strips non-IPA junk, equalizes sizes, writes cleaning_manifest.json)
poetry run python -m scripts.clean_corpus

# 2. Train the 7 base models on the cleaned corpora (zero-shot needs only these)
make train-bases

# 3. Generate the OOV assimilation table from the trained vocabs + snippets
poetry run python -m scripts.build_assimilation

# 4. Zero-shot eval, one CSV per OOV mode (skip is the headline metric)
poetry run python -m scripts.zero_shot_eval --oov-mode skip \
    --output-csv results/zero_shot_excess_ce_skip.csv
poetry run python -m scripts.zero_shot_eval --oov-mode unk \
    --output-csv results/zero_shot_excess_ce_unk.csv
poetry run python -m scripts.zero_shot_eval --oov-mode assimilate \
    --output-csv results/zero_shot_excess_ce_assimilate.csv

# 5. Character-trigram baseline (same matrix, simpler model)
poetry run python -m scripts.ngram_baseline

# 6. Validity battery (exit 0 only if every check passes)
poetry run python -m scripts.validate_method
```

### OOV modes (`--oov-mode`)

When a listener model meets a sound absent from its own vocabulary:

- `skip` — score only positions whose next character is in **every** model's
  vocabulary, so all models are scored on identical positions. Parameter-free;
  this is the headline metric.
- `unk` — score every position, mapping unseen characters to `<unk>`.
- `assimilate` — replace each unseen character with its nearest in-vocabulary
  segment (`data/assimilation.csv`) before scoring.

### Transfer-learning track

The base+fine-tune experiments (`make train`, or `train-1` … `train-7`)
pretrain on one language and fine-tune on each other with `--freeze-embed`.
Not required for the zero-shot result.

## Reading the results

`results/zero_shot_excess_ce_skip.csv`, one row per (listener, text) pair:

| column | meaning |
|--------|---------|
| `listener_language` | model doing the scoring |
| `text_language` | language of the scored text |
| `scoring_mode` | OOV mode used |
| `cross_entropy` | listener model's surprise on the text (lower = more predictable) |
| `native_cross_entropy` | the text's own model's surprise on the same positions (baseline) |
| `excess_cross_entropy` | `cross_entropy − native_cross_entropy` — **the distance** (passage difficulty removed) |
| `excess_confidence_interval_low` / `_high` | 95% paired-bootstrap CI for the distance |
| `fraction_of_positions_scored` | share of positions scored (identical for all listeners of a given text) |
| `number_of_positions_scored` | that share as a count |

Sort by `excess_cross_entropy`: the three smallest off-diagonal distances are
the within-branch pairs (az–tr, kk–ky, ug–uz); two distances differ reliably
only when their confidence intervals don't overlap.

`validate_method` writes `results/validity_report.json`: it must recover the
three sibling pairs on the perception text, replicate them on held-out corpus
slices, and find the branch signal collapse when character order is shuffled.

## CLI Reference: `char_lstm.train`

| Flag | Description | Default |
|------|-------------|---------|
| `--lang` | Language code (tr, az, kk, ky, uz, ug, fi) | Required |
| `--from-checkpoint` | Source checkpoint for fine-tuning | None (from scratch) |
| `--freeze-embed` | Freeze the embedding layer during fine-tuning | False |
| `--epochs` | Training epochs | 3 |
| `--lr` | Learning rate | 1e-4 |
| `--device` | `auto` / `cpu` / `cuda` (`auto` picks cuda when available) | `auto` |

## Model Architecture

Single source of truth: the `EMBED_DIM` / `HIDDEN_DIM` / `NUM_LAYERS` /
`DROPOUT` constants in `char_lstm/train.py`.

- **Type:** 2-layer character-level LSTM (~947K parameters)
- **Embedding dim:** 128
- **Hidden dim:** 256
- **Dropout:** 0.1
- **Vocab:** cleaned IPA characters + `<unk>`

## Outputs

- `checkpoints/{lang}_best.pt` + `{lang}_vocab.json` — the 7 base models
- `checkpoints/{src}_to_{tgt}*.pt` — transfer fine-tunes (transfer track only)
- `results/zero_shot_excess_ce_{skip,unk,assimilate}.csv` — eval matrices
- `results/ngram_excess_ce.csv` — trigram baseline
- `results/validity_report.json` — validity battery outcome

## Monitoring

Training logs to Weights & Biases (project `char-level-lstm`); set
`WANDB_MODE=disabled` to run offline (as the tests do).
