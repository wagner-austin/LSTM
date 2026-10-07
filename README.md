# Language Model Loss Captures Mutual Intelligibility in Turkic Languages

**The claim: how surprised a language model is by another language predicts how
well humans understand it.** Mutual intelligibility is normally measured with
human listener studies, which are expensive and hard to scale. This measures it
with cross-entropy loss instead.

The headline experiment is **zero-shot**: train one character-level LSTM per
language on only its own language, then measure how surprised each model is by
every other language's IPA-transcribed perception text. Less surprise = more
mutually predictable. A transfer-learning track (pretrain, then fine-tune) is
also supported.

Finnish is the control — Uralic, not Turkic — and appears as both listener and
target, so a genuine Turkic intelligibility gradient has to separate itself from
an unrelated language scoring the same way.

## Which corpus

**[`CORPORA.md`](CORPORA.md) — read before training, evaluating, or citing a
number.** Three generations of cleaned corpora live here and the directory
names do not say which is current. It is
`rebuild_2026-08/corpora_clean_v3/`; the other two are superseded.

## Paper

**Language Model Loss Captures Mutual Intelligibility Gradients in Turkic Languages**

Moldir Baidildinova, Shiva Upadhye, Austin Wagner (UC Irvine)

Accepted at two conferences, both held at MIT:

- **[HSP 2026](https://hsp2026.org/)** — 39th Annual Conference on Human Sentence
  Processing
- **[Tu+11](https://turkicworkshop.github.io/tu11/)** — 11th Workshop on Turkic
  and Languages in Contact with Turkic

Being extended with a transformer comparison and a human-subjects validation
study.

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
| `fi` | Finnish     | Uralic control (both listener and target as of 2026-06-30) |

## Data Layout

| Path | Contents |
|------|----------|
| `corpora_raw/oscar_{lang}_ipa.txt` | Raw OSCAR IPA corpora (inputs to cleaning; never modified) |
| `corpora_clean/oscar_{lang}_ipa.txt` | Cleaned corpora — what training reads |
| `data/perception/perception_{lang}.txt` | Original perception snippets (Turkic set from Moldir; Finnish added 2026-06-30 by Austin from B2 orthography via `turkic_translit.core.to_ipa`); not modified after arrival |
| `data/perception_clean/perception_{lang}.txt` | Snippets with the symbol map applied — what eval reads |
| — (symbol map) | Ships inside turkic-translit as package data; each row cited. One copy, no fork here |
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
# 1. Clean raw corpora -> corpora_clean/, and harmonize snippets
#    -> data/perception_clean/. The cleaner ships with turkic-translit
#    (>= 0.5.1) and uses its packaged, cited symbol map; run it from that
#    project's environment. Verified byte-identical to the script that
#    used to live here.
turkic-clean-corpus --input-dir corpora_raw --output-dir corpora_clean
turkic-clean-corpus --harmonize-dir data/perception \
    --harmonize-output-dir data/perception_clean --pattern "perception_??.txt"

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

One target per donor language — the number selects which language is pretrained
on first, then fine-tuned into the other six:

| Target | Donor | Target | Donor | Target | Donor |
|--------|-------|--------|-------|--------|-------|
| `train-1` | `tr` | `train-4` | `ky` | `train-7` | `fi` |
| `train-2` | `az` | `train-5` | `uz` | | |
| `train-3` | `kk` | `train-6` | `ug` | | |

Each fine-tune reads `checkpoints/<donor>_best.pt`, so the donor's own base run
has to have finished first — `make train-bases` covers that for all seven.

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
the within-branch pairs (az–tr, kk–ky, ug–uz).

Whether two distances differ is a question about their **difference**, and it
is read from a file that puts an interval on the difference, never from
whether the two rows' own intervals overlap:

- **Two listeners on one text** — `<stem>_listeners.csv`, one row per text and
  pair of foreign listeners. Both listeners were scored on the same sections
  and positions of that text against the same native baseline, so the test is
  a shared-index paired bootstrap (`bootstrap_listener_difference` in
  `scripts/zero_shot_bootstrap.py`): every resample draws one list of section
  indices and applies it to both listeners, passage difficulty cancels, and
  `paired_interval_excludes_zero` is the verdict.
- **One pair read in each direction** — `<stem>_asymmetry.csv`. The two
  directions read different texts, so their sections are resampled
  independently (`bootstrap_asymmetry`); `interval_excludes_zero` is the
  verdict.

`<stem>_section_scores.json` holds the per-section scores every interval was
pooled from, so a later question can be asked of the same numbers without
re-scoring.

**Correction, 2026-10-06.** Until this date this section said "two distances
differ reliably only when their confidence intervals don't overlap", which is
the test the code's own `bootstrap_asymmetry` docstring calls wrong. Its error
runs one way: intervals that do not overlap do imply a difference, but
overlapping ones imply nothing, and for two listeners on one text the rule
also discards the pairing that cancels passage difficulty. So it can only fail
to detect a difference, never invent one. Re-decided with the paired test on
the published matrix's own section scores
(`results/zero_shot_excess_ce_skip_section_scores.json`, which
`python -m scripts.compare_listeners` re-pools into
`results/zero_shot_excess_ce_skip.csv` byte for byte before it tests
anything), **20 of the 105 listener comparisons change verdict**: the overlap
rule separated 69, the paired test separates 89, all 20 changes go from
"unresolved" to "differ", and none of the 69 is reversed. By text: az 1, fi 4,
kk 3, ky 3, tr 5, ug 3, uz 1. On the Kazakh text, of the six comparisons the
rule left unresolved, az–fi, az–tr and az–uz now differ and fi–ug, fi–uz and
ug–uz still do not. The headline ordering ky→kk < tr→kk was separated under
both tests. Each interval is a 95% interval for one comparison; across 105 of
them, roughly one in twenty comparisons with no true difference would exclude
zero by chance, so read any single new separation with that in mind. The rows
are in `results/zero_shot_excess_ce_skip_listeners.csv`; the July pipeline
that produced the scores is rebuilt by
`analysis/listener-comparisons-2026-10/dump_published_section_scores.py`.

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
- `results/zero_shot_excess_ce_{skip,unk,assimilate}.csv` — eval matrices, each
  written with `_section_scores.json`, `_asymmetry.csv` and `_listeners.csv`
  beside it (see *Reading the results*)
- `results/ngram_excess_ce.csv` — trigram baseline
- `results/validity_report.json` — validity battery outcome

## Monitoring

Training logs to Weights & Biases (project `char-level-lstm`); set
`WANDB_MODE=disabled` to run offline (as the tests do).
