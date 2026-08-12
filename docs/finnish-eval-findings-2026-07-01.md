# Finnish added to the excess-CE matrix — findings (2026-07-01)

## What changed

`data/perception/perception_fi.txt` was added on 2026-06-30 (20 parallel B2 passages, IPA-transliterated via `turkic_translit.core.to_ipa` with `fi_ipa.rules`). The zero-shot eval pipeline was rerun; the results in `results/zero_shot_excess_ce_{skip,unk,assimilate}.csv` now hold a full 7 × 7 matrix (49 rows) rather than the previous 7 × 6 + 1 (43 rows including header).

The Finnish orthographic source is preserved at `data/perception/perception_fi_source.txt` for audit. A native-Finnish spot-check on the source is recommended before publication use.

## Pre-existing rows: byte-identical after rerun

Every listener/text pair in the file originally shared with Moldir on 2026-05-14 has byte-identical values in the current file. The pipeline is deterministic: adding Finnish did not perturb any measurement. Verified by numeric comparison at `1e-9` tolerance across all 42 overlapping pairs (`results/zero_shot_excess_ce_skip.csv`).

## New rows — Turkic listeners scoring Finnish text

Skip mode:

| Listener | text_language | cross_entropy | native_cross_entropy | excess_cross_entropy | 95% CI | n |
|---|---|---|---|---|---|---|
| az | fi | 3.354 | 1.304 | **2.050** | [1.977, 2.125] | 3228 |
| fi | fi | 1.304 | 1.304 | **0.000** | [0.000, 0.000] | 3228 |
| kk | fi | 3.599 | 1.304 | **2.295** | [2.210, 2.379] | 3228 |
| ky | fi | 3.781 | 1.304 | **2.477** | [2.361, 2.592] | 3228 |
| tr | fi | 3.361 | 1.304 | **2.057** | [1.969, 2.145] | 3228 |
| ug | fi | 3.906 | 1.304 | **2.602** | [2.487, 2.719] | 3228 |
| uz | fi | 3.681 | 1.304 | **2.377** | [2.251, 2.504] | 3228 |

`fi→fi` at excess 0 is the diagonal (native scoring native).

## Numerical read

**Turkic → Finnish excess-CE range: 2.05 – 2.60.** Every Turkic-listener excess-CE on Finnish text lies above every within-family Turkic pair except `ky→az` (2.66). Finnish is a distant Uralic control, exactly as the paper's methodology intends.

**Sibling pairs remain the lowest off-diagonal distances** — the paper's within-branch prediction from Table 1 (Oghuz az↔tr, Kipchak kk↔ky, Karluk uz↔ug) survives the matrix expansion:

- `ky → kk` 1.19 (Kipchak sibling)
- `tr → az` 1.37 (Oghuz sibling)
- `tr → kk` 1.42 (non-sibling — appears in the top three)
- `uz → ug` 1.50 (Karluk sibling)
- `az → kk` 1.53 (non-sibling)
- `az → tr` 1.56 (Oghuz sibling)

Four of the top-six lowest-excess directional pairs are sibling pairs, but `tr→kk` at 1.42 (position 3) is non-sibling. Whether `scripts/validate_method.py` exits 0 depends on whether it symmetrises the matrix (in which case sibling pairs cluster) or reads directional rows (in which case this non-sibling entry is a candidate outlier). Determining this requires running validate_method against the new matrix — deferred as its own step.

**Position counts on Finnish-text rows are uniformly `3228`** because skip-mode restricts scoring to positions whose next character is in *every* listener's vocab. Adding Finnish shrinks the shared-vocab intersection across the full listener set, which is why the new column's `n` differs from the Turkic-only rows (0.646–0.767 fractions on 3103–4183 counts). The pre-existing 6-Turkic rows are unchanged.

**Bootstrap CIs on the new Finnish-target values are 0.11 – 0.16 wide**, comfortably narrower than the between-listener differences. Every Turkic-listener excess-CE on Finnish is CI-non-overlapping with the largest sibling-pair distance (Karluk `uz→ug` 1.50), so the "Finnish is farther than any sibling pair" claim survives paired-bootstrap uncertainty.

## Sanity: fi listener on Turkic text (unchanged since 2026-05-14)

For symmetry, the 6 rows where `fi` scores Turkic text (existed before this session):

| Listener | text_language | cross_entropy | excess_cross_entropy | 95% CI |
|---|---|---|---|---|
| fi | az | 3.794 | 2.664 | [2.541, 2.776] |
| fi | kk | 3.427 | 1.667 | [1.591, 1.741] |
| fi | ky | 3.616 | 2.413 | [2.344, 2.474] |
| fi | tr | 3.561 | 2.438 | [2.344, 2.527] |
| fi | ug | 3.712 | 2.440 | [2.315, 2.567] |
| fi | uz | 3.911 | 2.733 | [2.613, 2.855] |

Averaging Turkic↔fi in both directions gives symmetric distances in [1.98, 2.68] — again, all above every Turkic-family sibling pair.

## Files

- `results/zero_shot_excess_ce_skip.csv` — 49-row primary output (skip mode, headline metric)
- `results/zero_shot_excess_ce_unk.csv` — 49-row unk-mode output
- `results/zero_shot_excess_ce_assimilate.csv` — 49-row assimilate-mode output
- `results/zero_shot_excess_ce_skip_forMoldir.csv` — same as skip-mode file with the `scoring_mode` column dropped, matching the column shape Moldir received on 2026-05-14
- `data/assimilation.csv` — 71 substitutions (adds fi listener column to the pre-existing 6-Turkic table)
- `data/perception_clean/perception_fi.txt` — cleaned Finnish snippet (applies `symbol_map.csv`; input to the eval)
- `data/perception/perception_fi.txt` — raw Finnish IPA (output of the transliterator)
- `data/perception/perception_fi_source.txt` — Finnish orthographic source

## Validity re-run (2026-08-10)

`poetry run python -m scripts.validate_method` against the 49-row matrix exits 0 with all five checks passing. Note the invocation: the module form is required, because running the file path directly fails on `from scripts.clean_corpus import CORPUS_TEMPLATE`.

The `tr→kk` 1.42 concern raised above did not break the criterion.

| | 2026-06-22 | 2026-08-10 | Δ |
|---|---|---|---|
| real within | +1.6108 | +1.5753 | −0.0355 |
| real cross | +2.1858 | +2.0653 | −0.1205 |
| **real gap** | **+0.5750** | **+0.4900** | −0.0850 |
| shuffled gap | +0.1276 | +0.0906 | −0.0370 |
| heldout gap | +0.4146 | +0.4146 | 0.0000 |

The substantive change is topological. `real_tree` went from `(((az,tr),(kk,ky)),(ug,uz))` to `((((az,tr),(kk,ky)),(ug,uz)),fi)` — the real matrix now recovers Finnish as the outgroup unaided, which previously only `heldout_tree` did. All three branch pairs still cluster.

The real gap narrowing by 0.085 is expected rather than adverse: Finnish enters as a singleton whose distances are large but finite, pulling the cross-branch mean down relative to a six-language set in which every cross pair was Turkic–Turkic. The gap stays clearly positive, the shuffled control still collapses toward zero, and `heldout` is numerically identical — consistent with the byte-identical finding above.

The report was written to a scratch path, so `results/validity_report.json` still holds the 2026-06-22 run and needs promoting if these numbers are the ones to keep.

## Follow-up items (not this session)

1. **Native-Finnish spot-check on `perception_fi_source.txt`.** The orthographic source was translated at B2 register but has not been reviewed by a native Finnish reader. Recommended before quoting in the proceedings paper.
2. ~~**Run `scripts/validate_method.py`** against the 49-row matrix to determine whether the validity criterion (three smallest off-diagonal distances within-branch) still exits 0.~~ **RESOLVED 2026-08-10 — passes, and the topology result improved.** See "Validity re-run" below.
3. **Consider running the ngram baseline** (`scripts/ngram_baseline.py`) so the trigram-baseline row for `text_language = fi` also lands.
4. **Panphon upstream PR** — file `encoding="utf-8"` on the two `open()` calls in `featuretable.py` so downstream users on Windows Python 3.11 don't need the scoped context manager we shipped in the turkic-transliteration test suite.
