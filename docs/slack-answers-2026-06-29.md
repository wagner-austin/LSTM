# Answers to the Tu+/SCiL Slack questions (2026-06-29)

Each section answers one of Moldir / Shiva's questions. Citations point at the wiki at `~/PROJECTS/wiki/pages/`, which carries the primary-source-backed analysis. Quotes from papers are verbatim from the cited page numbers.

---

## 1. Do we have Turkic → Finnish CE values?

**Status:** no, not yet. We have **Finnish-as-listener** values on Turkic text (because the Finnish model exists), but not **Finnish-as-target** for any of the Turkic listeners — Moldir confirmed she didn't make a Finnish IPA perception passage.

**What it would take to fill the asymmetry:**

1. Moldir writes a Finnish IPA perception snippet (20 passages, B2-level, 4-5 sentences each — the same protocol as the existing Turkic snippets).
2. We already have `corpora_clean/oscar_fi_ipa.txt` from the cleaned-OSCAR pass; the Finnish character vocabulary is in `checkpoints/fi_vocab.json`.
3. Each of the 6 Turkic listener models is run on the Finnish perception passages.
4. Add a `fi` column to the eval matrix; bootstrap CIs as usual.

**No retraining required.** The existing 7 base models suffice — we only need a new evaluation pass with Finnish target text.

**Why this matters for the paper:** Finnish is the non-Turkic Uralic control. The expected result: Turkic-listener CE on Finnish target should be the *highest* of all targets (since Finnish is most distant). That gives the eval matrix a clean "no genetic relatedness" baseline against which to interpret the Turkic-to-Turkic distances. Per [[suomi-toivanen-ylitalo-2008-finnish-sound-structure]], Finnish has 8-vowel inventory + harmony with neutral /i e/ + phonemic length — typologically similar to Turkic in some ways (agglutination, harmony) but with completely different participants.

The NeurIPS submission flagged this as future work ([[baidildinova-wagner-2024-neurips-lstm-mi]] p. 5); we should ship it for the Tu+ proceedings or the SCiL camera-ready.

---

## 2. CE estimated over whole text vs per-passage averaged?

**Answer (Austin's chat):** averaged the surprise over every character, rather than the 20 passages separately.

**More precisely** — for one (listener, target) pair:

1. Concatenate the 20 passages of target language into a stream.
2. For each character position i, compute `−log p(x_i | x_<i)` under the listener model.
3. Mean over all character positions = the CE value reported.

This is **char-pool-weighted**: long passages contribute more characters than short ones, so they dominate the average. The alternative (mean of per-passage means) weights all 20 passages equally. The two diverge when passages differ in length.

**Which is right?** Char-pool-weighted is the standard for LM evaluation; it matches what `torch.nn.functional.cross_entropy` returns with the default reduction. Per-passage-mean is what you want if you treat each passage as an independent sample and want a CI over passages.

**Recommendation:** keep char-pool-weighted for the point estimate (matches conventional LM-eval); use per-passage as the **resampling unit** in the bootstrap CI (which is what Austin already does — see Question 3).

---

## 3. Excess confidence intervals — what do they measure?

**Austin's chat (paraphrased):** paired bootstrap, 2000 draws, percentile method, captures **passage variability** not model variability.

**The exact procedure:**

For one language pair (e.g. az → tr):

1. Each of the 20 passages has the foreign model's total surprise and the native model's total surprise.
2. Draw 20 passage-slots with replacement (some passages picked multiple times, others not at all).
3. Foreign-pooled-surprise = sum of foreign surprises over the draw / sum of characters in the draw.
4. Native-pooled-surprise = sum of native surprises over the draw / sum of characters in the draw.
5. Excess = foreign − native = one excess-distance value for this draw.
6. Repeat 2000 times.
7. Sort the 2000 excess values; 2.5% percentile = low; 97.5% percentile = high.

**What the CI captures:** "if we had drawn a different set of 20 passages from the same passage population, how stable is the excess CE distance?" This is **passage variability**.

**What the CI does NOT capture:** "if we retrained the LSTM with a different random seed, how stable is excess CE?" That would be **model-initialization variability**. Per Shiva's chat note, both are valid; ours is passage variability.

**To clarify during presentation:** "These CIs reflect variability across the perception passages, not across model initializations. Re-running with different model seeds is named in our future work."

The methodology is grounded in the dialectometric MI tradition; see [[tang-heuven-2007-chinese-dialects-intelligibility]] for the methodological precedent (60 dialect pairs × 24 listeners × paired CI estimation) and [[gooskens-heuven-2021-mutual-intelligibility]] for the modern methodology review.

---

## 4. Why were excess CE values similar despite different target CE values?

**Moldir's confusion** (resolved in chat): in the az-listener row, native_CE was 1.13 for az on az; she expected `excess_CE = target_CE − 1.13` for every cell in the row. But the table showed different excess values for different targets even though native_CE was constant.

**The answer she eventually saw:** the **denominators differ across targets**. For each target language, the "number of positions scored" is the count of characters at positions where every listener model has the target's next character in its vocabulary. That count depends on which target language is being scored, so:

- For az → tr: scored over the positions where every model knows the Turkish next-char.
- For az → kk: scored over the positions where every model knows the Kazakh next-char.
- These are **different position sets**, so the native_CE recomputed for the "scored positions only" differs from native_CE on the full text.

So: `excess_CE(az, kk) ≠ raw_CE(az, kk) − 1.13`. Instead, `excess_CE = pooled_CE_over_scored_positions(foreign) − pooled_CE_over_scored_positions(native)`. The native_CE you see in the row "az on az" is the full-text value; the native_CE actually subtracted varies per target.

**Practical implication for the plot:** if you add a listener-on-native dot to the plot (Moldir's Q7), the value will be CE — not necessarily 0 for excess_CE, because the "scored positions" filter changes which positions count. If you want az-az to literally be 0 on the excess plot, you need the per-target filtered native CE, which is a separate column.

---

## 5. Use listener CE as the baseline (target_CE − listener_CE) instead?

**Austin's chat answer:** Yes, that works and gives the same symmetric distances as ours; only the directional cells change. When you average both directions of a pair it gives the exact same distance as what we used.

**More precisely:**

| Quantity | Formula | Interpretation |
|---|---|---|
| Native-as-baseline (current) | foreign_CE − native_CE | "how much extra did the foreigner struggle on the text the native made easy" |
| Listener-as-baseline (proposed) | target_CE − listener-on-listener_CE | "how much worse does the foreigner do on foreign text vs its own native text" |

**Symmetry:** for the symmetric distance (averaging Az→Tr with Tr→Az), both formulations give the same result, because:

- (foreign_CE − native_CE) + (reverse foreign_CE − reverse native_CE) summed and divided by 2
- (target_CE − listener_CE) + (reverse target_CE − reverse listener_CE) summed and divided by 2

The two-direction averages collapse to the same number. The **per-cell** numbers differ: the diagonal of native-as-baseline is 0, while the diagonal of listener-as-baseline is also 0 (because target = listener's own data). Off-diagonal cells differ in interpretation but average to the same matrix.

**Recommendation:** stay with the native-as-baseline framing. Two reasons: (a) it's the formulation in the NeurIPS submission already, (b) it's closer to the dialectometric tradition where "excess" is text-difficulty-controlled. The listener-as-baseline framing is an equivalent reframing for the symmetric-distance use case; for asymmetric MI analysis (a future-work item), the native-as-baseline is the appropriate framing because it makes the "target language difficulty" the natural pivot.

---

## 6. Mirea 2019 — feature-naive vs feature-aware LSTM. Is this relevant?

**Yes, directly.** The paper is Mirea & Bicknell, ACL 2019, "Using LSTMs to Assess the Obligatoriness of Phonological Distinctive Features for Phonotactic Learning" (paper at https://aclanthology.org/P19-1155/, PDF at https://research.duolingo.com/papers/mirea.acl19.pdf).

**Mirea's headline finding:** feature-naive LSTMs (character/segment-level, no explicit phonological features) **outperform** feature-aware LSTMs (with distinctive-feature vector inputs) on held-out English phonotactic test sets. The interpretation: distinctive features are not obligatory for learning phonotactic patterns at the segment level — the LSTM figures out the relevant feature structure from the data.

**Implications for our project:**

- Our **current architecture is feature-naive** (character-level over IPA characters — see [[character-level-language-modeling]]).
- Mirea & Bicknell directly support that this is the right call for phonotactic-style tasks. We can cite them in the Tu+ paper as justification.
- A **feature-aware extension** (Hayes-feature vectors as inputs, or a multi-task head that learns features simultaneously) is sometimes proposed as an upgrade — Mirea says: not obviously a win, possibly a loss, depending on data scale.

**Concrete recommendation for the Tu+ paper:** add a 1-2 sentence rationale in the IPA-transcription section noting that we use feature-naive char-level LSTM consistent with Mirea & Bicknell 2019's finding that explicit feature representations don't help phonotactic learning at this granularity. Avoids reviewers asking "why no Hayes features."

---

## 7. Add listener-on-native CE to the plot (az-on-az = 0)?

**Moldir's specific question:** "If I add listener language CE values on native target language to the plot, is it okay or does it look confusing? Here Azerbaijani on Azerbaijani excess CE is 0 and this value is reflected on the plot."

**Recommendation: YES, add the diagonal as 0-line markers** — but design carefully:

- **Pro:** makes the "excess CE" framing visually explicit. Readers see immediately that the diagonal is the zero-reference and off-diagonal values measure distance from it.
- **Pro:** avoids the visual ambiguity where readers wonder "where is az-on-az in this plot?"
- **Con:** adds visual clutter if the plot is already busy.

**Suggested visual treatment:**

- Plot the diagonal points (kk-on-kk, ky-on-ky, ...) as a thin horizontal line at y=0, NOT as the same marker style as the off-diagonal data.
- Add a "Native baseline (excess CE = 0)" label to the y=0 line.
- Optionally annotate "lower = more intelligible" on the y axis to anchor the direction.

This way readers see the zero reference without confusing it with the off-diagonal distances. Per Shiva's earlier feedback ("lead with the intuitive reading of terms"), this aligns: zero-line + lower-is-more-intelligible is the easiest mental model.

---

## 8. Tu+ proceedings — IPA transcription section

**Austin agreed to write it.** Suggested outline (15-page total limit; this section probably 2-4 pages):

1. **Motivation:** Turkic uses 3 scripts (Cyrillic, Latin, Arabic-script); we need shared phonological space for char-level cross-language comparison. → [[turkic-deterministic-transliteration-pipeline]]
2. **Per-language phonology sources:** cite the 7 primary papers — McCollum & Chen 2021 (Kk), McCollum 2020 (Ky), Ido 2025 (Uz), McCollum 2021 (Ug), Mokari & Werner 2017 (Az), Zimmer & Orgun 1992 (Tr), Suomi/Toivanen/Ylitalo 2008 (Fi). Each gives the vowel + consonant inventory we map into. → wiki pages exist for all except Zimmer & Orgun (no OA PDF surfaced).
3. **The rule format:** ICU UTS #35 LDML Transforms — context-sensitive grapheme→phoneme rules. → [[icu-uts35-ldml-transforms]]
4. **The pipeline:** OSCAR → FastText lid.176 ≥0.95 filter → Russian-token filter → deterministic ICU transliteration → SentencePiece tokenizer (optional). → [[turkic-deterministic-transliteration-pipeline]]
5. **Known limitations:** rule files are grapheme-driven; vowel harmony, allophonic alternation, positional reduction are unmodeled. → [[turkic-deterministic-transliteration-pipeline]] explicitly documents this.
6. **Pitch as a research resource:** propose `turkic-transliteration` as a centralized platform for Turkic IPA — "There is not currently a centralized platform like this for Turkic languages, so this tool could make the transcription process much more accessible and systematic" (Moldir's framing).

Everything cited above is now in `~/PROJECTS/wiki/hubs/computational-linguistics.md`. Use the wiki paper-pages as raw material; cite the primary sources directly in the Tu+ paper (NOT the wiki pages — wiki is internal infrastructure, not a citable source).

---

## 9. SCiL abstract / poster — outstanding items

From the chat:

- **Dataset size reporting** (Shiva): yes, report that corpora were controlled (equalized to ~10.2M chars per language).
- **Language legend on Figure 2** (Shiva): done (per chat).
- **Bottom-of-plot legend** (Moldir): approved (per Shiva).
- **Mirror plot of model performance** (Moldir's PDF): legend + axis labels worth double-checking before printing.
- **Line plot vs bar plot for zero-shot** (Moldir): Shiva approved line plot as more readable; confirmed.

---

## Sources for the answer

All claims above are grounded in:

- The wiki at `~/PROJECTS/wiki/pages/`, specifically the 39-page computational-linguistics hub.
- The chat transcript at `~/.claude/projects/C--Users-Test-PROJECTS-LSTM/3296285d-….jsonl` (the operational LSTM project's Jun 10-26 session log).
- The NeurIPS 2024 submission PDF at OneDrive `08_Project_Documents/Moldir Baidildinova & Austin Wagner NeurIPS_2024.pdf` → wiki page `baidildinova-wagner-2024-neurips-lstm-mi`.
- The lstm repo at `~/PROJECTS/lstm/` for code-level facts about the eval pipeline.

For any of these, the canonical move is to read the source page on the wiki and check its `local_pdf:` + `sources:` frontmatter — those resolve to PDFs you can verify directly.
