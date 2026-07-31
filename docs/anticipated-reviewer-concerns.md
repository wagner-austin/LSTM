# Anticipated reviewer concerns + responses

Drafted without the actual reviewer text in hand. These are the methodology vulnerabilities a careful Tu+ / SCiL / NeurIPS reviewer is most likely to surface, each paired with a defensible response grounded in the wiki at `~/PROJECTS/wiki/pages/`. When you have the actual reviewer text, cross-check against this list and use the relevant subset.

---

## R1. "Lindsay 2010 is a blog post, not peer-reviewed. Validating CE loss against blog-stated MI percentages is methodologically weak."

**Response:** Acknowledged in the paper's own footnote 3: "no human comprehension data have been collected on MI patterns in Turkic languages, except for the Swadesh-list cognate comparison by Lindsay [2010] and anecdotal impressions" [@baidildinova-wagner-2024-neurips, p. 5].

**Strengthening:** the wiki's [[swadesh-1952-lexicostatistic-dating]] page documents that Lindsay's percentages derive from Swadesh-list cognate counting, an established (though contested) lexicostatistical methodology going back to Swadesh 1952. The wiki's [[grimes-1992-vocab-similarity-intelligibility]] page documents that even careful Swadesh-list-based MI estimates correlate with measured intelligibility at only r = 0.34 (55 Philippine dialect pairs). So we're validating against a noisy proxy of a noisy proxy — but this is acknowledged, and the paper names collection of native-speaker MI judgments as future work.

**Stronger framing for the camera-ready:** rather than "our CE loss matches Lindsay's MI percentages," reframe as "our CE loss reproduces the genealogical-branch ordering that Lindsay's percentages also reflect." The genealogical ordering (Kipchak-Kipchak < Karluk-Karluk < Karluk-Kipchak < Oghuz-Kipchak — per [[turkic-language-family-classification]]) is uncontested; matching this ordering is a lower-bar but more defensible claim.

---

## R2. "MI is known to be asymmetric. Why test only one direction per pair?"

**Response:** Acknowledged limitation (the paper says: "only one training direction per pair; MI is known to be asymmetric") [@baidildinova-wagner-2024-neurips, p. 5]. The wiki's [[gooskens-heuven-2021-mutual-intelligibility]] page documents Gooskens & van Heuven's foundational asymmetry critique in detail. The paper does not claim to measure MI asymmetry; it measures symmetric distance.

**For the camera-ready:** explicitly state "symmetric distance" rather than "MI" in the headline result. The relevant column header is `excess_cross_entropy` averaged over the two directions — the paper's chat (msg #194) shows this is the convention used. The asymmetric per-cell values are reported in the supplementary tables but the headline distance is symmetric.

**Future work:** training both directions (Kk→Ky AND Ky→Kk) for the same pair would surface asymmetry. The pipeline supports this with no architectural changes — just additional training runs.

---

## R3. "B2 difficulty is human-calibrated. Why should it transfer to a character-level model?"

**Response:** Shiva's chat (msg #202): "model performance is known to be affected by things like token frequency, dependency distance etc. And so differences in IPA character frequencies, harmony patterns, word lengths etc. in different languages may make sequences easier or harder to predict for a character-level model, in a way that may not be captured by human comprehension metrics."

**Two defenses:** (a) the **excess CE** measure explicitly subtracts native-CE, which controls for the source-language-specific character-level statistical regularities; this is exactly what Shiva flagged as the point in favor of excess CE over raw CE. The native baseline absorbs whatever character-level easy/hard patterns exist in each language. (b) The cross-language pattern (Kk-Ky < Kk-Tr) is preserved across BOTH raw CE and excess CE per the paper's findings. Any difficulty-driven distortion would have to affect ALL Turkic→Turkic comparisons uniformly to leave the ordering intact, which is implausible.

**For the camera-ready:** include a short paragraph explicitly stating "excess CE controls for source-text character-level statistical regularities; the relative ordering of distances is robust to this source of difficulty" with a citation back to Shiva's framing.

---

## R4. "Why character-level LSTM and not Transformer?"

**Response:** Per [[character-level-language-modeling]] and [[hochreiter-schmidhuber-1997-lstm]]:

- **Compute budget.** The 7 corpora are ~10.2M chars each. LSTM ~947K params trains to convergence on each within reasonable budget; equivalently capable Transformer ~50M+ params requires more data to be data-efficient.
- **Tokenization confound.** Subword tokenizers reintroduce a vocab-size + segmentation choice that varies by language (per [[toraman-yilmaz-2022-turkish-mi]], [[kaya-tantug-2024-turkish-tokenization-granularity]], [[bayram-fincan-2025-turkish-tokenization-standards]]). Character-level on IPA puts all 7 languages in the same character vocabulary — a strict comparability requirement for cross-language CE.
- **Mirea & Bicknell 2019** explicitly find that feature-naive LSTMs outperform feature-aware LSTMs on phonotactic learning. Char-level LSTM on IPA is the feature-naive design they validate.

**Future work:** Transformer-based replication is named as future work [@baidildinova-wagner-2024-neurips, p. 5].

---

## R5. "Equalizing corpus sizes ≠ controlling for byte-premium per Arnett & Bergen 2024."

**Response:** Per [[arnett-bergen-2024-morphologically-complex-lms]], Arnett & Bergen 2024 find that for fair cross-language LM comparison, dataset size should be normalized by **byte-premium** (different scripts encode the same linguistic content with different byte counts). Cyrillic Kazakh uses more bytes per character than Latin Turkish.

**Defense:** the paper equalizes corpora to **character count**, not byte count. Since all 7 corpora are character-level over IPA after transliteration, every character is one phoneme regardless of source script — so character-equalization IS a byte-equivalent equalization at the model-input level. The byte-premium critique applies to comparisons at the *byte* level; the IPA-transliteration step (per [[turkic-deterministic-transliteration-pipeline]]) moves the comparison to the *phoneme* level, where character-equalization is appropriate.

**For the camera-ready:** add a short methodology footnote explaining "corpora equalized to character count at the IPA-normalized level; this is byte-premium-equivalent because transliteration moves the input to a single character vocabulary across all languages."

---

## R6. "6 of 41 Turkic languages is a small sample. The generalization claim is shaky."

**Response:** Acknowledged as a limitation [@baidildinova-wagner-2024-neurips, p. 5]. The 6 chosen languages cover three of the four major branches (Oghuz: Tr, Az; Kipchak: Kk, Ky; Karluk: Uz, Ug; Siberian: not represented). Per [[turkic-language-family-classification]], this is the standard reduced subset for cross-Turkic-NLP work (e.g., [[maxutov-2024-do-llms-speak-kazakh]] tests a similar subset).

**For the camera-ready:** scope the headline claim to "the 6 tested languages" rather than "Turkic languages broadly." Add adding more languages (Tatar, Bashkir, Karakalpak for Kipchak; Turkmen for Oghuz; Yakut/Tuvan for Siberian) to the future-work section.

---

## R7. "OSCAR data has noise + register variation. How do you control for it?"

**Response:** Per the paper's stated method [@baidildinova-wagner-2024-neurips, p. 2], FastText langid threshold 0.95 filters out lines where the language ID confidence is below threshold — this is the explicit noise filter. Per [[banon-bernabeu-2024-fastspell]], a more refined filter (FastSpell) exists for closely-related-language disambiguation and could be substituted; the current paper uses the standard lid.176 filter ([[joulin-2017-fasttext-bag-of-tricks]]).

**Acknowledged limitation:** "OSCAR web data, despite language identification filtering, can contain noise and register variation" [@baidildinova-wagner-2024-neurips, p. 5].

**For the camera-ready:** explicit threshold + filter described in §2 of the paper. Adding a single sentence about register heterogeneity being out of scope (we measure character-level distributional similarity; cross-register would require parallel-register corpora that don't exist for low-resource Turkic) heads off the obvious follow-up.

---

## R8. "Single random seed per model. CIs capture passage variability not model variability."

**Response:** Shiva flagged this directly in chat (msg #194): "we should just clarify during presentation that they're capturing passage variability, not model variability." This is a known limitation.

**For the camera-ready:** explicit single sentence in the CI methodology paragraph: "CIs reflect passage resampling variability, not model-initialization variability. Multi-seed replication is named as future work." This is a 1-sentence fix; no methodological change required.

**Strengthening:** the paper's headline result (Kk-Ky < Kk-Tr ordering) is preserved across **three** different metrics simultaneously (CE spike, AUC, test CE) per [@baidildinova-wagner-2024-neurips, p. 3]. The probability that single-seed noise produces consistent direction across three metrics is low; the ordering result is robust without multi-seed verification, even if the absolute values aren't.

---

## R9. "Finnish was promised in §5 but not delivered. Why?"

**Response:** Time constraints (paper acknowledges this) [@baidildinova-wagner-2024-neurips, p. 5]. The Finnish corpus is already transliterated to IPA (`oscar_fi_ipa.txt` per the lstm repo). The remaining work is: (1) Moldir writes a Finnish perception passage in the same B2 format; (2) the 6 Turkic listener models are run on the Finnish passages; (3) Finnish column added to the eval matrix.

**For the camera-ready:** if Moldir can produce a Finnish perception passage before the camera-ready deadline, the asymmetry (Finnish-as-listener exists; Finnish-as-target missing) can be filled. Per the wiki's [[suomi-toivanen-ylitalo-2008-finnish-sound-structure]] page, Finnish is the appropriate non-Turkic Uralic control with shared agglutination + vowel harmony but completely different participants. The expected result (Turkic listeners highly surprised by Finnish) gives a clean "no genetic relatedness" baseline that anchors the Turkic-to-Turkic distance interpretation.

---

## R10. "Bilingualism confound — Lindsay's Kk-Ky 91% likely reflects bilingual learning, not inherent intelligibility."

**Response:** Per [[grimes-1992-vocab-similarity-intelligibility]], Grimes (1992) flags this exact methodological concern: SIL field-survey MI scores often confuse inherent intelligibility (what a naive monolingual listener can understand) with bilingual proficiency (learned-after-the-fact comprehension). For post-Soviet Central Asia, Kazakh and Kyrgyz speakers have substantial mutual exposure due to geographic + political-cultural integration; Lindsay's 91% almost certainly includes bilingual contribution.

**Defense for our methodology:** the character-level LSTM measures what the model perceives from training data alone, with no contact-driven exposure. This is closer to Grimes's "inherent intelligibility" than to bilingual proficiency. So our CE-loss measure may actually be a CLEANER proxy for inherent intelligibility than Lindsay's blog estimates — even though the headline framing positions Lindsay as the ground truth.

**For the camera-ready:** add 1-2 sentences in the Discussion section explicitly distinguishing inherent intelligibility (what we measure) from bilingual proficiency (what survey-style MI percentages may conflate). Citing Grimes 1992 directly here would strengthen the position.

---

## Cross-cutting note on framing

Many of the above concerns can be defused by **tightening the headline claim from "CE loss is a MI proxy" to "CE loss reproduces the genealogical-branch ordering reflected in Lindsay 2010"**. The first is a strong epistemic claim (MI ≅ CE) that has all the proxy-of-proxy + asymmetry + bilingualism vulnerabilities. The second is a much narrower claim (ordering match) that's robust to most reviewer objections.

The supplementary tables can still report the full per-cell distances; the headline finding becomes the ordering reproduction.

---

## Sources

All claims above ground in the wiki at `~/PROJECTS/wiki/pages/`. Each `[[slug]]` link resolves to a paper-page with `local_pdf:` + verified sha256. For a reviewer asking "what's your source for claim X", the wiki's claim → primary-source chain is intact.
