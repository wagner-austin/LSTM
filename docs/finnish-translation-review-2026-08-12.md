# Review of the Finnish perception passages against the parallel versions

**Artifact reviewed:** `data/perception/perception_fi_source.txt` (20 passages, 4 texts).
**Date:** 2026-08-12.
**Method:** passage-by-passage comparison against the six Turkic translations
of the same source texts (`data/perception_sources/`), each of which was
checked by a native speaker of its language (paper §4.1.4). Six
independently native-verified versions of a passage jointly fix its
meaning, so agreement with them is the standard applied. Turkish served
as the primary comparison text, with Uzbek and the others consulted where
the two diverged.

## Checks and results

1. **Meaning equivalence, all 20 passages.** Each Finnish passage was
   read against its Turkish counterpart clause by clause. All 20 carry
   the same propositional content; no omissions, no additions.
2. **Numeral audit (objective, scripted).** Every digit token in every
   passage compared across all seven versions: **0 disagreements**
   (`scripts run 2026-08-12; five hours of training, two-to-three rides a
   week, doubled and tripled braking distances, etc. all agree`).
3. **Grammaticality and register.** The Finnish is grammatical and
   idiomatic throughout, at the intermediate register of the source
   materials (e.g., *hengästyminen*, *jarrutusmatka*, *huippu-urheilija*
   used correctly; consistent formal *sinä*-address matching the
   instructional genre).
4. **Structure.** 4 texts × 5 passages, headers and markers intact —
   independently enforced by `tests/test_perception_files.py`.

## Notes

- One deliberate localisation: the advice columnist is *Christina* in the
  Turkish text and *Kristiina* in the Finnish, a conventional name
  adaptation with no effect on the character-level measures.
- One nuance within normal translation variance: passage 17's Turkish has
  "more efficient muscles" where the Finnish has "strengthens your
  muscles"; both clauses also state "makes the heart stronger", so the
  content is preserved.
- A native-speaker read remains worthwhile before any *human* Finnish
  listening experiment is run; for the model-side use the passages are
  fit for purpose, and this review supersedes the open "spot-check
  recommended" flag for that use.

## Incidental finding

The Uzbek source docx (`UZBEK_test_items.docx`) lacks section breaks
between its texts: text 2 begins mid-passage after text 1's passage 5,
so a structural parse yields 18 sections instead of 20. All content is
present. Regeneration from that file must restore the breaks by hand and
verify against `data/perception/perception_uz.txt`'s section structure.
