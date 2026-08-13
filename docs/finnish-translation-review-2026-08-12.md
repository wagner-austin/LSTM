# Author-side review of the Finnish perception passages

**Artifact reviewed:** `data/perception/perception_fi_source.txt` (20 passages, 4 texts).
**Date:** 2026-08-12.
**Method:** passage-by-passage comparison against the six Turkic translations
of the same source texts (`data/perception_sources/`), each of which was
checked by a native speaker of its language (paper §4.1.4). Six
independently native-verified versions of a passage jointly fix its
meaning, so agreement with them is the comparison standard used here. Turkish served
as the primary comparison text, with Uzbek and the others consulted where
the two diverged.

## Checks and results

1. **Meaning equivalence, all 20 passages.** Each Finnish passage was
   read against its Turkish counterpart clause by clause. No meaning divergences,
   omissions or additions were found in the 20 passages.
2. **Numeral audit (scripted, narrow).** Digit tokens in every passage
   compared across all seven versions: 0 disagreements. Most quantities
   in these passages are written as words, not digits, so this check's
   coverage is small; the word-written quantities (five hours of
   training, two-to-three rides a week, doubled and tripled braking
   distances) were compared in the reading of check 1, not by script.
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
- This is an author-side review against the parallel versions, not a
  native-speaker read, and it should not be represented as one. It found
  no issues affecting the character-level measures. A native read remains
  recommended, and is a precondition for any *human* Finnish listening
  experiment.

## Incidental finding

The Uzbek source docx (`UZBEK_test_items.docx`) lacks section breaks
between its texts: text 2 begins mid-passage after text 1's passage 5,
so a structural parse yields 18 sections instead of 20. All content is
present. Regeneration from that file must restore the breaks by hand and
verify against `data/perception/perception_uz.txt`'s section structure.


## Native-speaker review (2026-08-12, same day)

A native Finnish speaker, recruited through a public request, read all
four texts. The findings match the scope this review predicted for
itself: no meaning divergences, omissions or additions; one grammatical
agreement error (predicative *vaarallinen* for *vaarallista* with an
abstract subject); and unnatural phrasing at six points, including
*vakava* for goal-directed training where Finnish uses *tavoitteellinen*.

All seven edits were applied to ``perception_fi_source.txt`` and the IPA
file was regenerated and re-harmonised the same day, passage structure
verified by the perception-file tests. The Finnish passages now carry
the same level of review as the six Turkic translations.
