# Primary sources behind `results/human_mean_ratings.csv`

These two files are the evidence for numbers this repository reports and
cannot otherwise regenerate. They were fetched from Slack on 2026-09-04 and
copied here because the Slack mount they landed in (`MCPs/files/`) is
gitignored, so the derived CSV was under version control while the thing it
was derived FROM was not. That is the shape of gap this directory exists to
close.

| file | sha256 (first 16) | Slack id | posted |
|---|---|---|---|
| `F0BQEA45FNV-human-ce-FINAL.pdf` | `3c6e05e1ed249fe0` | F0BQEA45FNV | 2026-08-14 19:41 |
| `F0BQPCK53QC-zero_shot_excess_ce_FINAL_merged.csv` | `0cf86cb638c2ecf2` | F0BQPCK53QC | 2026-08-16 21:43 |

Both were uploaded by Moldir Baidildinova to `#mutual-intelligibilty-llm-project`.

## What each one carries

**`zero_shot_excess_ce_FINAL_merged.csv`** is the v3 excess-CE matrix joined
with `cophenetic_distance` per ordered pair. Its excess-CE column matches
`results/v3_full_skip.csv` exactly, which is what identifies it as the
published-era matrix. The cophenetic column appears nowhere else on this
machine, so without this file the genealogical-distance correlation cannot
be computed at all.

**`human-ce-FINAL.pdf`** is Figure 4, mean human rating against model excess
CE. It matters for two separate reasons.

It carries its own statistics, `r = -0.62, p = 0.004`, which do not match the
draft's prose (`r = -.53, p = .016`). The figure was regenerated after the
transcription corrections and the text was not, so the paper currently
disagrees with itself and the prose is the stale half.

And it is a VECTOR pdf, so its twenty markers are real coordinates rather
than pixels. Six mean ratings exist in no other file here -- Turkish and
Azerbaijani listeners on Kyrgyz, Uzbek and Uyghur -- and were recovered by
calibrating those markers against the axis ticks. `human_mean_ratings.csv`
records per row whether a value was quoted from the draft's prose or
reconstructed from this geometry, because the two are not equally strong and
a reader must be able to tell them apart.

## How the reconstruction was checked

Three ways, independently, because reconstructed data earns less trust than
reported data:

1. The correlation over the recovered points is `-0.6162`, against the
   `-0.62` the figure prints of itself.
2. Eleven of the fourteen ratings the draft states in prose are reproduced
   with a **uniform** `-0.0100` offset -- minimum equal to maximum, which is
   a calibration constant rather than scatter.
3. The completed twenty-point set reproduces the paper's published
   `r = -.53, p = .016` on the v3 matrix.

The two values that did not reproduce were a clean mutual swap, `ky_tr`
against `ky_ug`, caused by label displacement: `ky_ug` had the largest
label-to-marker distance in the figure. The draft's prose is authoritative
for those two. The figure is trusted only where nothing else exists.

## What is still missing

Participant-level responses, and the six recovered means as Moldir reported
them rather than as this repository reconstructed them. Both are hers. If
they arrive, replace the six reconstructed rows and delete this paragraph.
