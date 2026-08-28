# Which corpus is which

**Read this before training, evaluating, or citing a number.**

There are three generations of cleaned corpora in this repository. They differ
in size, in which language binds the equalisation, and in what they record
about how they were built. Nothing in the directory names says which is
current, and on 2026-08-28 that cost real time: an HPC3 sweep was built
against the wrong one, because `slurm/train_base.sub` named it and no other
signal existed.

| directory | equalised budget | binding language | status |
|---|---|---|---|
| `corpora_clean_2026-02/` | 10,215,670 | Uyghur | **superseded** |
| `corpora_clean/` | 12,642,807 | Uzbek | **superseded** |
| `rebuild_2026-08/corpora_clean_v3/` | **11,658,775** | **Uzbek** | **CURRENT** |

## Use `rebuild_2026-08/corpora_clean_v3/`

It is what the paper reports and what recent training used.

- `overleaf-tu-paper/LM_MI_LSA_template.tex` states 11,658,775 characters with
  Uzbek binding, and every language's `chars_written` matches its file exactly.
- `train_v3.log` trained against it.
- On HPC3 it is staged at
  `/pub/wagnera3/LSTM/rebuild_2026-08/corpora_clean_v3`, digest-verified
  against `API/tools/hpc3/runs/turkic-v3-corpus-digests.txt`, and it is what
  `runs/sweep-turkic-bases.json` trains from.

## Why the other two are still here

`corpora_clean_2026-02/` produced the models behind the earlier draft. It is
what `turkic-transliteration/docs/tu-proceedings-datasets-section.tex` still
describes — that draft section is **stale relative to the live paper**, which
moved to v3.

`corpora_clean/` is an intermediate rebuild. Nothing current cites it. It has
the weakest provenance of the three: its `cleaning_manifest.json` records only
the cleaning parameters and per-language counts, and **no `rules_fingerprint`
at all**, so nothing says which transliteration rules produced it.

Neither is deleted, because both trained checkpoints that exist
(`checkpoints_2026-02/`, `checkpoints/`) and deleting the inputs to results
that were reported would be worse than the confusion of keeping them. This
file is the fix for the confusion.

## What none of them can do

**No generation is byte-reproducible from version control today.**

- v3 records eight rule digests. The seven `*.rules` files still match
  `turkic-transliteration` exactly. `symbol_map` matches **neither** version
  in that repository's history, in any line-ending form — so the map that
  built v3 is not in git.
- `corpora_clean/` records no rule digests at all.
- `corpora_clean_2026-02/` was produced by a script that
  `tu-proceedings-datasets-section.tex` states "is not in either repository",
  using the `lid218e` classifier that was never wired into the released
  package.

The paper says "the corpora in Table 2 can be rebuilt from source." That is
not true today for `symbol_map`. It is one file; recovering or re-deriving it
would make the claim true.

One thing that is **not** wrong: the 2026-08-12 fix merging the `U+02A6`
ligature into `t͡s` for Kyrgyz landed before every corpus here. Checked
directly — zero `U+02A6` in any Kyrgyz file; 19,421 merged forms in v3.

## Where corpora come from

`~/PROJECTS/turkic-transliteration` is the **engine**, not a corpus store —
`src/turkic_translit/rules/*.rules` plus the cleaner. Its `data/` is empty.
The corpora are that engine's output over OSCAR text and live **here**.
