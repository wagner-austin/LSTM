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

## What each one records about how it was built

**v3's transliteration inputs are fully accounted for.** Its
`cleaning_manifest.json` records eight digests and **all eight match
`turkic-transliteration` as it stands today**: the seven `*_ipa.rules` files
by file digest, and `symbol_map` by table digest.

`symbol_map` is worth a note, because it is easy to check wrongly. It is
**not** the SHA-256 of `symbol_map.csv`. `corpus/clean.py` hashes the parsed
rows re-encoded as JSON:

```python
encoded = json.dumps([encode_symbol_rule(rule) for rule in rules], ensure_ascii=False)
fingerprint["symbol_map"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

Its seven siblings *are* file digests, so comparing the CSV's hash and
concluding the map has drifted is a mistake someone will make — it was made
on 2026-08-28. Reproduce it with `read_symbol_map()` and the snippet above;
18 rows currently hash to `9a3b98c8…`, which is what v3 records.

**`corpora_clean/` records no rule digests at all**, so there is nothing to
check and no way to ask what produced it. That is the real provenance gap
among the three.

**`corpora_clean_2026-02/` has an open item that is the author's, not a
tooling bug.** `tu-proceedings-datasets-section.tex` states that the script
which produced it "is not in either repository", and that it used the
`lid218e` classifier which was never wired into the released package. That
concerns the raw-corpus language filtering, upstream of cleaning. Whether it
also applies to `corpora_raw_v3` is **not established here** — do not assume
either way.

One thing checked and cleared: the 2026-08-12 fix merging the `U+02A6`
ligature into `t͡s` for Kyrgyz landed before every corpus here. Zero
`U+02A6` in any Kyrgyz file; 19,421 merged forms in v3.

## Where corpora come from

`~/PROJECTS/turkic-transliteration` is the **engine**, not a corpus store —
`src/turkic_translit/rules/*.rules` plus the cleaner. Its `data/` is empty.
The corpora are that engine's output over OSCAR text and live **here**.
