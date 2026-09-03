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
| `rebuild_2026-08/corpora_clean_v3/` | 11,658,775 | Uzbek | **superseded, and wrong** |
| `rebuild_2026-09/corpora_clean_v4/` | **11,658,775** | **Uzbek** | **CURRENT** |

## v4 corrects a segment that was outside the Kazakh inventory

v3 is not merely superseded. Kazakh, Kyrgyz and Uzbek-Cyrillic wrote ‹щ› as
the alveolo-palatal ɕː. McCollum & Chen's Kazakh chart (p. 277) has no
alveolo-palatal column at all, and the prose under it enumerates what Kazakh
borrows -- [f] and [v] from Russian, [h] from Arabic -- with ɕ absent. The
rules now emit ʃː, the post-alveolar the chart prints.

It is not cosmetic. ɕ occurred in Kazakh and Kyrgyz and nowhere else in the
seven, so it was a character each model saw in two languages and never in
the other five -- the same shape as the Uyghur ʔ artifact that
`symbol_map.csv` records as having "systematically inflated Uyghur
distances". 987 occurrences in Kazakh, 891 in Kyrgyz.

**The budget did not move.** ɕː and ʃː are both two characters, so no file's
length changes, no `chars_written` changes, and Uzbek still binds at
11,658,775. Five of the seven cleaned corpora are byte-identical to v3 and
their digests are literally the same strings. What does change: Kazakh and
Kyrgyz each drop from 39 distinct characters to 38, because ɕ merges into a
ʃ the corpus already had. **The paper's corpora table reports Vocabulary;
those two cells are now wrong.**

v4 was built by substituting on v3's raw transliterated text, not by
re-downloading OSCAR -- a fresh download samples different lines and would
give a different corpus rather than a corrected one. The substitution is
exact rather than approximate: the only difference between the old and new
rule files is the ‹щ› line, and new-rules was checked against
old-rules-then-substitute over 2,944 exhaustive probes per language with
zero mismatches.

## Use `rebuild_2026-09/corpora_clean_v4/`

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
