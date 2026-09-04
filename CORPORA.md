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
| `rebuild_2026-09/corpora_clean_v4/` | 11,658,775 | Uzbek | seven languages |
| `rebuild_2026-09/corpora_clean_v5/` | 11,658,775 | Uzbek | eight, mixed provenance |
| `rebuild_2026-09/corpora_clean_v6/` | **11,658,775** | **Uzbek** | **CURRENT, eight, fully provenanced** |

## v6 rebuilds all eight from recorded downloads, and five come out identical

Built 2026-09-03. Every language re-downloaded from OSCAR-2301 with the
parameters its own manifest records, then transliterated and cleaned with
the current tool. Every v6 raw corpus carries a manifest that pins its own
`output_sha256` and names the normaliser that shaped it, which is what no
earlier generation did.

**The budget did not move.** 11,658,775, set by Uzbek, for the third
generation running.

**Five of the eight cleaned corpora are byte-identical to v5:**

| lang | raw lines equal to the archive | cleaned corpus |
|---|---|---|
| az | 6,987 / 10,000 (69.9%) | identical |
| ky | 7,317 / 10,000 (73.2%) | identical |
| uz | 7,270 / 7,708 (72.7%) | identical |
| ug | 4,821 / 10,000 (48.2%) | identical |
| ru | built the same day | identical |
| **fi** | **0 / 10,000** | **CHANGED** |
| **tr** | **1 / 10,000** | **CHANGED** |
| **kk** | **10 / 10,000** | **CHANGED** |

Two things in that table correct what this file said earlier on 2026-09-03.

**Kazakh was not the only unreproducible archive.** Finnish and Turkish are
too, and by the same margin: near-zero line overlap with a fresh download
of the same source under the same recorded parameters. The earlier entry
concluded kk was the exception, from a line-count check. A line count
cannot see this, because all three archives have exactly the 10,000 lines
their manifests claim.

**Normalisation does not reach the cleaned corpora at all.** The four
languages whose raw text differs by 27 to 52 percent of lines produce
cleaned corpora that are byte-identical. That difference is the format
characters `3bfd89e` began stripping on 2026-08-14, and the cleaner's
filtering absorbs it completely. So the earlier expectation here, that
every language would shift slightly under the new normaliser, was wrong:
it shifts no cleaned corpus at all.

What follows is that the v5 to v6 delta has ONE cause rather than two.
Five languages reproduce bit for bit through a fresh download, which is
strong evidence the pipeline is sound end to end; three archives were never
reproducible samples, which is a property of those three files.

**Only fi, tr and kk were retrained on v6.** The other five corpora are
byte-identical, and training on this stack reproduces bitwise (see the
personal wiki, `seeded-lstm-training-reproduces-bitwise-on-one-card`), so
retraining them would spend three hours reproducing checkpoints that
already exist.

## v5 is v4 plus Russian, and the seven are byte-identical

Russian was added on 2026-09-03 as a second non-Turkic control. Finnish is
the agglutinative control; Russian is the CONTACT language the Cyrillic
corpora borrow from, which is what makes it worth having. If cross-entropy
transfer tracks genealogy, Russian sits far from all seven. If it tracks
shared loan vocabulary, it sits closer to the Cyrillic-script Turkic
languages than to the Latin-script ones. The current seven-language design
cannot separate those.

**The seven did not move, and that is checked rather than asserted.** All
seven cleaned files in v5 are byte-identical to their v4 counterparts:

| lang | digest (first 12) |
|---|---|
| az | `e3dbdaa758d2` |
| fi | `dff6e3e4aed8` |
| kk | `d1fe98d2fc7e` |
| ky | `ea4b907bb95f` |
| tr | `19049664c3dd` |
| ug | `63cd9a8dda95` |
| uz | `35d3d502ffde` |
| **ru** | **`3455997e9295`** (new) |

That is a property of the design, not luck, and it was the reason to clean
all eight together rather than Russian alone. `truncate_to_budget` keeps
whole lines from the start, and the budget is the smallest `chars_kept` in
the set. Russian kept 11,642,181 characters after cleaning, above Uzbek's
11,658,775 before truncation, so Uzbek still binds and every other corpus
is cut at the same place it was before. Had Russian come in BELOW that, it
would have rebound the budget and re-truncated all eight, invalidating the
seven checkpoints trained on v4. Running the eight together is what turns
that risk into a digest comparison.

The seven v4 checkpoints therefore remain valid against v5's files. Only
`ru_best.pt` had to be trained.

**How the Russian raw corpus was built**, matching the seven exactly:
OSCAR-2301 via `turkic-download-corpus`, `lid218e` script-aware at threshold
0.95, 10,000 lines (10,000 of 11,645 seen), then `turkic-translit translit
--lang ru`, then the same cleaner at `min-line-chars 30` and
`min-ipa-ratio 0.95`. OSCAR-2301 is gated: the downloader reads the
credential from the `HF_TOKEN` environment variable and not from
`~/.cache/huggingface/token`, so an unset variable fails as HTTP 401 rather
than as a missing-credential message.

## kk's download manifest describes a corpus that is not there

Found 2026-09-03, recorded because it is a provenance defect rather than a
data one. `rebuild_2026-08/orthographic/kk.txt.manifest.json` states
`lines_written: 150000`. The file beside it has 10,000 lines.

| lang | manifest | actual |
|---|---|---|
| az | 10,000 | 10,000 |
| fi | 10,000 | 10,000 |
| **kk** | **150,000** | **10,000** |
| ky | 10,000 | 10,000 |
| tr | 10,000 | 10,000 |
| ug | 10,000 | 10,000 |
| uz | 7,708 | 7,708 |

Six of seven agree, and Uzbek's 7,708 is the source running out, which is
exactly why it binds the budget. So the real cap was 10,000.

**This first read as an abandoned run's leftover manifest. It is not, and
the difference matters.** Re-running the download with every parameter the
manifest records, on classifier weights whose byte count still matches it
exactly, produced text that differs from `kk.txt` at line 0. Only 10 of
10,000 lines coincide. The archived file is not the first 10,000 lines of
that stream, so it was not produced by truncating a longer run of it either.

**The pipeline is not the problem, and that was measured rather than
assumed.** Two fresh Kazakh downloads run back to back are byte-identical
to each other, 97,598,509 bytes, SHA-256 `9effd770764aad4d7157...`. The
downloader is deterministic: same parameters, same bytes. `hub.shard_paths`
orders shards by part number rather than as text, so the stream order is
fixed.

What follows is narrower and worse than "the count is stale": `kk.txt` was
produced by some step nobody recorded, and no parameters exist that
reproduce it. Its selection is unknown, not merely undocumented.

Consequences, stated exactly:

- The cleaned corpora are unaffected. Cleaning reads the `.txt`, every
  `.txt` is internally consistent, and every cleaned file is digest-pinned.
- Every model stays reproducible **from the cleaned corpus**, which is what
  training actually consumes.
- What is lost is the ability to regenerate Kazakh's raw text from scratch.
- **Do not "fix" this by re-downloading kk.** A fresh download is different
  text, which would change the cleaned Kazakh corpus, which would invalidate
  every Kazakh checkpoint and the paper's Kazakh numbers. The gap is worth
  less than that.
- No manifest was rewritten to look correct. There is no set of fields that
  would be true, and inventing one is the failure the manifest module's own
  docstring exists to prevent.

Future runs cannot repeat this. `CorpusRunManifest` now carries a required
`output_sha256` taken from the bytes on disk after the writer closes, so a
manifest that stops describing its file fails to verify instead of going
unnoticed. Russian's manifest matches its file, and carries that digest.

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

## Use `rebuild_2026-09/corpora_clean_v6/`

It is the only generation in which every language carries a manifest pinning
its own `output_sha256` and naming the normaliser that shaped it. Train and
evaluate from it; cite numbers from the arm that produced them, not from the
newest arm.

- The equalised budget it reports, 11,658,775 characters with Uzbek binding,
  is the same one `overleaf-tu-paper/LM_MI_LSA_template.tex` states, and every
  language's `chars_written` matches its file exactly. The budget has not
  moved since v3, so the paper's corpus sentence is still true of v6.
- Checkpoints: `checkpoints_v6/` holds the three bases retrained on it (fi,
  kk, tr — the three whose cleaned corpora changed), and
  `checkpoints_v6_matrix/` holds the seven perception languages assembled for
  the 7x7 evaluation. Russian is a base, not a matrix member: no human
  intelligibility ratings were collected for it, so it has no row or column to
  score against.
- Evaluation output: `results/v6_full_skip.csv` and
  `results/v6_full_skip_asymmetry.csv`.

**The three superseded generations are not interchangeable with it, and the
difference is not cosmetic.** v3 and everything before it transcribe ‹щ› as
ɕː in Kazakh, Kyrgyz and Uzbek-Cyrillic, which is a segment no evaluation
snippet contains; correcting it to ʃː in v4 moved Kyrgyz-as-reader by 0.22
and removed roughly two thirds of one of the draft's reported asymmetries.
Numbers computed on v3 and numbers computed on v4 or later are not
subtractable. The personal wiki records the measurement at
`transcription-error-inflated-a-directional-asymmetry`.

**On HPC3 only v3 is staged**, at
`/pub/wagnera3/LSTM/rebuild_2026-08/corpora_clean_v3`, digest-verified
against `API/tools/hpc3/runs/turkic-v3-corpus-digests.txt`, and it is what
`runs/sweep-turkic-bases.json` trains from. That sweep therefore trains the
uncorrected corpus and its seven members predate Russian. Staging v6 and
repointing the sweep is open work, not something this file describes as
done.

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
