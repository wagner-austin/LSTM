"""Diagnostic: vowel-harmony MI + missing-phoneme tables.

(a) Mutual information between consecutive vowels within words, per language
    (computed on 1M chars of each TRAIN split). Vowel harmony languages
    should show high MI; this is exactly the structure a charLM can exploit
    across word-length distances.
(b) For each source vocab, which chars of each target snippet are missing,
    weighted by how often they occur (what `coverage` actually measured).
"""

import math
import re
from collections import Counter
from pathlib import Path

from scripts.zero_shot_eval import LANGS, load_model_with_vocab

CKPT = Path("checkpoints")
SNIP = Path("data/perception")
CORP = Path("09_Downloaded_Corpora_2")
TARGETS = [lang for lang in LANGS if lang != "fi"]
SAMPLE = 1_000_000

VOWELS = set("aɑæeiɯoøuyəɵʊɪœɛɔʏãỹ")
WORD_RE = re.compile(r"[^\s\d\.,;:!?\"'«»()\[\]–—-]+")


def vowel_mi(text: str) -> tuple[float, float]:
    """MI (bits) between consecutive vowels within words + vowel entropy."""
    pairs: Counter[tuple[str, str]] = Counter()
    singles: Counter[str] = Counter()
    for word in WORD_RE.findall(text):
        vs = [c for c in word if c in VOWELS]
        singles.update(vs)
        pairs.update(zip(vs, vs[1:]))
    n_p = sum(pairs.values())
    n_s = sum(singles.values())
    if n_p == 0:
        return 0.0, 0.0
    mi = 0.0
    for (v1, v2), c in pairs.items():
        p12 = c / n_p
        p1 = singles[v1] / n_s
        p2 = singles[v2] / n_s
        mi += p12 * math.log2(p12 / (p1 * p2))
    ent = -sum((c / n_s) * math.log2(c / n_s) for c in singles.values())
    return mi, ent


print("=== Vowel-harmony MI per language (train split, 1M chars) ===")
print(f"{'lang':>4}  {'MI(bits)':>8}  {'H(vowel)':>8}  {'MI/H':>6}")
for lang in LANGS:
    text = (CORP / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")
    train = text[: int(len(text) * 0.70)][:SAMPLE]
    mi, ent = vowel_mi(train)
    print(f"{lang:>4}  {mi:8.3f}  {ent:8.3f}  {mi / ent:6.3f}")

print("\n(same measure on the perception snippets themselves)")
for t in TARGETS:
    mi, ent = vowel_mi((SNIP / f"perception_{t}.txt").read_text(encoding="utf-8"))
    print(f"{t:>4}  {mi:8.3f}  {ent:8.3f}  {mi / ent:6.3f}")

print("\n=== Missing chars per (source vocab, target snippet) ===")
print("(chars >0.3% of target snippet that the source vocab lacks)")
vocabs = {
    s: set(
        load_model_with_vocab(CKPT / f"{s}_best.pt", CKPT / f"{s}_vocab.json")["stoi"]
    )
    for s in LANGS
}
for t in TARGETS:
    text = (SNIP / f"perception_{t}.txt").read_text(encoding="utf-8")
    freq = Counter(text)
    n = sum(freq.values())
    print(f"\n  target {t}:")
    for s in LANGS:
        if s == t:
            continue
        missing = [
            (ch, cnt / n)
            for ch, cnt in freq.most_common()
            if ch not in vocabs[s] and cnt / n > 0.003
        ]
        if missing:
            desc = ", ".join(f"{ch!r}:{p:.1%}" for ch, p in missing)
            total = sum(p for _, p in missing)
            print(f"    {s} lacks ({total:.1%} of snippet): {desc}")
