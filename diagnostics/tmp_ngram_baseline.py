"""Diagnostic: char n-gram baseline for the zero-shot CE matrix.

Trains interpolated char trigram models on each language's TRAIN split
(first 70% of corpus, same as the LSTM saw), scores the perception sections
on the same common-support mask, and builds the same excess-CE UPGMA tree.

If the n-gram tree already recovers Oghuz/Kipchak/Karluk, the genealogical
signal is in surface statistics; if not, the LSTM's longer context earned it.
"""

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scripts.zero_shot_eval import LANGS, load_model_with_vocab

CKPT = Path("checkpoints")
SNIP = Path("data/perception")
CORP = Path("09_Downloaded_Corpora_2")
TARGETS = [lang for lang in LANGS if lang != "fi"]
TRAIN_RATIO = 0.70
TRAIN_CHARS = 4_000_000  # counting cap per language, plenty for trigrams
LAMBDAS = (0.05, 0.25, 0.70)  # weights for orders 1, 2, 3
MARKER_RE = re.compile(r"^\s*[1-5]\s*$")
HEADER_RE = re.compile(r"^\s*te?xt\s*\d", re.IGNORECASE)


def parse_sections(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[list[str]] = []
    current: list[str] | None = None
    for i, line in enumerate(lines):
        if MARKER_RE.match(line):
            if current:
                sections.append(current)
            current = []
            continue
        if i == 0 or HEADER_RE.match(line) or not line.strip():
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if len(line.strip()) < 40 and MARKER_RE.match(nxt) and nxt.strip() == "1":
            continue
        if current is not None:
            current.append(line.strip())
    if current:
        sections.append(current)
    return [" ".join(s) for s in sections if s and len(" ".join(s)) >= 20]


class TrigramLM:
    def __init__(self, text: str) -> None:
        self.uni = Counter(text)
        self.bi = Counter(zip(text, text[1:]))
        self.tri = Counter(zip(text, text[1:], text[2:]))
        self.bi_ctx = Counter(text[:-1])
        self.tri_ctx = Counter(zip(text, text[1:]))
        self.n = len(text)
        self.v = len(self.uni)

    def logp(self, c1: str, c2: str, c3: str) -> float:
        """Interpolated log P(c3 | c1 c2) with a uniform floor."""
        p1 = self.uni.get(c3, 0) / self.n
        ctx2 = self.bi_ctx.get(c2, 0)
        p2 = self.bi.get((c2, c3), 0) / ctx2 if ctx2 else 0.0
        ctx3 = self.tri_ctx.get((c1, c2), 0)
        p3 = self.tri.get((c1, c2, c3), 0) / ctx3 if ctx3 else 0.0
        p = LAMBDAS[0] * p1 + LAMBDAS[1] * p2 + LAMBDAS[2] * p3
        p = 0.999 * p + 0.001 / max(self.v, 100)
        return math.log(p)


# common-support mask uses the SAME vocabs as the LSTM eval
vocabs = [
    set(
        load_model_with_vocab(CKPT / f"{s}_best.pt", CKPT / f"{s}_vocab.json")["stoi"]
    )
    for s in LANGS
]

print("Training trigram models on train splits...")
lms: dict[str, TrigramLM] = {}
for s in LANGS:
    text = (CORP / f"oscar_{s}_ipa.txt").read_text(encoding="utf-8")
    train = text[: int(len(text) * TRAIN_RATIO)][:TRAIN_CHARS]
    lms[s] = TrigramLM(train)
    print(f"  {s}: {len(train)} chars, {lms[s].v} unique, {len(lms[s].tri)} trigrams")

sections = {t: parse_sections(SNIP / f"perception_{t}.txt") for t in TARGETS}

print("\n=== Trigram common-support CE matrix (section-only) ===")
ce: dict[str, dict[str, float]] = {s: {} for s in LANGS}
print("src\\tgt " + "  ".join(f"{t:>6}" for t in TARGETS))
for s in LANGS:
    for t in TARGETS:
        total, count = 0.0, 0
        for sec in sections[t]:
            for k in range(2, len(sec)):
                if all(sec[k] in v for v in vocabs):
                    total -= lms[s].logp(sec[k - 2], sec[k - 1], sec[k])
                    count += 1
        ce[s][t] = total / count
    print(f"  {s}    " + "  ".join(f"{ce[s][t]:6.3f}" for t in TARGETS))

print("\n=== Trigram excess CE  d(s,t) = CE(s->t) - CE(t->t) ===")
excess: dict[str, dict[str, float]] = {s: {} for s in LANGS}
print("src\\tgt " + "  ".join(f"{t:>6}" for t in TARGETS))
for s in LANGS:
    for t in TARGETS:
        excess[s][t] = ce[s][t] - ce[t][t]
    print(f"  {s}    " + "  ".join(f"{excess[s][t]:6.3f}" for t in TARGETS))

print("\n=== UPGMA tree, symmetrized trigram excess CE ===")
langs6 = list(TARGETS)
members: dict[int, list[int]] = {i: [i] for i in range(6)}
names = {i: langs6[i] for i in range(6)}
dist = np.zeros((6, 6))
for i, a in enumerate(langs6):
    for j, b in enumerate(langs6):
        if i != j:
            dist[i, j] = 0.5 * (excess[a][b] + excess[b][a])
active = list(range(6))
next_id = 6
while len(active) > 1:
    best = None
    for ai in range(len(active)):
        for bi in range(ai + 1, len(active)):
            a, b = active[ai], active[bi]
            d = np.mean([dist[x, y] for x in members[a] for y in members[b]])
            if best is None or d < best[0]:
                best = (d, a, b)
    d, a, b = best
    names[next_id] = f"({names[a]},{names[b]}):{d:.2f}"
    members[next_id] = members[a] + members[b]
    active = [x for x in active if x not in (a, b)] + [next_id]
    next_id += 1
print("  " + names[active[0]])
