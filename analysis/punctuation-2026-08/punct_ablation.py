"""Punctuation/digit ablation for the zero-shot excess-CE matrix.

Reproduces the shipped skip-mode evaluation exactly, then re-scores with
every punctuation, symbol, and digit position removed from the masks.
Nothing in the repository is modified; models, corpora, and perception
texts are read as released.
"""

import csv
import sys
import unicodedata
from pathlib import Path

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
sys.path.insert(0, str(LSTM))

from scripts.zero_shot_eval import (  # noqa: E402
    CORPUS_TEMPLATE,
    MIN_ATTESTED,
    attested_chars,
    ce_from_scores,
    common_support_mask,
    load_sources,
    parse_sections,
    score_section,
)

OUT = Path(
    r"C:\Users\Test\AppData\Local\Temp\claude\C--Users-Test-PROJECTS-turkic-transliteration"
    r"\6f031a76-36d1-47c9-9ff7-40e012dd3579\scratchpad\punct_ablation_results.txt"
)
SHIPPED = LSTM / "results" / "zero_shot_excess_ce_skip_2026-08-13.csv"

def is_nonlinguistic(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith(("P", "S")) or cat == "Nd"

def matrix(scores, sources, targets):
    m = {}
    for src in sources:
        for tgt in targets:
            m[(src, tgt)] = ce_from_scores(scores[(src, tgt)]) - ce_from_scores(scores[(tgt, tgt)])
    return m

lines = []
sources = load_sources(LSTM / "checkpoints")
langs = list(sources)
targets = {
    lang: parse_sections((LSTM / "data" / "perception_clean" / f"perception_{lang}.txt").read_text(encoding="utf-8"))
    for lang in langs
}
vocabs = [
    set(loaded["stoi"]) & attested_chars(LSTM / "corpora_clean" / CORPUS_TEMPLATE.format(lang=lang), MIN_ATTESTED)
    for lang, loaded in sources.items()
]

std_masks = {lang: [common_support_mask(s, vocabs) for s in secs] for lang, secs in targets.items()}
abl_masks = {
    lang: [
        [m and not is_nonlinguistic(sec[i + 1]) for i, m in enumerate(mask)]
        for sec, mask in zip(secs, std_masks[lang], strict=True)
    ]
    for lang, secs in targets.items()
}

for label, masks in (("baseline", std_masks), ("ablated", abl_masks)):
    dropped = sum(1 for lang in langs for m in masks[lang] if sum(m) == 0)
    scored = sum(sum(m) for lang in langs for m in masks[lang])
    lines.append(f"{label}: scored positions {scored}, zero-support sections {dropped}")

scores_std, scores_abl = {}, {}
for src, loaded in sources.items():
    for tgt in langs:
        secs = targets[tgt]
        scores_std[(src, tgt)] = [score_section(loaded, s, m) for s, m in zip(secs, std_masks[tgt], strict=True)]
        scores_abl[(src, tgt)] = [
            score_section(loaded, s, m) for s, m in zip(secs, abl_masks[tgt], strict=True) if sum(m) > 0
        ]

m_std = matrix(scores_std, langs, langs)
m_abl = matrix(scores_abl, langs, langs)

shipped = {}
with SHIPPED.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        shipped[(row["listener_language"], row["text_language"])] = float(row["excess_cross_entropy"])

max_dev = max(abs(m_std[k] - shipped[k]) for k in shipped)
lines.append(f"\nbaseline reproduction: max |delta| vs shipped CSV = {max_dev:.6f} over {len(shipped)} cells")

off = [(s, t) for s in langs for t in langs if s != t]
lines.append("\nsrc tgt  baseline  ablated   delta")
for s, t in off:
    lines.append(f"{s} {t}   {m_std[(s, t)]:.3f}    {m_abl[(s, t)]:.3f}   {m_abl[(s, t)] - m_std[(s, t)]:+.3f}")

def mean(v):
    return sum(v) / len(v)

turkic = [lang for lang in langs if lang != "fi"]
within_pairs = [(a, b) for a in turkic for b in turkic if a != b]
cross = [(a, "fi") for a in turkic] + [("fi", b) for b in turkic]
for label, m in (("baseline", m_std), ("ablated", m_abl)):
    w = mean([m[p] for p in within_pairs])
    c = mean([m[p] for p in cross])
    lines.append(f"\n{label}: within-Turkic {w:.3f}  cross-to-Finnish {c:.3f}  gap {c - w:+.3f}")

sib = {"az": "tr", "tr": "az", "kk": "ky", "ky": "kk", "ug": "uz", "uz": "ug"}
for label, m in (("baseline", m_std), ("ablated", m_abl)):
    ok = []
    for s, expected in sib.items():
        best = min((t for t in langs if t != s), key=lambda t: m[(s, t)])
        ok.append(f"{s}->{best}{'==' if best == expected else '!=' + expected}")
    lines.append(f"{label} nearest-neighbour: " + "  ".join(ok))

def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den

rho_all = spearman([m_std[p] for p in off], [m_abl[p] for p in off])
lines.append(f"\nSpearman rho over 42 off-diagonal cells: {rho_all:.4f}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("done")
