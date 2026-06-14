"""Diagnostic: section-level bootstrap CIs + normalized distance clustering.

1. Parse perception snippets into numbered sections (dropping headers/titles).
2. Score every (src model, tgt section) on the common-support mask
   (next char present in all 7 vocabs).
3. Paired bootstrap over sections -> CIs for pair CE, excess CE
   d(s,t) = CE(s->t) - CE(t->t), and key comparisons.
4. UPGMA clustering of the symmetrized excess-CE matrix (6 Turkic langs),
   plus profile-correlation clustering of all 7 sources.
"""

import re
from pathlib import Path

import numpy as np
import torch
from scripts.zero_shot_eval import LANGS, load_model_with_vocab
from torch.nn import functional

from char_lstm.data import encode

CKPT = Path("checkpoints")
SNIP = Path("data/perception")
TARGETS = [lang for lang in LANGS if lang != "fi"]
MARKER_RE = re.compile(r"^\s*[1-5]\s*$")
HEADER_RE = re.compile(r"^\s*te?xt\s*\d", re.IGNORECASE)
N_BOOT = 4000
RNG = np.random.default_rng(0)


def parse_sections(path: Path) -> list[str]:
    """Split a perception file into numbered sections, dropping headers."""
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
        # short non-sentence line right before a "1" marker = stray title
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if len(line.strip()) < 40 and MARKER_RE.match(nxt) and nxt.strip() == "1":
            continue
        if current is not None:
            current.append(line.strip())
    if current:
        sections.append(current)
    return [" ".join(s) for s in sections if s and len(" ".join(s)) >= 20]


# ---- load everything -------------------------------------------------------
loaded = {
    s: load_model_with_vocab(CKPT / f"{s}_best.pt", CKPT / f"{s}_vocab.json")
    for s in LANGS
}
sections: dict[str, list[str]] = {
    t: parse_sections(SNIP / f"perception_{t}.txt") for t in TARGETS
}
print("=== Section parse check ===")
for t in TARGETS:
    total = sum(len(s) for s in sections[t])
    raw = len((SNIP / f"perception_{t}.txt").read_text(encoding="utf-8"))
    print(
        f"  {t}: {len(sections[t])} sections, {total} chars "
        f"({total / raw:.1%} of raw file)"
    )

# ---- per-section common-support losses -------------------------------------
# loss_sum[s][t][k], n[t][k] for section k of target t scored by source s
all_chars_everywhere = [set(loaded[s]["stoi"]) for s in LANGS]


def common_mask(text: str) -> list[bool]:
    return [all(c in v for v in all_chars_everywhere) for c in text[1:]]


loss_sum: dict[str, dict[str, np.ndarray]] = {s: {} for s in LANGS}
n_common: dict[str, np.ndarray] = {}
for t in TARGETS:
    masks = [torch.tensor(common_mask(sec)) for sec in sections[t]]
    n_common[t] = np.array([int(m.sum()) for m in masks])
    for s in LANGS:
        stoi, vs = loaded[s]["stoi"], loaded[s]["vocab_size"]
        sums = []
        for sec, mask in zip(sections[t], masks):
            idx = encode(sec, stoi)
            inputs = torch.tensor([idx[:-1]], dtype=torch.long)
            tgts = torch.tensor(idx[1:], dtype=torch.long)
            with torch.no_grad():
                logits, _ = loaded[s]["model"](inputs)
            losses = functional.cross_entropy(
                logits.view(-1, vs), tgts, reduction="none"
            )
            sums.append(float(losses[mask].sum()))
        loss_sum[s][t] = np.array(sums)

# ---- point estimates + paired bootstrap -------------------------------------
def pair_ce(s: str, t: str, sec_idx: np.ndarray) -> float:
    return float(loss_sum[s][t][sec_idx].sum() / n_common[t][sec_idx].sum())


print("\n=== Section-only common-support CE (point estimates) ===")
print("src\\tgt " + "  ".join(f"{t:>6}" for t in TARGETS))
full_idx = {t: np.arange(len(sections[t])) for t in TARGETS}
for s in LANGS:
    print(
        f"  {s}    "
        + "  ".join(f"{pair_ce(s, t, full_idx[t]):6.3f}" for t in TARGETS)
    )

# bootstrap: resample section indices per target, shared across sources (paired)
boot_ce = {s: {t: np.empty(N_BOOT) for t in TARGETS} for s in LANGS}
for b in range(N_BOOT):
    idx = {
        t: RNG.integers(0, len(sections[t]), len(sections[t])) for t in TARGETS
    }
    for s in LANGS:
        for t in TARGETS:
            boot_ce[s][t][b] = pair_ce(s, t, idx[t])

print("\n=== 95% CIs for excess CE  d(s,t) = CE(s->t) - CE(t->t) ===")
print("src\\tgt " + "        ".join(f"{t:>6}" for t in TARGETS))
excess_pt: dict[str, dict[str, float]] = {s: {} for s in LANGS}
for s in LANGS:
    cells = []
    for t in TARGETS:
        pt = pair_ce(s, t, full_idx[t]) - pair_ce(t, t, full_idx[t])
        excess_pt[s][t] = pt
        if s == t:
            cells.append("     --       ")
            continue
        d = boot_ce[s][t] - boot_ce[t][t]
        lo, hi = np.percentile(d, [2.5, 97.5])
        cells.append(f"{pt:5.2f}[{lo:4.2f},{hi:4.2f}]")
    print(f"  {s}  " + " ".join(cells))

print("\n=== Key comparisons (bootstrap prob that difference > 0) ===")
COMPARISONS = [
    ("az->tr vs tr->az asymmetry", ("tr", "az"), ("az", "tr")),
    ("fi beats az on kk target", ("az", "kk"), ("fi", "kk")),
    ("fi beats kk-model on ky target", ("kk", "ky"), ("fi", "ky")),
    ("tr->uz beats ug->uz (Karluk sib)", ("ug", "uz"), ("tr", "uz")),
    ("uz->tr beats az->tr", ("az", "tr"), ("uz", "tr")),
    ("fi worst-on-tr vs next (az)", ("fi", "tr"), ("az", "tr")),
    ("fi worst-on-uz vs next (ky)", ("fi", "uz"), ("ky", "uz")),
]
for label, (s1, t1), (s2, t2) in COMPARISONS:
    diff = boot_ce[s1][t1] - boot_ce[s2][t2]
    pt = pair_ce(s1, t1, full_idx[t1]) - pair_ce(s2, t2, full_idx[t2])
    p_gt = float((diff > 0).mean())
    print(f"  {label}: diff={pt:+.3f}, P(>0)={p_gt:.3f}")

# ---- UPGMA on symmetrized excess CE (6 Turkic) ------------------------------
print("\n=== UPGMA tree, symmetrized excess CE, 6 Turkic languages ===")
langs6 = list(TARGETS)
dist = np.zeros((6, 6))
for i, a in enumerate(langs6):
    for j, b in enumerate(langs6):
        if i != j:
            dist[i, j] = 0.5 * (excess_pt[a][b] + excess_pt[b][a])

clusters: list[tuple[str, list[int]]] = [(lang, [i]) for i, lang in enumerate(langs6)]
work = dist.copy()
active = list(range(6))
names = {i: langs6[i] for i in range(6)}
members: dict[int, list[int]] = {i: [i] for i in range(6)}
next_id = 6
while len(active) > 1:
    best = None
    for ai in range(len(active)):
        for bi in range(ai + 1, len(active)):
            a, b = active[ai], active[bi]
            d = np.mean(
                [dist[x, y] for x in members[a] for y in members[b]]
            )
            if best is None or d < best[0]:
                best = (d, a, b)
    d, a, b = best
    names[next_id] = f"({names[a]},{names[b]}):{d:.2f}"
    members[next_id] = members[a] + members[b]
    active = [x for x in active if x not in (a, b)] + [next_id]
    next_id += 1
print("  " + names[active[0]])

# ---- profile clustering of all 7 sources ------------------------------------
print("\n=== Source-profile distances (corr. over shared non-self targets) ===")
print("(how similarly two models rank the six target texts)")
profs = {s: np.array([excess_pt[s][t] for t in TARGETS]) for s in LANGS}
print("       " + "  ".join(f"{s:>5}" for s in LANGS))
for a in LANGS:
    row = []
    for b in LANGS:
        dims = [
            k for k, t in enumerate(TARGETS) if t != a and t != b
        ]
        va, vb = profs[a][dims], profs[b][dims]
        r = float(np.corrcoef(va, vb)[0, 1])
        row.append(f"{1 - r:5.2f}" if a != b else "   --")
    print(f"  {a}   " + "  ".join(row))
