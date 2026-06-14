"""Validity tests for the zero-shot excess-CE method.

Test A (negative control): shuffle chars within each perception section --
same characters/frequencies, phonotactic order destroyed. The branch
structure should collapse; if it survives, the method measures alphabet
overlap, not phonotactics.

Test B (replication): use 5k-char slices from each language's held-out TEST
split (middle of last 15% of corpus) as targets instead of Moldir's
snippets. The tree should replicate on totally different text. Finnish can
be a target here too.

Both report the symmetrized excess-CE matrix, UPGMA tree, and the "branch
gap" = mean cross-branch minus mean within-branch distance (positive and
large = real branch signal).
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
CORP = Path("09_Downloaded_Corpora_2")
TARGETS6 = [lang for lang in LANGS if lang != "fi"]
BRANCH = {"az": "oghuz", "tr": "oghuz", "kk": "kipchak", "ky": "kipchak", "ug": "karluk", "uz": "karluk"}
MARKER_RE = re.compile(r"^\s*[1-5]\s*$")
HEADER_RE = re.compile(r"^\s*te?xt\s*\d", re.IGNORECASE)
RNG = np.random.default_rng(1)


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


loaded = {
    s: load_model_with_vocab(CKPT / f"{s}_best.pt", CKPT / f"{s}_vocab.json")
    for s in LANGS
}
all_vocabs = [set(loaded[s]["stoi"]) for s in LANGS]


def ce_matrix(targets: dict[str, str]) -> dict[str, dict[str, float]]:
    """Common-support CE of every source model on every target text."""
    out: dict[str, dict[str, float]] = {s: {} for s in LANGS}
    masks = {
        t: torch.tensor([all(c in v for v in all_vocabs) for c in text[1:]])
        for t, text in targets.items()
    }
    for s in LANGS:
        stoi, vs = loaded[s]["stoi"], loaded[s]["vocab_size"]
        for t, text in targets.items():
            idx = encode(text, stoi)
            inputs = torch.tensor([idx[:-1]], dtype=torch.long)
            tgts = torch.tensor(idx[1:], dtype=torch.long)
            with torch.no_grad():
                logits, _ = loaded[s]["model"](inputs)
            losses = functional.cross_entropy(logits.view(-1, vs), tgts, reduction="none")
            out[s][t] = float(losses[masks[t]].mean())
    return out


def report(ce: dict[str, dict[str, float]], targets: list[str], label: str) -> None:
    excess = {
        s: {t: ce[s][t] - ce[t][t] for t in targets} for s in LANGS if s in targets or s == "fi"
    }
    print(f"\n=== {label}: excess CE ===")
    print("src\\tgt " + "  ".join(f"{t:>6}" for t in targets))
    for s in [x for x in LANGS if x in targets] + (["fi"] if "fi" not in targets else []):
        print(f"  {s}    " + "  ".join(f"{excess[s][t]:6.3f}" for t in targets))

    sym_langs = [t for t in targets if t in BRANCH]
    dist = {
        (a, b): 0.5 * (excess[a][b] + excess[b][a])
        for a in sym_langs
        for b in sym_langs
        if a != b
    }
    within = [d for (a, b), d in dist.items() if BRANCH[a] == BRANCH[b]]
    cross = [d for (a, b), d in dist.items() if BRANCH[a] != BRANCH[b]]
    gap = float(np.mean(cross) - np.mean(within))
    print(
        f"  branch gap = {gap:+.3f}  "
        f"(within-branch mean {np.mean(within):.3f}, cross-branch mean {np.mean(cross):.3f})"
    )

    # UPGMA
    n = len(sym_langs)
    members = {i: [i] for i in range(n)}
    names = {i: sym_langs[i] for i in range(n)}
    mat = np.zeros((n, n))
    for i, a in enumerate(sym_langs):
        for j, b in enumerate(sym_langs):
            if i != j:
                mat[i, j] = dist[(a, b)]
    active = list(range(n))
    nid = n
    while len(active) > 1:
        best = None
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                d = np.mean([mat[x, y] for x in members[a] for y in members[b]])
                if best is None or d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        names[nid] = f"({names[a]},{names[b]}):{d:.2f}"
        members[nid] = members[a] + members[b]
        active = [x for x in active if x not in (a, b)] + [nid]
        nid += 1
    print(f"  tree: {names[active[0]]}")


# --- Test A: shuffled snippets (negative control) ---------------------------
real = {}
shuf = {}
for t in TARGETS6:
    secs = parse_sections(SNIP / f"perception_{t}.txt")
    real[t] = " ".join(secs)
    shuffled_secs = []
    for sec in secs:
        chars = list(sec)
        RNG.shuffle(chars)
        shuffled_secs.append("".join(chars))
    shuf[t] = " ".join(shuffled_secs)

report(ce_matrix(real), TARGETS6, "REAL snippets (known-answer reference)")
report(ce_matrix(shuf), TARGETS6, "SHUFFLED snippets (negative control: tree should collapse)")

# --- Test B: held-out corpus slices (replication) ----------------------------
slices = {}
for t in LANGS:  # fi included as target here
    text = (CORP / f"oscar_{t}_ipa.txt").read_text(encoding="utf-8")
    test_region = text[int(len(text) * 0.85):]
    mid = len(test_region) // 2
    slices[t] = test_region[mid : mid + 5000]

report(ce_matrix(slices), list(LANGS), "HELD-OUT corpus slices (replication, fi included)")
