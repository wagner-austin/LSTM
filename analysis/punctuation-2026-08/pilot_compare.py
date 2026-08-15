"""Helpers comparing the pilot punctuation variants on excess-CE matrices.

Loads the per-variant pilot CSVs and the full-scale matrix, and provides
foreign-cell extraction, Spearman correlation, and nearest-neighbour
tables. Run from the repo root; writes its two-variant summary next to
this file. The archived pilot_compare_results.txt holds the three-way
comparison over a (raw), b (collapsed), c (stripped)."""

import csv
from pathlib import Path

RESULTS = Path("results")
OUT = Path(__file__).with_name("pilot_compare_a_vs_c.txt")


def load(name):
    cells = {}
    with (RESULTS / name).open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            key = (row["listener_language"], row["text_language"])
            cells[key] = float(row["excess_cross_entropy"])
    return cells


def foreign(cells):
    return {k: v for k, v in cells.items() if k[0] != k[1]}


def spearman(x, y):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy)


def nearest(cells):
    listeners = sorted({k[0] for k in cells})
    out = {}
    for src in listeners:
        row = {t: v for (s, t), v in cells.items() if s == src and s != t}
        out[src] = min(row, key=row.get)
    return out


a = load("pilot_a_skip.csv")
c = load("pilot_c_skip.csv")
full = load("zero_shot_excess_ce_skip.csv")

fa, fc, ff = foreign(a), foreign(c), foreign(full)
keys = sorted(fa)
xa = [fa[k] for k in keys]
xc = [fc[k] for k in keys]
xf = [ff[k] for k in keys]

lines = []
lines.append(f"mean foreign excess  a={sum(xa)/len(xa):.4f}  c={sum(xc)/len(xc):.4f}")
na = sum(v for (s, t), v in a.items() if s == t) / 7
nc = sum(v for (s, t), v in c.items() if s == t) / 7
lines.append("")
lines.append(f"spearman(a, c) over 42 foreign cells      = {spearman(xa, xc):.4f}")
lines.append(f"spearman(a, full-scale) [both raw punct]  = {spearman(xa, xf):.4f}")
lines.append(f"spearman(c, full-scale)                   = {spearman(xc, xf):.4f}")
lines.append("")
diffs = sorted(((fc[k] - fa[k], k) for k in keys), key=lambda p: abs(p[0]), reverse=True)
lines.append("largest c-minus-a shifts:")
for d, k in diffs[:8]:
    lines.append(f"  {k[0]}->{k[1]}: {d:+.4f}  (a={fa[k]:.4f}, c={fc[k]:.4f})")
lines.append("")
lines.append("nearest foreign text per listener:")
n_a, n_c, n_f = nearest(a), nearest(c), nearest(full)
for src in sorted(n_a):
    flag = "" if n_a[src] == n_c[src] == n_f[src] else "  <-- differs"
    lines.append(f"  {src}: a={n_a[src]}  c={n_c[src]}  full={n_f[src]}{flag}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("done")
