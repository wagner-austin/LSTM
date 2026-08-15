"""Compare the rebuilt-pipeline matrix against the shipped 2026-08-13 run."""

import csv
import json
from pathlib import Path

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
OUT = LSTM / "analysis" / "rebuild-2026-08" / "compare_v3.txt"
BRANCH = {"kk": "kipchak", "ky": "kipchak", "tr": "oghuz", "az": "oghuz", "uz": "karluk", "ug": "karluk"}


def load(name: str) -> dict[tuple[str, str], float]:
    cells = {}
    with (LSTM / "results" / name).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["listener_language"], row["text_language"])
            cells[key] = float(row["excess_cross_entropy"])
    return cells


def foreign(cells):
    return {k: v for k, v in cells.items() if k[0] != k[1]}


def nearest(cells):
    out = {}
    for src in sorted({k[0] for k in cells}):
        row = {t: v for (s, t), v in cells.items() if s == src and s != t}
        out[src] = min(row, key=row.get)
    return out


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


def stats(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return m, sd


v3 = load("v3_full_skip.csv")
old = load("zero_shot_excess_ce_skip_2026-08-13.csv")
f3, fo = foreign(v3), foreign(old)
keys = sorted(f3)
lines = []

lines.append("== nearest foreign text per listener ==")
n3, no = nearest(v3), nearest(old)
for src in sorted(n3):
    flag = "" if n3[src] == no[src] else "  <-- CHANGED"
    lines.append(f"  {src}: v3={n3[src]}  shipped={no[src]}{flag}")

lines.append("")
lines.append("== branch structure (Turkic-only cells) ==")
for name, cells in (("v3", f3), ("shipped", fo)):
    within = [v for (s, t), v in cells.items() if s in BRANCH and t in BRANCH and BRANCH[s] == BRANCH[t]]
    between = [v for (s, t), v in cells.items() if s in BRANCH and t in BRANCH and BRANCH[s] != BRANCH[t]]
    wm, wsd = stats(within)
    bm, bsd = stats(between)
    lines.append(f"  {name}: within {wm:.2f} (sd {wsd:.2f}) | between {bm:.2f} (sd {bsd:.2f})")

lines.append("")
lines.append("== the control question: fi vs uz per Turkic listener ==")
for src in ("az", "tr", "kk", "ky", "ug"):
    v3f, v3u = f3[(src, "fi")], f3[(src, "uz")]
    olf, olu = fo[(src, "fi")], fo[(src, "uz")]
    v3v = "uz easier" if v3u < v3f else "fi easier"
    olv = "uz easier" if olu < olf else "fi easier"
    mark = "  <-- flipped" if v3v != olv else ""
    lines.append(
        f"  {src}: v3 fi={v3f:.2f} uz={v3u:.2f} ({v3v}) | shipped fi={olf:.2f} uz={olu:.2f} ({olv}){mark}"
    )

lines.append("")
lines.append("== fi rank among each Turkic listener's foreign texts (1=easiest, 6=hardest) ==")
for src in ("az", "tr", "kk", "ky", "ug", "uz"):
    row3 = sorted((v, t) for (s, t), v in f3.items() if s == src)
    rowo = sorted((v, t) for (s, t), v in fo.items() if s == src)
    r3 = [t for _, t in row3].index("fi") + 1
    ro = [t for _, t in rowo].index("fi") + 1
    lines.append(f"  {src}: v3 rank {r3} of 6 | shipped rank {ro} of 6")

lines.append("")
lines.append(f"== spearman(v3, shipped) over 42 foreign cells: "
             f"{spearman([f3[k] for k in keys], [fo[k] for k in keys]):.4f} ==")

lines.append("")
lines.append("== largest cell moves (v3 minus shipped) ==")
moves = sorted(((f3[k] - fo[k], k) for k in keys), key=lambda p: abs(p[0]), reverse=True)
for d, (s, t) in moves[:10]:
    lines.append(f"  {s}->{t}: {d:+.3f}  (shipped {fo[(s, t)]:.2f} -> v3 {f3[(s, t)]:.2f})")

lines.append("")
lines.append("== v3 vocab sizes (model vocab incl. unk) ==")
sizes = []
for path in sorted((LSTM / "checkpoints_v3").glob("*_vocab.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    sizes.append(f"{path.name.split('_')[0]}:{len(data['itos'])}")
lines.append("  " + "  ".join(sizes))

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("done")
