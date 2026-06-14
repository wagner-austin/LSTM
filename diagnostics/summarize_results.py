"""One-off: summary table across eval modes from the results CSVs."""

import csv
from pathlib import Path

BRANCHES = {
    "az": "oghuz",
    "tr": "oghuz",
    "kk": "kipchak",
    "ky": "kipchak",
    "ug": "karluk",
    "uz": "karluk",
}
FILES = {
    "LSTM skip (headline)": "results/zero_shot_excess_ce_skip.csv",
    "LSTM unk (deafness)": "results/zero_shot_excess_ce_unk.csv",
    "LSTM assimilate": "results/zero_shot_excess_ce_assimilate.csv",
    "Trigram skip (control)": "results/ngram_excess_ce.csv",
}

print(f"{'condition':<24} {'within':>7} {'cross':>7} {'gap':>7} {'separation':>11} {'pairs'}")
for label, path in FILES.items():
    excess: dict[tuple[str, str], float] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            excess[(row["listener_language"], row["text_language"])] = float(
                row["excess_cross_entropy"]
            )
    langs = sorted({s for s, _ in excess} & set(BRANCHES))
    sym = {
        (a, b): 0.5 * (excess[(a, b)] + excess[(b, a)])
        for i, a in enumerate(langs)
        for b in langs[i + 1 :]
    }
    within = {p: d for p, d in sym.items() if BRANCHES[p[0]] == BRANCHES[p[1]]}
    cross = {p: d for p, d in sym.items() if BRANCHES[p[0]] != BRANCHES[p[1]]}
    w = sum(within.values()) / len(within)
    c = sum(cross.values()) / len(cross)
    perfect = max(within.values()) < min(cross.values())
    closest = {min(within, key=within.get), min(cross, key=cross.get)}
    pairs = ", ".join(f"{a}-{b}={d:.2f}" for (a, b), d in sorted(within.items()))
    print(
        f"{label:<24} {w:7.3f} {c:7.3f} {c - w:+7.3f} "
        f"{'PERFECT' if perfect else 'overlap':>11} {pairs}"
    )

print()
print("Self-CE diagonals (snippet difficulty per language, skip mode):")
with Path(FILES["LSTM skip (headline)"]).open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["listener_language"] == row["text_language"]:
            print(f"  {row['listener_language']}: {float(row['cross_entropy']):.3f}")
