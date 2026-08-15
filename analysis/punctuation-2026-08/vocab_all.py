"""Character-class breakdown for all seven training corpora."""

import collections
import unicodedata

BASE = r"C:\Users\Test\PROJECTS\lstm\corpora_clean"
OUT = r"C:\Users\Test\AppData\Local\Temp\claude\C--Users-Test-PROJECTS-turkic-transliteration\6f031a76-36d1-47c9-9ff7-40e012dd3579\scratchpad\vocab_all.txt"

IPA_ASCII = set("abdefghijklmnopqrstuvwyzx")  # ascii letters that ARE IPA output glyphs somewhere
rows = ["lang total | letters+mods punct digit ws | strays (each <0.01%): list"]
for lang in ["tr", "az", "kk", "ky", "uz", "ug", "fi"]:
    counts: collections.Counter[str] = collections.Counter()
    with open(rf"{BASE}\oscar_{lang}_ipa.txt", encoding="utf-8") as fh:
        for line in fh:
            counts.update(line)
    total_chars = sum(counts.values())
    letters, punct, digit, ws, stray = [], [], [], [], []
    for ch, n in counts.items():
        cat = unicodedata.category(ch)
        if ch in "\n ":
            ws.append(ch)
        elif cat.startswith("P") or cat in ("Sm", "Sc", "Sk"):
            punct.append(ch)
        elif cat == "Nd":
            digit.append(ch)
        elif n / total_chars < 0.0001:
            stray.append((ch, n))
        else:
            letters.append(ch)
    stray.sort(key=lambda e: -e[1])
    stray_str = " ".join(f"{c}:{n}" for c, n in stray[:12])
    rows.append(
        f"{lang} {len(counts):3d} | {len(letters):2d} {len(punct):2d} {len(digit):2d} {len(ws)} | {len(stray)} strays: {stray_str}"
    )
open(OUT, "w", encoding="utf-8").write("\n".join(rows))
print("done")
