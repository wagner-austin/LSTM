"""Punctuation usage profiles: training corpora vs perception texts."""

import collections
import unicodedata
from pathlib import Path

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
OUT = Path(
    r"C:\Users\Test\AppData\Local\Temp\claude\C--Users-Test-PROJECTS-turkic-transliteration"
    r"\6f031a76-36d1-47c9-9ff7-40e012dd3579\scratchpad\punct_profile.txt"
)
LANGS = ["tr", "az", "kk", "ky", "uz", "ug", "fi"]
MARKS = [".", ",", "-", ":", "«", "»", '"', "'", "(", ")", "!", "?", ";", "–", "—"]

def profile(text: str) -> tuple[dict[str, float], float]:
    counts = collections.Counter(text)
    total = len(text)
    per1k = {m: 1000 * counts[m] / total for m in MARKS}
    all_punct = 1000 * sum(n for ch, n in counts.items() if unicodedata.category(ch).startswith(("P", "S")) or unicodedata.category(ch) == "Nd") / total
    return per1k, all_punct

rows = ["=== training corpora (per 1,000 chars) ==="]
header = "lang  total " + " ".join(f"{m:>5}" for m in MARKS)
rows.append(header)
for lang in LANGS:
    text = (LSTM / "corpora_clean" / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")
    per1k, tot = profile(text)
    rows.append(f"{lang:4}  {tot:5.1f} " + " ".join(f"{per1k[m]:5.2f}" for m in MARKS))

rows.append("\n=== perception texts (per 1,000 chars) ===")
rows.append(header)
for lang in LANGS:
    text = (LSTM / "data" / "perception_clean" / f"perception_{lang}.txt").read_text(encoding="utf-8")
    per1k, tot = profile(text)
    rows.append(f"{lang:4}  {tot:5.1f} " + " ".join(f"{per1k[m]:5.2f}" for m in MARKS))

OUT.write_text("\n".join(rows), encoding="utf-8")
print("done")
