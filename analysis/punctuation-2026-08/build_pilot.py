"""Build the punctuation-method pilot: variants A/B/C at 25% corpus scale.

A = raw cleaned text, B = punctuation collapsed to one symbol and digits
to another, C = punctuation and digits replaced by a space with space
runs collapsed. Each pilot corpus is the first 2.5M characters of its
full variant, cut at a line boundary. Perception texts get the matching
transform per variant; A reuses the released files.
"""

import re
import sys
import unicodedata
from pathlib import Path

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
sys.path.insert(0, str(LSTM))

from scripts.zero_shot_eval import parse_sections  # noqa: E402

LANGS = ("az", "fi", "kk", "ky", "tr", "ug", "uz")
PILOT_CHARS = 2_500_000
PUNCT_SYMBOL = "\u00b6"
DIGIT_SYMBOL = "#"
SPACE_RUN = re.compile(r" {2,}")


def collapse(text: str) -> str:
    """Variant B: punctuation/symbols to pilcrow, digits to hash."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Nd":
            out.append(DIGIT_SYMBOL)
        elif cat.startswith(("P", "S")):
            out.append(PUNCT_SYMBOL)
        else:
            out.append(ch)
    return "".join(out)


def strip(text: str) -> str:
    """Variant C: punctuation/symbols/digits to space, runs collapsed."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Nd" or cat.startswith(("P", "S")):
            out.append(" ")
        else:
            out.append(ch)
    lines = "".join(out).split("\n")
    return "\n".join(SPACE_RUN.sub(" ", ln).strip() for ln in lines)


def truncate_at_line(text: str, limit: int) -> str:
    """First `limit` characters, cut back to the last full line."""
    cut = text[:limit]
    return cut[: cut.rfind("\n") + 1]


TRANSFORMS = {"a": lambda t: t, "b": collapse, "c": strip}

for variant, fn in TRANSFORMS.items():
    corpus_dir = LSTM / f"corpora_pilot_{variant}"
    snip_dir = LSTM / "data" / f"perception_pilot_{variant}"
    corpus_dir.mkdir(exist_ok=True)
    snip_dir.mkdir(exist_ok=True)
    for lang in LANGS:
        full = (LSTM / "corpora_clean" / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")
        pilot = truncate_at_line(fn(full), PILOT_CHARS)
        (corpus_dir / f"oscar_{lang}_ipa.txt").write_text(pilot, encoding="utf-8")

        snippet = (LSTM / "data" / "perception_clean" / f"perception_{lang}.txt").read_text(
            encoding="utf-8"
        )
        sections = parse_sections(snippet)
        lines = [lang.upper()]
        for i, section in enumerate(sections):
            lines.append(str(i % 5 + 1))
            lines.append(fn(section))
        (snip_dir / f"perception_{lang}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(variant, "built")

# Round-trip: per variant, section counts match and text is the transform.
for variant, fn in TRANSFORMS.items():
    for lang in LANGS:
        orig = parse_sections(
            (LSTM / "data" / "perception_clean" / f"perception_{lang}.txt").read_text(
                encoding="utf-8"
            )
        )
        var = parse_sections(
            (LSTM / "data" / f"perception_pilot_{variant}" / f"perception_{lang}.txt").read_text(
                encoding="utf-8"
            )
        )
        assert len(orig) == len(var), f"{variant}/{lang}: {len(orig)} vs {len(var)}"
        assert all(fn(o).strip() == v.strip() for o, v in zip(orig, var)), f"{variant}/{lang} text"
print("round-trip ok")
