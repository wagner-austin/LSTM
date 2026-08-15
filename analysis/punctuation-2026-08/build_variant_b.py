"""Build corpus variant B: punctuation collapsed to one symbol, digits to another.

Every Unicode punctuation/symbol character becomes the pilcrow and every
decimal digit becomes the hash, in both the training corpora and the
perception texts, so the boundary information survives while every
corpus-specific typographic habit disappears. Perception files are
re-emitted in the minimal marker format parse_sections consumes, with
section text transformed and markers left intact.
"""

import sys
import unicodedata
from pathlib import Path

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
sys.path.insert(0, str(LSTM))

from scripts.zero_shot_eval import parse_sections  # noqa: E402

VARIANT_CORPUS = LSTM / "corpora_variant_b"
VARIANT_SNIPPETS = LSTM / "data" / "perception_variant_b"
LANGS = ("az", "fi", "kk", "ky", "tr", "ug", "uz")

PUNCT_SYMBOL = "\u00b6"  # pilcrow
DIGIT_SYMBOL = "#"


def collapse(text: str) -> str:
    """Collapse punctuation/symbols to the pilcrow and digits to the hash.

    Args:
        text: Input text.

    Returns:
        The transformed text; letters, marks, and whitespace untouched.
    """
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


VARIANT_CORPUS.mkdir(exist_ok=True)
VARIANT_SNIPPETS.mkdir(exist_ok=True)

for lang in LANGS:
    src = (LSTM / "corpora_clean" / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")
    (VARIANT_CORPUS / f"oscar_{lang}_ipa.txt").write_text(collapse(src), encoding="utf-8")

    snippet = (LSTM / "data" / "perception_clean" / f"perception_{lang}.txt").read_text(
        encoding="utf-8"
    )
    sections = parse_sections(snippet)
    lines = [lang.upper()]
    for i, section in enumerate(sections):
        lines.append(str(i % 5 + 1))
        lines.append(collapse(section))
    (VARIANT_SNIPPETS / f"perception_{lang}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(lang, len(src), "chars,", len(sections), "sections")

# Sanity: round-trip section counts must match the originals.
for lang in LANGS:
    orig = parse_sections(
        (LSTM / "data" / "perception_clean" / f"perception_{lang}.txt").read_text(encoding="utf-8")
    )
    var = parse_sections(
        (VARIANT_SNIPPETS / f"perception_{lang}.txt").read_text(encoding="utf-8")
    )
    assert len(orig) == len(var), f"{lang}: {len(orig)} vs {len(var)} sections"
    assert all(collapse(o) == v for o, v in zip(orig, var)), f"{lang}: section text mismatch"
print("round-trip ok")
