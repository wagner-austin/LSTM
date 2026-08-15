"""Gate the rebuilt corpora and snippets before any training run.

Run inside the turkic-rebuild worktree environment, after
retransliterate.py and the clean CLI. Every check answers one of the
defects this rebuild exists to remove; any failure prints and exits
nonzero, so the training launcher can refuse to start.
"""

import json
import sys
import unicodedata as ud
from pathlib import Path

from turkic_translit.corpus.clean import harmonized_emitted
from turkic_translit.corpus.symbols import read_symbol_map

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
CLEAN = LSTM / "rebuild_2026-08" / "corpora_clean_v3"
SNIPPETS = LSTM / "rebuild_2026-08" / "perception_v3"
LANGS = ("az", "fi", "kk", "ky", "tr", "ug", "uz")

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record a failed check.

    Args:
        condition: True when the property holds.
        message: What failed, when it did.
    """
    if not condition:
        failures.append(message)


SYMBOL_RULES = read_symbol_map()

for lang in LANGS:
    text = (CLEAN / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")
    allowed = harmonized_emitted(lang, SYMBOL_RULES) | {" ", "\n"}
    residue = set(text) - allowed
    check(residue == set(), f"{lang}: cleaned corpus carries {sorted(residue)[:10]}")
    check("\u00ad" not in text, f"{lang}: soft hyphen survived")
    check("\u02bb" not in text, f"{lang}: stranded okina survived")
    presentation = sum(1 for c in text if 0xFB50 <= ord(c) <= 0xFEFF)
    check(presentation == 0, f"{lang}: {presentation} presentation forms survived")
    digits = sum(1 for c in text if c.isdigit())
    check(digits == 0, f"{lang}: {digits} digits survived")

report = json.loads((CLEAN / "cleaning_manifest.json").read_text(encoding="utf-8"))
check(
    set(report["languages"]) == set(LANGS),
    f"manifest covers {sorted(report['languages'])}",
)
check(
    "rules_fingerprint" in report and len(report["rules_fingerprint"]) == len(LANGS) + 1,
    "manifest lacks the rules fingerprint",
)

# The corpus may legitimately carry j\u0254q: informal web Uzbek writes
# yoq without the apostrophe, and <yo> is j\u0254. What the seam fix
# guarantees is that the apostrophe spelling yields joq at scale.
uz_corpus = (CLEAN / "oscar_uz_ipa.txt").read_text(encoding="utf-8")
check(" joq " in uz_corpus, "uz corpus lacks joq -- the seam fix did not reach it")
check(" jol" in uz_corpus, "uz corpus lacks jol-words -- the seam fix did not reach it")

kk_corpus = (CLEAN / "oscar_kk_ipa.txt").read_text(encoding="utf-8")
check("ja" not in kk_corpus, "kk corpus still carries ASCII a after j — the я fix missed")

for lang in LANGS:
    snippet = (SNIPPETS / f"perception_{lang}.txt").read_text(encoding="utf-8")
    markers = sum(1 for ln in snippet.splitlines() if ln.strip() in "12345" and ln.strip())
    expected = 19 if lang == "uz" else 20
    check(
        markers == expected,
        f"{lang}: snippet has {markers} section markers, expected {expected}",
    )
    cyrillic = [c for c in snippet if ud.name(c, "").startswith("CYRILLIC")]
    check(cyrillic == [], f"{lang}: snippet keeps Cyrillic {sorted(set(cyrillic))[:5]}")

# The passages contain yo\u02bbl-words, which the previous generation
# rendered with the wrong vowel and a stranded quote mark.
uz_snippet = (SNIPPETS / "perception_uz.txt").read_text(encoding="utf-8")
check("joldan" in uz_snippet, "uz snippet lacks joldan -- the seam fix missed it")
check("jolni" in uz_snippet, "uz snippet lacks jolni")
for mark in ("\u02bb", "\u2018", "\u2019", "\u00ad"):
    for lang in LANGS:
        snippet_text = (SNIPPETS / f"perception_{lang}.txt").read_text(encoding="utf-8")
        check(
            mark not in snippet_text,
            f"{lang}: snippet strands U+{ord(mark):04X}",
        )

fi_snippet = (SNIPPETS / "perception_fi.txt").read_text(encoding="utf-8")
check("\u00e6" in fi_snippet, "fi snippet lacks æ — source encoding suspect")

if failures:
    for failure in failures:
        print(f"FAIL {failure}")
    sys.exit(1)
print("audit clean: corpora and snippets carry only what the rules emit")
sys.exit(0)
