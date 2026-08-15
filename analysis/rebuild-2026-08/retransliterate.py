"""Re-transliterate the orthographic downloads with the fixed pipeline.

Run inside the turkic-rebuild worktree environment. Each line is
normalised (format characters deleted, compatibility folded), repaired
with the language's fold table, and transliterated with the corrected
rules; structure lines in the perception sources (language name, TEXT
headers, section markers, blanks) pass through the same path, which is
what the previous generation did.

Inputs and outputs are absolute so the script can run from either repo.
"""

import sys
import time
from pathlib import Path

from turkic_translit.core import to_ipa
from turkic_translit.corpus.normalize import PACKAGED_FOLDS, normalize_line
from turkic_translit.corpus.symbols import (
    PACKAGED_SYMBOL_MAP,
    apply_substitutions,
    read_symbol_map,
    substitutions_for,
)

LSTM = Path(r"C:\Users\Test\PROJECTS\lstm")
ORTHO = LSTM / "rebuild_2026-08" / "orthographic"
RAW_OUT = LSTM / "rebuild_2026-08" / "corpora_raw_v3"
SOURCES = LSTM / "data" / "perception_sources"
SNIPPET_OUT = LSTM / "rebuild_2026-08" / "perception_v3"

LANGS = ("az", "fi", "kk", "ky", "tr", "ug", "uz")


def convert_file(
    source: Path,
    target: Path,
    lang: str,
    folds: dict[str, str],
    harmonize: dict[str, str],
) -> int:
    """Normalise, fold, transliterate and harmonise one file line by line.

    Args:
        source: Orthographic input file.
        target: Transliterated output file.
        lang: Rule language code.
        folds: The language's misencoding repairs, applied before the rules.
        harmonize: The language's symbol-map rewrites, applied after. The
            corpora pass an empty mapping because the cleaner applies the
            map itself; the snippets receive it here.

    Returns:
        Number of lines written.
    """
    lines_out = []
    for line in source.read_text(encoding="utf-8").splitlines():
        prepared = apply_substitutions(normalize_line(line), folds)
        lines_out.append(apply_substitutions(to_ipa(prepared, lang), harmonize))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def main() -> int:
    """Convert the seven corpora and the perception sources."""
    fold_rules = read_symbol_map(PACKAGED_FOLDS)
    symbol_rules = read_symbol_map(PACKAGED_SYMBOL_MAP)
    for lang in LANGS:
        folds = substitutions_for(fold_rules, lang)
        started = time.monotonic()
        count = convert_file(
            ORTHO / f"{lang}.txt", RAW_OUT / f"oscar_{lang}_ipa.txt", lang, folds, {}
        )
        print(f"corpus {lang}: {count} lines in {time.monotonic() - started:.0f}s", flush=True)

    for lang in LANGS:
        if lang == "fi":
            source = LSTM / "data" / "perception" / "perception_fi_source.txt"
        else:
            source = SOURCES / f"source_{lang}.txt"
        folds = substitutions_for(fold_rules, lang)
        harmonize = substitutions_for(symbol_rules, lang)
        count = convert_file(
            source, SNIPPET_OUT / f"perception_{lang}.txt", lang, folds, harmonize
        )
        print(f"snippet {lang}: {count} lines", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
