"""Clean IPA corpora: dedup, junk filtering, symbol harmonization, equalization.

Pipeline per language (order matters):

1. Apply the symbol-harmonization map (``data/symbol_map.csv``, rows with
   ``action=merge``) so all languages spell the same sounds the same way.
2. Drop lines whose IPA-character ratio is below ``--min-ipa-ratio`` --
   removes emoji/CJK/foreign-script leakage.
3. Sanitize surviving lines: residual non-IPA characters become spaces, so
   no junk symbol ever reaches the training vocabulary.
4. Drop short lines (< ``--min-line-chars``) -- menu words, section markers.
5. Drop duplicate lines (keep first occurrence) -- removes site boilerplate
   such as navigation menus, footers, and registry dumps.
6. Truncate every cleaned corpus to the size of the smallest one (whole
   lines), so all languages train on equal data.

Also re-writes the perception snippets with the same symbol map (no line
filtering -- their section structure is preserved) so snippets and corpora
stay in the same symbol space.

Usage::

    poetry run python -m scripts.clean_corpus \\
        --input-dir 09_Downloaded_Corpora_2 \\
        --output-dir 10_Cleaned_Corpora

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TypedDict

LANGS: tuple[str, ...] = ("az", "fi", "kk", "ky", "tr", "ug", "uz")

DEFAULT_INPUT_DIR = Path("09_Downloaded_Corpora_2")
DEFAULT_OUTPUT_DIR = Path("10_Cleaned_Corpora")
DEFAULT_SYMBOL_MAP = Path("data/symbol_map.csv")
DEFAULT_SNIPPET_DIR = Path("data/perception")
DEFAULT_SNIPPET_OUTPUT_DIR = Path("data/perception_clean")
DEFAULT_MIN_LINE_CHARS = 30
DEFAULT_MIN_IPA_RATIO = 0.95
CORPUS_TEMPLATE = "oscar_{lang}_ipa.txt"
SNIPPET_TEMPLATE = "perception_{lang}.txt"

# Characters considered legitimate transcription content. Latin base letters
# cover plain IPA segments; the extension block covers every IPA symbol that
# the seven languages' phonologies use (per the JIPA Illustrations cited in
# data/symbol_map.csv); digits and basic punctuation occur in running prose.
_IPA_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_IPA_EXTENSIONS = (
    "ɑæɐɒəɘɵɛɜɪɨɯɔœøʊʏʌ"  # vowels
    "ʁʔʕɣɟɡɢɦɥɰʝŋɲɴɸɹɾʀʂʃʈʋʍχʐʑʒθðβçħɕ"  # consonants
    "ʧʤ"  # affricate ligatures (mapped away, but legal pre-map)
)
_DIACRITICS = "ː͡ʲʼ̥̃̆"
_PUNCTUATION = " \t.,!?;:()«»\"'’`-–—%/0123456789"
ALLOWED_CHARS = frozenset(_IPA_LETTERS + _IPA_EXTENSIONS + _DIACRITICS + _PUNCTUATION)


class LangStats(TypedDict):
    """Cleaning statistics for one language.

    Attributes:
        lines_in: Raw line count before cleaning.
        dropped_duplicate: Lines dropped as exact duplicates of earlier lines.
        dropped_short: Lines dropped for being shorter than the minimum.
        dropped_low_ipa: Lines dropped for a sub-threshold IPA-char ratio.
        lines_kept: Lines surviving all filters.
        chars_kept: Total characters surviving all filters.
        chars_written: Characters actually written after equalization.
    """

    lines_in: int
    dropped_duplicate: int
    dropped_short: int
    dropped_low_ipa: int
    lines_kept: int
    chars_kept: int
    chars_written: int


class CleanArgs(TypedDict):
    """Parsed and validated CLI arguments.

    Attributes:
        input_dir: Directory with raw ``oscar_{lang}_ipa.txt`` corpora.
        output_dir: Destination directory for cleaned corpora.
        symbol_map: Path to the symbol-harmonization CSV.
        snippet_dir: Directory with raw perception snippets.
        snippet_output_dir: Destination for harmonized snippets.
        min_line_chars: Minimum surviving line length.
        min_ipa_ratio: Minimum fraction of allowed chars per line.
    """

    input_dir: Path
    output_dir: Path
    symbol_map: Path
    snippet_dir: Path
    snippet_output_dir: Path
    min_line_chars: int
    min_ipa_ratio: float


# ---------------------------------------------------------------------------
# Symbol map
# ---------------------------------------------------------------------------


def load_symbol_map(path: Path) -> dict[str, dict[str, str]]:
    """Load per-language character substitutions from the symbol-map CSV.

    Only rows with ``action=merge`` produce substitutions; ``keep`` and any
    other action rows are documentation. A row's ``scope`` is either ``all``
    or a ``+``-separated list of language codes.

    Args:
        path: CSV with columns ``action,scope,from,to,...``.

    Returns:
        Mapping ``lang -> {from_char: to_str}`` covering every language code
        in :data:`LANGS` (possibly with empty dicts).

    Raises:
        ValueError: If a merge row has an empty ``from`` field or its scope
            names an unknown language code.
    """
    mapping: dict[str, dict[str, str]] = {lang: {} for lang in LANGS}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["action"] != "merge":
                continue
            source = row["from"]
            if source == "":
                msg = f"symbol map {path}: merge row with empty 'from' field"
                raise ValueError(msg)
            scopes = list(LANGS) if row["scope"] == "all" else row["scope"].split("+")
            for lang in scopes:
                if lang not in mapping:
                    msg = f"symbol map {path}: unknown language {lang!r} in scope"
                    raise ValueError(msg)
                mapping[lang][source] = row["to"]
    return mapping


def apply_symbol_map(text: str, substitutions: dict[str, str]) -> str:
    """Apply character substitutions to a text.

    Args:
        text: Input text.
        substitutions: Mapping of source string to replacement (empty string
            deletes the source).

    Returns:
        Text with every substitution applied.
    """
    for source, target in substitutions.items():
        text = text.replace(source, target)
    return text


# ---------------------------------------------------------------------------
# Line filtering
# ---------------------------------------------------------------------------


def sanitize_line(line: str) -> str:
    """Replace residual disallowed characters with spaces and collapse runs.

    Applied only to lines that already passed the IPA-ratio gate, so at most
    a few stray characters per line are affected. Replacing with a space
    (rather than deleting) avoids creating false character adjacencies in
    the phonotactic training signal.

    Args:
        line: Line that passed the IPA-ratio filter.

    Returns:
        Line containing only allowed characters, single-spaced, stripped.
    """
    replaced = "".join(ch if ch in ALLOWED_CHARS else " " for ch in line)
    return " ".join(replaced.split())


def ipa_ratio(line: str) -> float:
    """Fraction of a line's characters that are allowed transcription chars.

    Args:
        line: Non-empty line.

    Returns:
        Value in [0, 1].
    """
    allowed = sum(1 for ch in line if ch in ALLOWED_CHARS)
    return allowed / len(line)


def clean_lines(
    lines: list[str],
    substitutions: dict[str, str],
    min_line_chars: int,
    min_ipa_ratio: float,
) -> tuple[list[str], LangStats]:
    """Run the per-line cleaning pipeline for one language.

    Args:
        lines: Raw corpus lines.
        substitutions: Symbol-map substitutions for this language.
        min_line_chars: Minimum surviving line length (after strip).
        min_ipa_ratio: Minimum allowed-character ratio per line.

    Returns:
        Tuple of (kept lines, statistics). ``chars_written`` is initialized
        to 0; equalization fills it in later.
    """
    seen: set[str] = set()
    kept: list[str] = []
    stats: LangStats = {
        "lines_in": len(lines),
        "dropped_duplicate": 0,
        "dropped_short": 0,
        "dropped_low_ipa": 0,
        "lines_kept": 0,
        "chars_kept": 0,
        "chars_written": 0,
    }
    for raw in lines:
        line = apply_symbol_map(raw, substitutions).strip()
        if len(line) < min_line_chars:
            stats["dropped_short"] += 1
            continue
        if ipa_ratio(line) < min_ipa_ratio:
            stats["dropped_low_ipa"] += 1
            continue
        line = sanitize_line(line)
        if len(line) < min_line_chars:
            stats["dropped_short"] += 1
            continue
        if line in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(line)
        kept.append(line)
    stats["lines_kept"] = len(kept)
    stats["chars_kept"] = sum(len(line) + 1 for line in kept)
    return kept, stats


def truncate_to_budget(lines: list[str], budget: int) -> list[str]:
    """Keep whole lines from the start until the character budget is reached.

    Args:
        lines: Cleaned lines.
        budget: Maximum total characters (each line costs ``len + 1`` for
            its newline).

    Returns:
        Longest prefix of ``lines`` whose total cost is <= ``budget``.
    """
    total = 0
    out: list[str] = []
    for line in lines:
        cost = len(line) + 1
        if total + cost > budget:
            break
        total += cost
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def clean_corpora(args: CleanArgs) -> dict[str, LangStats]:
    """Clean all corpora, equalize sizes, and write outputs plus a manifest.

    Args:
        args: Validated CLI arguments.

    Returns:
        Per-language statistics.

    Raises:
        FileNotFoundError: If a corpus file is missing.
    """
    symbol_map = load_symbol_map(args["symbol_map"])
    cleaned: dict[str, list[str]] = {}
    stats: dict[str, LangStats] = {}
    for lang in LANGS:
        corpus_path = args["input_dir"] / CORPUS_TEMPLATE.format(lang=lang)
        raw_lines = corpus_path.read_text(encoding="utf-8").splitlines()
        kept, lang_stats = clean_lines(
            raw_lines,
            symbol_map[lang],
            args["min_line_chars"],
            args["min_ipa_ratio"],
        )
        cleaned[lang] = kept
        stats[lang] = lang_stats

    budget = min(stats[lang]["chars_kept"] for lang in LANGS)
    args["output_dir"].mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        final_lines = truncate_to_budget(cleaned[lang], budget)
        text = "\n".join(final_lines) + "\n" if final_lines else ""
        out_path = args["output_dir"] / CORPUS_TEMPLATE.format(lang=lang)
        out_path.write_text(text, encoding="utf-8")
        stats[lang]["chars_written"] = len(text)
        print(
            f"  {lang}: {stats[lang]['lines_in']} lines -> "
            f"{stats[lang]['lines_kept']} kept "
            f"(dup {stats[lang]['dropped_duplicate']}, "
            f"short {stats[lang]['dropped_short']}, "
            f"junk {stats[lang]['dropped_low_ipa']}), "
            f"wrote {stats[lang]['chars_written']} chars"
        )

    manifest = {
        "config": {
            "input_dir": str(args["input_dir"]),
            "symbol_map": str(args["symbol_map"]),
            "min_line_chars": args["min_line_chars"],
            "min_ipa_ratio": args["min_ipa_ratio"],
            "equalized_char_budget": budget,
        },
        "stats": stats,
    }
    manifest_path = args["output_dir"] / "cleaning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats


def clean_snippets(args: CleanArgs) -> list[str]:
    """Re-write perception snippets with the symbol map applied.

    No line filtering happens here: snippet section structure is preserved
    verbatim so the eval's section parser keeps working.

    Args:
        args: Validated CLI arguments.

    Returns:
        Language codes whose snippet was found and converted.
    """
    symbol_map = load_symbol_map(args["symbol_map"])
    converted: list[str] = []
    args["snippet_output_dir"].mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        snippet_path = args["snippet_dir"] / SNIPPET_TEMPLATE.format(lang=lang)
        if not snippet_path.exists():
            print(f"  snippet missing for {lang}: {snippet_path}; skipping")
            continue
        text = snippet_path.read_text(encoding="utf-8")
        out_path = args["snippet_output_dir"] / SNIPPET_TEMPLATE.format(lang=lang)
        out_path.write_text(apply_symbol_map(text, symbol_map[lang]), encoding="utf-8")
        converted.append(lang)
    return converted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Clean IPA corpora: dedup, junk filter, symbol map, equalize.",
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbol-map", type=str, default=str(DEFAULT_SYMBOL_MAP))
    parser.add_argument("--snippet-dir", type=str, default=str(DEFAULT_SNIPPET_DIR))
    parser.add_argument("--snippet-output-dir", type=str, default=str(DEFAULT_SNIPPET_OUTPUT_DIR))
    parser.add_argument("--min-line-chars", type=int, default=DEFAULT_MIN_LINE_CHARS)
    parser.add_argument("--min-ipa-ratio", type=float, default=DEFAULT_MIN_IPA_RATIO)
    return parser


def _extract_args(namespace: argparse.Namespace) -> CleanArgs:
    """Validate and convert an argparse Namespace into typed CleanArgs.

    Args:
        namespace: Parsed argparse namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`CleanArgs`.

    Raises:
        TypeError: If any argument has an unexpected type.
    """
    str_fields = {
        "input_dir": namespace.input_dir,
        "output_dir": namespace.output_dir,
        "symbol_map": namespace.symbol_map,
        "snippet_dir": namespace.snippet_dir,
        "snippet_output_dir": namespace.snippet_output_dir,
    }
    for name, value in str_fields.items():
        if not isinstance(value, str):
            msg = f"Expected str for --{name.replace('_', '-')}, got {type(value).__name__}"
            raise TypeError(msg)
    if not isinstance(namespace.min_line_chars, int):
        msg = f"Expected int for --min-line-chars, got {type(namespace.min_line_chars).__name__}"
        raise TypeError(msg)
    if not isinstance(namespace.min_ipa_ratio, float):
        msg = f"Expected float for --min-ipa-ratio, got {type(namespace.min_ipa_ratio).__name__}"
        raise TypeError(msg)
    return {
        "input_dir": Path(str_fields["input_dir"]),
        "output_dir": Path(str_fields["output_dir"]),
        "symbol_map": Path(str_fields["symbol_map"]),
        "snippet_dir": Path(str_fields["snippet_dir"]),
        "snippet_output_dir": Path(str_fields["snippet_output_dir"]),
        "min_line_chars": namespace.min_line_chars,
        "min_ipa_ratio": namespace.min_ipa_ratio,
    }


def parse_args(argv: list[str] | None = None) -> CleanArgs:
    """Parse CLI arguments into typed :class:`CleanArgs`.

    Args:
        argv: Optional CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`CleanArgs`.
    """
    parser = _build_arg_parser()
    return _extract_args(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Optional CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success.
    """
    args = parse_args(argv)
    print("Cleaning corpora...")
    clean_corpora(args)
    print("Harmonizing perception snippets...")
    converted = clean_snippets(args)
    print(f"Done: snippets converted for {', '.join(converted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
