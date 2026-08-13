"""Generate the OOV assimilation table from trained vocabs and snippets.

For every listener language (a trained model's vocabulary) and every IPA
segment that occurs in the perception snippets but is missing from that
vocabulary, finds the nearest in-vocabulary segment in articulatory feature
space -- the modeling analogue of perceptual assimilation, where a listener
maps an unfamiliar foreign sound onto their closest native category.

Features are coarse articulatory coordinates:

- Vowels: (height 0=close..4=open, backness 0=front..2=back, rounding 0/1),
  distance ``2*|dh| + 2*|db| + |dr|``.
- Consonants: (place 0=bilabial..9=glottal, manner 0=stop..5=approximant,
  voicing 0/1), distance ``|dplace| + 2*|dmanner| + |dvoice|``.

Vowels only map to vowels and consonants to consonants. Ties break on the
lexicographically smallest candidate so output is fully deterministic.
Non-phoneme characters (digits, punctuation, length/tie diacritics) are not
mapped: they remain OOV at eval time.

Usage::

    poetry run python -m scripts.build_assimilation \\
        --checkpoint-dir checkpoints \\
        --snippet-dir data/perception_clean \\
        --output-csv data/assimilation.csv

"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from pathlib import Path
from typing import IO, TypedDict

from char_lstm.data import load_vocab_json
from scripts.corpora import LANGS

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_SNIPPET_DIR = Path("data/perception_clean")
DEFAULT_OUTPUT_CSV = Path("data/assimilation.csv")
DEFAULT_SNIPPET_TEMPLATE = "perception_{lang}.txt"
DEFAULT_MIN_COUNT = 5

# (height, backness, rounding)
VOWELS: dict[str, tuple[int, int, int]] = {
    "i": (0, 0, 0),
    "y": (0, 0, 1),
    "ɨ": (0, 1, 0),
    "ɯ": (0, 2, 0),
    "u": (0, 2, 1),
    "ɪ": (1, 0, 0),
    "ʏ": (1, 0, 1),
    "ʊ": (1, 2, 1),
    "e": (2, 0, 0),
    "ø": (2, 0, 1),
    "ɘ": (2, 1, 0),
    "ə": (2, 1, 0),
    "ɵ": (2, 1, 1),
    "o": (2, 2, 1),
    "ɛ": (3, 0, 0),
    "œ": (3, 0, 1),
    "ɜ": (3, 1, 0),
    "ɐ": (3, 1, 0),
    "ʌ": (3, 2, 0),
    "ɔ": (3, 2, 1),
    "æ": (4, 0, 0),
    "a": (4, 1, 0),
    "ɑ": (4, 2, 0),
    "ɒ": (4, 2, 1),
}

# (place, manner, voicing); place 0=bilabial 1=labiodental 2=dental 3=alveolar
# 4=postalveolar 5=palatal 6=velar 7=uvular 8=pharyngeal 9=glottal; manner
# 0=stop 1=affricate 2=fricative 3=nasal 4=trill/tap 5=approximant/lateral
CONSONANTS: dict[str, tuple[int, int, int]] = {
    "p": (0, 0, 0),
    "b": (0, 0, 1),
    "m": (0, 3, 1),
    "ɸ": (0, 2, 0),
    "β": (0, 2, 1),
    "f": (1, 2, 0),
    "v": (1, 2, 1),
    "ʋ": (1, 5, 1),
    "θ": (2, 2, 0),
    "ð": (2, 2, 1),
    "t": (3, 0, 0),
    "d": (3, 0, 1),
    "s": (3, 2, 0),
    "z": (3, 2, 1),
    "n": (3, 3, 1),
    "r": (3, 4, 1),
    "ɾ": (3, 4, 1),
    "l": (3, 5, 1),
    "ɹ": (3, 5, 1),
    "ʃ": (4, 2, 0),
    "ʒ": (4, 2, 1),
    "ɕ": (4, 2, 0),
    "ʑ": (4, 2, 1),
    "ʈ": (5, 0, 0),
    "ʂ": (5, 2, 0),
    "ʐ": (5, 2, 1),
    "c": (5, 0, 0),
    "ɟ": (5, 0, 1),
    "ç": (5, 2, 0),
    "ʝ": (5, 2, 1),
    "j": (5, 5, 1),
    "ɲ": (5, 3, 1),
    "k": (6, 0, 0),
    "g": (6, 0, 1),
    "ɡ": (6, 0, 1),
    "x": (6, 2, 0),
    "ɣ": (6, 2, 1),
    "ŋ": (6, 3, 1),
    "w": (6, 5, 1),
    "ɰ": (6, 5, 1),
    "ʍ": (6, 2, 0),
    "q": (7, 0, 0),
    "ɢ": (7, 0, 1),
    "χ": (7, 2, 0),
    "ʁ": (7, 2, 1),
    "ɴ": (7, 3, 1),
    "ʀ": (7, 4, 1),
    "ħ": (8, 2, 0),
    "ʕ": (8, 2, 1),
    "ʔ": (9, 0, 0),
    "h": (9, 2, 0),
    "ɦ": (9, 2, 1),
}


class AssimilationRow(TypedDict):
    """One generated substitution.

    Attributes:
        listener: Language whose model lacks the segment.
        missing: The OOV segment.
        replacement: Nearest in-vocabulary segment.
        distance: Articulatory feature distance of the chosen replacement.
        n_occurrences: How often the segment occurs across all snippets.
    """

    listener: str
    missing: str
    replacement: str
    distance: int
    n_occurrences: int


class BuildArgs(TypedDict):
    """Parsed and validated CLI arguments.

    Attributes:
        checkpoint_dir: Directory with ``{lang}_vocab.json`` files.
        snippet_dir: Directory with perception snippet files.
        output_csv: Destination CSV path.
        snippet_template: Filename template containing ``{lang}``.
        min_count: Minimum snippet occurrences for a segment to get a row.
    """

    checkpoint_dir: Path
    snippet_dir: Path
    output_csv: Path
    snippet_template: str
    min_count: int


def vowel_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """Articulatory distance between two vowels.

    Args:
        a: (height, backness, rounding) of the first vowel.
        b: (height, backness, rounding) of the second vowel.

    Returns:
        ``2*|dh| + 2*|db| + |dr|``.
    """
    return 2 * abs(a[0] - b[0]) + 2 * abs(a[1] - b[1]) + abs(a[2] - b[2])


def consonant_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """Articulatory distance between two consonants.

    Args:
        a: (place, manner, voicing) of the first consonant.
        b: (place, manner, voicing) of the second consonant.

    Returns:
        ``|dplace| + 2*|dmanner| + |dvoice|``.
    """
    return abs(a[0] - b[0]) + 2 * abs(a[1] - b[1]) + abs(a[2] - b[2])


def nearest_segment(missing: str, available: set[str]) -> tuple[str, int]:
    """Find the nearest in-vocabulary segment of the same class.

    Args:
        missing: OOV segment; must be in :data:`VOWELS` or :data:`CONSONANTS`.
        available: Characters present in the listener's vocabulary.

    Returns:
        (replacement, distance); ties break on the lexicographically
        smallest replacement.

    Raises:
        ValueError: If ``missing`` has no feature entry, or no same-class
            candidate exists in ``available``.
    """
    if missing in VOWELS:
        table = VOWELS
        features = VOWELS[missing]
        distance_fn = vowel_distance
    elif missing in CONSONANTS:
        table = CONSONANTS
        features = CONSONANTS[missing]
        distance_fn = consonant_distance
    else:
        msg = f"No feature entry for segment {missing!r}; cannot assimilate."
        raise ValueError(msg)
    candidates = sorted(ch for ch in available if ch in table and ch != missing)
    if not candidates:
        msg = f"No same-class candidate for {missing!r} in listener vocabulary."
        raise ValueError(msg)
    best = min(candidates, key=lambda ch: (distance_fn(features, table[ch]), ch))
    return best, distance_fn(features, table[best])


def count_snippet_segments(snippet_dir: Path, snippet_template: str) -> Counter[str]:
    """Count phoneme-table characters across all perception snippets.

    Args:
        snippet_dir: Directory with snippet files.
        snippet_template: Filename template containing ``{lang}``.

    Returns:
        Counts for every character that has a feature entry; characters
        without one (digits, punctuation, diacritics) are not counted.
    """
    counts: Counter[str] = Counter()
    for lang in LANGS:
        path = snippet_dir / snippet_template.format(lang=lang)
        if not path.exists():
            continue
        for ch in path.read_text(encoding="utf-8"):
            if ch in VOWELS or ch in CONSONANTS:
                counts[ch] += 1
    return counts


def build_rows(args: BuildArgs) -> list[AssimilationRow]:
    """Generate assimilation rows for every listener with a vocabulary file.

    Args:
        args: Validated CLI arguments.

    Returns:
        Rows sorted by (listener, missing segment).
    """
    counts = count_snippet_segments(args["snippet_dir"], args["snippet_template"])
    rows: list[AssimilationRow] = []
    for lang in LANGS:
        vocab_path = args["checkpoint_dir"] / f"{lang}_vocab.json"
        if not vocab_path.exists():
            print(f"  vocab missing for {lang}: {vocab_path}; skipping listener")
            continue
        stoi, _itos, _size, _unk = load_vocab_json(vocab_path)
        available = set(stoi)
        for segment, count in sorted(counts.items()):
            if count < args["min_count"] or segment in available:
                continue
            replacement, distance = nearest_segment(segment, available)
            rows.append(
                {
                    "listener": lang,
                    "missing": segment,
                    "replacement": replacement,
                    "distance": distance,
                    "n_occurrences": count,
                }
            )
    return rows


def render_csv(rows: list[AssimilationRow]) -> str:
    """Render assimilation rows as CSV text.

    Args:
        rows: Generated rows in output order.

    Returns:
        CSV text including header and trailing newline.
    """
    lines = ["listener,missing,replacement,distance,n_occurrences"]
    for row in rows:
        lines.append(
            f"{row['listener']},{row['missing']},{row['replacement']},"
            f"{row['distance']},{row['n_occurrences']}"
        )
    return "\n".join(lines) + "\n"


def run(args: BuildArgs) -> list[AssimilationRow]:
    """Generate the table and write it to disk.

    Args:
        args: Validated CLI arguments.

    Returns:
        The generated rows.
    """
    rows = build_rows(args)
    args["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    args["output_csv"].write_text(render_csv(rows), encoding="utf-8")
    for row in rows:
        print(
            f"  {row['listener']}: {row['missing']} -> {row['replacement']} "
            f"(distance {row['distance']}, {row['n_occurrences']} occurrences)"
        )
    print(f"Wrote {len(rows)} substitution(s) to {args['output_csv']}")
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Generate the OOV assimilation table from vocabs + snippets.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--snippet-dir", type=str, default=str(DEFAULT_SNIPPET_DIR))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--snippet-template", type=str, default=DEFAULT_SNIPPET_TEMPLATE)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    return parser


def _extract_args(namespace: argparse.Namespace) -> BuildArgs:
    """Validate and convert an argparse Namespace into typed BuildArgs.

    Args:
        namespace: Parsed argparse namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`BuildArgs`.

    Raises:
        TypeError: If any argument has an unexpected type.
        ValueError: If ``min_count`` is not positive.
    """
    str_fields = {
        "checkpoint_dir": namespace.checkpoint_dir,
        "snippet_dir": namespace.snippet_dir,
        "output_csv": namespace.output_csv,
        "snippet_template": namespace.snippet_template,
    }
    for name, value in str_fields.items():
        if not isinstance(value, str):
            msg = f"Expected str for --{name.replace('_', '-')}, got {type(value).__name__}"
            raise TypeError(msg)
    if not isinstance(namespace.min_count, int):
        msg = f"Expected int for --min-count, got {type(namespace.min_count).__name__}"
        raise TypeError(msg)
    if namespace.min_count < 1:
        msg = f"--min-count must be >= 1, got {namespace.min_count}"
        raise ValueError(msg)
    return {
        "checkpoint_dir": Path(str_fields["checkpoint_dir"]),
        "snippet_dir": Path(str_fields["snippet_dir"]),
        "output_csv": Path(str_fields["output_csv"]),
        "snippet_template": str_fields["snippet_template"],
        "min_count": namespace.min_count,
    }


def parse_args(argv: list[str] | None = None) -> BuildArgs:
    """Parse CLI arguments into typed :class:`BuildArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`BuildArgs`.
    """
    parser = _build_arg_parser()
    return _extract_args(parser.parse_args(argv))


def _configure_stream_utf8(stream: IO[str]) -> None:
    """Reconfigure ``stream`` to UTF-8 encoding when it supports it.

    ``run()`` prints IPA characters (``ə``, ``t͡ʃ`` etc.) drawn from the
    assimilation table. On Windows Python 3.11's default cp1252
    stdout encoding, those characters raise
    :class:`UnicodeEncodeError` at ``print`` time and abort the
    script. This helper reconfigures the passed stream to UTF-8 so
    IPA output round-trips reliably. The ``isinstance`` guard leaves
    in-memory captures (``io.StringIO`` and similar) alone: those
    types do not carry a text encoding, so reconfiguration is
    inapplicable rather than best-effort.

    Args:
        stream: The text stream to reconfigure.
    """
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success.
    """
    _configure_stream_utf8(sys.stdout)
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
