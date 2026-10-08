"""The letter-rate profile of each training corpus, as a committed document.

``tests/test_train_eval_consistency.py`` refuses a training corpus and an
evaluation text written under two transliteration conventions, and it needs
only each corpus's letter rates to do so. It used to read the corpora
themselves, which are 104 MB and untracked, so on any checkout but the
workstation that built them the test failed on a missing file instead of
checking anything. The profile is small, so it is committed, and it names
each corpus by :func:`char_lstm.training_record.corpus_label` -- the same
generation-and-digest label a checkpoint's training record carries -- so
the corpus a profile describes is the one a model was trained on, and
``--check`` refuses a profile the corpora on disk no longer match.

The document is::

    {"corpora": [{"lang": "az", "corpus": "corpora_clean:0123456789ab",
                  "rates": {"a": 81.2, ...}},
                 ...]}

with one entry per perception language, in
:data:`char_lstm.corpora.PERCEPTION_LANGS` order, and each table's letters
sorted.

Usage::

    poetry run python -m scripts.letter_rates --corpus-dir corpora_clean
    poetry run python -m scripts.letter_rates --corpus-dir corpora_clean --check
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_float,
    require_dict,
    require_list,
    require_str,
)

from char_lstm.corpora import PERCEPTION_LANGS, corpus_file
from char_lstm.training_record import corpus_label

DEFAULT_CORPUS_DIR = Path("corpora_clean")
DEFAULT_PROFILE = Path("data") / "training_letter_rates.json"


class CorpusLetterRates(TypedDict):
    """One training corpus's letter rates.

    Attributes:
        lang: Language code of the corpus.
        corpus: The corpus's label, generation directory and digest, as a
            training record names it.
        rates: Occurrences per 1,000 characters of every lowercase letter.
    """

    lang: str
    corpus: str
    rates: dict[str, float]


class LetterRatesArgs(TypedDict):
    """Parsed and validated CLI arguments.

    Attributes:
        corpus_dir: Directory holding the cleaned training corpora.
        profile: The profile document to write or check.
        check: Compare against ``profile`` instead of writing it.
    """

    corpus_dir: Path
    profile: Path
    check: bool


class ProfileMismatchError(ValueError):
    """The committed profile does not describe the corpora on disk."""


def letter_rates(text: str) -> dict[str, float]:
    """Occurrences per 1,000 characters for every letter in a text.

    Case, digits and punctuation are excluded: headers contribute
    uppercase, and genre legitimately moves punctuation. Letters are
    where a notation difference would live.

    Args:
        text: The text to profile; must not be empty.

    Returns:
        Letter to rate per 1,000 characters of the text, letters sorted.

    Raises:
        ValueError: If ``text`` is empty, since it has no rate to take.
    """
    if not text:
        msg = "Cannot take letter rates of an empty text."
        raise ValueError(msg)
    counts = collections.Counter(ch for ch in text if ch.isalpha() and not ch.isupper())
    scale = len(text) / 1000
    return {ch: counts[ch] / scale for ch in sorted(counts)}


def profile_corpus(corpus_dir: Path, lang: str) -> CorpusLetterRates:
    """Profile one language's training corpus.

    Args:
        corpus_dir: Directory holding the cleaned corpora.
        lang: Language code.

    Returns:
        The corpus's label and letter rates.
    """
    path = corpus_file(corpus_dir, lang)
    return CorpusLetterRates(
        lang=lang,
        corpus=corpus_label(path),
        rates=letter_rates(path.read_text(encoding="utf-8")),
    )


def _decode_rates(obj: JSONObject) -> dict[str, float]:
    """Decode one corpus's rate table.

    Args:
        obj: The ``rates`` object.

    Returns:
        Letter to rate.

    Raises:
        ValueError: If a key is not a single character or a rate is negative.
    """
    rates: dict[str, float] = {}
    for ch, value in obj.items():
        rate = narrow_json_to_float(value)
        if len(ch) != 1 or rate < 0:
            msg = f"Rate entry {ch!r}: {rate} is not one character at a non-negative rate."
            raise ValueError(msg)
        rates[ch] = rate
    return rates


def decode_letter_rate_profile(value: JSONValue) -> tuple[CorpusLetterRates, ...]:
    """Decode and validate a letter-rate profile document.

    Args:
        value: The parsed JSON document.

    Returns:
        One entry per corpus, in document order.

    Raises:
        ValueError: If a language appears twice.
    """
    profiles: list[CorpusLetterRates] = []
    for entry in require_list(narrow_json_to_dict(value), "corpora"):
        obj = narrow_json_to_dict(entry)
        lang = require_str(obj, "lang")
        if any(p["lang"] == lang for p in profiles):
            msg = f"Language {lang} appears twice in the profile."
            raise ValueError(msg)
        profiles.append(
            CorpusLetterRates(
                lang=lang,
                corpus=require_str(obj, "corpus"),
                rates=_decode_rates(require_dict(obj, "rates")),
            )
        )
    return tuple(profiles)


def load_letter_rate_profile(raw: str) -> tuple[CorpusLetterRates, ...]:
    """Parse and decode a letter-rate profile from its text.

    Args:
        raw: The document's JSON text.

    Returns:
        The profile, validated by :func:`decode_letter_rate_profile`.
    """
    return decode_letter_rate_profile(load_json_str(raw))


def encode_letter_rate_profile(profiles: tuple[CorpusLetterRates, ...]) -> str:
    """Encode a profile as the document :func:`load_letter_rate_profile` reads.

    Args:
        profiles: One entry per corpus.

    Returns:
        The JSON text, indented, with a trailing newline.
    """
    corpora: list[JSONValue] = []
    for p in profiles:
        rates: JSONObject = dict(p["rates"])
        corpora.append({"lang": p["lang"], "corpus": p["corpus"], "rates": rates})
    document: JSONObject = {"corpora": corpora}
    return dump_json_str(document, indent=1) + "\n"


def require_matches(
    committed: tuple[CorpusLetterRates, ...], measured: tuple[CorpusLetterRates, ...]
) -> None:
    """Refuse a committed profile that is not the one the corpora give.

    Args:
        committed: The profile read from disk.
        measured: The profile just taken from the corpora.

    Raises:
        ProfileMismatchError: Naming each language whose entry differs.
    """
    by_lang = {p["lang"]: p for p in committed}
    stale = [m["lang"] for m in measured if by_lang.get(m["lang"]) != m]
    extra = sorted(set(by_lang) - {m["lang"] for m in measured})
    if stale or extra:
        msg = (
            f"The committed profile does not describe these corpora: it differs for {stale} "
            f"and names {extra}, which have no corpus here. Rewrite it without --check."
        )
        raise ProfileMismatchError(msg)


def run(args: LetterRatesArgs) -> tuple[CorpusLetterRates, ...]:
    """Profile every perception language's corpus, then write or check.

    Args:
        args: Validated CLI arguments.

    Returns:
        The profile taken from the corpora.

    Raises:
        ProfileMismatchError: Under ``check``, if the committed profile differs.
    """
    measured = tuple(profile_corpus(args["corpus_dir"], lang) for lang in PERCEPTION_LANGS)
    if args["check"]:
        committed = load_letter_rate_profile(args["profile"].read_text(encoding="utf-8"))
        require_matches(committed, measured)
        print(f"{args['profile']} describes {args['corpus_dir']}")
        return measured
    args["profile"].parent.mkdir(parents=True, exist_ok=True)
    args["profile"].write_text(encode_letter_rate_profile(measured), encoding="utf-8")
    print(f"Wrote letter rates of {len(measured)} corpora to {args['profile']}")
    return measured


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Write or check the letter-rate profile of the training corpora.",
    )
    parser.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--profile", type=str, default=str(DEFAULT_PROFILE))
    parser.add_argument("--check", action="store_true")
    return parser


def _extract_args(namespace: argparse.Namespace) -> LetterRatesArgs:
    """Validate and convert an argparse Namespace into typed LetterRatesArgs.

    Args:
        namespace: Parsed namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`LetterRatesArgs`.

    Raises:
        TypeError: If an argument has an unexpected type.
    """
    corpus_dir = namespace.corpus_dir
    profile = namespace.profile
    check = namespace.check
    if not isinstance(corpus_dir, str) or not isinstance(profile, str):
        msg = "Expected str for --corpus-dir and --profile."
        raise TypeError(msg)
    if not isinstance(check, bool):
        msg = "Expected bool for --check."
        raise TypeError(msg)
    return LetterRatesArgs(corpus_dir=Path(corpus_dir), profile=Path(profile), check=check)


def parse_args(argv: list[str] | None = None) -> LetterRatesArgs:
    """Parse CLI arguments into a typed :class:`LetterRatesArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`LetterRatesArgs`.
    """
    return _extract_args(_build_arg_parser().parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success.
    """
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
