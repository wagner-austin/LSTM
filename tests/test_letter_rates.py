"""Tests for scripts.letter_rates."""

from __future__ import annotations

import argparse
import hashlib
import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str
from scripts.letter_rates import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_PROFILE,
    CorpusLetterRates,
    LetterRatesArgs,
    ProfileMismatchError,
    _extract_args,
    decode_letter_rate_profile,
    encode_letter_rate_profile,
    letter_rates,
    load_letter_rate_profile,
    main,
    parse_args,
    profile_corpus,
    require_matches,
    run,
)

from char_lstm.corpora import PERCEPTION_LANGS

#: The profile ``run`` takes of the corpora :func:`_corpora` writes.
_TEXT = "Ab1, ba!b\n"


def _corpora(tmp_path: Path) -> Path:
    """Test helper: one identical corpus per perception language."""
    corpus_dir = tmp_path / "corpora_clean"
    corpus_dir.mkdir()
    for lang in PERCEPTION_LANGS:
        (corpus_dir / f"oscar_{lang}_ipa.txt").write_text(_TEXT * 100, encoding="utf-8")
    return corpus_dir


def _args(tmp_path: Path, check: bool) -> LetterRatesArgs:
    """Test helper: arguments over :func:`_corpora` and a profile beside them."""
    corpus_dir = tmp_path / "corpora_clean"
    if not corpus_dir.is_dir():
        _corpora(tmp_path)
    return {"corpus_dir": corpus_dir, "profile": tmp_path / "data" / "rates.json", "check": check}


def _entry(lang: str, corpus: str, rates: dict[str, float]) -> CorpusLetterRates:
    """Test helper: one profile entry."""
    return {"lang": lang, "corpus": corpus, "rates": rates}


def test_rates_count_lowercase_letters_per_thousand_characters() -> None:
    """Uppercase, digits, punctuation and whitespace count toward length only.

    ``_TEXT`` is ten characters of which one lowercase a and three b are
    counted, so a is 100 per 1,000 and b is 300; the capital A is not.
    """
    rates = letter_rates(_TEXT * 100)

    assert rates == {"a": 100.0, "b": 300.0}
    assert list(rates) == ["a", "b"]


def test_letters_are_sorted_whatever_order_they_appear_in() -> None:
    assert list(letter_rates("ɑzba")) == ["a", "b", "z", "ɑ"]


def test_an_empty_text_has_no_rate() -> None:
    with pytest.raises(ValueError, match="empty text"):
        letter_rates("")


def test_a_corpus_is_named_by_its_generation_and_digest(tmp_path: Path) -> None:
    """The label is the one a training record gives the same file."""
    corpus_dir = _corpora(tmp_path)
    path = corpus_dir / "oscar_kk_ipa.txt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    profile = profile_corpus(corpus_dir, "kk")

    assert profile == _entry("kk", f"corpora_clean:{digest}", {"a": 100.0, "b": 300.0})


def test_a_profile_survives_a_round_trip() -> None:
    profiles = (
        _entry("az", "corpora_clean:000000000001", {"a": 1.5, "ɑ": 80.25}),
        _entry("kk", "corpora_clean:000000000002", {"w": 0.0}),
    )

    assert load_letter_rate_profile(encode_letter_rate_profile(profiles)) == profiles


def test_a_language_profiled_twice_is_refused() -> None:
    document = load_json_str(
        '{"corpora": [{"lang": "az", "corpus": "c", "rates": {}},'
        ' {"lang": "az", "corpus": "c", "rates": {}}]}'
    )
    with pytest.raises(ValueError, match="Language az appears twice"):
        decode_letter_rate_profile(document)


@pytest.mark.parametrize("rates", ['{"ab": 1.0}', '{"a": -1.0}'])
def test_a_rate_entry_must_be_one_character_at_a_non_negative_rate(rates: str) -> None:
    document = load_json_str(f'{{"corpora": [{{"lang": "az", "corpus": "c", "rates": {rates}}}]}}')
    with pytest.raises(ValueError, match="is not one character at a non-negative rate"):
        decode_letter_rate_profile(document)


def test_a_profile_matching_the_corpora_is_accepted() -> None:
    profiles = (_entry("az", "corpora_clean:000000000001", {"a": 1.0}),)
    require_matches(profiles, profiles)


def test_a_profile_of_another_corpus_is_refused() -> None:
    committed = (_entry("az", "corpora_clean:000000000001", {"a": 1.0}),)
    measured = (_entry("az", "corpora_clean:000000000002", {"a": 1.0}),)
    with pytest.raises(ProfileMismatchError, match=r"differs for \['az'\] and names \[\]"):
        require_matches(committed, measured)


def test_a_profile_naming_a_language_with_no_corpus_is_refused() -> None:
    kept = _entry("az", "corpora_clean:000000000001", {"a": 1.0})
    extra = _entry("ru", "corpora_clean:000000000003", {"a": 1.0})
    with pytest.raises(ProfileMismatchError, match=r"differs for \[\] and names \['ru'\]"):
        require_matches((kept, extra), (kept,))


def test_run_writes_one_entry_per_perception_language(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(tmp_path, check=False)

    measured = run(args)

    assert [p["lang"] for p in measured] == list(PERCEPTION_LANGS)
    assert load_letter_rate_profile(args["profile"].read_text(encoding="utf-8")) == measured
    out = capsys.readouterr().out
    assert out == f"Wrote letter rates of 7 corpora to {args['profile']}\n"


def test_run_checks_a_profile_without_rewriting_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(_args(tmp_path, check=False))
    args = _args(tmp_path, check=True)
    written = args["profile"].read_bytes()

    run(args)

    assert args["profile"].read_bytes() == written
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == f"{args['profile']} describes {args['corpus_dir']}"


def test_run_check_refuses_a_profile_the_corpora_have_moved_past(tmp_path: Path) -> None:
    """A corpus rebuilt after its profile was written changes its digest."""
    run(_args(tmp_path, check=False))
    args = _args(tmp_path, check=True)
    (args["corpus_dir"] / "oscar_tr_ipa.txt").write_text(_TEXT * 99, encoding="utf-8")

    with pytest.raises(ProfileMismatchError, match=r"differs for \['tr'\]"):
        run(args)


def test_parse_args_defaults_to_the_published_corpora_and_profile() -> None:
    args = parse_args([])
    assert args == {"corpus_dir": DEFAULT_CORPUS_DIR, "profile": DEFAULT_PROFILE, "check": False}


def test_parse_args_reads_every_flag() -> None:
    args = parse_args(["--corpus-dir", "c", "--profile", "p.json", "--check"])
    assert args == {"corpus_dir": Path("c"), "profile": Path("p.json"), "check": True}


def test_extract_args_rejects_a_non_string_path() -> None:
    namespace = argparse.Namespace(corpus_dir=1, profile="p", check=False)
    with pytest.raises(TypeError, match="Expected str for --corpus-dir and --profile"):
        _extract_args(namespace)


def test_extract_args_rejects_a_non_bool_check() -> None:
    namespace = argparse.Namespace(corpus_dir="c", profile="p", check="yes")
    with pytest.raises(TypeError, match="Expected bool for --check"):
        _extract_args(namespace)


def test_main_writes_the_profile(tmp_path: Path) -> None:
    args = _args(tmp_path, check=False)
    argv = ["--corpus-dir", str(args["corpus_dir"]), "--profile", str(args["profile"])]

    assert main(argv) == 0
    assert len(load_letter_rate_profile(args["profile"].read_text(encoding="utf-8"))) == 7


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path, check=False)
    argv = ["--corpus-dir", str(args["corpus_dir"]), "--profile", str(args["profile"])]
    monkeypatch.setattr(sys, "argv", ["letter_rates", *argv])
    monkeypatch.delitem(sys.modules, "scripts.letter_rates")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.letter_rates", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
