"""Tests for scripts.build_assimilation."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pytest
from scripts.build_assimilation import (
    CONSONANTS,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MIN_COUNT,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SNIPPET_DIR,
    DEFAULT_SNIPPET_TEMPLATE,
    VOWELS,
    AssimilationRow,
    BuildArgs,
    _extract_args,
    build_rows,
    consonant_distance,
    count_snippet_segments,
    main,
    nearest_segment,
    parse_args,
    render_csv,
    run,
    vowel_distance,
)

from char_lstm.data import UNK, save_vocab_json


def _write_vocab(checkpoint_dir: Path, lang: str, vocab: list[str]) -> None:
    """Test helper: write a ``{lang}_vocab.json`` with UNK appended."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    itos: dict[int, str] = dict(enumerate([*vocab, UNK]))
    save_vocab_json(itos, checkpoint_dir / f"{lang}_vocab.json")


def _make_args(tmp_path: Path) -> BuildArgs:
    """Test helper: BuildArgs pointing at tmp_path locations."""
    return {
        "checkpoint_dir": tmp_path / "ckpt",
        "snippet_dir": tmp_path / "snip",
        "output_csv": tmp_path / "out" / "assimilation.csv",
        "snippet_template": "perception_{lang}.txt",
        "min_count": 2,
    }


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------


def test_feature_tables_are_disjoint_and_well_formed() -> None:
    assert set(VOWELS) & set(CONSONANTS) == set()
    for features in VOWELS.values():
        assert 0 <= features[0] <= 4
        assert 0 <= features[1] <= 2
        assert features[2] in (0, 1)
    for features in CONSONANTS.values():
        assert 0 <= features[0] <= 9
        assert 0 <= features[1] <= 5
        assert features[2] in (0, 1)


def test_vowel_distance_weights_height_and_backness_double() -> None:
    assert vowel_distance(VOWELS["i"], VOWELS["i"]) == 0
    assert vowel_distance(VOWELS["ɯ"], VOWELS["u"]) == 1  # rounding only
    assert vowel_distance(VOWELS["ɯ"], VOWELS["i"]) == 4  # backness 2 doubled


def test_consonant_distance_weights_manner_double() -> None:
    assert consonant_distance(CONSONANTS["q"], CONSONANTS["k"]) == 1  # place only
    assert consonant_distance(CONSONANTS["t"], CONSONANTS["s"]) == 4  # manner 2 doubled
    assert consonant_distance(CONSONANTS["t"], CONSONANTS["d"]) == 1  # voicing only


# ---------------------------------------------------------------------------
# nearest_segment
# ---------------------------------------------------------------------------


def test_nearest_segment_finnish_hears_turkic_unrounded_u_as_u() -> None:
    assert nearest_segment("ɯ", {"u", "i", "y", "k", "s"}) == ("u", 1)


def test_nearest_segment_turkish_hears_uvular_q_as_k() -> None:
    assert nearest_segment("q", {"k", "b", "a"}) == ("k", 1)


def test_nearest_segment_breaks_ties_lexicographically() -> None:
    # ə (2,1,0): both e (2,0,0) and ɜ (3,1,0) are at distance 2.
    assert nearest_segment("ə", {"e", "ɜ"}) == ("e", 2)


def test_nearest_segment_excludes_the_missing_char_itself() -> None:
    assert nearest_segment("ə", {"ə", "e"}) == ("e", 2)


def test_nearest_segment_rejects_unknown_segment() -> None:
    with pytest.raises(ValueError, match="No feature entry"):
        nearest_segment("ː", {"a"})


def test_nearest_segment_rejects_empty_candidate_class() -> None:
    with pytest.raises(ValueError, match="No same-class candidate"):
        nearest_segment("ɯ", {"k", "s"})


# ---------------------------------------------------------------------------
# count_snippet_segments / build_rows
# ---------------------------------------------------------------------------


def test_count_snippet_segments_counts_only_feature_chars(tmp_path: Path) -> None:
    snip = tmp_path / "snip"
    snip.mkdir()
    (snip / "perception_az.txt").write_text("qq aa 12 ːː\n", encoding="utf-8")
    counts = count_snippet_segments(snip, "perception_{lang}.txt")
    assert counts == {"q": 2, "a": 2}


def test_build_rows_generates_expected_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(tmp_path)
    _write_vocab(args["checkpoint_dir"], "az", ["a", "k", "u"])
    _write_vocab(args["checkpoint_dir"], "tr", ["a", "b", "k"])
    args["snippet_dir"].mkdir(parents=True)
    (args["snippet_dir"] / "perception_az.txt").write_text("qqqqqq ɯɯɯ uuuu\n", encoding="utf-8")

    rows = build_rows(args)

    expected: list[AssimilationRow] = [
        {"listener": "az", "missing": "q", "replacement": "k", "distance": 1, "n_occurrences": 6},
        {"listener": "az", "missing": "ɯ", "replacement": "u", "distance": 1, "n_occurrences": 3},
        {"listener": "tr", "missing": "q", "replacement": "k", "distance": 1, "n_occurrences": 6},
        {"listener": "tr", "missing": "u", "replacement": "a", "distance": 11, "n_occurrences": 4},
        {"listener": "tr", "missing": "ɯ", "replacement": "a", "distance": 10, "n_occurrences": 3},
    ]
    assert rows == expected
    out = capsys.readouterr().out
    assert "vocab missing for fi" in out


def test_build_rows_min_count_filters_rare_segments(tmp_path: Path) -> None:
    args = _make_args(tmp_path)
    args["min_count"] = 10
    _write_vocab(args["checkpoint_dir"], "az", ["a", "k", "u"])
    args["snippet_dir"].mkdir(parents=True)
    (args["snippet_dir"] / "perception_az.txt").write_text("qqqqqq\n", encoding="utf-8")
    assert build_rows(args) == []


# ---------------------------------------------------------------------------
# render_csv / run
# ---------------------------------------------------------------------------


def test_render_csv_exact_output() -> None:
    rows: list[AssimilationRow] = [
        {"listener": "fi", "missing": "ɯ", "replacement": "u", "distance": 1, "n_occurrences": 9},
    ]
    assert render_csv(rows) == ("listener,missing,replacement,distance,n_occurrences\nfi,ɯ,u,1,9\n")


def test_run_writes_csv_and_prints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(tmp_path)
    _write_vocab(args["checkpoint_dir"], "az", ["a", "k", "u"])
    args["snippet_dir"].mkdir(parents=True)
    (args["snippet_dir"] / "perception_az.txt").write_text("qqqqqq\n", encoding="utf-8")

    rows = run(args)

    assert len(rows) == 1
    csv_text = args["output_csv"].read_text(encoding="utf-8")
    assert csv_text == ("listener,missing,replacement,distance,n_occurrences\naz,q,k,1,6\n")
    out = capsys.readouterr().out
    assert "az: q -> k (distance 1, 6 occurrences)" in out
    assert "Wrote 1 substitution(s)" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args == {
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
        "snippet_dir": DEFAULT_SNIPPET_DIR,
        "output_csv": DEFAULT_OUTPUT_CSV,
        "snippet_template": DEFAULT_SNIPPET_TEMPLATE,
        "min_count": DEFAULT_MIN_COUNT,
    }


def test_parse_args_overrides() -> None:
    args = parse_args(["--min-count", "3", "--output-csv", "x.csv"])
    assert args["min_count"] == 3
    assert args["output_csv"] == Path("x.csv")


def _good_namespace() -> argparse.Namespace:
    """Test helper: namespace with all-valid argument types."""
    return argparse.Namespace(
        checkpoint_dir="a",
        snippet_dir="b",
        output_csv="c",
        snippet_template="{lang}",
        min_count=5,
    )


def test_extract_args_rejects_bad_str_field() -> None:
    namespace = _good_namespace()
    namespace.output_csv = 3
    with pytest.raises(TypeError, match="Expected str for --output-csv"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_min_count_type() -> None:
    namespace = _good_namespace()
    namespace.min_count = "5"
    with pytest.raises(TypeError, match="Expected int for --min-count"):
        _extract_args(namespace)


def test_extract_args_rejects_nonpositive_min_count() -> None:
    namespace = _good_namespace()
    namespace.min_count = 0
    with pytest.raises(ValueError, match="--min-count must be >= 1"):
        _extract_args(namespace)


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(tmp_path)
    _write_vocab(args["checkpoint_dir"], "az", ["a", "k", "u"])
    args["snippet_dir"].mkdir(parents=True)
    (args["snippet_dir"] / "perception_az.txt").write_text("qqqqqq\n", encoding="utf-8")
    code = main(
        [
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-csv",
            str(args["output_csv"]),
            "--min-count",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Wrote 1 substitution(s)" in out


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(tmp_path)
    _write_vocab(args["checkpoint_dir"], "az", ["a", "k", "u"])
    args["snippet_dir"].mkdir(parents=True)
    (args["snippet_dir"] / "perception_az.txt").write_text("qqqqqq\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_assimilation",
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-csv",
            str(args["output_csv"]),
            "--min-count",
            "2",
        ],
    )
    monkeypatch.delitem(sys.modules, "scripts.build_assimilation")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.build_assimilation", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
