"""Tests for scripts.clean_corpus."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import pytest
from scripts.clean_corpus import (
    CORPUS_TEMPLATE,
    DEFAULT_INPUT_DIR,
    DEFAULT_MIN_IPA_RATIO,
    DEFAULT_MIN_LINE_CHARS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SNIPPET_DIR,
    DEFAULT_SNIPPET_OUTPUT_DIR,
    DEFAULT_SYMBOL_MAP,
    LANGS,
    SNIPPET_TEMPLATE,
    CleanArgs,
    _extract_args,
    apply_symbol_map,
    clean_corpora,
    clean_lines,
    clean_snippets,
    ipa_ratio,
    load_symbol_map,
    main,
    parse_args,
    sanitize_line,
    truncate_to_budget,
)

LINE_40 = "ɑ" * 40
LINE_50 = "æ" * 50
LINE_60 = "ʒɑŋɑləqtɑrʒɑŋɑləqtɑrʒɑŋɑləqtɑrʒɑŋɑləqtɑrʒɑŋɑləqtɑrʒɑŋɑləqtɑr"
SHORT_LINE = "ʒɑŋɑləqtɑr"
JUNK_LINE = "微" * 40

MAP_CSV = (
    "action,scope,from,to,verdict,rationale,citation\n"
    "merge,all,ʧ,t͡ʃ,N,affricate ligature,IPA\n"
    "merge,tr,a,ɑ,N,low vowel,JIPA\n"
    "merge,ug,ʔ,,N,pipeline artifact,n/a\n"
    "keep,uz,ɔ,ɔ,R,real contrast,JIPA\n"
)


def _write_map(tmp_path: Path) -> Path:
    """Test helper: write the standard symbol-map CSV fixture."""
    path = tmp_path / "symbol_map.csv"
    path.write_text(MAP_CSV, encoding="utf-8")
    return path


def _write_corpora(input_dir: Path, lines_by_lang: dict[str, list[str]]) -> None:
    """Test helper: write one corpus file per language."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for lang, lines in lines_by_lang.items():
        path = input_dir / CORPUS_TEMPLATE.format(lang=lang)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_args(tmp_path: Path) -> CleanArgs:
    """Test helper: CleanArgs pointing at tmp_path locations."""
    return {
        "input_dir": tmp_path / "raw",
        "output_dir": tmp_path / "clean",
        "symbol_map": _write_map(tmp_path),
        "snippet_dir": tmp_path / "snippets",
        "snippet_output_dir": tmp_path / "snippets_clean",
        "min_line_chars": 30,
        "min_ipa_ratio": 0.95,
    }


# ---------------------------------------------------------------------------
# Symbol map
# ---------------------------------------------------------------------------


def test_load_symbol_map_scopes_and_actions(tmp_path: Path) -> None:
    mapping = load_symbol_map(_write_map(tmp_path))
    assert mapping["ky"]["ʧ"] == "t͡ʃ"
    assert mapping["tr"] == {"ʧ": "t͡ʃ", "a": "ɑ"}
    assert mapping["ug"] == {"ʧ": "t͡ʃ", "ʔ": ""}
    assert mapping["uz"] == {"ʧ": "t͡ʃ"}  # keep row produces no substitution


def test_load_symbol_map_rejects_empty_from(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "action,scope,from,to,verdict,rationale,citation\nmerge,all,,x,N,r,c\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty 'from' field"):
        load_symbol_map(path)


def test_load_symbol_map_rejects_unknown_language(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "action,scope,from,to,verdict,rationale,citation\nmerge,zz,ʧ,t,N,r,c\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown language 'zz'"):
        load_symbol_map(path)


def test_repo_symbol_map_loads_with_decided_rows() -> None:
    mapping = load_symbol_map(Path("data/symbol_map.csv"))
    assert mapping["ky"]["ʧ"] == "t͡ʃ"
    assert mapping["tr"]["a"] == "ɑ"
    assert mapping["tr"]["ɾ"] == "r"
    assert mapping["ug"]["ʔ"] == ""
    assert mapping["ug"]["،"] == ","


def test_apply_symbol_map_replaces_and_deletes() -> None:
    subs = {"ʧ": "t͡ʃ", "ʔ": ""}
    assert apply_symbol_map("ʧɑʔbiri", subs) == "t͡ʃɑbiri"
    assert apply_symbol_map("unchanged", {}) == "unchanged"


# ---------------------------------------------------------------------------
# Line filtering
# ---------------------------------------------------------------------------


def test_ipa_ratio_values() -> None:
    assert ipa_ratio(LINE_40) == 1.0
    assert ipa_ratio(JUNK_LINE) == 0.0
    assert ipa_ratio("ɑɑ微微") == 0.5


def test_clean_lines_filters_and_stats() -> None:
    lines = [LINE_40, LINE_40, SHORT_LINE, JUNK_LINE, LINE_50]
    kept, stats = clean_lines(lines, {}, 30, 0.95)
    assert kept == [LINE_40, LINE_50]
    assert stats == {
        "lines_in": 5,
        "dropped_duplicate": 1,
        "dropped_short": 1,
        "dropped_low_ipa": 1,
        "lines_kept": 2,
        "chars_kept": 41 + 51,
        "chars_written": 0,
    }


def test_sanitize_line_replaces_strays_and_collapses_whitespace() -> None:
    assert sanitize_line("ɑ" * 39 + "微") == "ɑ" * 39
    assert sanitize_line("ɑɑ 微 ɑɑ") == "ɑɑ ɑɑ"
    assert sanitize_line("ɑ\t\tɑ") == "ɑ ɑ"


def test_clean_lines_sanitizes_residual_stray_chars() -> None:
    kept, stats = clean_lines(["ɑ" * 39 + "微"], {}, 30, 0.95)
    assert kept == ["ɑ" * 39]
    assert stats["dropped_low_ipa"] == 0


def test_clean_lines_drops_line_that_collapses_below_minimum() -> None:
    kept, stats = clean_lines(["ɑ" + " " * 38 + "ɑ"], {}, 30, 0.95)
    assert kept == []
    assert stats["dropped_short"] == 1


def test_clean_lines_dedups_on_sanitized_form() -> None:
    kept, stats = clean_lines(["ɑ" * 39 + "微", "微" + "ɑ" * 39], {}, 30, 0.95)
    assert kept == ["ɑ" * 39]
    assert stats["dropped_duplicate"] == 1


def test_clean_lines_applies_map_before_filtering() -> None:
    kept, stats = clean_lines(["a" * 40], {"a": "ɑ"}, 30, 0.95)
    assert kept == ["ɑ" * 40]
    assert stats["lines_kept"] == 1


def test_truncate_to_budget_stops_at_boundary() -> None:
    lines = ["x" * 60, "y" * 60]
    assert truncate_to_budget(lines, 92) == ["x" * 60]
    assert truncate_to_budget(lines, 122) == lines
    assert truncate_to_budget([], 100) == []


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_clean_corpora_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(tmp_path)
    lines_by_lang = {lang: [LINE_60, LINE_60[::-1], LINE_60.upper()] for lang in LANGS}
    lines_by_lang["az"] = [LINE_40, LINE_40, SHORT_LINE, JUNK_LINE, LINE_50]
    lines_by_lang["tr"] = ["a" * 60, "b" + "a" * 59, "c" + "a" * 59]
    _write_corpora(args["input_dir"], lines_by_lang)

    stats = clean_corpora(args)

    # az is the bottleneck: 41 + 51 = 92 chars survive.
    assert stats["az"] == {
        "lines_in": 5,
        "dropped_duplicate": 1,
        "dropped_short": 1,
        "dropped_low_ipa": 1,
        "lines_kept": 2,
        "chars_kept": 92,
        "chars_written": 92,
    }
    az_out = (args["output_dir"] / CORPUS_TEMPLATE.format(lang="az")).read_text(encoding="utf-8")
    assert az_out == LINE_40 + "\n" + LINE_50 + "\n"

    # tr had its 'a' chars mapped to 'ɑ' and is truncated to one line (61 <= 92).
    tr_out = (args["output_dir"] / CORPUS_TEMPLATE.format(lang="tr")).read_text(encoding="utf-8")
    assert tr_out == "ɑ" * 60 + "\n"
    assert stats["tr"]["chars_written"] == 61

    manifest = json.loads(
        (args["output_dir"] / "cleaning_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config"]["equalized_char_budget"] == 92
    assert manifest["stats"]["az"]["chars_written"] == 92
    out = capsys.readouterr().out
    assert "az: 5 lines -> 2 kept" in out


def test_clean_corpora_all_junk_writes_empty_files(tmp_path: Path) -> None:
    args = _make_args(tmp_path)
    _write_corpora(args["input_dir"], {lang: [JUNK_LINE, SHORT_LINE] for lang in LANGS})
    stats = clean_corpora(args)
    assert stats["kk"]["chars_written"] == 0
    kk_out = (args["output_dir"] / CORPUS_TEMPLATE.format(lang="kk")).read_text(encoding="utf-8")
    assert kk_out == ""


def test_clean_snippets_converts_and_skips_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(tmp_path)
    args["snippet_dir"].mkdir(parents=True)
    snippet = args["snippet_dir"] / SNIPPET_TEMPLATE.format(lang="ky")
    snippet.write_text("KYRGYZ\nʧɑʧ\n1\nʧɑʧ ʧɑʧ\n", encoding="utf-8")

    converted = clean_snippets(args)

    assert converted == ["ky"]
    out_text = (args["snippet_output_dir"] / SNIPPET_TEMPLATE.format(lang="ky")).read_text(
        encoding="utf-8"
    )
    assert out_text == "KYRGYZ\nt͡ʃɑt͡ʃ\n1\nt͡ʃɑt͡ʃ t͡ʃɑt͡ʃ\n"
    out = capsys.readouterr().out
    assert "snippet missing for az" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args == {
        "input_dir": DEFAULT_INPUT_DIR,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "symbol_map": DEFAULT_SYMBOL_MAP,
        "snippet_dir": DEFAULT_SNIPPET_DIR,
        "snippet_output_dir": DEFAULT_SNIPPET_OUTPUT_DIR,
        "min_line_chars": DEFAULT_MIN_LINE_CHARS,
        "min_ipa_ratio": DEFAULT_MIN_IPA_RATIO,
    }


def test_parse_args_overrides(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--input-dir",
            str(tmp_path / "in"),
            "--min-line-chars",
            "10",
            "--min-ipa-ratio",
            "0.5",
        ]
    )
    assert args["input_dir"] == tmp_path / "in"
    assert args["min_line_chars"] == 10
    assert args["min_ipa_ratio"] == 0.5


def _good_namespace() -> argparse.Namespace:
    """Test helper: namespace with all-valid argument types."""
    return argparse.Namespace(
        input_dir="a",
        output_dir="b",
        symbol_map="c",
        snippet_dir="d",
        snippet_output_dir="e",
        min_line_chars=30,
        min_ipa_ratio=0.95,
    )


def test_extract_args_rejects_bad_str_field() -> None:
    namespace = _good_namespace()
    namespace.input_dir = 123
    with pytest.raises(TypeError, match="Expected str for --input-dir"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_int_field() -> None:
    namespace = _good_namespace()
    namespace.min_line_chars = "30"
    with pytest.raises(TypeError, match="Expected int for --min-line-chars"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_float_field() -> None:
    namespace = _good_namespace()
    namespace.min_ipa_ratio = 1
    with pytest.raises(TypeError, match="Expected float for --min-ipa-ratio"):
        _extract_args(namespace)


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(tmp_path)
    _write_corpora(args["input_dir"], {lang: [LINE_60, LINE_50] for lang in LANGS})
    args["snippet_dir"].mkdir(parents=True)
    (args["snippet_dir"] / SNIPPET_TEMPLATE.format(lang="az")).write_text(
        "ɑzeɾɯ\n1\nʧɑʧ\n", encoding="utf-8"
    )
    code = main(
        [
            "--input-dir",
            str(args["input_dir"]),
            "--output-dir",
            str(args["output_dir"]),
            "--symbol-map",
            str(args["symbol_map"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--snippet-output-dir",
            str(args["snippet_output_dir"]),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Done: snippets converted for az" in out
    az_corpus = (args["output_dir"] / CORPUS_TEMPLATE.format(lang="az")).read_text(encoding="utf-8")
    assert az_corpus == LINE_60 + "\n" + LINE_50 + "\n"


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(tmp_path)
    _write_corpora(args["input_dir"], {lang: [LINE_60] for lang in LANGS})
    args["snippet_dir"].mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_corpus",
            "--input-dir",
            str(args["input_dir"]),
            "--output-dir",
            str(args["output_dir"]),
            "--symbol-map",
            str(args["symbol_map"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--snippet-output-dir",
            str(args["snippet_output_dir"]),
        ],
    )
    monkeypatch.delitem(sys.modules, "scripts.clean_corpus")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.clean_corpus", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
