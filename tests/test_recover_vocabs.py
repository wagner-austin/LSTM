"""Tests for scripts.recover_vocabs."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.recover_vocabs import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CORPUS_DIR,
    MAX_CHARS,
    _build_arg_parser,
    _extract_args,
    build_union_vocab,
    coverage_of,
    main,
    parse_args,
    reconstruct_vocab,
    render_coverage_csv,
    run,
)

from char_lstm.corpora import LANGS
from char_lstm.data import UNK, load_vocab_json

# ---------------------------------------------------------------------------
# reconstruct_vocab
# ---------------------------------------------------------------------------


def test_reconstruct_vocab_returns_sorted_chars_plus_unk(tmp_path: Path) -> None:
    """reconstruct_vocab returns sorted unique chars with UNK appended at end."""
    corpus = tmp_path / "oscar_xx_ipa.txt"
    corpus.write_text("cab", encoding="utf-8")
    stoi, itos, vocab_size = reconstruct_vocab(corpus)
    assert vocab_size == 4
    assert stoi == {"a": 0, "b": 1, "c": 2, UNK: 3}
    assert itos == {0: "a", 1: "b", 2: "c", 3: UNK}


def test_reconstruct_vocab_truncates_to_max_chars(tmp_path: Path) -> None:
    """reconstruct_vocab honors max_chars truncation, ignoring later chars."""
    corpus = tmp_path / "oscar_xx_ipa.txt"
    corpus.write_text("aaazzz", encoding="utf-8")
    stoi, _itos, vocab_size = reconstruct_vocab(corpus, max_chars=3)
    assert vocab_size == 2
    assert "a" in stoi
    assert "z" not in stoi


# ---------------------------------------------------------------------------
# build_union_vocab
# ---------------------------------------------------------------------------


def test_build_union_vocab_empty_input_yields_only_unk() -> None:
    """Empty per_lang_stoi yields a vocab containing only UNK."""
    stoi, itos, size = build_union_vocab({})
    assert size == 1
    assert stoi == {UNK: 0}
    assert itos == {0: UNK}


def test_build_union_vocab_excludes_unk_from_source_stoi() -> None:
    """UNK present in source stoi is excluded so it appears exactly once."""
    per_lang = {
        "x": {"a": 0, "b": 1, UNK: 2},
        "y": {"b": 0, "c": 1, UNK: 2},
    }
    stoi, itos, size = build_union_vocab(per_lang)
    assert size == 4
    assert stoi == {"a": 0, "b": 1, "c": 2, UNK: 3}
    assert itos[size - 1] == UNK


def test_build_union_vocab_handles_input_without_unk() -> None:
    """Source stoi without UNK still yields a union with UNK at the end."""
    per_lang = {"x": {"a": 0, "b": 1}}
    stoi, _itos, size = build_union_vocab(per_lang)
    assert size == 3
    assert stoi[UNK] == 2


def test_build_union_vocab_uses_custom_unk() -> None:
    """A custom UNK token is appended at the end of the union vocab."""
    per_lang = {"x": {"a": 0, "b": 1}}
    custom = "[X]"
    stoi, _itos, size = build_union_vocab(per_lang, unk=custom)
    assert size == 3
    assert stoi[custom] == 2


# ---------------------------------------------------------------------------
# coverage_of
# ---------------------------------------------------------------------------


def test_coverage_of_returns_one_for_identical_sets() -> None:
    """Identical character sets yield coverage of 1.0."""
    chars = {"a", "b", "c"}
    assert coverage_of(chars, chars) == 1.0


def test_coverage_of_returns_one_for_empty_target_after_unk_removal() -> None:
    """Target containing only UNK is treated as vacuously fully covered."""
    src: set[str] = {"a"}
    tgt = {UNK}
    assert coverage_of(src, tgt) == 1.0


def test_coverage_of_returns_zero_for_disjoint_sets() -> None:
    """Disjoint character sets yield coverage of 0.0."""
    assert coverage_of({"a", "b"}, {"c", "d"}) == 0.0


def test_coverage_of_partial_overlap_is_fraction() -> None:
    """Partial overlap yields the correct fraction of target chars present."""
    src = {"a", "b"}
    tgt = {"a", "c", "d"}
    assert coverage_of(src, tgt) == pytest.approx(1 / 3)


def test_coverage_of_excludes_unk_from_both_sides() -> None:
    """UNK in either side does not influence the coverage fraction."""
    src = {"a", "b", UNK}
    tgt = {"a", "b", UNK}
    assert coverage_of(src, tgt) == 1.0


# ---------------------------------------------------------------------------
# render_coverage_csv
# ---------------------------------------------------------------------------


def test_render_coverage_csv_produces_expected_layout() -> None:
    """Coverage CSV header lists languages alphabetically and rows match."""
    per_lang = {
        "y": {"a": 0, "c": 1, UNK: 2},
        "x": {"a": 0, "b": 1, UNK: 2},
    }
    csv = render_coverage_csv(per_lang)
    expected = "src,x,y\nx,1.0000,0.5000\ny,0.5000,1.0000\n"
    assert csv == expected


# ---------------------------------------------------------------------------
# _build_arg_parser / _extract_args / parse_args
# ---------------------------------------------------------------------------


def test_build_arg_parser_defaults_match_module_constants() -> None:
    """Default values from the parser match the module-level constants."""
    parser = _build_arg_parser()
    namespace = parser.parse_args([])
    assert namespace.corpus_dir == str(DEFAULT_CORPUS_DIR)
    assert namespace.checkpoint_dir == str(DEFAULT_CHECKPOINT_DIR)


def test_extract_args_returns_paths_for_valid_strings() -> None:
    """_extract_args turns valid string args into Path values."""
    namespace = argparse.Namespace(corpus_dir="some/corpus", checkpoint_dir="some/ckpt")
    args = _extract_args(namespace)
    assert args["corpus_dir"] == Path("some/corpus")
    assert args["checkpoint_dir"] == Path("some/ckpt")


def test_extract_args_rejects_non_string_corpus_dir() -> None:
    """_extract_args raises TypeError if corpus_dir is not a string."""
    namespace = argparse.Namespace(corpus_dir=123, checkpoint_dir="ckpt")
    with pytest.raises(TypeError, match="Expected str for --corpus-dir"):
        _extract_args(namespace)


def test_extract_args_rejects_non_string_checkpoint_dir() -> None:
    """_extract_args raises TypeError if checkpoint_dir is not a string."""
    namespace = argparse.Namespace(corpus_dir="corpus", checkpoint_dir=42)
    with pytest.raises(TypeError, match="Expected str for --checkpoint-dir"):
        _extract_args(namespace)


def test_parse_args_with_explicit_argv_returns_typed_paths() -> None:
    """parse_args produces a RecoveryArgs with both fields as Path."""
    args = parse_args(["--corpus-dir", "a", "--checkpoint-dir", "b"])
    assert args["corpus_dir"] == Path("a")
    assert args["checkpoint_dir"] == Path("b")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _write_corpus(corpus_dir: Path, lang: str, text: str) -> None:
    """Test helper: write a corpus file for a single language."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"oscar_{lang}_ipa.txt").write_text(text, encoding="utf-8")


def test_run_writes_per_language_union_and_coverage(tmp_path: Path) -> None:
    """run writes per-lang vocabs, the union, and the coverage CSV."""
    corpus_dir = tmp_path / "corpora"
    ckpt_dir = tmp_path / "ckpts"
    _write_corpus(corpus_dir, "az", "abc")
    _write_corpus(corpus_dir, "fi", "bcd")

    sizes = run(corpus_dir, ckpt_dir, langs=("az", "fi"))

    assert sizes == {"az": 4, "fi": 4}
    az_stoi, _, az_size, _ = load_vocab_json(ckpt_dir / "az_vocab.json")
    fi_stoi, _, fi_size, _ = load_vocab_json(ckpt_dir / "fi_vocab.json")
    assert az_size == 4
    assert fi_size == 4
    assert set(az_stoi.keys()) == {"a", "b", "c", UNK}
    assert set(fi_stoi.keys()) == {"b", "c", "d", UNK}

    union_stoi, _, union_size, _ = load_vocab_json(ckpt_dir / "union_vocab.json")
    assert union_size == 5
    assert set(union_stoi.keys()) == {"a", "b", "c", "d", UNK}

    csv_text = (ckpt_dir / "vocab_coverage.csv").read_text(encoding="utf-8")
    assert csv_text.startswith("src,az,fi\n")
    assert "az,1.0000," in csv_text
    assert "fi," in csv_text


def test_run_skips_missing_corpora(tmp_path: Path) -> None:
    """run skips languages whose corpus file is missing."""
    corpus_dir = tmp_path / "corpora"
    ckpt_dir = tmp_path / "ckpts"
    _write_corpus(corpus_dir, "az", "abc")
    sizes = run(corpus_dir, ckpt_dir, langs=("az", "fi"))
    assert sizes == {"az": 4}
    assert (ckpt_dir / "az_vocab.json").exists()
    assert not (ckpt_dir / "fi_vocab.json").exists()


def test_run_with_no_corpora_writes_nothing_extra(tmp_path: Path) -> None:
    """run returns empty mapping and writes no union/coverage when empty."""
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    ckpt_dir = tmp_path / "ckpts"
    sizes = run(corpus_dir, ckpt_dir, langs=("az", "fi"))
    assert sizes == {}
    assert not (ckpt_dir / "union_vocab.json").exists()
    assert not (ckpt_dir / "vocab_coverage.csv").exists()


# ---------------------------------------------------------------------------
# main / __main__ block
# ---------------------------------------------------------------------------


def test_main_returns_zero_and_writes_outputs(tmp_path: Path) -> None:
    """main returns 0 and writes the same outputs run() would write."""
    corpus_dir = tmp_path / "corpora"
    ckpt_dir = tmp_path / "ckpts"
    _write_corpus(corpus_dir, "az", "abc")
    _write_corpus(corpus_dir, "fi", "bcd")

    code = main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )

    assert code == 0
    union = json.loads((ckpt_dir / "union_vocab.json").read_text(encoding="utf-8"))
    assert union["unk"] == UNK


def test_invocation_as_script_invokes_main(tmp_path: Path) -> None:
    """Running ``python -m scripts.recover_vocabs`` invokes main."""
    corpus_dir = tmp_path / "corpora"
    ckpt_dir = tmp_path / "ckpts"
    _write_corpus(corpus_dir, "az", "abc")

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.recover_vocabs",
            "--corpus-dir",
            str(corpus_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert (ckpt_dir / "az_vocab.json").exists()


def test_run_path_invokes_main_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runpy.run_path exercises the ``if __name__ == '__main__'`` block.

    Uses ``run_path`` rather than ``run_module`` so the script does not get
    re-imported through ``sys.modules`` after the test file's own import,
    avoiding the runpy double-import RuntimeWarning.
    """
    corpus_dir = tmp_path / "corpora"
    ckpt_dir = tmp_path / "ckpts"
    _write_corpus(corpus_dir, "az", "abc")
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "recover_vocabs.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--corpus-dir",
            str(corpus_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(script_path), run_name="__main__")
    assert excinfo.value.code == 0
    assert (ckpt_dir / "az_vocab.json").exists()


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_module_constants_are_consistent() -> None:
    """Module-level constants match the documented contract."""
    assert LANGS == ("az", "fi", "kk", "ky", "ru", "tr", "ug", "uz")
    assert MAX_CHARS == 10_000_000
    assert Path("corpora_raw") == DEFAULT_CORPUS_DIR
    assert Path("checkpoints") == DEFAULT_CHECKPOINT_DIR
