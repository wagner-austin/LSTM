"""Tests for scripts.validate_method."""

from __future__ import annotations

import argparse
import json
import random
import runpy
import sys
from pathlib import Path

import pytest
import torch
from scripts.validate_method import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CORPUS_DIR,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_SEED,
    DEFAULT_SLICE_CHARS,
    DEFAULT_SNIPPET_DIR,
    EXPECTED_SIBLINGS,
    ValidateArgs,
    _extract_args,
    branch_gap,
    canonical,
    excess_matrix,
    leaf_sibling_pairs,
    main,
    parse_args,
    run,
    shuffle_text,
    symmetrize,
    upgma,
)
from scripts.zero_shot_eval import load_model_with_vocab

from char_lstm.data import UNK, save_vocab_json
from char_lstm.model import CharLSTM


def _write_tiny_model(checkpoint_dir: Path, lang: str, vocab: list[str], seed: int) -> None:
    """Test helper: build a tiny CharLSTM, save its state dict and vocab.

    Args:
        checkpoint_dir: Directory to write into. Created if missing.
        lang: Language code used to name the output files.
        vocab: List of characters (UNK appended automatically) defining stoi.
        seed: Torch seed so each language gets distinct weights.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chars = [*vocab, UNK]
    itos: dict[int, str] = dict(enumerate(chars))
    save_vocab_json(itos, checkpoint_dir / f"{lang}_vocab.json")
    torch.manual_seed(seed)
    model = CharLSTM(vocab_size=len(chars), embed_dim=4, hidden_dim=8, num_layers=2, dropout=0.0)
    torch.save(model.state_dict(), checkpoint_dir / f"{lang}_best.pt")


# ---------------------------------------------------------------------------
# Distance geometry
# ---------------------------------------------------------------------------


def test_symmetrize_averages_both_directions() -> None:
    excess = {"a": {"a": 0.0, "b": 2.0}, "b": {"a": 4.0, "b": 0.0}}
    assert symmetrize(excess) == {("a", "b"): 3.0}


def test_branch_gap_exact_means_and_ignores_non_branch_langs() -> None:
    dist = {
        ("az", "tr"): 1.0,  # within oghuz
        ("az", "kk"): 2.0,  # cross
        ("kk", "tr"): 4.0,  # cross
        ("az", "fi"): 99.0,  # fi has no branch: ignored
    }
    stats = branch_gap(dist)
    assert stats == {"within": 1.0, "cross": 3.0, "gap": 2.0}


def test_branch_gap_requires_both_pair_kinds() -> None:
    with pytest.raises(ValueError, match="needs both pair kinds"):
        branch_gap({("az", "tr"): 1.0})


def test_upgma_recovers_two_pair_topology() -> None:
    dist = {
        ("az", "tr"): 1.0,
        ("ug", "uz"): 1.0,
        ("az", "ug"): 5.0,
        ("az", "uz"): 5.0,
        ("tr", "ug"): 5.0,
        ("tr", "uz"): 5.0,
    }
    assert canonical(upgma(dist)) == "((az,tr),(ug,uz))"


def test_upgma_two_languages() -> None:
    assert canonical(upgma({("a", "b"): 1.0})) == "(a,b)"


def test_upgma_rejects_fewer_than_two_languages() -> None:
    with pytest.raises(ValueError, match="at least two languages"):
        upgma({})


def test_canonical_is_child_order_independent() -> None:
    assert canonical(("b", "a")) == "(a,b)"
    assert canonical(("b", "a")) == canonical(("a", "b"))


def test_leaf_sibling_pairs_finds_only_cherries() -> None:
    # (kk,ky) and (ug,uz) are cherries; the az-(...) node is not a two-leaf clade.
    tree = (("az", ("kk", "ky")), ("ug", "uz"))
    assert leaf_sibling_pairs(tree) == {("kk", "ky"), ("ug", "uz")}


def test_leaf_sibling_pairs_recovers_three_branch_cherries() -> None:
    tree = (("az", "tr"), (("kk", "ky"), ("ug", "uz")))
    assert leaf_sibling_pairs(tree) == {("az", "tr"), ("kk", "ky"), ("ug", "uz")}


def test_leaf_sibling_pairs_single_leaf_has_none() -> None:
    assert leaf_sibling_pairs("az") == set()


def test_expected_siblings_are_the_three_branches() -> None:
    assert {tuple(sorted(p)) for p in EXPECTED_SIBLINGS} == {
        ("az", "tr"),
        ("kk", "ky"),
        ("ug", "uz"),
    }


def test_shuffle_text_preserves_multiset_deterministically() -> None:
    text = "abcdefgh" * 5
    first = shuffle_text(text, random.Random(0))
    second = shuffle_text(text, random.Random(0))
    assert first == second
    assert sorted(first) == sorted(text)
    assert first != text


# ---------------------------------------------------------------------------
# excess_matrix
# ---------------------------------------------------------------------------


def test_excess_matrix_diagonal_is_zero_and_square(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "az", ["a", "b"], seed=0)
    _write_tiny_model(tmp_path, "tr", ["a", "b"], seed=1)
    models = {
        lang: load_model_with_vocab(tmp_path / f"{lang}_best.pt", tmp_path / f"{lang}_vocab.json")
        for lang in ("az", "tr")
    }
    targets = {"az": ["ab" * 20], "tr": ["ba" * 20]}
    excess = excess_matrix(models, targets)
    assert sorted(excess) == ["az", "tr"]
    assert sorted(excess["az"]) == ["az", "tr"]
    assert excess["az"]["az"] == 0.0
    assert excess["tr"]["tr"] == 0.0


def test_excess_matrix_rejects_mismatched_language_sets(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "az", ["a", "b"], seed=0)
    models = {"az": load_model_with_vocab(tmp_path / "az_best.pt", tmp_path / "az_vocab.json")}
    with pytest.raises(ValueError, match="identical language sets"):
        excess_matrix(models, {"az": ["abab"], "tr": ["abab"]})


# ---------------------------------------------------------------------------
# run (end-to-end with tiny models)
# ---------------------------------------------------------------------------


def _setup_battery(tmp_path: Path, langs: list[str]) -> ValidateArgs:
    """Test helper: full battery fixture for the given fully-equipped langs.

    Every language in ``langs`` gets a model, a 3-section snippet, and a
    corpus. Additionally: uz gets a model with a sectionless snippet and an
    empty corpus (exercising the silent-skip branches), and ky gets a model
    with no snippet and no corpus (exercising the notice branches) -- unless
    those codes are already in ``langs``.
    """
    ckpt = tmp_path / "ckpt"
    snip = tmp_path / "snip"
    corp = tmp_path / "corp"
    snip.mkdir()
    corp.mkdir()
    sections = ["ab" * 20, "ba" * 20, "aabb" * 10]
    for i, lang in enumerate(langs):
        _write_tiny_model(ckpt, lang, ["a", "b"], seed=i)
        body = "\n2\n".join(sections)
        (snip / f"perception_{lang}.txt").write_text(f"LANG\n1\n{body}\n", encoding="utf-8")
        (corp / f"oscar_{lang}_ipa.txt").write_text("ab" * 2000, encoding="utf-8")
    if "uz" not in langs:
        _write_tiny_model(ckpt, "uz", ["a", "b"], seed=90)
        (snip / "perception_uz.txt").write_text("UZBEK\n1\n2\n", encoding="utf-8")
        (corp / "oscar_uz_ipa.txt").write_text("", encoding="utf-8")
    if "ky" not in langs:
        _write_tiny_model(ckpt, "ky", ["a", "b"], seed=91)
    return {
        "checkpoint_dir": ckpt,
        "corpus_dir": corp,
        "snippet_dir": snip,
        "snippet_template": "perception_{lang}.txt",
        "output_json": tmp_path / "out" / "validity.json",
        "seed": 1,
        "slice_chars": 100,
    }


def test_run_partial_language_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _setup_battery(tmp_path, ["az", "tr", "kk"])
    report = run(args)

    # Sibling-clade checks require all six Turkic languages; only gaps are checked.
    assert sorted(report["checks"]) == [
        "heldout_gap_positive",
        "real_gap_positive",
        "shuffled_gap_collapsed",
    ]
    assert report["passed"] == all(report["checks"].values())
    assert report["expected_siblings"] == [["az", "tr"], ["kk", "ky"], ["ug", "uz"]]

    on_disk = json.loads(args["output_json"].read_text(encoding="utf-8"))
    assert on_disk == report

    out = capsys.readouterr().out
    assert "snippet missing for ky" in out
    assert "corpus missing for ky" in out
    assert "Validity battery:" in out


def test_run_full_turkic_set_includes_sibling_checks(tmp_path: Path) -> None:
    args = _setup_battery(tmp_path, ["az", "tr", "kk", "ky", "ug", "uz"])
    report = run(args)
    assert sorted(report["checks"]) == [
        "heldout_gap_positive",
        "heldout_siblings_recovered",
        "real_gap_positive",
        "real_siblings_recovered",
        "shuffled_gap_collapsed",
    ]
    assert report["passed"] == all(report["checks"].values())


def test_main_exit_code_matches_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _setup_battery(tmp_path, ["az", "tr", "kk"])
    code = main(
        [
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--corpus-dir",
            str(args["corpus_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-json",
            str(args["output_json"]),
            "--slice-chars",
            "100",
        ]
    )
    report = json.loads(args["output_json"].read_text(encoding="utf-8"))
    assert code == (0 if report["passed"] else 1)
    out = capsys.readouterr().out
    assert "Validity battery:" in out


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _setup_battery(tmp_path, ["az", "tr", "kk"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_method",
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--corpus-dir",
            str(args["corpus_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-json",
            str(args["output_json"]),
            "--slice-chars",
            "100",
        ],
    )
    monkeypatch.delitem(sys.modules, "scripts.validate_method")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.validate_method", run_name="__main__", alter_sys=True)
    report = json.loads(args["output_json"].read_text(encoding="utf-8"))
    assert excinfo.value.code == (0 if report["passed"] else 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args == {
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
        "corpus_dir": DEFAULT_CORPUS_DIR,
        "snippet_dir": DEFAULT_SNIPPET_DIR,
        "snippet_template": "perception_{lang}.txt",
        "output_json": DEFAULT_OUTPUT_JSON,
        "seed": DEFAULT_SEED,
        "slice_chars": DEFAULT_SLICE_CHARS,
    }


def _good_namespace() -> argparse.Namespace:
    """Test helper: namespace with all-valid argument types."""
    return argparse.Namespace(
        checkpoint_dir="a",
        corpus_dir="b",
        snippet_dir="c",
        snippet_template="{lang}",
        output_json="d",
        seed=1,
        slice_chars=100,
    )


def test_extract_args_rejects_bad_str_field() -> None:
    namespace = _good_namespace()
    namespace.corpus_dir = 9
    with pytest.raises(TypeError, match="Expected str for --corpus-dir"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_seed_type() -> None:
    namespace = _good_namespace()
    namespace.seed = "1"
    with pytest.raises(TypeError, match="Expected int for --seed"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_slice_chars_type() -> None:
    namespace = _good_namespace()
    namespace.slice_chars = 5.0
    with pytest.raises(TypeError, match="Expected int for --slice-chars"):
        _extract_args(namespace)


def test_extract_args_rejects_too_small_slice_chars() -> None:
    namespace = _good_namespace()
    namespace.slice_chars = 1
    with pytest.raises(ValueError, match="--slice-chars must be >= 2"):
        _extract_args(namespace)
