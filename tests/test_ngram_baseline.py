"""Tests for scripts.ngram_baseline."""

from __future__ import annotations

import argparse
import math
import runpy
import sys
from pathlib import Path

import pytest
from scripts.corpora import CORPUS_TEMPLATE
from scripts.ngram_baseline import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CORPUS_DIR,
    DEFAULT_MAX_TRAIN_CHARS,
    DEFAULT_N_BOOT,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SEED,
    DEFAULT_SNIPPET_DIR,
    FLOOR_MASS,
    LAMBDAS,
    NgramArgs,
    TrigramModel,
    _extract_args,
    main,
    parse_args,
    run,
    score_section_ngram,
)
from scripts.zero_shot_eval import DEFAULT_SNIPPET_TEMPLATE

from char_lstm.data import UNK, save_vocab_json


def _floored(p: float) -> float:
    """Test helper: apply the model's uniform floor to a raw probability."""
    return (1.0 - FLOOR_MASS) * p + FLOOR_MASS / 100


# ---------------------------------------------------------------------------
# TrigramModel
# ---------------------------------------------------------------------------


def test_trigram_model_rejects_too_short_text() -> None:
    with pytest.raises(ValueError, match="too short for trigrams"):
        TrigramModel("ab")


def test_trigram_model_counts_are_exact() -> None:
    model = TrigramModel("abab")
    assert model.unigrams == {"a": 2, "b": 2}
    assert model.bigrams == {("a", "b"): 2, ("b", "a"): 1}
    assert model.trigrams == {("a", "b", "a"): 1, ("b", "a", "b"): 1}
    assert model.bigram_contexts == {"a": 2, "b": 1}
    assert model.trigram_contexts == {("a", "b"): 1, ("b", "a"): 1}
    assert model.n_chars == 4
    assert model.n_types == 2


def test_neg_logp_interpolates_all_orders_exactly() -> None:
    model = TrigramModel("abab")
    # P(a | a,b): p1 = 2/4, p2 = bi(b,a)/ctx(b) = 1/1, p3 = tri(a,b,a)/ctx(a,b) = 1/1
    expected = -math.log(_floored(LAMBDAS[0] * 0.5 + LAMBDAS[1] * 1.0 + LAMBDAS[2] * 1.0))
    assert model.neg_logp("a", "b", "a") == pytest.approx(expected)


def test_neg_logp_unseen_char_gets_floor_probability() -> None:
    model = TrigramModel("abab")
    assert model.neg_logp("a", "b", "z") == pytest.approx(-math.log(FLOOR_MASS / 100))


def test_neg_logp_unseen_contexts_fall_back_to_unigram() -> None:
    model = TrigramModel("abab")
    # 'z' contexts unseen: p2 = p3 = 0, only the unigram term survives.
    expected = -math.log(_floored(LAMBDAS[0] * 0.5))
    assert model.neg_logp("z", "z", "a") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# score_section_ngram
# ---------------------------------------------------------------------------


def test_score_section_ngram_sums_per_position_losses() -> None:
    model = TrigramModel("abababab")
    section = "abab"
    full = score_section_ngram(model, section, [True, True, True])
    expected = (
        model.neg_logp(" ", "a", "b")
        + model.neg_logp("a", "b", "a")
        + model.neg_logp("b", "a", "b")
    )
    assert full["loss_sum"] == pytest.approx(expected)
    assert full["n_scored"] == 3
    assert full["n_total"] == 3


def test_score_section_ngram_respects_mask() -> None:
    model = TrigramModel("abababab")
    partial = score_section_ngram(model, "abab", [False, True, False])
    assert partial["n_scored"] == 1
    assert partial["loss_sum"] == pytest.approx(model.neg_logp("a", "b", "a"))


def test_score_section_ngram_rejects_mask_mismatch() -> None:
    model = TrigramModel("abababab")
    with pytest.raises(ValueError, match="Mask length"):
        score_section_ngram(model, "abab", [True])


# ---------------------------------------------------------------------------
# run (end-to-end)
# ---------------------------------------------------------------------------


def _setup_dirs(tmp_path: Path) -> NgramArgs:
    """Test helper: corpora/vocabs/snippets covering every run() branch.

    az: corpus + vocab + zero-support snippet (all 'z').
    kk: corpus + vocab, no snippet.
    tr: corpus, NO vocab (support falls back to corpus chars), good snippet.
    Other languages: no corpus at all.
    """
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    for lang in ("az", "kk", "tr"):
        (corpus_dir / CORPUS_TEMPLATE.format(lang=lang)).write_text("ab" * 50, encoding="utf-8")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for lang in ("az", "kk"):
        itos: dict[int, str] = dict(enumerate(["a", "b", UNK]))
        save_vocab_json(itos, ckpt / f"{lang}_vocab.json")
    snip = tmp_path / "snip"
    snip.mkdir()
    (snip / "perception_az.txt").write_text("AZERI\n1\n" + "z" * 40 + "\n", encoding="utf-8")
    (snip / "perception_tr.txt").write_text(
        "TURKISH\n1\n" + "ab" * 20 + "\n2\n" + "ba" * 20 + "\n", encoding="utf-8"
    )
    return {
        "corpus_dir": corpus_dir,
        "checkpoint_dir": ckpt,
        "snippet_dir": snip,
        "output_csv": tmp_path / "out" / "ngram.csv",
        "snippet_template": "perception_{lang}.txt",
        "max_train_chars": 1000,
        "n_boot": 10,
        "seed": 0,
    }


def test_run_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _setup_dirs(tmp_path)
    results = run(args)

    # Sources az/kk/tr; only tr is a usable target.
    assert [(r["src"], r["tgt"]) for r in results] == [
        ("az", "tr"),
        ("kk", "tr"),
        ("tr", "tr"),
    ]
    for r in results:
        assert r["mode"] == "trigram-skip"
        assert r["support"] == 1.0
        if r["src"] == r["tgt"]:
            assert r["excess_ce"] == 0.0
            assert (r["excess_lo"], r["excess_hi"]) == (0.0, 0.0)
    # az and kk share training text, vocab, and support: identical scores.
    assert results[0]["ce"] == pytest.approx(results[1]["ce"])

    csv_text = args["output_csv"].read_text(encoding="utf-8")
    assert len(csv_text.strip().splitlines()) == 4  # header + 3 rows

    out = capsys.readouterr().out
    assert "corpus missing for fi" in out
    assert "vocab missing for tr" in out
    assert "snippet missing for kk" in out
    assert "snippet for az has no supported sections" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args == {
        "corpus_dir": DEFAULT_CORPUS_DIR,
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
        "snippet_dir": DEFAULT_SNIPPET_DIR,
        "output_csv": DEFAULT_OUTPUT_CSV,
        "snippet_template": DEFAULT_SNIPPET_TEMPLATE,
        "max_train_chars": DEFAULT_MAX_TRAIN_CHARS,
        "n_boot": DEFAULT_N_BOOT,
        "seed": DEFAULT_SEED,
    }


def test_parse_args_overrides() -> None:
    args = parse_args(["--max-train-chars", "500", "--n-boot", "10", "--seed", "3"])
    assert args["max_train_chars"] == 500
    assert args["n_boot"] == 10
    assert args["seed"] == 3


def _good_namespace() -> argparse.Namespace:
    """Test helper: namespace with all-valid argument types."""
    return argparse.Namespace(
        corpus_dir="a",
        checkpoint_dir="b",
        snippet_dir="c",
        output_csv="d",
        snippet_template="{lang}",
        max_train_chars=100,
        n_boot=10,
        seed=0,
    )


def test_extract_args_rejects_bad_str_field() -> None:
    namespace = _good_namespace()
    namespace.corpus_dir = 1
    with pytest.raises(TypeError, match="Expected str for --corpus-dir"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_int_field() -> None:
    namespace = _good_namespace()
    namespace.max_train_chars = "100"
    with pytest.raises(TypeError, match="Expected int for --max-train-chars"):
        _extract_args(namespace)


def test_extract_args_rejects_nonpositive_n_boot() -> None:
    namespace = _good_namespace()
    namespace.n_boot = 0
    with pytest.raises(ValueError, match="--n-boot must be >= 1"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_seed_type() -> None:
    namespace = _good_namespace()
    namespace.seed = "0"
    with pytest.raises(TypeError, match="Expected int for --seed"):
        _extract_args(namespace)


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _setup_dirs(tmp_path)
    code = main(
        [
            "--corpus-dir",
            str(args["corpus_dir"]),
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-csv",
            str(args["output_csv"]),
            "--n-boot",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Wrote 3 pair(s)" in out


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _setup_dirs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ngram_baseline",
            "--corpus-dir",
            str(args["corpus_dir"]),
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-csv",
            str(args["output_csv"]),
            "--n-boot",
            "5",
        ],
    )
    monkeypatch.delitem(sys.modules, "scripts.ngram_baseline")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.ngram_baseline", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
