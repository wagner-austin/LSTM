"""Tests for scripts.zero_shot_eval."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pytest
import torch
from scripts.zero_shot_eval import (
    CSV_HEADER,
    DEFAULT_ASSIMILATION_CSV,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_N_BOOT,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SEED,
    DEFAULT_SNIPPET_DIR,
    DEFAULT_SNIPPET_TEMPLATE,
    OOV_MODES,
    EvalArgs,
    PairResult,
    SectionScore,
    _extract_args,
    bootstrap_excess,
    ce_from_scores,
    common_support_mask,
    infer_num_layers,
    load_assimilation_map,
    load_model_with_vocab,
    main,
    parse_args,
    parse_sections,
    render_results_csv,
    run,
    score_section,
    snippet_path,
)

from char_lstm.data import UNK, save_vocab_json
from char_lstm.model import CharLSTM

# Tiny architecture used across tests for fast model construction.
TEST_VOCAB_SIZE = 5  # 4 chars + UNK
TEST_EMBED = 4
TEST_HIDDEN = 8


def _write_tiny_model(checkpoint_dir: Path, lang: str, vocab: list[str]) -> None:
    """Test helper: build a tiny CharLSTM, save its state dict and vocab.

    Args:
        checkpoint_dir: Directory to write into. Created if missing.
        lang: Language code used to name the output files.
        vocab: List of characters (UNK appended automatically) defining stoi.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chars = [*vocab, UNK]
    itos: dict[int, str] = dict(enumerate(chars))
    save_vocab_json(itos, checkpoint_dir / f"{lang}_vocab.json")

    torch.manual_seed(0)
    model = CharLSTM(
        vocab_size=len(chars),
        embed_dim=TEST_EMBED,
        hidden_dim=TEST_HIDDEN,
        num_layers=2,
        dropout=0.0,
    )
    torch.save(model.state_dict(), checkpoint_dir / f"{lang}_best.pt")


# ---------------------------------------------------------------------------
# infer_num_layers / load_model_with_vocab
# ---------------------------------------------------------------------------


def test_infer_num_layers_returns_one_for_single_layer() -> None:
    state = {"lstm.weight_ih_l0": torch.zeros(4)}
    assert infer_num_layers(state) == 1


def test_infer_num_layers_returns_max_index_plus_one() -> None:
    state = {
        "lstm.weight_ih_l0": torch.zeros(4),
        "lstm.weight_ih_l1": torch.zeros(4),
        "lstm.weight_ih_l2": torch.zeros(4),
    }
    assert infer_num_layers(state) == 3


def test_infer_num_layers_raises_when_no_lstm_keys() -> None:
    state = {"linear.weight": torch.zeros(4)}
    with pytest.raises(ValueError, match=r"No 'lstm\.weight_ih_l"):
        infer_num_layers(state)


def test_load_model_with_vocab_round_trips_state(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "x", ["a", "b", "c", "d"])
    loaded = load_model_with_vocab(tmp_path / "x_best.pt", tmp_path / "x_vocab.json")
    assert loaded["vocab_size"] == TEST_VOCAB_SIZE
    assert loaded["stoi"][UNK] == TEST_VOCAB_SIZE - 1
    assert loaded["model"].embedding.weight.shape == (TEST_VOCAB_SIZE, TEST_EMBED)


def test_load_model_with_vocab_raises_on_size_mismatch(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "x", ["a", "b", "c", "d"])
    smaller_itos: dict[int, str] = {0: "a", 1: "b", 2: UNK}
    save_vocab_json(smaller_itos, tmp_path / "x_vocab.json")
    with pytest.raises(ValueError, match="Vocab/checkpoint mismatch"):
        load_model_with_vocab(tmp_path / "x_best.pt", tmp_path / "x_vocab.json")


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


def test_parse_sections_full_structure() -> None:
    text = (
        "KAZAKH\n"  # language-name first line: dropped
        "TEXT 1: child athletes\n"  # header line: dropped
        "1\n"
        + "ab" * 20
        + "\n\n2\n"
        + "cd" * 20
        + "\nshort title\n"  # stray short line right before a "1": dropped
        + "1\n"
        + "ef" * 20
        + "\n"
    )
    assert parse_sections(text) == ["ab" * 20, "cd" * 20, "ef" * 20]


def test_parse_sections_drops_short_sections_and_preamble() -> None:
    # A long non-header line before any marker has no open section: dropped.
    text = "LANG\n" + "z" * 50 + "\n1\nshort\n2\n" + "gh" * 20 + "\n"
    assert parse_sections(text) == ["gh" * 20]


def test_parse_sections_joins_multiline_sections() -> None:
    text = "LANG\n1\n" + "a" * 25 + "\n" + "b" * 25 + "\n"
    assert parse_sections(text) == ["a" * 25 + " " + "b" * 25]


def test_parse_sections_empty_marker_runs_produce_nothing() -> None:
    assert parse_sections("LANG\n1\n2\n3\n") == []


# ---------------------------------------------------------------------------
# snippet_path
# ---------------------------------------------------------------------------


def test_snippet_path_substitutes_language() -> None:
    path = snippet_path(Path("d"), "perception_{lang}.txt", "kk")
    assert path == Path("d") / "perception_kk.txt"


def test_snippet_path_rejects_template_without_placeholder() -> None:
    with pytest.raises(ValueError, match="must contain"):
        snippet_path(Path("d"), "perception.txt", "kk")


# ---------------------------------------------------------------------------
# load_assimilation_map
# ---------------------------------------------------------------------------


def test_load_assimilation_map_builds_per_listener_maps(tmp_path: Path) -> None:
    path = tmp_path / "assim.csv"
    path.write_text(
        "listener,missing,replacement,rationale\ntr,q,k,uvular to velar\nfi,w,v,extra\n",
        encoding="utf-8",
    )
    mapping = load_assimilation_map(path)
    assert mapping["tr"] == {"q": "k"}
    assert mapping["fi"] == {"w": "v"}
    assert mapping["kk"] == {}


def test_load_assimilation_map_rejects_empty_field(tmp_path: Path) -> None:
    path = tmp_path / "assim.csv"
    path.write_text("listener,missing,replacement\ntr,q,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty field"):
        load_assimilation_map(path)


def test_load_assimilation_map_rejects_unknown_listener(tmp_path: Path) -> None:
    path = tmp_path / "assim.csv"
    path.write_text("listener,missing,replacement\nzz,q,k\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown listener"):
        load_assimilation_map(path)


# ---------------------------------------------------------------------------
# common_support_mask / score_section
# ---------------------------------------------------------------------------


def test_common_support_mask_checks_every_vocab() -> None:
    vocabs = [{"a", "b"}, {"a", "b", "q"}]
    assert common_support_mask("abqb", vocabs) == [True, False, True]


def test_score_section_mask_partition_sums_to_full(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "x", ["a", "b", "c", "d"])
    loaded = load_model_with_vocab(tmp_path / "x_best.pt", tmp_path / "x_vocab.json")
    section = "abcdabcd"
    full = score_section(loaded, section, None)
    mask = [True, False, True, False, True, False, True]
    inverse = [not m for m in mask]
    part_a = score_section(loaded, section, mask)
    part_b = score_section(loaded, section, inverse)
    assert full["n_scored"] == 7
    assert full["n_total"] == 7
    assert part_a["n_scored"] == 4
    assert part_b["n_scored"] == 3
    assert part_a["loss_sum"] + part_b["loss_sum"] == pytest.approx(full["loss_sum"])


def test_score_section_assimilated_text_scores_as_native(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "x", ["a", "b", "c", "d"])
    loaded = load_model_with_vocab(tmp_path / "x_best.pt", tmp_path / "x_vocab.json")
    native = score_section(loaded, "abab", None)
    substituted = score_section(loaded, "azaz".replace("z", "b"), None)
    assert substituted["loss_sum"] == pytest.approx(native["loss_sum"])
    assert substituted["n_scored"] == native["n_scored"]


def test_score_section_raises_for_short_section(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "x", ["a", "b", "c", "d"])
    loaded = load_model_with_vocab(tmp_path / "x_best.pt", tmp_path / "x_vocab.json")
    with pytest.raises(ValueError, match="too short"):
        score_section(loaded, "a", None)


def test_score_section_raises_on_mask_length_mismatch(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path, "x", ["a", "b", "c", "d"])
    loaded = load_model_with_vocab(tmp_path / "x_best.pt", tmp_path / "x_vocab.json")
    with pytest.raises(ValueError, match="Mask length"):
        score_section(loaded, "abcd", [True])


# ---------------------------------------------------------------------------
# ce_from_scores / bootstrap_excess
# ---------------------------------------------------------------------------


def _score(loss_sum: float, n_scored: int, n_total: int) -> SectionScore:
    """Test helper: literal SectionScore."""
    return {"loss_sum": loss_sum, "n_scored": n_scored, "n_total": n_total}


def test_ce_from_scores_pools_loss_over_positions() -> None:
    scores = [_score(2.0, 2, 2), _score(4.0, 2, 4)]
    assert ce_from_scores(scores) == 1.5
    assert ce_from_scores(scores, [1, 1]) == 2.0


def test_ce_from_scores_raises_on_zero_positions() -> None:
    with pytest.raises(ValueError, match="No scored positions"):
        ce_from_scores([_score(0.0, 0, 4)])


def test_bootstrap_excess_is_zero_for_identical_scores() -> None:
    scores = [_score(1.0, 1, 1), _score(3.0, 1, 1)]
    assert bootstrap_excess(scores, scores, 50, 0) == (0.0, 0.0)


def test_bootstrap_excess_recovers_constant_offset_exactly() -> None:
    self_scores = [_score(1.0, 1, 1), _score(3.0, 1, 1)]
    pair_scores = [_score(2.0, 1, 1), _score(4.0, 1, 1)]
    lo, hi = bootstrap_excess(pair_scores, self_scores, 200, 0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_excess_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        bootstrap_excess([_score(1.0, 1, 1)], [], 10, 0)


# ---------------------------------------------------------------------------
# render_results_csv
# ---------------------------------------------------------------------------


def test_render_results_csv_exact_row() -> None:
    result: PairResult = {
        "src": "az",
        "tgt": "tr",
        "mode": "skip",
        "ce": 3.5,
        "self_ce": 1.25,
        "excess_ce": 2.25,
        "excess_lo": 2.0,
        "excess_hi": 2.5,
        "support": 0.75,
        "n_scored": 120,
    }
    expected = (
        CSV_HEADER + "\naz,tr,skip,3.500000,1.250000,2.250000,2.000000,2.500000,0.750000,120\n"
    )
    assert render_results_csv([result]) == expected


# ---------------------------------------------------------------------------
# run (end-to-end with tiny models)
# ---------------------------------------------------------------------------

AB_SECTION = "ab" * 20
AD_SECTION = "ad" * 20
C_SECTION = "c" * 40
AQ_SECTION = "aq" * 20


def _setup_eval_dirs(tmp_path: Path) -> EvalArgs:
    """Test helper: tiny checkpoints + snippets covering every run() branch.

    Models: az (a,b,c,d), tr (a,b,q,d), uz (a,b,c,d,q superset).
    Snippets: az (3 sections, one all-'c'), tr (2 sections), kk (no model),
    uz (no parseable sections). fi/ky/ug have no snippet files.
    """
    ckpt = tmp_path / "ckpt"
    _write_tiny_model(ckpt, "az", ["a", "b", "c", "d"])
    _write_tiny_model(ckpt, "tr", ["a", "b", "q", "d"])
    _write_tiny_model(ckpt, "uz", ["a", "b", "c", "d", "q"])

    snip = tmp_path / "snip"
    snip.mkdir()
    (snip / "perception_az.txt").write_text(
        f"AZERI\n1\n{AB_SECTION}\n2\n{AD_SECTION}\n3\n{C_SECTION}\n", encoding="utf-8"
    )
    (snip / "perception_tr.txt").write_text(
        f"TURKISH\n1\n{AB_SECTION}\n2\n{AQ_SECTION}\n", encoding="utf-8"
    )
    (snip / "perception_kk.txt").write_text(f"KAZAKH\n1\n{AB_SECTION}\n", encoding="utf-8")
    (snip / "perception_uz.txt").write_text("UZBEK\n1\n2\n", encoding="utf-8")

    assim = tmp_path / "assim.csv"
    assim.write_text("listener,missing,replacement\ntr,c,a\naz,q,a\nuz,w,v\n", encoding="utf-8")
    return {
        "checkpoint_dir": ckpt,
        "snippet_dir": snip,
        "output_csv": tmp_path / "out" / "results.csv",
        "snippet_template": "perception_{lang}.txt",
        "oov_mode": "unk",
        "assimilation_csv": assim,
        "n_boot": 25,
        "seed": 0,
    }


def test_run_unk_mode_scores_all_pairs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _setup_eval_dirs(tmp_path)
    results = run(args)

    # 3 sources (az, tr, uz) x 2 targets (az, tr): uz/kk targets are skipped.
    assert [(r["src"], r["tgt"]) for r in results] == [
        ("az", "az"),
        ("az", "tr"),
        ("tr", "az"),
        ("tr", "tr"),
        ("uz", "az"),
        ("uz", "tr"),
    ]
    for r in results:
        assert r["mode"] == "unk"
        assert r["support"] == 1.0
        if r["src"] == r["tgt"]:
            assert r["excess_ce"] == 0.0
            assert (r["excess_lo"], r["excess_hi"]) == (0.0, 0.0)

    csv_text = args["output_csv"].read_text(encoding="utf-8")
    assert csv_text.startswith(CSV_HEADER + "\n")
    assert len(csv_text.strip().splitlines()) == 7  # header + 6 rows

    out = capsys.readouterr().out
    assert "model missing for fi" in out
    assert "no self model for kk" in out
    assert "snippet for uz parsed to zero sections" in out
    assert "snippet missing for ky" in out


def test_run_skip_mode_drops_zero_support_sections_and_masks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _setup_eval_dirs(tmp_path)
    args["oov_mode"] = "skip"
    results = run(args)

    by_pair = {(r["src"], r["tgt"]): r for r in results}
    # az's all-'c' section has no common-support position (tr lacks 'c'): dropped.
    out = capsys.readouterr().out
    assert "az: dropped 1 zero-support section(s)" in out

    # az target keeps sections ab/ad: 'd' is in every vocab, 'c'/'q' are not.
    az_row = by_pair[("tr", "az")]
    assert az_row["support"] == 1.0  # surviving az sections are fully covered
    # tr target's aq section: 'q' positions (20 of 39) lack common support.
    tr_row = by_pair[("az", "tr")]
    assert tr_row["n_scored"] == 39 + 19
    assert tr_row["support"] == pytest.approx((39 + 19) / 78)
    for r in results:
        if r["src"] == r["tgt"]:
            assert r["excess_ce"] == 0.0


def test_run_assimilate_mode_substitutes_before_scoring(tmp_path: Path) -> None:
    args = _setup_eval_dirs(tmp_path)
    unk_results = run(args)
    args["oov_mode"] = "assimilate"
    assim_results = run(args)

    unk_by_pair = {(r["src"], r["tgt"]): r for r in unk_results}
    assim_by_pair = {(r["src"], r["tgt"]): r for r in assim_results}
    # tr scoring az text: the all-'c' section becomes all-'a' instead of all-UNK,
    # so the assimilated CE must differ from the deafness CE.
    assert assim_by_pair[("tr", "az")]["ce"] != unk_by_pair[("tr", "az")]["ce"]
    # az scoring tr text: 'q' -> 'a' likewise changes the score.
    assert assim_by_pair[("az", "tr")]["ce"] != unk_by_pair[("az", "tr")]["ce"]
    # Pairs with no OOV chars are identical under both modes.
    assert assim_by_pair[("uz", "az")]["ce"] == pytest.approx(unk_by_pair[("uz", "az")]["ce"])
    for r in assim_results:
        assert r["support"] == 1.0


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
        "oov_mode": "skip",
        "assimilation_csv": DEFAULT_ASSIMILATION_CSV,
        "n_boot": DEFAULT_N_BOOT,
        "seed": DEFAULT_SEED,
    }


def test_parse_args_overrides(tmp_path: Path) -> None:
    args = parse_args(
        ["--oov-mode", "unk", "--n-boot", "10", "--seed", "7", "--output-csv", "x.csv"]
    )
    assert args["oov_mode"] == "unk"
    assert args["n_boot"] == 10
    assert args["seed"] == 7
    assert args["output_csv"] == Path("x.csv")


def _good_namespace() -> argparse.Namespace:
    """Test helper: namespace with all-valid argument types."""
    return argparse.Namespace(
        checkpoint_dir="a",
        snippet_dir="b",
        output_csv="c",
        snippet_template="{lang}",
        oov_mode="unk",
        assimilation_csv="d",
        n_boot=10,
        seed=0,
    )


def test_extract_args_rejects_bad_str_field() -> None:
    namespace = _good_namespace()
    namespace.snippet_dir = 5
    with pytest.raises(TypeError, match="Expected str for --snippet-dir"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_n_boot_type() -> None:
    namespace = _good_namespace()
    namespace.n_boot = "10"
    with pytest.raises(TypeError, match="Expected int for --n-boot"):
        _extract_args(namespace)


def test_extract_args_rejects_bad_seed_type() -> None:
    namespace = _good_namespace()
    namespace.seed = "0"
    with pytest.raises(TypeError, match="Expected int for --seed"):
        _extract_args(namespace)


def test_extract_args_rejects_unknown_mode() -> None:
    namespace = _good_namespace()
    namespace.oov_mode = "mask"
    with pytest.raises(ValueError, match="Unknown --oov-mode"):
        _extract_args(namespace)


def test_extract_args_rejects_nonpositive_n_boot() -> None:
    namespace = _good_namespace()
    namespace.n_boot = 0
    with pytest.raises(ValueError, match="--n-boot must be >= 1"):
        _extract_args(namespace)


def test_oov_modes_are_the_documented_three() -> None:
    assert OOV_MODES == ("unk", "skip", "assimilate")


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _setup_eval_dirs(tmp_path)
    code = main(
        [
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-csv",
            str(args["output_csv"]),
            "--oov-mode",
            "skip",
            "--n-boot",
            "10",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Wrote 6 pair(s)" in out
    assert args["output_csv"].read_text(encoding="utf-8").startswith(CSV_HEADER)


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _setup_eval_dirs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "zero_shot_eval",
            "--checkpoint-dir",
            str(args["checkpoint_dir"]),
            "--snippet-dir",
            str(args["snippet_dir"]),
            "--output-csv",
            str(args["output_csv"]),
            "--oov-mode",
            "unk",
            "--n-boot",
            "5",
        ],
    )
    monkeypatch.delitem(sys.modules, "scripts.zero_shot_eval")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.zero_shot_eval", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
