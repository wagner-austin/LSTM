"""Tests for scripts.compare_listeners."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record
from scripts.compare_listeners import (
    CompareArgs,
    MatrixMismatchError,
    _extract_args,
    main,
    parse_args,
    require_reproduces,
    run,
    verdict_changes,
)
from scripts.section_scores import SectionScoreTable, encode_section_score_table
from scripts.zero_shot_bootstrap import SectionScore
from scripts.zero_shot_comparisons import (
    LISTENER_CSV_HEADER,
    LISTENER_EXPERIMENT,
    ListenerComparison,
)
from scripts.zero_shot_eval import DEFAULT_N_BOOT, DEFAULT_SEED, pair_results, render_results_csv

from char_lstm.provenance import sidecar_path

#: Section difficulty of the one text, ``kk``, shared by every listener.
_KK = [1.0, 4.0, 2.0, 3.0, 2.5]


def _flat(losses: list[float]) -> list[SectionScore]:
    """Test helper: one single-position section per loss."""
    return [{"loss_sum": loss, "n_scored": 1, "n_total": 1} for loss in losses]


def _table() -> SectionScoreTable:
    """Three models of which ``kk`` is the only text.

    ``ky`` and ``tr`` swing with difficulty and differ by a flat 0.3, so
    their own intervals overlap while the shared-index test separates
    them: one verdict the overlap rule got wrong, by construction.
    """
    return {
        "mode": "skip",
        "listeners": ("kk", "ky", "tr"),
        "texts": ("kk",),
        "scores": {
            ("kk", "kk"): _flat(_KK),
            ("ky", "kk"): _flat([2.0 * d for d in _KK]),
            ("tr", "kk"): _flat([2.0 * d + 0.3 for d in _KK]),
        },
    }


def _published(tmp_path: Path) -> CompareArgs:
    """Write a table and the matrix it pools to, as a publication would hold them."""
    table = _table()
    scores_path = tmp_path / "matrix_section_scores.json"
    scores_path.write_text(encode_section_score_table(table), encoding="utf-8")
    matrix_csv = tmp_path / "matrix.csv"
    matrix_csv.write_text(render_results_csv(pair_results(table, 200, 0)), encoding="utf-8")
    return {"section_scores": scores_path, "matrix_csv": matrix_csv, "n_boot": 200, "seed": 0}


def test_run_writes_the_comparisons_of_a_reproduced_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _published(tmp_path)
    rows = run(args)

    assert [(r["listener_a"], r["listener_b"]) for r in rows] == [("ky", "tr")]
    assert rows[0]["differs"] is True
    assert rows[0]["intervals_overlap"] is True
    written = (tmp_path / "matrix_listeners.csv").read_text(encoding="utf-8")
    assert written.splitlines()[0] == LISTENER_CSV_HEADER
    assert written.splitlines()[1].startswith("kk,ky,tr,skip,")
    assert written.splitlines()[1].endswith(",yes,yes")
    record = decode_run_record(
        load_json_str(sidecar_path(tmp_path / "matrix_listeners.csv").read_text("utf-8"))
    )
    assert record["experiment"] == LISTENER_EXPERIMENT
    assert record["label"] == "skip"

    out = capsys.readouterr().out
    assert "reproduce" in out
    assert (
        "Verdicts changed from the interval-overlap rule: 1 of 1 (1 now differ, 0 no longer" in out
    )


def test_run_refuses_scores_that_pool_to_different_numbers(tmp_path: Path) -> None:
    """A re-scoring that resembles the publication is not the publication.

    The published ky row is moved by one unit in the sixth decimal of its
    cross-entropy, the smallest change the CSV can carry.
    """
    args = _published(tmp_path)
    lines = args["matrix_csv"].read_text(encoding="utf-8").splitlines()
    fields = lines[2].split(",")
    assert fields[:2] == ["ky", "kk"]
    fields[3] = f"{float(fields[3]) + 0.000001:.6f}"
    lines[2] = ",".join(fields)
    args["matrix_csv"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(MatrixMismatchError, match="at line 3"):
        run(args)
    assert not (tmp_path / "matrix_listeners.csv").exists()


def test_require_reproduces_accepts_identical_text() -> None:
    require_reproduces("h\na\n", "h\na\n", Path("m.csv"))


def test_require_reproduces_names_the_first_differing_line() -> None:
    with pytest.raises(MatrixMismatchError, match=r"line 2: re-pooled 'b', published 'c'"):
        require_reproduces("h\nb\nx\n", "h\nc\ny\n", Path("m.csv"))


def test_require_reproduces_refuses_a_matrix_with_rows_missing() -> None:
    with pytest.raises(MatrixMismatchError, match="re-pooled 1 line"):
        require_reproduces("h\n", "h\na\n", Path("m.csv"))


def _verdicts(differs: bool, intervals_overlap: bool) -> ListenerComparison:
    """Test helper: a comparison row carrying only the two verdicts that matter."""
    return {
        "text": "kk",
        "listener_a": "ky",
        "listener_b": "tr",
        "mode": "skip",
        "excess_a": 1.0,
        "excess_b": 1.2,
        "difference": -0.2,
        "difference_lo": -0.3,
        "difference_hi": -0.1,
        "differs": differs,
        "intervals_overlap": intervals_overlap,
    }


def test_verdict_changes_counts_each_direction_of_disagreement() -> None:
    """Gained: separated though overlapping. Lost: overlap-separated, paired-tied."""
    rows = [
        _verdicts(differs=True, intervals_overlap=True),
        _verdicts(differs=False, intervals_overlap=False),
        _verdicts(differs=False, intervals_overlap=False),
        _verdicts(differs=True, intervals_overlap=False),
        _verdicts(differs=False, intervals_overlap=True),
    ]
    assert verdict_changes(rows) == (1, 2)


def test_parse_args_defaults_to_the_matrix_bootstrap_settings() -> None:
    args = parse_args(["--section-scores", "s.json", "--matrix-csv", "m.csv"])
    assert args == {
        "section_scores": Path("s.json"),
        "matrix_csv": Path("m.csv"),
        "n_boot": DEFAULT_N_BOOT,
        "seed": DEFAULT_SEED,
    }


def _namespace(section_scores: str | int, n_boot: int, seed: str | int) -> argparse.Namespace:
    """Test helper: a namespace as argparse would build one, with chosen types."""
    return argparse.Namespace(
        section_scores=section_scores, matrix_csv="m.csv", n_boot=n_boot, seed=seed
    )


def test_extract_args_rejects_a_non_string_path() -> None:
    with pytest.raises(TypeError, match="Expected str for --section-scores"):
        _extract_args(_namespace(3, 10, 0))


def test_extract_args_rejects_a_non_integer_count() -> None:
    with pytest.raises(TypeError, match="Expected int for --n-boot and --seed"):
        _extract_args(_namespace("s.json", 10, "0"))


def test_extract_args_rejects_a_nonpositive_n_boot() -> None:
    with pytest.raises(ValueError, match="--n-boot must be >= 1, got 0"):
        _extract_args(_namespace("s.json", 0, 0))


def test_main_end_to_end(tmp_path: Path) -> None:
    args = _published(tmp_path)
    argv = [
        "--section-scores",
        str(args["section_scores"]),
        "--matrix-csv",
        str(args["matrix_csv"]),
        "--n-boot",
        "200",
    ]
    assert main(argv) == 0
    assert (tmp_path / "matrix_listeners.csv").read_text("utf-8").count("\n") == 2


def test_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _published(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_listeners",
            "--section-scores",
            str(args["section_scores"]),
            "--matrix-csv",
            str(args["matrix_csv"]),
            "--n-boot",
            "200",
        ],
    )
    monkeypatch.delitem(sys.modules, "scripts.compare_listeners")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.compare_listeners", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
