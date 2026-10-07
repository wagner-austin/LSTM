"""Tests for scripts.section_scores."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str
from scripts.section_scores import (
    SectionScoreTable,
    decode_section_score_table,
    encode_section_score_table,
    load_section_score_table,
)
from scripts.zero_shot_bootstrap import SectionScore


def _section(loss_sum: float, n_scored: int, n_total: int) -> SectionScore:
    """Test helper: literal SectionScore."""
    return {"loss_sum": loss_sum, "n_scored": n_scored, "n_total": n_total}


def _table() -> SectionScoreTable:
    """Two models, ``tr`` and ``az``, reading two texts of two sections each."""
    return {
        "mode": "skip",
        "listeners": ("tr", "az"),
        "texts": ("tr", "az"),
        "scores": {
            ("tr", "tr"): [_section(1.5, 3, 4), _section(2.25, 2, 2)],
            ("tr", "az"): [_section(4.0, 2, 3), _section(0.125, 1, 1)],
            ("az", "tr"): [_section(6.0, 3, 4), _section(3.5, 2, 2)],
            ("az", "az"): [_section(2.0, 2, 3), _section(0.0625, 1, 1)],
        },
    }


def _pair(listener: str, text: str, sections: list[JSONValue]) -> JSONValue:
    """Test helper: one cell of the document."""
    return {"listener": listener, "text": text, "sections": sections}


_ONE: JSONValue = {"loss_sum": 1.0, "n_scored": 1, "n_total": 1}


def test_a_table_round_trips_through_its_document_exactly() -> None:
    """Order is carried by the document, so listener and text order survive."""
    text = encode_section_score_table(_table())
    assert load_section_score_table(text) == _table()
    assert text.endswith("\n")


def test_the_document_lists_cells_in_matrix_row_order() -> None:
    """Listeners outer, texts inner: the order the matrix CSV is written in."""
    document = decode_section_score_table(
        {
            "mode": "unk",
            "pairs": [
                _pair("tr", "tr", [_ONE]),
                _pair("tr", "az", [_ONE]),
                _pair("az", "tr", [_ONE]),
                _pair("az", "az", [_ONE]),
            ],
        }
    )
    assert document["listeners"] == ("tr", "az")
    assert document["texts"] == ("tr", "az")
    assert document["mode"] == "unk"
    assert document["scores"][("az", "tr")] == [_section(1.0, 1, 1)]


def test_an_integer_loss_is_read_as_a_float() -> None:
    """JSON writes 0.0 as 0 in some encoders; the count is still a loss."""
    table = decode_section_score_table(
        {
            "mode": "skip",
            "pairs": [_pair("kk", "kk", [{"loss_sum": 0, "n_scored": 0, "n_total": 3}])],
        }
    )
    assert table["scores"][("kk", "kk")] == [_section(0.0, 0, 3)]


def test_a_cell_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="Cell kk->kk appears twice"):
        decode_section_score_table(
            {"mode": "skip", "pairs": [_pair("kk", "kk", [_ONE]), _pair("kk", "kk", [_ONE])]}
        )


def test_a_text_with_no_self_scores_is_refused() -> None:
    """Excess is taken against the text's own model; without it there is none."""
    with pytest.raises(ValueError, match="Text ky has no self-scores"):
        decode_section_score_table({"mode": "skip", "pairs": [_pair("kk", "ky", [_ONE])]})


def test_a_missing_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="Cell tr->az is missing"):
        decode_section_score_table(
            {
                "mode": "skip",
                "pairs": [
                    _pair("az", "az", [_ONE]),
                    _pair("tr", "tr", [_ONE]),
                    _pair("az", "tr", [_ONE]),
                ],
            }
        )


def test_listeners_of_one_text_with_different_section_counts_are_refused() -> None:
    """Every listener of a text read the same sections, so the counts agree."""
    with pytest.raises(
        ValueError, match=r"Listeners of text az carry different section counts: \[1, 2\]"
    ):
        decode_section_score_table(
            {
                "mode": "skip",
                "pairs": [
                    _pair("az", "az", [_ONE, _ONE]),
                    _pair("az", "tr", [_ONE]),
                    _pair("tr", "az", [_ONE]),
                    _pair("tr", "tr", [_ONE]),
                ],
            }
        )


@pytest.mark.parametrize(("n_scored", "n_total"), [(-1, 3), (4, 3)])
def test_a_scored_count_outside_the_section_is_refused(n_scored: int, n_total: int) -> None:
    section: JSONValue = {"loss_sum": 1.0, "n_scored": n_scored, "n_total": n_total}
    with pytest.raises(ValueError, match=f"Section scored {n_scored} of {n_total} positions"):
        decode_section_score_table({"mode": "skip", "pairs": [_pair("kk", "kk", [section])]})


def test_a_count_written_as_a_float_is_refused() -> None:
    section: JSONValue = {"loss_sum": 1.0, "n_scored": 1.0, "n_total": 1}
    with pytest.raises(JSONTypeError, match="'n_scored' must be an integer"):
        decode_section_score_table({"mode": "skip", "pairs": [_pair("kk", "kk", [section])]})


def test_a_document_without_a_mode_is_refused() -> None:
    with pytest.raises(JSONTypeError, match="Missing required field 'mode'"):
        load_section_score_table(dump_json_str({"pairs": []}))
