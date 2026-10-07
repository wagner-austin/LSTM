"""Tests for scripts.zero_shot_comparisons."""

from __future__ import annotations

import pytest
from scripts.zero_shot_bootstrap import SectionScore, bootstrap_listener_difference
from scripts.zero_shot_comparisons import (
    ASYMMETRY_CSV_HEADER,
    LISTENER_CSV_HEADER,
    AsymmetryResult,
    ListenerComparison,
    asymmetry_observations,
    asymmetry_results,
    listener_comparisons,
    listener_observations,
    render_asymmetry_csv,
    render_listener_csv,
)


def _flat(losses: list[float]) -> list[SectionScore]:
    """Test helper: one single-position section per loss."""
    return [{"loss_sum": loss, "n_scored": 1, "n_total": 1} for loss in losses]


# ---------------------------------------------------------------------------
# asymmetry_results / asymmetry_observations / render_asymmetry_csv
# ---------------------------------------------------------------------------


def test_asymmetry_results_covers_every_unordered_pair_once() -> None:
    """Three languages give three pairs, alphabetical within each."""
    flat = _flat([1.0, 1.0])
    langs = ("ky", "az", "tr")
    scores = {(a, b): flat for a in langs for b in langs}
    rows = asymmetry_results(scores, langs, "skip", 20, 0)
    assert [(r["lang_a"], r["lang_b"]) for r in rows] == [
        ("az", "ky"),
        ("az", "tr"),
        ("ky", "tr"),
    ]


def test_asymmetry_results_reports_a_real_difference_as_excluding_zero() -> None:
    """One direction costlier than the other, with no overlap of zero."""
    scores = {
        ("az", "az"): _flat([1.0, 1.0]),
        ("tr", "tr"): _flat([1.0, 1.0]),
        ("az", "tr"): _flat([4.0, 4.0]),
        ("tr", "az"): _flat([2.0, 2.0]),
    }
    rows = asymmetry_results(scores, ("az", "tr"), "skip", 100, 0)
    assert len(rows) == 1
    assert rows[0]["excess_ab"] == pytest.approx(3.0)
    assert rows[0]["excess_ba"] == pytest.approx(1.0)
    assert rows[0]["difference"] == pytest.approx(2.0)
    assert rows[0]["excludes_zero"] is True


def test_asymmetry_results_reports_no_difference_as_including_zero() -> None:
    """Symmetric directions must not be reported as directional."""
    flat = _flat([2.0, 2.0])
    self_flat = _flat([1.0, 1.0])
    scores = {
        ("az", "az"): self_flat,
        ("tr", "tr"): self_flat,
        ("az", "tr"): flat,
        ("tr", "az"): flat,
    }
    rows = asymmetry_results(scores, ("az", "tr"), "skip", 100, 0)
    assert rows[0]["difference"] == pytest.approx(0.0)
    assert rows[0]["excludes_zero"] is False


_ASYMMETRY_ROW: AsymmetryResult = {
    "lang_a": "az",
    "lang_b": "tr",
    "mode": "skip",
    "excess_ab": 1.7885,
    "excess_ba": 1.3070,
    "difference": 0.4815,
    "difference_lo": 0.2,
    "difference_hi": 0.75,
    "excludes_zero": True,
}


def test_asymmetry_observations_name_the_estimate_and_both_bounds() -> None:
    """The interval is the verdict, so it travels with the estimate."""
    assert asymmetry_observations([_ASYMMETRY_ROW]) == (
        {"name": "asymmetry.az.tr", "value": 0.4815},
        {"name": "asymmetry_lo.az.tr", "value": 0.2},
        {"name": "asymmetry_hi.az.tr", "value": 0.75},
    )


def test_render_asymmetry_csv_exact_row() -> None:
    """The rendered row is the contract a reader parses."""
    text = render_asymmetry_csv([_ASYMMETRY_ROW])
    assert text.splitlines()[0] == ASYMMETRY_CSV_HEADER
    assert text.splitlines()[1] == ("az,tr,skip,1.788500,1.307000,0.481500,0.200000,0.750000,yes")
    assert text.endswith("\n")


def test_render_asymmetry_csv_writes_no_for_an_interval_spanning_zero() -> None:
    """The flag is rendered as a word, so both values must be exercised."""
    row: AsymmetryResult = {**_ASYMMETRY_ROW, "excludes_zero": False}
    assert render_asymmetry_csv([row]).splitlines()[1].endswith(",no")


# ---------------------------------------------------------------------------
# listener_comparisons
# ---------------------------------------------------------------------------

#: Per-section difficulty shared by every reading of text ``kk``.
_KK = [1.0, 4.0, 2.0, 3.0]


def _kk_scores() -> dict[tuple[str, str], list[SectionScore]]:
    """Three listeners on one Kazakh text, built so the answers are known.

    ``ky`` and ``tr`` swing with section difficulty and differ by a flat
    0.25 per section, so their own intervals overlap while their difference
    is exact; ``uz`` is a flat 3.0 above the baseline everywhere.
    """
    return {
        ("kk", "kk"): _flat(_KK),
        ("ky", "kk"): _flat([2.0 * d for d in _KK]),
        ("tr", "kk"): _flat([2.0 * d + 0.25 for d in _KK]),
        ("uz", "kk"): _flat([d + 3.0 for d in _KK]),
    }


def _marginals() -> dict[tuple[str, str], tuple[float, float]]:
    """The cells' own intervals, as a matrix would report them."""
    return {
        ("ky", "kk"): (1.5, 3.5),
        ("tr", "kk"): (1.75, 3.75),
        ("uz", "kk"): (3.0, 3.0),
    }


def test_listener_comparisons_pair_every_two_foreign_listeners_once() -> None:
    """The text's own model is a baseline, not a listener to compare."""
    rows = listener_comparisons(
        _kk_scores(), _marginals(), ("uz", "kk", "tr", "ky"), ("kk",), "skip", 200, 0
    )
    assert [(r["text"], r["listener_a"], r["listener_b"]) for r in rows] == [
        ("kk", "ky", "tr"),
        ("kk", "ky", "uz"),
        ("kk", "tr", "uz"),
    ]
    assert {r["mode"] for r in rows} == {"skip"}


def test_listener_comparisons_separate_overlapping_listeners_by_their_difference() -> None:
    """The row the old README rule would have refused, decided."""
    rows = listener_comparisons(
        _kk_scores(), _marginals(), ("kk", "ky", "tr", "uz"), ("kk",), "skip", 200, 0
    )
    ky_tr = rows[0]
    assert ky_tr["excess_a"] == pytest.approx(2.5)
    assert ky_tr["excess_b"] == pytest.approx(2.75)
    assert ky_tr["difference"] == pytest.approx(-0.25)
    assert (ky_tr["difference_lo"], ky_tr["difference_hi"]) == pytest.approx((-0.25, -0.25))
    assert ky_tr["differs"] is True
    assert ky_tr["intervals_overlap"] is True


def test_listener_comparisons_report_disjoint_intervals_as_not_overlapping() -> None:
    """Moving ``uz``'s interval clear of the other two leaves only ky-tr overlapping."""
    marginals = {**_marginals(), ("uz", "kk"): (4.0, 4.5)}
    rows = listener_comparisons(
        _kk_scores(), marginals, ("kk", "ky", "tr", "uz"), ("kk",), "skip", 200, 0
    )
    assert [r["intervals_overlap"] for r in rows] == [True, False, False]


def test_listener_comparisons_include_zero_for_listeners_that_tie() -> None:
    """Two listeners whose sections swing in opposite directions around one mean."""
    scores = {
        ("kk", "kk"): _flat([1.0, 1.0, 1.0, 1.0]),
        ("ky", "kk"): _flat([2.0, 4.0, 2.0, 4.0]),
        ("tr", "kk"): _flat([4.0, 2.0, 4.0, 2.0]),
    }
    marginals = {("ky", "kk"): (1.0, 3.0), ("tr", "kk"): (1.0, 3.0)}
    rows = listener_comparisons(scores, marginals, ("kk", "ky", "tr"), ("kk",), "skip", 400, 0)
    assert rows[0]["difference"] == pytest.approx(0.0)
    assert rows[0]["difference_lo"] < 0.0 < rows[0]["difference_hi"]
    assert rows[0]["differs"] is False


def test_listener_comparisons_offset_the_seed_per_comparison_across_texts() -> None:
    """Comparison k is resampled with seed + k, counted across every text.

    Each row's interval is checked against a direct call at the seed it
    must have used, so a counter that restarted per text, or did not
    advance, would put a wrong interval on some row.
    """
    a_text = _flat([1.0, 2.0, 1.5])
    b_text = _flat([2.0, 1.0, 3.0])
    scores = {
        ("aa", "aa"): a_text,
        ("bb", "aa"): _flat([1.0, 3.0, 2.0]),
        ("cc", "aa"): _flat([2.0, 1.0, 4.0]),
        ("dd", "aa"): _flat([3.0, 1.0, 1.0]),
        ("aa", "bb"): _flat([4.0, 2.0, 5.0]),
        ("bb", "bb"): b_text,
        ("cc", "bb"): _flat([3.0, 3.0, 4.0]),
        ("dd", "bb"): _flat([5.0, 1.0, 4.0]),
    }
    marginals = dict.fromkeys(scores, (0.0, 1.0))
    rows = listener_comparisons(
        scores, marginals, ("aa", "bb", "cc", "dd"), ("aa", "bb"), "unk", 50, 10
    )
    expected = [
        ("aa", "bb", "cc"),
        ("aa", "bb", "dd"),
        ("aa", "cc", "dd"),
        ("bb", "aa", "cc"),
        ("bb", "aa", "dd"),
        ("bb", "cc", "dd"),
    ]
    assert [(r["text"], r["listener_a"], r["listener_b"]) for r in rows] == expected
    for offset, (row, (text, a, b)) in enumerate(zip(rows, expected, strict=True)):
        direct = bootstrap_listener_difference(
            scores[(a, text)], scores[(b, text)], scores[(text, text)], 50, 10 + offset
        )
        assert (row["difference_lo"], row["difference_hi"]) == direct


# ---------------------------------------------------------------------------
# listener_observations / render_listener_csv
# ---------------------------------------------------------------------------

_LISTENER_ROW: ListenerComparison = {
    "text": "kk",
    "listener_a": "ky",
    "listener_b": "tr",
    "mode": "skip",
    "excess_a": 1.190986,
    "excess_b": 1.417665,
    "difference": -0.226679,
    "difference_lo": -0.3,
    "difference_hi": -0.15,
    "differs": True,
    "intervals_overlap": False,
}


def test_listener_observations_name_the_estimate_and_both_bounds() -> None:
    assert listener_observations([_LISTENER_ROW]) == (
        {"name": "listener_difference.kk.ky.tr", "value": -0.226679},
        {"name": "listener_difference_lo.kk.ky.tr", "value": -0.3},
        {"name": "listener_difference_hi.kk.ky.tr", "value": -0.15},
    )


def test_render_listener_csv_exact_row() -> None:
    text = render_listener_csv([_LISTENER_ROW])
    assert text == (
        LISTENER_CSV_HEADER
        + "\nkk,ky,tr,skip,1.190986,1.417665,-0.226679,-0.300000,-0.150000,yes,no\n"
    )


def test_render_listener_csv_writes_both_verdict_words() -> None:
    row: ListenerComparison = {**_LISTENER_ROW, "differs": False, "intervals_overlap": True}
    assert render_listener_csv([row]).splitlines()[1].endswith(",no,yes")
