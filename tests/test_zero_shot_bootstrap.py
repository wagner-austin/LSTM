"""Tests for scripts.zero_shot_bootstrap."""

from __future__ import annotations

import pytest
from scripts.zero_shot_bootstrap import (
    SectionScore,
    bootstrap_asymmetry,
    bootstrap_excess,
    bootstrap_listener_difference,
    ce_from_scores,
    percentile_interval,
)


def _score(loss_sum: float, n_scored: int, n_total: int) -> SectionScore:
    """Test helper: literal SectionScore."""
    return {"loss_sum": loss_sum, "n_scored": n_scored, "n_total": n_total}


def _flat(losses: list[float]) -> list[SectionScore]:
    """Test helper: one single-position section per loss."""
    return [_score(loss, 1, 1) for loss in losses]


# ---------------------------------------------------------------------------
# ce_from_scores / percentile_interval
# ---------------------------------------------------------------------------


def test_ce_from_scores_pools_loss_over_positions() -> None:
    scores = [_score(2.0, 2, 2), _score(4.0, 2, 4)]
    assert ce_from_scores(scores) == 1.5
    assert ce_from_scores(scores, [1, 1]) == 2.0


def test_ce_from_scores_raises_on_zero_positions() -> None:
    with pytest.raises(ValueError, match="No scored positions"):
        ce_from_scores([_score(0.0, 0, 4)])


def test_percentile_interval_takes_the_outer_order_statistics() -> None:
    """Of 201 replicates 0..200, the bounds are the 5th and the 195th.

    floor(0.025 * 200) is 5 and ceil(0.975 * 200) is 195, and the input
    order must not matter, so the replicates are given reversed.
    """
    samples = [float(v) for v in range(200, -1, -1)]
    assert percentile_interval(samples) == (5.0, 195.0)


def test_percentile_interval_rounds_outward_between_order_statistics() -> None:
    """With 2000 replicates the indices 49.975 and 1949.025 round outward."""
    samples = [float(v) for v in range(2000)]
    assert percentile_interval(samples) == (49.0, 1950.0)


def test_percentile_interval_refuses_no_replicates() -> None:
    with pytest.raises(ValueError, match="No bootstrap replicates"):
        percentile_interval([])


# ---------------------------------------------------------------------------
# bootstrap_excess
# ---------------------------------------------------------------------------


def test_bootstrap_excess_is_zero_for_identical_scores() -> None:
    scores = _flat([1.0, 3.0])
    assert bootstrap_excess(scores, scores, 50, 0) == (0.0, 0.0)


def test_bootstrap_excess_recovers_constant_offset_exactly() -> None:
    lo, hi = bootstrap_excess(_flat([2.0, 4.0]), _flat([1.0, 3.0]), 200, 0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_excess_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        bootstrap_excess(_flat([1.0]), [], 10, 0)


# ---------------------------------------------------------------------------
# bootstrap_asymmetry
# ---------------------------------------------------------------------------


def test_bootstrap_asymmetry_is_zero_when_both_directions_match() -> None:
    """Two directions built from identical scores differ by exactly zero."""
    scores = _flat([1.0, 3.0])
    assert bootstrap_asymmetry(scores, scores, scores, scores, 50, 0) == (0.0, 0.0)


def test_bootstrap_asymmetry_recovers_a_constant_difference_exactly() -> None:
    """Constant offsets per direction leave the difference free of spread.

    Forward excess is a flat +2.0 and reverse a flat +0.5 whatever indices
    are drawn, so every resample yields 1.5 and the interval collapses onto
    it. A test with per-section variation could only assert a range, which
    would not distinguish a correct implementation from one that resampled
    the wrong list.
    """
    fwd_self = _flat([1.0, 3.0])
    fwd_pair = _flat([3.0, 5.0])
    rev_self = _flat([2.0, 4.0, 6.0])
    rev_pair = _flat([2.5, 4.5, 6.5])
    lo, hi = bootstrap_asymmetry(fwd_pair, fwd_self, rev_pair, rev_self, 200, 0)
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(1.5)


def test_bootstrap_asymmetry_accepts_directions_of_different_length() -> None:
    """The two directions read different languages' sections.

    This is the property that separates it from bootstrap_excess: there is
    no correspondence between index i on one side and index i on the other,
    so equal lengths must not be required and must not be assumed.
    """
    lo, hi = bootstrap_asymmetry(
        _flat([2.0]), _flat([1.0]), _flat([1.5, 1.5, 1.5]), _flat([1.0, 1.0, 1.0]), 100, 0
    )
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_asymmetry_rejects_mismatched_forward_lengths() -> None:
    """A pair and its self-scores must cover the same sections."""
    good = _flat([1.0])
    with pytest.raises(ValueError, match="Forward score lists differ in length"):
        bootstrap_asymmetry(_flat([1.0]), [], good, good, 10, 0)


def test_bootstrap_asymmetry_rejects_mismatched_reverse_lengths() -> None:
    """The reverse direction is checked separately, and names itself."""
    good = _flat([1.0])
    with pytest.raises(ValueError, match="Reverse score lists differ in length"):
        bootstrap_asymmetry(good, good, _flat([1.0]), [], 10, 0)


# ---------------------------------------------------------------------------
# bootstrap_listener_difference
# ---------------------------------------------------------------------------

#: Section losses that vary five-fold, the way passage difficulty does. The
#: self-scores carry the same difficulty, so each listener's excess still
#: varies by section and its own interval is wide.
_DIFFICULTY = [1.0, 5.0, 2.0, 4.0, 3.0, 1.5, 4.5, 2.5]


def test_listener_difference_resolves_what_the_overlap_rule_cannot() -> None:
    """The case the README's old rule got wrong, built so the answer is known.

    Listener B is worse than listener A by exactly 0.2 on every section,
    but both listeners' excesses swing with section difficulty, so their
    two marginal intervals are wide and overlap: the overlap rule calls
    them indistinguishable. The shared-index interval sees the same
    sections on both sides, the difficulty cancels, and the interval
    collapses onto the true difference.
    """
    self_scores = _flat(_DIFFICULTY)
    first = _flat([2.0 * d for d in _DIFFICULTY])
    second = _flat([2.0 * d + 0.2 for d in _DIFFICULTY])

    first_lo, first_hi = bootstrap_excess(first, self_scores, 400, 0)
    second_lo, second_hi = bootstrap_excess(second, self_scores, 400, 0)
    assert first_lo <= second_hi
    assert second_lo <= first_hi

    lo, hi = bootstrap_listener_difference(first, second, self_scores, 400, 0)
    assert lo == pytest.approx(-0.2)
    assert hi == pytest.approx(-0.2)


def test_listener_difference_does_not_depend_on_the_self_scores() -> None:
    """Both excesses subtract one pooled baseline, so it cancels exactly.

    Any baseline at all must give the interval that the two pair-scorings
    alone give; a function that resampled the self-scores separately from
    the pairs would fail this.
    """
    first = _flat([1.0, 2.0, 4.0, 3.0])
    second = _flat([1.5, 1.0, 5.0, 2.0])
    low_baseline = bootstrap_listener_difference(first, second, _flat([0.1] * 4), 300, 3)
    high_baseline = bootstrap_listener_difference(first, second, _flat([9.0, 1, 7, 2]), 300, 3)
    assert low_baseline == pytest.approx(high_baseline)


def test_listener_difference_is_the_matrix_difference_when_sections_agree() -> None:
    """Identical listeners differ by exactly zero in every resample."""
    scores = _flat(_DIFFICULTY)
    assert bootstrap_listener_difference(scores, scores, scores, 100, 0) == (0.0, 0.0)


def test_listener_difference_rejects_lists_of_different_length() -> None:
    with pytest.raises(ValueError, match=r"Listener score lists differ in length: 2, 1 and 2"):
        bootstrap_listener_difference(_flat([1.0, 2.0]), _flat([1.0]), _flat([1.0, 2.0]), 10, 0)


def test_listener_difference_rejects_a_self_list_of_different_length() -> None:
    with pytest.raises(ValueError, match=r"1, 1 and 2 \(self\)"):
        bootstrap_listener_difference(_flat([1.0]), _flat([1.0]), _flat([1.0, 2.0]), 10, 0)


def test_listener_difference_refuses_sections_scored_over_different_positions() -> None:
    """Index i must name the same positions for both listeners.

    Two listeners of one text are scored on one mask, so a section whose
    scored-position counts differ means the lists describe different
    readings, and sharing an index would invent a correspondence.
    """
    first = [_score(1.0, 2, 3), _score(2.0, 3, 3)]
    second = [_score(1.0, 2, 3), _score(2.0, 2, 3)]
    with pytest.raises(ValueError, match="Section 1 scored 3 positions for one listener and 2"):
        bootstrap_listener_difference(first, second, first, 10, 0)
