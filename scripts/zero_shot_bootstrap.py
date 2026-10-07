"""Section-level bootstrap intervals for the zero-shot excess-CE matrix.

Every interval the evaluation reports is a percentile bootstrap over
perception SECTIONS: a resample draws section indices with replacement and
pools loss over the positions those sections scored. The three functions
here differ only in which quantity they resample and in which score lists
share an index list, and that sharing is the whole of the statistics:

- :func:`bootstrap_excess` -- one listener against the text's own model on
  the same sections, indices SHARED, so passage difficulty cancels.
- :func:`bootstrap_asymmetry` -- one pair read in both directions, which
  covers two different texts, indices drawn INDEPENDENTLY per direction.
- :func:`bootstrap_listener_difference` -- two listeners on the same text,
  indices SHARED, so passage difficulty cancels in the difference.

Whether two numbers differ is a question about their DIFFERENCE, and is
answered by an interval on it. Comparing two separate intervals by eye is
the wrong test: non-overlap does imply a difference, but overlap implies
nothing, so it can only ever fail to detect one.
"""

from __future__ import annotations

import math
import random
from typing import TypedDict


class SectionScore(TypedDict):
    """Cross-entropy sums for one snippet section under one (src, tgt) pair.

    Attributes:
        loss_sum: Summed per-position cross-entropy over scored positions.
        n_scored: Number of positions actually scored.
        n_total: Number of next-char positions in the section.
    """

    loss_sum: float
    n_scored: int
    n_total: int


def ce_from_scores(scores: list[SectionScore], idx: list[int] | None = None) -> float:
    """Pooled cross-entropy over a (re)sample of sections.

    Args:
        scores: Per-section scores for one (src, tgt) pair.
        idx: Section indices to pool; ``None`` pools all sections once.

    Returns:
        Total loss divided by total scored positions.

    Raises:
        ValueError: If the selection contains zero scored positions.
    """
    selected = scores if idx is None else [scores[i] for i in idx]
    n = sum(s["n_scored"] for s in selected)
    if n == 0:
        msg = "No scored positions in selection; cannot compute cross-entropy."
        raise ValueError(msg)
    return sum(s["loss_sum"] for s in selected) / n


def percentile_interval(samples: list[float]) -> tuple[float, float]:
    """The 95% percentile interval of a list of bootstrap replicates.

    The bounds are the order statistics at ``floor(0.025 * (n - 1))`` and
    ``ceil(0.975 * (n - 1))``, so the interval never narrows below what
    the replicates support.

    Args:
        samples: One value per resample, in any order; not modified.

    Returns:
        (lower, upper) bounds of the 95% interval.

    Raises:
        ValueError: If there are no replicates to take an interval of.
    """
    if not samples:
        msg = "No bootstrap replicates; cannot take an interval."
        raise ValueError(msg)
    ordered = sorted(samples)
    last = len(ordered) - 1
    return ordered[math.floor(0.025 * last)], ordered[math.ceil(0.975 * last)]


def bootstrap_excess(
    pair_scores: list[SectionScore],
    self_scores: list[SectionScore],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """95% paired bootstrap CI for excess CE over sections.

    Each resample draws section indices with replacement and applies the
    SAME indices to both the pair and the self scores, so per-section
    difficulty cancels within every resample.

    Args:
        pair_scores: Per-section scores for (src, tgt).
        self_scores: Per-section scores for (tgt, tgt) on identical sections.
        n_boot: Number of resamples.
        seed: RNG seed.

    Returns:
        (lower, upper) bounds of the 95% interval.

    Raises:
        ValueError: If the two score lists have different lengths.
    """
    if len(pair_scores) != len(self_scores):
        msg = f"Score lists differ in length: {len(pair_scores)} vs {len(self_scores)}."
        raise ValueError(msg)
    n = len(pair_scores)
    rng = random.Random(seed)
    excesses: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        excesses.append(ce_from_scores(pair_scores, idx) - ce_from_scores(self_scores, idx))
    return percentile_interval(excesses)


def bootstrap_asymmetry(
    forward_pair: list[SectionScore],
    forward_self: list[SectionScore],
    reverse_pair: list[SectionScore],
    reverse_self: list[SectionScore],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """95% bootstrap CI for the DIFFERENCE between two excess CEs.

    The directional claim about a language pair is that ``excess(a, b)``
    and ``excess(b, a)`` differ. Answering it by checking whether their
    two intervals overlap is the wrong test and errs in one direction:
    non-overlap does imply a difference, but overlap implies nothing, so
    that check can only ever fail to detect one. This resamples the
    difference itself, which is the quantity the claim is about.

    Unlike :func:`bootstrap_excess` the two halves are NOT paired. Each
    excess is measured over a different language's sections, so there is
    no correspondence between index ``i`` on one side and index ``i`` on
    the other, and reusing one index list would invent one. The two are
    resampled independently within each iteration; each half stays
    internally paired against its own self-scores, so per-section
    difficulty still cancels where it genuinely can.

    Args:
        forward_pair: Per-section scores for (a, b).
        forward_self: Per-section scores for (b, b), same sections.
        reverse_pair: Per-section scores for (b, a).
        reverse_self: Per-section scores for (a, a), same sections.
        n_boot: Number of resamples.
        seed: RNG seed.

    Returns:
        (lower, upper) bounds of the 95% interval for
        ``excess(a, b) - excess(b, a)``. An interval excluding zero is
        the evidence a directional asymmetry claim needs.

    Raises:
        ValueError: If either side's score lists differ in length, since
            a pair and its self-scores must cover the same sections.
    """
    if len(forward_pair) != len(forward_self):
        msg = f"Forward score lists differ in length: {len(forward_pair)} vs {len(forward_self)}."
        raise ValueError(msg)
    if len(reverse_pair) != len(reverse_self):
        msg = f"Reverse score lists differ in length: {len(reverse_pair)} vs {len(reverse_self)}."
        raise ValueError(msg)
    n_fwd = len(forward_pair)
    n_rev = len(reverse_pair)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        fwd_idx = [rng.randrange(n_fwd) for _ in range(n_fwd)]
        rev_idx = [rng.randrange(n_rev) for _ in range(n_rev)]
        forward = ce_from_scores(forward_pair, fwd_idx) - ce_from_scores(forward_self, fwd_idx)
        reverse = ce_from_scores(reverse_pair, rev_idx) - ce_from_scores(reverse_self, rev_idx)
        diffs.append(forward - reverse)
    return percentile_interval(diffs)


def bootstrap_listener_difference(
    first_pair: list[SectionScore],
    second_pair: list[SectionScore],
    self_scores: list[SectionScore],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """95% shared-index bootstrap CI for ``excess(first) - excess(second)``.

    The comparison is between two listeners reading ONE text, which is the
    opposite case to :func:`bootstrap_asymmetry`: both excesses are
    measured over the same sections and the same positions of that text,
    and both subtract the same self-scores. Index ``i`` therefore names the
    same section on both sides, and every resample applies one index list
    to all three score lists, as :func:`bootstrap_excess` does for a
    listener and its baseline. Passage difficulty is shared and cancels in
    the difference, which is why this interval is narrower than either
    listener's own, and why reading the verdict off those two intervals
    instead discards exactly the information that resolves it.

    Within a resample the self-scores cancel exactly, since both excesses
    subtract the same pooled value; they are still taken, so that each
    replicate is literally the difference of two excesses and the function
    can refuse score lists that do not describe one text.

    Args:
        first_pair: Per-section scores for (first listener, text).
        second_pair: Per-section scores for (second listener, text).
        self_scores: Per-section scores for (text, text), same sections.
        n_boot: Number of resamples.
        seed: RNG seed.

    Returns:
        (lower, upper) bounds of the 95% interval for the difference. An
        interval excluding zero is the evidence that the two listeners'
        distances to the text differ.

    Raises:
        ValueError: If the three lists differ in length, or if any section
            scored a different number of positions for the two listeners,
            since then index ``i`` does not name the same positions.
    """
    if not len(first_pair) == len(second_pair) == len(self_scores):
        msg = (
            "Listener score lists differ in length: "
            f"{len(first_pair)}, {len(second_pair)} and {len(self_scores)} (self)."
        )
        raise ValueError(msg)
    for i, (first, second) in enumerate(zip(first_pair, second_pair, strict=True)):
        if first["n_scored"] != second["n_scored"]:
            msg = (
                f"Section {i} scored {first['n_scored']} positions for one listener and "
                f"{second['n_scored']} for the other; the listeners did not read the same text."
            )
            raise ValueError(msg)
    n = len(first_pair)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        baseline = ce_from_scores(self_scores, idx)
        first_excess = ce_from_scores(first_pair, idx) - baseline
        second_excess = ce_from_scores(second_pair, idx) - baseline
        diffs.append(first_excess - second_excess)
    return percentile_interval(diffs)
