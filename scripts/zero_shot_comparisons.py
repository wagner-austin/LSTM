"""Comparisons between cells of the zero-shot excess-CE matrix.

A matrix cell is a LEVEL: one listener's excess cross-entropy on one text.
The claims made about the matrix are about DIFFERENCES between cells, and
each difference gets its own interval here rather than being read off two
cells' intervals:

- an ASYMMETRY row compares one pair read in both directions,
  ``excess(a, b)`` against ``excess(b, a)``;
- a LISTENER row compares two listeners on one text,
  ``excess(a, text)`` against ``excess(b, text)``.

Both write a CSV beside the matrix and a run record beside that, each under
its own experiment name, because a level and a difference of levels must
never be subtracted from one another.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.run_record import Observation

from scripts.zero_shot_bootstrap import (
    SectionScore,
    bootstrap_asymmetry,
    bootstrap_listener_difference,
    ce_from_scores,
)


class AsymmetryResult(TypedDict):
    """Whether one unordered language pair reads differently in each direction.

    The paper-level claim this exists to test is that transfer is
    directional: that a model of ``lang_a`` reading ``lang_b`` is not
    interchangeable with the reverse. That claim is about the DIFFERENCE
    between two excess cross-entropies, so the interval belongs to the
    difference rather than to either side.

    Attributes:
        lang_a: First language of the unordered pair, alphabetically.
        lang_b: Second language of the pair.
        mode: OOV regime the matrix was scored under.
        excess_ab: Excess CE of ``lang_a`` reading ``lang_b``.
        excess_ba: Excess CE of ``lang_b`` reading ``lang_a``.
        difference: ``excess_ab - excess_ba``.
        difference_lo: Lower bound of the 95% bootstrap CI for it.
        difference_hi: Upper bound of the same interval.
        excludes_zero: Whether that interval excludes zero, which is the
            condition a directional claim actually needs. Stored rather
            than left to the reader because comparing two separate
            intervals by eye is the mistake this row exists to replace.
    """

    lang_a: str
    lang_b: str
    mode: str
    excess_ab: float
    excess_ba: float
    difference: float
    difference_lo: float
    difference_hi: float
    excludes_zero: bool


class ListenerComparison(TypedDict):
    """Whether two listeners are at different distances from one text.

    Attributes:
        text: Language of the scored text.
        listener_a: First listener, alphabetically.
        listener_b: Second listener.
        mode: OOV regime the matrix was scored under.
        excess_a: Excess CE of ``listener_a`` reading ``text``.
        excess_b: Excess CE of ``listener_b`` reading ``text``.
        difference: ``excess_a - excess_b``.
        difference_lo: Lower bound of the 95% shared-index bootstrap CI
            for the difference (:func:`bootstrap_listener_difference`).
        difference_hi: Upper bound of the same interval.
        differs: Whether that interval excludes zero. This is the verdict.
        intervals_overlap: Whether the two listeners' own excess intervals
            in the matrix overlap. Kept beside the verdict because until
            2026-10-06 the README told readers two distances differ only
            when these do NOT overlap, and a row carrying both lets anyone
            count where that rule and the test disagree.
    """

    text: str
    listener_a: str
    listener_b: str
    mode: str
    excess_a: float
    excess_b: float
    difference: float
    difference_lo: float
    difference_hi: float
    differs: bool
    intervals_overlap: bool


#: The asymmetry record's experiment, distinct from the matrix's on purpose. A
#: matrix row is a LEVEL (one model's excess cross-entropy reading one text)
#: and an asymmetry row is a DIFFERENCE between two of them with its own
#: interval; ``compare_run_records`` refuses to subtract records from different
#: experiments, which is exactly the refusal wanted between a level and a
#: difference of levels.
ASYMMETRY_EXPERIMENT = "turkic-zero-shot-asymmetry"

#: The listener-comparison record's experiment, distinct from both the
#: matrix's and the asymmetry's for the same reason: its rows are differences
#: between two listeners on one text, which are neither levels nor
#: directional differences.
LISTENER_EXPERIMENT = "turkic-zero-shot-listener-difference"

ASYMMETRY_CSV_HEADER = (
    "language_a,language_b,scoring_mode,excess_a_reading_b,excess_b_reading_a,"
    "difference,difference_confidence_interval_low,difference_confidence_interval_high,"
    "interval_excludes_zero"
)

LISTENER_CSV_HEADER = (
    "text_language,listener_a,listener_b,scoring_mode,excess_a,excess_b,"
    "difference,difference_confidence_interval_low,difference_confidence_interval_high,"
    "paired_interval_excludes_zero,marginal_intervals_overlap"
)


def _excess(scores: dict[tuple[str, str], list[SectionScore]], src: str, tgt: str) -> float:
    """Excess CE of ``src`` reading ``tgt``, pooled over every section once.

    Args:
        scores: Per-section scores keyed by ordered (source, target) pair.
        src: Listener language.
        tgt: Text language.

    Returns:
        ``ce(src, tgt) - ce(tgt, tgt)``.
    """
    return ce_from_scores(scores[(src, tgt)]) - ce_from_scores(scores[(tgt, tgt)])


def _yes_no(flag: bool) -> str:
    """Render a verdict as the word a CSV reader parses.

    Args:
        flag: The verdict.

    Returns:
        ``"yes"`` or ``"no"``.
    """
    return "yes" if flag else "no"


def asymmetry_results(
    scores: dict[tuple[str, str], list[SectionScore]],
    languages: tuple[str, ...],
    mode: str,
    n_boot: int,
    seed: int,
) -> list[AsymmetryResult]:
    """Test every unordered language pair for a directional difference.

    Args:
        scores: Per-section scores keyed by ordered (source, target) pair.
        languages: Language codes to pair up, in the order rows appear.
        mode: OOV regime, recorded on every row.
        n_boot: Number of bootstrap resamples per pair.
        seed: Base RNG seed; each pair is offset from it so that two pairs
            do not share a resampling pattern.

    Returns:
        One row per unordered pair, in alphabetical order within the pair.
    """
    ordered = sorted(languages)
    rows: list[AsymmetryResult] = []
    for offset, (a, b) in enumerate(
        (a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]
    ):
        excess_ab = _excess(scores, a, b)
        excess_ba = _excess(scores, b, a)
        lo, hi = bootstrap_asymmetry(
            scores[(a, b)],
            scores[(b, b)],
            scores[(b, a)],
            scores[(a, a)],
            n_boot,
            seed + offset,
        )
        rows.append(
            AsymmetryResult(
                lang_a=a,
                lang_b=b,
                mode=mode,
                excess_ab=excess_ab,
                excess_ba=excess_ba,
                difference=excess_ab - excess_ba,
                difference_lo=lo,
                difference_hi=hi,
                excludes_zero=lo > 0.0 or hi < 0.0,
            )
        )
    return rows


def listener_comparisons(
    scores: dict[tuple[str, str], list[SectionScore]],
    marginals: dict[tuple[str, str], tuple[float, float]],
    listeners: tuple[str, ...],
    texts: tuple[str, ...],
    mode: str,
    n_boot: int,
    seed: int,
) -> list[ListenerComparison]:
    """Compare every two foreign listeners on every text.

    A text's own model is not a listener here: its excess is zero by
    construction, so comparing it would test nothing.

    Args:
        scores: Per-section scores keyed by ordered (listener, text) pair,
            including each text's (text, text) self-scores.
        marginals: Each (listener, text) cell's own 95% excess interval, as
            the matrix reports it; read only for ``intervals_overlap``.
        listeners: Listener languages to compare.
        texts: Text languages, in the order their rows appear.
        mode: OOV regime, recorded on every row.
        n_boot: Number of bootstrap resamples per comparison.
        seed: Base RNG seed; each comparison is offset from it so that two
            comparisons do not share a resampling pattern.

    Returns:
        One row per text and unordered pair of foreign listeners,
        alphabetical within the pair.
    """
    rows: list[ListenerComparison] = []
    offset = 0
    for text in texts:
        foreign = sorted(lang for lang in listeners if lang != text)
        for i, a in enumerate(foreign):
            for b in foreign[i + 1 :]:
                lo, hi = bootstrap_listener_difference(
                    scores[(a, text)],
                    scores[(b, text)],
                    scores[(text, text)],
                    n_boot,
                    seed + offset,
                )
                offset += 1
                excess_a = _excess(scores, a, text)
                excess_b = _excess(scores, b, text)
                a_lo, a_hi = marginals[(a, text)]
                b_lo, b_hi = marginals[(b, text)]
                rows.append(
                    ListenerComparison(
                        text=text,
                        listener_a=a,
                        listener_b=b,
                        mode=mode,
                        excess_a=excess_a,
                        excess_b=excess_b,
                        difference=excess_a - excess_b,
                        difference_lo=lo,
                        difference_hi=hi,
                        differs=lo > 0.0 or hi < 0.0,
                        intervals_overlap=a_lo <= b_hi and b_lo <= a_hi,
                    )
                )
    return rows


def asymmetry_observations(results: list[AsymmetryResult]) -> tuple[Observation, ...]:
    """Name every asymmetry number so two runs can be paired by it.

    Three observations per unordered pair rather than one: the difference is
    the estimate and the interval is the verdict, and a record carrying only
    the estimate could be compared against another run without either reader
    being able to tell whether the sign had ever excluded zero.

    Args:
        results: One entry per unordered language pair.

    Returns:
        The observations, named ``asymmetry.<a>.<b>``,
        ``asymmetry_lo.<a>.<b>`` and ``asymmetry_hi.<a>.<b>``. Sorting is
        left to :func:`~platform_core.run_record.run_record`.
    """
    observations: list[Observation] = []
    for r in results:
        pair = f"{r['lang_a']}.{r['lang_b']}"
        observations.append(Observation(name=f"asymmetry.{pair}", value=r["difference"]))
        observations.append(Observation(name=f"asymmetry_lo.{pair}", value=r["difference_lo"]))
        observations.append(Observation(name=f"asymmetry_hi.{pair}", value=r["difference_hi"]))
    return tuple(observations)


def listener_observations(results: list[ListenerComparison]) -> tuple[Observation, ...]:
    """Name every listener-comparison number so two runs can be paired by it.

    Three per row, for the reason :func:`asymmetry_observations` gives: the
    difference is the estimate and its interval is the verdict.

    Args:
        results: One entry per text and pair of listeners.

    Returns:
        The observations, named ``listener_difference.<text>.<a>.<b>`` and
        the same with ``_lo`` and ``_hi``. Sorting is left to
        :func:`~platform_core.run_record.run_record`.
    """
    observations: list[Observation] = []
    for r in results:
        key = f"{r['text']}.{r['listener_a']}.{r['listener_b']}"
        observations.append(Observation(name=f"listener_difference.{key}", value=r["difference"]))
        observations.append(
            Observation(name=f"listener_difference_lo.{key}", value=r["difference_lo"])
        )
        observations.append(
            Observation(name=f"listener_difference_hi.{key}", value=r["difference_hi"])
        )
    return tuple(observations)


def render_asymmetry_csv(results: list[AsymmetryResult]) -> str:
    """Render a list of :class:`AsymmetryResult` as a CSV string.

    Args:
        results: Asymmetry results; output rows preserve the input order.

    Returns:
        CSV text including header and trailing newline.
    """
    lines = [ASYMMETRY_CSV_HEADER]
    for r in results:
        lines.append(
            ",".join(
                [
                    r["lang_a"],
                    r["lang_b"],
                    r["mode"],
                    f"{r['excess_ab']:.6f}",
                    f"{r['excess_ba']:.6f}",
                    f"{r['difference']:.6f}",
                    f"{r['difference_lo']:.6f}",
                    f"{r['difference_hi']:.6f}",
                    _yes_no(r["excludes_zero"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_listener_csv(results: list[ListenerComparison]) -> str:
    """Render a list of :class:`ListenerComparison` as a CSV string.

    Args:
        results: Listener comparisons; output rows preserve the input order.

    Returns:
        CSV text including header and trailing newline.
    """
    lines = [LISTENER_CSV_HEADER]
    for r in results:
        lines.append(
            ",".join(
                [
                    r["text"],
                    r["listener_a"],
                    r["listener_b"],
                    r["mode"],
                    f"{r['excess_a']:.6f}",
                    f"{r['excess_b']:.6f}",
                    f"{r['difference']:.6f}",
                    f"{r['difference_lo']:.6f}",
                    f"{r['difference_hi']:.6f}",
                    _yes_no(r["differs"]),
                    _yes_no(r["intervals_overlap"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"
