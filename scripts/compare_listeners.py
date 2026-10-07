"""Re-decide a published matrix's listener comparisons from its section scores.

``scripts.zero_shot_eval`` writes the listener comparisons for every run it
makes. This is for a matrix that was published before it did: given the
section scores behind it, it re-pools them into the matrix and REFUSES to go
on unless the result is the published CSV byte for byte, so the comparisons
it then writes are provably about the published numbers and not about a
re-scoring that merely resembles them.

Usage::

    poetry run python -m scripts.compare_listeners \\
        --section-scores results/zero_shot_excess_ce_skip_section_scores.json \\
        --matrix-csv results/zero_shot_excess_ce_skip.csv

writes ``results/zero_shot_excess_ce_skip_listeners.csv`` and its run record.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

from char_lstm.provenance import scoring_fingerprint
from scripts.section_scores import load_section_score_table
from scripts.zero_shot_comparisons import ListenerComparison
from scripts.zero_shot_eval import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    listener_results,
    pair_results,
    render_results_csv,
    write_listener_comparisons,
)


class CompareArgs(TypedDict):
    """Parsed and validated CLI arguments.

    Attributes:
        section_scores: The section-score document behind the matrix.
        matrix_csv: The published matrix those scores must reproduce.
        n_boot: Bootstrap resamples, as the matrix was made with.
        seed: Bootstrap seed, as the matrix was made with.
    """

    section_scores: Path
    matrix_csv: Path
    n_boot: int
    seed: int


class MatrixMismatchError(ValueError):
    """The section scores do not reproduce the published matrix."""


def require_reproduces(rendered: str, published: str, matrix_csv: Path) -> None:
    """Refuse section scores whose pooled matrix is not the published one.

    Args:
        rendered: The matrix re-pooled from the section scores.
        published: The published matrix's text.
        matrix_csv: Where the published matrix was read, for the message.

    Raises:
        MatrixMismatchError: Naming the first line that differs, if any.
    """
    if rendered == published:
        return
    ours = rendered.splitlines()
    theirs = published.splitlines()
    for number, (mine, published_line) in enumerate(zip(ours, theirs, strict=False), start=1):
        if mine != published_line:
            msg = (
                f"Section scores do not reproduce {matrix_csv} at line {number}: "
                f"re-pooled {mine!r}, published {published_line!r}."
            )
            raise MatrixMismatchError(msg)
    msg = (
        f"Section scores do not reproduce {matrix_csv}: re-pooled {len(ours)} line(s), "
        f"published {len(theirs)}."
    )
    raise MatrixMismatchError(msg)


def verdict_changes(rows: list[ListenerComparison]) -> tuple[int, int]:
    """Count the comparisons where the overlap rule and the paired test disagree.

    Args:
        rows: Listener comparisons.

    Returns:
        ``(gained, lost)``: comparisons the paired test separates though the
        two intervals overlap, and comparisons the overlap rule separates
        though the paired interval includes zero.
    """
    gained = sum(1 for r in rows if r["differs"] and r["intervals_overlap"])
    lost = sum(1 for r in rows if not r["differs"] and not r["intervals_overlap"])
    return gained, lost


def run(args: CompareArgs) -> list[ListenerComparison]:
    """Re-pool, check against the published matrix, and write the comparisons.

    Args:
        args: Validated CLI arguments.

    Returns:
        The listener comparisons written.

    Raises:
        MatrixMismatchError: If the scores do not reproduce the matrix.
    """
    table = load_section_score_table(args["section_scores"].read_text(encoding="utf-8"))
    results = pair_results(table, args["n_boot"], args["seed"])
    require_reproduces(
        render_results_csv(results),
        args["matrix_csv"].read_text(encoding="utf-8"),
        args["matrix_csv"],
    )
    print(f"Section scores reproduce {args['matrix_csv']} byte for byte")
    rows = listener_results(table, results, args["n_boot"], args["seed"])
    write_listener_comparisons(args["matrix_csv"], rows, table["mode"], scoring_fingerprint())
    gained, lost = verdict_changes(rows)
    print(
        f"Verdicts changed from the interval-overlap rule: {gained + lost} of {len(rows)} "
        f"({gained} now differ, {lost} no longer differ)"
    )
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Re-decide a published matrix's listener comparisons from its section scores.",
    )
    parser.add_argument("--section-scores", type=str, required=True)
    parser.add_argument("--matrix-csv", type=str, required=True)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def _extract_args(namespace: argparse.Namespace) -> CompareArgs:
    """Validate and convert an argparse Namespace into typed CompareArgs.

    Args:
        namespace: Parsed namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`CompareArgs`.

    Raises:
        TypeError: If an argument has an unexpected type.
        ValueError: If ``--n-boot`` is not positive.
    """
    section_scores = namespace.section_scores
    matrix_csv = namespace.matrix_csv
    n_boot = namespace.n_boot
    seed = namespace.seed
    if not isinstance(section_scores, str) or not isinstance(matrix_csv, str):
        msg = "Expected str for --section-scores and --matrix-csv."
        raise TypeError(msg)
    if not isinstance(n_boot, int) or not isinstance(seed, int):
        msg = "Expected int for --n-boot and --seed."
        raise TypeError(msg)
    if n_boot < 1:
        msg = f"--n-boot must be >= 1, got {n_boot}"
        raise ValueError(msg)
    return CompareArgs(
        section_scores=Path(section_scores), matrix_csv=Path(matrix_csv), n_boot=n_boot, seed=seed
    )


def parse_args(argv: list[str] | None = None) -> CompareArgs:
    """Parse CLI arguments into a typed :class:`CompareArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`CompareArgs`.
    """
    return _extract_args(_build_arg_parser().parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success.
    """
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
