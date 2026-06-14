"""Validity battery for the zero-shot excess-CE method.

Three falsification tests, run after every retrain so a change that breaks
the method fails loudly instead of producing quietly wrong numbers:

1. Known-answer: on the real perception sections, the symmetrized excess-CE
   distances must recover every SETTLED sibling pair -- (az,tr) Oghuz,
   (kk,ky) Kipchak, (ug,uz) Karluk -- as a clade, with Finnish never inside
   a sibling pair (checked only when all six Turkic models are present).
   The higher-order arrangement of the three branches is contested in the
   historical-linguistics literature, so it is reported but not asserted.
   Within-branch distances must also be smaller than cross-branch ones
   (positive "branch gap").
2. Negative control: with each section's characters shuffled (same symbols,
   same frequencies, phonotactic order destroyed), the branch gap must
   collapse below half the real gap. If structure survives shuffling, the
   method is measuring alphabet overlap, not phonotactics.
3. Replication: on held-out corpus slices (middle of each language's test
   split -- text the models never trained on, in a different register), the
   branch gap must again be positive, and the tree must match when all six
   Turkic models are present.

All scoring uses the parameter-free common-support regime, identical to the
LSTM eval's ``skip`` mode. Exit code is 0 only if every check passes.

Usage::

    poetry run python -m scripts.validate_method \\
        --checkpoint-dir checkpoints \\
        --corpus-dir 10_Cleaned_Corpora \\
        --snippet-dir data/perception_clean

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TypedDict

from scripts.clean_corpus import CORPUS_TEMPLATE
from scripts.zero_shot_eval import (
    DEFAULT_SNIPPET_TEMPLATE,
    LoadedModel,
    ce_from_scores,
    common_support_mask,
    load_sources,
    parse_sections,
    score_section,
    snippet_path,
)

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CORPUS_DIR = Path("10_Cleaned_Corpora")
DEFAULT_SNIPPET_DIR = Path("data/perception_clean")
DEFAULT_OUTPUT_JSON = Path("results/validity_report.json")
DEFAULT_SEED = 1
DEFAULT_SLICE_CHARS = 5000

TEST_SPLIT_START = 0.85  # train 0.70 + val 0.15, matching char_lstm.train
SHUFFLED_GAP_FACTOR = 0.5  # shuffled gap must fall below real gap times this

BRANCHES: dict[str, str] = {
    "az": "oghuz",
    "tr": "oghuz",
    "kk": "kipchak",
    "ky": "kipchak",
    "ug": "karluk",
    "uz": "karluk",
}

# str leaves; 2-tuples for internal nodes (recursive, like UnknownJson in _types).
Tree = str | tuple["Tree", "Tree"]

# The uncontested claims: each branch's pair must form a clade. The
# higher-order subgrouping of the three branches is NOT settled linguistics
# and is deliberately not asserted.
EXPECTED_SIBLINGS: tuple[tuple[str, str], ...] = (("az", "tr"), ("kk", "ky"), ("ug", "uz"))


class GapStats(TypedDict):
    """Branch-gap statistics for one distance matrix.

    Attributes:
        within: Mean symmetrized distance between same-branch pairs.
        cross: Mean symmetrized distance between cross-branch pairs.
        gap: ``cross - within``; positive means branch structure exists.
    """

    within: float
    cross: float
    gap: float


class ValidityReport(TypedDict):
    """Full battery outcome.

    Attributes:
        real: Gap stats on the real perception sections.
        shuffled: Gap stats on the character-shuffled sections.
        heldout: Gap stats on the held-out corpus slices.
        real_tree: Canonical tree string from the real sections.
        heldout_tree: Canonical tree string from the held-out slices.
        expected_siblings: The settled sibling pairs that must be clades.
        checks: Named boolean outcomes of every criterion.
        passed: True iff every check passed.
    """

    real: GapStats
    shuffled: GapStats
    heldout: GapStats
    real_tree: str
    heldout_tree: str
    expected_siblings: list[list[str]]
    checks: dict[str, bool]
    passed: bool


class ValidateArgs(TypedDict):
    """Parsed and validated CLI arguments.

    Attributes:
        checkpoint_dir: Directory with model checkpoints and vocabs.
        corpus_dir: Directory with cleaned corpora (held-out slices).
        snippet_dir: Directory with perception snippets.
        snippet_template: Filename template containing ``{lang}``.
        output_json: Destination for the JSON report.
        seed: RNG seed for the shuffle control.
        slice_chars: Held-out slice length per language.
    """

    checkpoint_dir: Path
    corpus_dir: Path
    snippet_dir: Path
    snippet_template: str
    output_json: Path
    seed: int
    slice_chars: int


# ---------------------------------------------------------------------------
# Distance geometry
# ---------------------------------------------------------------------------


def symmetrize(excess: dict[str, dict[str, float]]) -> dict[tuple[str, str], float]:
    """Symmetrized distances from an excess-CE matrix.

    Args:
        excess: ``excess[src][tgt]`` over a common language set.

    Returns:
        ``{(a, b): (excess[a][b] + excess[b][a]) / 2}`` for every unordered
        pair, keyed with ``a < b``.
    """
    langs = sorted(excess)
    return {
        (a, b): 0.5 * (excess[a][b] + excess[b][a])
        for i, a in enumerate(langs)
        for b in langs[i + 1 :]
    }


def branch_gap(dist: dict[tuple[str, str], float]) -> GapStats:
    """Within- vs cross-branch distance means for Turkic language pairs.

    Args:
        dist: Symmetrized distances from :func:`symmetrize`. Pairs where
            either language has no branch assignment are ignored.

    Returns:
        :class:`GapStats`.

    Raises:
        ValueError: If there is no within-branch or no cross-branch pair.
    """
    within = [
        d
        for (a, b), d in dist.items()
        if a in BRANCHES and b in BRANCHES and BRANCHES[a] == BRANCHES[b]
    ]
    cross = [
        d
        for (a, b), d in dist.items()
        if a in BRANCHES and b in BRANCHES and BRANCHES[a] != BRANCHES[b]
    ]
    if not within or not cross:
        msg = (
            f"Branch gap needs both pair kinds; got {len(within)} within-branch "
            f"and {len(cross)} cross-branch pairs."
        )
        raise ValueError(msg)
    within_mean = sum(within) / len(within)
    cross_mean = sum(cross) / len(cross)
    return {"within": within_mean, "cross": cross_mean, "gap": cross_mean - within_mean}


def upgma(dist: dict[tuple[str, str], float]) -> Tree:
    """Average-linkage agglomerative clustering of a distance matrix.

    Args:
        dist: Symmetrized distances keyed with ``a < b``.

    Returns:
        Nested-tuple tree over the languages appearing in ``dist``.

    Raises:
        ValueError: If fewer than two languages are present.
    """
    langs = sorted({lang for pair in dist for lang in pair})
    if len(langs) < 2:
        msg = f"UPGMA needs at least two languages; got {langs}."
        raise ValueError(msg)

    def lookup(a: str, b: str) -> float:
        return dist[(a, b)] if (a, b) in dist else dist[(b, a)]

    clusters: dict[int, Tree] = dict(enumerate(langs))
    members: dict[int, list[str]] = {i: [lang] for i, lang in enumerate(langs)}
    active = list(clusters)
    next_id = len(langs)
    while len(active) > 1:
        best_pair = (active[0], active[1])
        best_dist = float("inf")
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                pair_dists = [lookup(x, y) for x in members[a] for y in members[b]]
                mean = sum(pair_dists) / len(pair_dists)
                if mean < best_dist:
                    best_dist = mean
                    best_pair = (a, b)
        a, b = best_pair
        clusters[next_id] = (clusters[a], clusters[b])
        members[next_id] = members[a] + members[b]
        active = [x for x in active if x not in (a, b)] + [next_id]
        next_id += 1
    return clusters[active[0]]


def canonical(tree: Tree) -> str:
    """Order-independent string form of a tree, for topology comparison.

    Args:
        tree: Leaf name or nested 2-tuple.

    Returns:
        Canonical string: children of every node sorted lexicographically.
    """
    if isinstance(tree, str):
        return tree
    left, right = canonical(tree[0]), canonical(tree[1])
    first, second = sorted([left, right])
    return f"({first},{second})"


def leaf_sibling_pairs(tree: Tree) -> set[tuple[str, str]]:
    """All two-leaf subtrees (cherries) of a tree, each as a sorted tuple.

    A pair is included only when both of a node's children are leaves, i.e.
    the two languages are each other's closest relatives in the tree.

    Args:
        tree: Leaf name or nested 2-tuple.

    Returns:
        Set of sorted (leaf, leaf) pairs that form a two-leaf clade.
    """
    if isinstance(tree, str):
        return set()
    left, right = tree
    pairs = leaf_sibling_pairs(left) | leaf_sibling_pairs(right)
    if isinstance(left, str) and isinstance(right, str):
        lo, hi = sorted((left, right))
        pairs.add((lo, hi))
    return pairs


def shuffle_text(text: str, rng: random.Random) -> str:
    """Shuffle a text's characters in position-independent order.

    Args:
        text: Section text.
        rng: Seeded RNG (shared across sections for determinism).

    Returns:
        Same characters, random order.
    """
    chars = list(text)
    rng.shuffle(chars)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Excess matrices
# ---------------------------------------------------------------------------


def excess_matrix(
    models: dict[str, LoadedModel],
    targets: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Common-support excess-CE matrix over a target set.

    Args:
        models: Loaded models; must cover exactly the target languages so
            the resulting matrix is square (symmetrization needs both
            directions of every pair).
        targets: Section lists per target language.

    Returns:
        ``excess[src][tgt]`` over the common language set.

    Raises:
        ValueError: If the model and target language sets differ, or a
            target's sections have no common-support position.
    """
    if set(models) != set(targets):
        msg = (
            f"excess_matrix needs identical language sets; models={sorted(models)} "
            f"targets={sorted(targets)}."
        )
        raise ValueError(msg)
    vocabs = [set(loaded["stoi"]) for loaded in models.values()]
    masks = {
        tgt: [common_support_mask(s, vocabs) for s in sections] for tgt, sections in targets.items()
    }
    ce: dict[str, dict[str, float]] = {}
    for src, loaded in models.items():
        ce[src] = {}
        for tgt, sections in targets.items():
            scores = [
                score_section(loaded, section, mask)
                for section, mask in zip(sections, masks[tgt], strict=True)
            ]
            ce[src][tgt] = ce_from_scores(scores)
    return {src: {tgt: ce[src][tgt] - ce[tgt][tgt] for tgt in targets} for src in models}


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def _load_real_targets(args: ValidateArgs, models: dict[str, LoadedModel]) -> dict[str, list[str]]:
    """Parse perception sections for every language with a model.

    Args:
        args: Validated CLI arguments.
        models: Loaded models.

    Returns:
        Section lists per language; languages without a usable snippet are
        skipped with a notice.
    """
    targets: dict[str, list[str]] = {}
    for lang in models:
        path = snippet_path(args["snippet_dir"], args["snippet_template"], lang)
        if not path.exists():
            print(f"  snippet missing for {lang}: {path}; excluded from battery")
            continue
        sections = parse_sections(path.read_text(encoding="utf-8"))
        if sections:
            targets[lang] = sections
    return targets


def _load_heldout_targets(
    args: ValidateArgs, models: dict[str, LoadedModel]
) -> dict[str, list[str]]:
    """Cut one held-out slice per language from its corpus test region.

    Args:
        args: Validated CLI arguments.
        models: Loaded models.

    Returns:
        Single-section target per language with a corpus file.
    """
    targets: dict[str, list[str]] = {}
    for lang in models:
        path = args["corpus_dir"] / CORPUS_TEMPLATE.format(lang=lang)
        if not path.exists():
            print(f"  corpus missing for {lang}: {path}; excluded from replication")
            continue
        text = path.read_text(encoding="utf-8")
        region = text[int(len(text) * TEST_SPLIT_START) :]
        mid = max(0, len(region) // 2 - args["slice_chars"] // 2)
        slice_text = region[mid : mid + args["slice_chars"]]
        if len(slice_text) >= 2:
            targets[lang] = [slice_text]
    return targets


def run(args: ValidateArgs) -> ValidityReport:
    """Run the three-test battery and write the JSON report.

    Args:
        args: Validated CLI arguments.

    Returns:
        :class:`ValidityReport`.
    """
    models = load_sources(args["checkpoint_dir"])
    real_targets = _load_real_targets(args, models)
    rng = random.Random(args["seed"])
    shuffled_targets = {
        lang: [shuffle_text(s, rng) for s in sections] for lang, sections in real_targets.items()
    }
    heldout_targets = _load_heldout_targets(args, models)

    real_models = {lang: models[lang] for lang in real_targets}
    heldout_models = {lang: models[lang] for lang in heldout_targets}
    real_dist = symmetrize(excess_matrix(real_models, real_targets))
    shuffled_dist = symmetrize(excess_matrix(real_models, shuffled_targets))
    heldout_dist = symmetrize(excess_matrix(heldout_models, heldout_targets))

    real = branch_gap(real_dist)
    shuffled = branch_gap(shuffled_dist)
    heldout = branch_gap(heldout_dist)
    real_upgma = upgma(real_dist)
    heldout_upgma = upgma(heldout_dist)
    real_tree = canonical(real_upgma)
    heldout_tree = canonical(heldout_upgma)
    expected_pairs = {tuple(sorted(pair)) for pair in EXPECTED_SIBLINGS}

    checks: dict[str, bool] = {
        "real_gap_positive": real["gap"] > 0,
        "heldout_gap_positive": heldout["gap"] > 0,
        "shuffled_gap_collapsed": shuffled["gap"] < SHUFFLED_GAP_FACTOR * real["gap"],
    }
    if set(BRANCHES) <= set(real_targets):
        checks["real_siblings_recovered"] = expected_pairs <= leaf_sibling_pairs(real_upgma)
    if set(BRANCHES) <= set(heldout_targets):
        checks["heldout_siblings_recovered"] = expected_pairs <= leaf_sibling_pairs(heldout_upgma)

    report: ValidityReport = {
        "real": real,
        "shuffled": shuffled,
        "heldout": heldout,
        "real_tree": real_tree,
        "heldout_tree": heldout_tree,
        "expected_siblings": [list(pair) for pair in EXPECTED_SIBLINGS],
        "checks": checks,
        "passed": all(checks.values()),
    }
    args["output_json"].parent.mkdir(parents=True, exist_ok=True)
    args["output_json"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  real:     gap={real['gap']:+.3f}  tree={real_tree}")
    print(f"  shuffled: gap={shuffled['gap']:+.3f}")
    print(f"  heldout:  gap={heldout['gap']:+.3f}  tree={heldout_tree}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"Validity battery: {'PASSED' if report['passed'] else 'FAILED'}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Validity battery for the zero-shot excess-CE method.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--snippet-dir", type=str, default=str(DEFAULT_SNIPPET_DIR))
    parser.add_argument("--snippet-template", type=str, default=DEFAULT_SNIPPET_TEMPLATE)
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--slice-chars", type=int, default=DEFAULT_SLICE_CHARS)
    return parser


def _extract_args(namespace: argparse.Namespace) -> ValidateArgs:
    """Validate and convert an argparse Namespace into typed ValidateArgs.

    Args:
        namespace: Parsed argparse namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`ValidateArgs`.

    Raises:
        TypeError: If any argument has an unexpected type.
        ValueError: If ``slice_chars`` is smaller than 2.
    """
    str_fields = {
        "checkpoint_dir": namespace.checkpoint_dir,
        "corpus_dir": namespace.corpus_dir,
        "snippet_dir": namespace.snippet_dir,
        "snippet_template": namespace.snippet_template,
        "output_json": namespace.output_json,
    }
    for name, value in str_fields.items():
        if not isinstance(value, str):
            msg = f"Expected str for --{name.replace('_', '-')}, got {type(value).__name__}"
            raise TypeError(msg)
    if not isinstance(namespace.seed, int):
        msg = f"Expected int for --seed, got {type(namespace.seed).__name__}"
        raise TypeError(msg)
    if not isinstance(namespace.slice_chars, int):
        msg = f"Expected int for --slice-chars, got {type(namespace.slice_chars).__name__}"
        raise TypeError(msg)
    if namespace.slice_chars < 2:
        msg = f"--slice-chars must be >= 2, got {namespace.slice_chars}"
        raise ValueError(msg)
    return {
        "checkpoint_dir": Path(str_fields["checkpoint_dir"]),
        "corpus_dir": Path(str_fields["corpus_dir"]),
        "snippet_dir": Path(str_fields["snippet_dir"]),
        "snippet_template": str_fields["snippet_template"],
        "output_json": Path(str_fields["output_json"]),
        "seed": namespace.seed,
        "slice_chars": namespace.slice_chars,
    }


def parse_args(argv: list[str] | None = None) -> ValidateArgs:
    """Parse CLI arguments into typed :class:`ValidateArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`ValidateArgs`.
    """
    parser = _build_arg_parser()
    return _extract_args(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Process exit code: ``0`` if every validity check passed, else ``1``.
    """
    report = run(parse_args(argv))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
