"""Character-trigram baseline for the zero-shot excess-CE matrix.

Answers the control question a reviewer will ask of the LSTM result: does
the genealogical signal require a neural model, or is it already present in
surface trigram statistics? Trains an interpolated character-trigram model
per language on the TRAIN split of its cleaned corpus (first 70%, matching
``char_lstm.train``), scores the perception sections on the same
common-support positions as the LSTM eval's ``skip`` mode, and writes a CSV
with identical columns so the two matrices are directly comparable.

Only the parameter-free ``skip`` regime is implemented: the baseline exists
to benchmark the headline measurement, not the OOV sensitivity analyses.

Usage::

    poetry run python -m scripts.ngram_baseline \\
        --corpus-dir corpora_clean \\
        --snippet-dir data/perception_clean \\
        --output-csv results/ngram_excess_ce.csv

"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

from char_lstm.data import load_vocab_json
from scripts.corpora import CORPUS_TEMPLATE, LANGS
from scripts.zero_shot_eval import (
    DEFAULT_SNIPPET_TEMPLATE,
    PairResult,
    SectionScore,
    bootstrap_excess,
    ce_from_scores,
    common_support_mask,
    parse_sections,
    render_results_csv,
    snippet_path,
)

DEFAULT_CORPUS_DIR = Path("corpora_clean")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_SNIPPET_DIR = Path("data/perception_clean")
DEFAULT_OUTPUT_CSV = Path("results/ngram_excess_ce.csv")
DEFAULT_MAX_TRAIN_CHARS = 4_000_000
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0

TRAIN_RATIO = 0.70  # must match char_lstm.train's train_ratio
LAMBDAS = (0.05, 0.25, 0.70)  # interpolation weights for orders 1, 2, 3
FLOOR_MASS = 0.001  # probability mass reserved for unseen events


class NgramArgs(TypedDict):
    """Parsed and validated CLI arguments.

    Attributes:
        corpus_dir: Directory with cleaned ``oscar_{lang}_ipa.txt`` corpora.
        checkpoint_dir: Directory with ``{lang}_vocab.json`` files (defines
            the common-support positions, identical to the LSTM eval).
        snippet_dir: Directory with perception snippet files.
        output_csv: Destination CSV path.
        snippet_template: Filename template containing ``{lang}``.
        max_train_chars: Cap on training characters per language.
        n_boot: Number of bootstrap resamples.
        seed: Bootstrap RNG seed.
    """

    corpus_dir: Path
    checkpoint_dir: Path
    snippet_dir: Path
    output_csv: Path
    snippet_template: str
    max_train_chars: int
    n_boot: int
    seed: int


class TrigramModel:
    """Interpolated character-trigram language model.

    Probabilities mix unigram, bigram, and trigram maximum-likelihood
    estimates with fixed weights (:data:`LAMBDAS`), plus a uniform floor of
    :data:`FLOOR_MASS` so unseen events never get zero probability.
    """

    def __init__(self, text: str) -> None:
        """Count n-grams in a training text.

        Args:
            text: Training characters.

        Raises:
            ValueError: If the text is shorter than 3 characters (no
                trigram can be counted).
        """
        if len(text) < 3:
            msg = f"Training text too short for trigrams: {len(text)} chars."
            raise ValueError(msg)
        self.unigrams: Counter[str] = Counter(text)
        self.bigrams: Counter[tuple[str, str]] = Counter(pairwise(text))
        self.trigrams: Counter[tuple[str, str, str]] = Counter(
            (text[i], text[i + 1], text[i + 2]) for i in range(len(text) - 2)
        )
        self.bigram_contexts: Counter[str] = Counter(text[:-1])
        self.trigram_contexts: Counter[tuple[str, str]] = Counter(pairwise(text[:-1]))
        self.n_chars = len(text)
        self.n_types = len(self.unigrams)

    def neg_logp(self, c1: str, c2: str, c3: str) -> float:
        """Negative log-probability of ``c3`` following ``c1 c2``.

        Args:
            c1: Character two positions back.
            c2: Previous character.
            c3: Character being predicted.

        Returns:
            ``-log P(c3 | c1, c2)`` in nats (comparable to torch CE).
        """
        p1 = self.unigrams.get(c3, 0) / self.n_chars
        bigram_ctx = self.bigram_contexts.get(c2, 0)
        p2 = self.bigrams.get((c2, c3), 0) / bigram_ctx if bigram_ctx else 0.0
        trigram_ctx = self.trigram_contexts.get((c1, c2), 0)
        p3 = self.trigrams.get((c1, c2, c3), 0) / trigram_ctx if trigram_ctx else 0.0
        p = LAMBDAS[0] * p1 + LAMBDAS[1] * p2 + LAMBDAS[2] * p3
        p = (1.0 - FLOOR_MASS) * p + FLOOR_MASS / max(self.n_types, 100)
        return -math.log(p)


def score_section_ngram(model: TrigramModel, section: str, mask: list[bool]) -> SectionScore:
    """Score one section with a trigram model on masked positions.

    Positions are indexed like the LSTM eval: position ``k`` predicts
    ``section[k+1]``, so the mask has ``len(section) - 1`` entries. The
    first position has only one context character; its missing ``c1`` is a
    space, the neutral boundary character.

    Args:
        model: Trained trigram model.
        section: Section text.
        mask: Positions to score (common-support mask).

    Returns:
        :class:`SectionScore` with summed loss over scored positions.

    Raises:
        ValueError: If the mask length does not match the section.
    """
    n_total = len(section) - 1
    if len(mask) != n_total:
        msg = f"Mask length {len(mask)} != position count {n_total}."
        raise ValueError(msg)
    loss_sum = 0.0
    n_scored = 0
    for k in range(n_total):
        if not mask[k]:
            continue
        c1 = section[k - 1] if k >= 1 else " "
        loss_sum += model.neg_logp(c1, section[k], section[k + 1])
        n_scored += 1
    return {"loss_sum": loss_sum, "n_scored": n_scored, "n_total": n_total}


def _train_models(args: NgramArgs) -> dict[str, TrigramModel]:
    """Train one trigram model per language with an available corpus.

    Args:
        args: Validated CLI arguments.

    Returns:
        Trained models in :data:`LANGS` order.
    """
    models: dict[str, TrigramModel] = {}
    for lang in LANGS:
        corpus_path = args["corpus_dir"] / CORPUS_TEMPLATE.format(lang=lang)
        if not corpus_path.exists():
            print(f"  corpus missing for {lang}: {corpus_path}; skipping as source")
            continue
        text = corpus_path.read_text(encoding="utf-8")
        train_text = text[: int(len(text) * TRAIN_RATIO)][: args["max_train_chars"]]
        models[lang] = TrigramModel(train_text)
        print(f"  {lang}: trained on {len(train_text)} chars, {models[lang].n_types} types")
    return models


def _load_vocab_sets(args: NgramArgs, models: dict[str, TrigramModel]) -> list[set[str]]:
    """Character sets defining common support, one per source language.

    Uses the trained LSTM vocab files when present (identical support to the
    LSTM eval); falls back to the trigram training characters otherwise,
    with a notice.

    Args:
        args: Validated CLI arguments.
        models: Trained trigram models.

    Returns:
        One character set per model, in model order.
    """
    vocabs: list[set[str]] = []
    for lang in models:
        vocab_path = args["checkpoint_dir"] / f"{lang}_vocab.json"
        if not vocab_path.exists():
            print(f"  vocab missing for {lang}: {vocab_path}; support uses corpus chars")
            vocabs.append(set(models[lang].unigrams))
            continue
        stoi, _itos, _size, _unk = load_vocab_json(vocab_path)
        vocabs.append(set(stoi))
    return vocabs


def _load_targets(
    args: NgramArgs,
    models: dict[str, TrigramModel],
    vocabs: list[set[str]],
) -> tuple[dict[str, list[str]], dict[str, list[list[bool]]]]:
    """Parse target sections and their common-support masks.

    Args:
        args: Validated CLI arguments.
        models: Trained trigram models (targets must also be sources).
        vocabs: Character sets defining common support.

    Returns:
        (sections per target, masks per target), zero-support sections
        dropped.
    """
    targets: dict[str, list[str]] = {}
    masks: dict[str, list[list[bool]]] = {}
    for lang in models:
        path = snippet_path(args["snippet_dir"], args["snippet_template"], lang)
        if not path.exists():
            print(f"  snippet missing for {lang}: {path}; skipping as target")
            continue
        sections = parse_sections(path.read_text(encoding="utf-8"))
        section_masks = [common_support_mask(s, vocabs) for s in sections]
        keep = [i for i, m in enumerate(section_masks) if sum(m) > 0]
        if not keep:
            print(f"  snippet for {lang} has no supported sections; skipping as target")
            continue
        targets[lang] = [sections[i] for i in keep]
        masks[lang] = [section_masks[i] for i in keep]
    return targets, masks


def run(args: NgramArgs) -> list[PairResult]:
    """Train trigram models, score all pairs on common support, write CSV.

    Mirrors the LSTM eval's ``skip`` mode: a language participates as a
    source if its corpus exists, and as a target if its snippet exists, its
    vocab file exists (vocabs define common support), and its own corpus
    exists (excess CE needs the self baseline).

    Args:
        args: Validated CLI arguments.

    Returns:
        List of :class:`PairResult` for every evaluated pair.
    """
    models = _train_models(args)
    vocabs = _load_vocab_sets(args, models)
    targets, masks = _load_targets(args, models, vocabs)

    scores: dict[tuple[str, str], list[SectionScore]] = {}
    for src, model in models.items():
        for tgt, sections in targets.items():
            scores[(src, tgt)] = [
                score_section_ngram(model, section, mask)
                for section, mask in zip(sections, masks[tgt], strict=True)
            ]

    results: list[PairResult] = []
    for src in models:
        for tgt in targets:
            pair = scores[(src, tgt)]
            self_scores = scores[(tgt, tgt)]
            ce = ce_from_scores(pair)
            self_ce = ce_from_scores(self_scores)
            lo, hi = bootstrap_excess(pair, self_scores, args["n_boot"], args["seed"])
            n_scored = sum(s["n_scored"] for s in pair)
            n_total = sum(s["n_total"] for s in pair)
            results.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "mode": "trigram-skip",
                    "ce": ce,
                    "self_ce": self_ce,
                    "excess_ce": ce - self_ce,
                    "excess_lo": lo,
                    "excess_hi": hi,
                    "support": n_scored / n_total,
                    "n_scored": n_scored,
                }
            )
            print(
                f"  {src}->{tgt} [trigram] ce={ce:.4f} excess={ce - self_ce:+.4f} "
                f"[{lo:+.4f},{hi:+.4f}]"
            )

    args["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    args["output_csv"].write_text(render_results_csv(results), encoding="utf-8")
    print(f"Wrote {len(results)} pair(s) to {args['output_csv']}")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Character-trigram baseline for the zero-shot excess-CE matrix.",
    )
    parser.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--snippet-dir", type=str, default=str(DEFAULT_SNIPPET_DIR))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--snippet-template", type=str, default=DEFAULT_SNIPPET_TEMPLATE)
    parser.add_argument("--max-train-chars", type=int, default=DEFAULT_MAX_TRAIN_CHARS)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def _extract_args(namespace: argparse.Namespace) -> NgramArgs:
    """Validate and convert an argparse Namespace into typed NgramArgs.

    Args:
        namespace: Parsed argparse namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`NgramArgs`.

    Raises:
        TypeError: If any argument has an unexpected type.
        ValueError: If a count argument is not positive.
    """
    str_fields = {
        "corpus_dir": namespace.corpus_dir,
        "checkpoint_dir": namespace.checkpoint_dir,
        "snippet_dir": namespace.snippet_dir,
        "output_csv": namespace.output_csv,
        "snippet_template": namespace.snippet_template,
    }
    for name, value in str_fields.items():
        if not isinstance(value, str):
            msg = f"Expected str for --{name.replace('_', '-')}, got {type(value).__name__}"
            raise TypeError(msg)
    int_fields = {
        "max_train_chars": namespace.max_train_chars,
        "n_boot": namespace.n_boot,
    }
    for name, int_value in int_fields.items():
        if not isinstance(int_value, int):
            msg = f"Expected int for --{name.replace('_', '-')}, got {type(int_value).__name__}"
            raise TypeError(msg)
        if int_value < 1:
            msg = f"--{name.replace('_', '-')} must be >= 1, got {int_value}"
            raise ValueError(msg)
    if not isinstance(namespace.seed, int):
        msg = f"Expected int for --seed, got {type(namespace.seed).__name__}"
        raise TypeError(msg)
    return {
        "corpus_dir": Path(str_fields["corpus_dir"]),
        "checkpoint_dir": Path(str_fields["checkpoint_dir"]),
        "snippet_dir": Path(str_fields["snippet_dir"]),
        "output_csv": Path(str_fields["output_csv"]),
        "snippet_template": str_fields["snippet_template"],
        "max_train_chars": namespace.max_train_chars,
        "n_boot": namespace.n_boot,
        "seed": namespace.seed,
    }


def parse_args(argv: list[str] | None = None) -> NgramArgs:
    """Parse CLI arguments into typed :class:`NgramArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`NgramArgs`.
    """
    parser = _build_arg_parser()
    return _extract_args(parser.parse_args(argv))


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
