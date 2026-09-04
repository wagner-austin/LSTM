"""Zero-shot cross-entropy evaluation across language model pairs.

For each (source, target) language pair, loads the source's frozen base
checkpoint together with its vocabulary, scores the target language's
perception sections (headers and section markers excluded), and reports
cross-entropy under one of three out-of-vocabulary regimes:

- ``unk``         Chars missing from the source vocab become ``<unk>`` and
                  every next-char position is scored. The "deafness"
                  baseline: unfamiliar sounds carry no information.
- ``skip``        Only positions whose true next char is in EVERY loaded
                  model's vocab are scored (common support). All models see
                  the identical positions, eliminating per-source selection
                  bias.
- ``assimilate``  Chars missing from the source vocab are first replaced by
                  the nearest perceptual category (``data/assimilation.csv``)
                  and every position is scored. The listener analogue:
                  unfamiliar sounds are heard as the closest familiar one.

The headline metric is excess CE: ``ce(src->tgt) - ce(tgt->tgt)`` -- how much
more surprised the foreign model is than the target's own model on the same
sections -- with a paired bootstrap confidence interval over sections.

Usage::

    poetry run python -m scripts.zero_shot_eval \\
        --oov-mode skip \\
        --snippet-dir data/perception_clean \\
        --output-csv results/zero_shot_excess_ce_skip.csv

"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import random
import re
from pathlib import Path
from typing import TypedDict

import torch
from platform_core.run_record import Observation
from torch import Tensor
from torch.nn import functional

from char_lstm._types import _get_torch_load
from char_lstm.corpora import CORPUS_TEMPLATE, LANGS
from char_lstm.data import encode, load_vocab_json
from char_lstm.model import CharLSTM
from char_lstm.provenance import scoring_fingerprint, write_run_record

OOV_MODES: tuple[str, ...] = ("unk", "skip", "assimilate")

EXPERIMENT = "turkic-zero-shot-excess-ce"
"""What these runs are, stable across the OOV regimes and the corpus rebuilds.

Two records comparing under different experiment names are not comparable at
all, which is what makes the name load-bearing rather than decorative. The
OOV regime is the LABEL, not part of this: ``unk``, ``skip`` and
``assimilate`` score the same models on the same passages under three
different treatments of unfamiliar characters, and comparing across them is
a thing someone may legitimately want to do.
"""

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_SNIPPET_DIR = Path("data/perception_clean")
DEFAULT_OUTPUT_CSV = Path("results/zero_shot_excess_ce.csv")
DEFAULT_SNIPPET_TEMPLATE = "perception_{lang}.txt"
DEFAULT_ASSIMILATION_CSV = Path("data/assimilation.csv")
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0
DEFAULT_CORPUS_DIR = Path("corpora_clean")

# A character counts as attested for a listener when its training corpus
# contains it at least this many times. Real phonemes of a training
# language occur thousands of times per ten million characters and
# contamination occurs tens of times, so the floor sits between the two
# with an order of magnitude to spare on each side.
MIN_ATTESTED = 100

MIN_SECTION_CHARS = 20
DROPOUT = 0.0  # eval-only path; dropout is bypassed by model.eval() anyway

MARKER_RE = re.compile(r"^\s*[1-5]\s*$")

# The snippet files carry their title lines in transliterated form,
# because the word TEXT was passed through the transliterator along with
# the passages. Each language therefore spells it differently: Turkish
# leaves "teXt", Uzbek "text1", and Finnish "tekst", since Finnish rules
# write x as ks. A pattern matching only "text" silently let the Finnish
# titles through, and they were then appended to the preceding passage.
# The k and s are optional here so every spelling seen in the data is
# recognised; test_perception_files.py checks that against the real
# files rather than against a fixture.
HEADER_RE = re.compile(r"^\s*te?(ks|x)t\s*\d", re.IGNORECASE)
_LSTM_LAYER_RE = re.compile(r"^lstm\.weight_ih_l(\d+)$")


class LoadedModel(TypedDict):
    """A loaded model paired with its vocabulary.

    Attributes:
        model: The frozen :class:`CharLSTM` in eval mode on CPU.
        stoi: String-to-index mapping for the model's vocabulary.
        vocab_size: Size of the vocabulary including UNK.
    """

    model: CharLSTM
    stoi: dict[str, int]
    vocab_size: int


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


class PairResult(TypedDict):
    """One (src, tgt) evaluation result under a given OOV mode.

    Attributes:
        src: Source language code (model that does the scoring).
        tgt: Target language code (whose sections are being scored).
        mode: OOV regime, one of :data:`OOV_MODES`.
        ce: Mean cross-entropy of the source model on the target sections.
        self_ce: Mean cross-entropy of the target's own model on the same
            sections and positions.
        excess_ce: ``ce - self_ce``.
        excess_lo: Lower bound of the 95% paired bootstrap CI for excess_ce.
        excess_hi: Upper bound of the 95% paired bootstrap CI for excess_ce.
        support: Fraction of next-char positions scored (1.0 except ``skip``).
        n_scored: Total number of scored positions.
    """

    src: str
    tgt: str
    mode: str
    ce: float
    self_ce: float
    excess_ce: float
    excess_lo: float
    excess_hi: float
    support: float
    n_scored: int


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
        mode: OOV regime, one of :data:`OOV_MODES`.
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


class EvalArgs(TypedDict):
    """Parsed and validated CLI arguments for zero-shot evaluation.

    Attributes:
        checkpoint_dir: Directory containing ``{lang}_best.pt`` and
            ``{lang}_vocab.json`` files.
        snippet_dir: Directory containing perception snippet files.
        output_csv: Path to write the results CSV.
        snippet_template: Filename template containing ``{lang}``.
        oov_mode: One of :data:`OOV_MODES`.
        corpus_dir: Training corpora, read in skip mode to decide which
            characters count as attested for a listener.
        assimilation_csv: Substitution table, read only in assimilate mode.
        n_boot: Number of bootstrap resamples.
        seed: Bootstrap RNG seed.
    """

    checkpoint_dir: Path
    snippet_dir: Path
    output_csv: Path
    snippet_template: str
    oov_mode: str
    corpus_dir: Path
    assimilation_csv: Path
    n_boot: int
    seed: int


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def infer_num_layers(state_dict: dict[str, Tensor]) -> int:
    """Infer the number of LSTM layers from a saved state dict.

    Args:
        state_dict: Dictionary returned by ``torch.load`` for a CharLSTM
            checkpoint.

    Returns:
        Number of stacked LSTM layers (the maximum ``l<N>`` index plus 1).

    Raises:
        ValueError: If no ``lstm.weight_ih_l<N>`` key is present.
    """
    indices: list[int] = []
    for key in state_dict:
        match = _LSTM_LAYER_RE.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        msg = "No 'lstm.weight_ih_l<N>' keys in state dict; not a CharLSTM checkpoint."
        raise ValueError(msg)
    return max(indices) + 1


def load_model_with_vocab(checkpoint_path: Path, vocab_path: Path) -> LoadedModel:
    """Load a CharLSTM checkpoint together with its vocabulary.

    Architecture is inferred from the saved state dict so the returned
    model has the exact embedding, hidden, and layer dims that were used
    at training time.

    Args:
        checkpoint_path: Path to the saved ``*_best.pt`` model state dict.
        vocab_path: Path to the matching ``*_vocab.json`` file.

    Returns:
        :class:`LoadedModel` with model in eval mode and stoi mapping.

    Raises:
        ValueError: If the vocab size in the JSON does not match the
            checkpoint's embedding row count, or if the checkpoint lacks
            recognizable LSTM layer keys.
    """
    stoi, _itos, vocab_size, _unk = load_vocab_json(vocab_path)
    load_fn = _get_torch_load()
    state_dict = load_fn(str(checkpoint_path), map_location="cpu", weights_only=True)
    embed_weight = state_dict["embedding.weight"]
    embed_vocab_size = int(embed_weight.shape[0])
    if embed_vocab_size != vocab_size:
        msg = (
            f"Vocab/checkpoint mismatch: {vocab_path} has {vocab_size} tokens but "
            f"{checkpoint_path} embedding has {embed_vocab_size} rows."
        )
        raise ValueError(msg)
    embed_dim = int(embed_weight.shape[1])
    num_layers = infer_num_layers(state_dict)
    lstm_weight = state_dict["lstm.weight_ih_l0"]
    hidden_dim = int(lstm_weight.shape[0]) // 4

    model = CharLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=DROPOUT,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return {"model": model, "stoi": stoi, "vocab_size": vocab_size}


# ---------------------------------------------------------------------------
# Snippet parsing
# ---------------------------------------------------------------------------


def parse_sections(text: str) -> list[str]:
    """Split a perception snippet into numbered sections, dropping headers.

    Sections are delimited by lines that contain only a digit 1-5. The
    first line (language name), ``TEXT N`` title lines, blank lines, and
    short stray title lines immediately preceding a ``1`` marker are
    discarded. Sections shorter than :data:`MIN_SECTION_CHARS` are dropped.

    Args:
        text: Full snippet file content.

    Returns:
        Section texts in file order.
    """
    lines = text.splitlines()
    sections: list[list[str]] = []
    current: list[str] | None = None
    for i, line in enumerate(lines):
        if MARKER_RE.match(line):
            if current:
                sections.append(current)
            current = []
            continue
        if i == 0 or HEADER_RE.match(line) or not line.strip():
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if len(line.strip()) < 40 and MARKER_RE.match(nxt) and nxt.strip() == "1":
            continue
        if current is not None:
            current.append(line.strip())
    if current:
        sections.append(current)
    joined = (" ".join(s) for s in sections if s)
    return [s for s in joined if len(s) >= MIN_SECTION_CHARS]


def snippet_path(snippet_dir: Path, snippet_template: str, lang: str) -> Path:
    """Resolve the snippet file path for a given language.

    Args:
        snippet_dir: Directory containing snippet files.
        snippet_template: Filename template; must contain ``{lang}``.
        lang: Language code to substitute.

    Returns:
        Resolved :class:`Path` ``snippet_dir/<template-with-lang>``.

    Raises:
        ValueError: If ``snippet_template`` does not contain ``{lang}``.
    """
    if "{lang}" not in snippet_template:
        msg = f"snippet_template must contain '{{lang}}'; got {snippet_template!r}"
        raise ValueError(msg)
    return snippet_dir / snippet_template.format(lang=lang)


# ---------------------------------------------------------------------------
# Assimilation table
# ---------------------------------------------------------------------------


def load_assimilation_map(path: Path) -> dict[str, dict[str, str]]:
    """Load the per-listener OOV substitution table.

    The CSV must have columns ``listener,missing,replacement`` (extra
    documentation columns are ignored). Each row says: when the listener
    language's model encounters ``missing``, replace it with ``replacement``
    before encoding.

    Args:
        path: CSV path.

    Returns:
        Mapping ``listener -> {missing_char: replacement}`` covering every
        language in :data:`LANGS` (possibly with empty dicts).

    Raises:
        ValueError: If a row has an empty field or an unknown listener code.
    """
    mapping: dict[str, dict[str, str]] = {lang: {} for lang in LANGS}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            listener = row["listener"]
            missing = row["missing"]
            replacement = row["replacement"]
            if listener == "" or missing == "" or replacement == "":
                msg = f"assimilation table {path}: row with empty field: {row!r}"
                raise ValueError(msg)
            if listener not in mapping:
                msg = f"assimilation table {path}: unknown listener {listener!r}"
                raise ValueError(msg)
            mapping[listener][missing] = replacement
    return mapping


def apply_assimilation(text: str, substitutions: dict[str, str]) -> str:
    """Apply a listener's OOV substitutions to a section.

    Replacements run in the table's row order, which matters when one
    row's replacement is another row's missing character.

    Args:
        text: Section text.
        substitutions: One listener's ``missing -> replacement`` table.

    Returns:
        The section with every substitution applied.
    """
    for missing, replacement in substitutions.items():
        text = text.replace(missing, replacement)
    return text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def common_support_mask(section: str, vocabs: list[set[str]]) -> list[bool]:
    """Mask of next-char positions whose true target is in every vocab.

    Args:
        section: Section text.
        vocabs: Character sets of all loaded models.

    Returns:
        One bool per next-char position (``len(section) - 1`` entries).
    """
    return [all(ch in vocab for vocab in vocabs) for ch in section[1:]]


def score_section(
    loaded: LoadedModel,
    section: str,
    mask: list[bool] | None,
) -> SectionScore:
    """Score one section with a frozen model.

    The section is encoded into source-vocab indices (chars not in the
    vocab become UNK), shifted by one for next-character prediction, and
    run through the model in a single forward pass with zero initial
    hidden state.

    Args:
        loaded: Result of :func:`load_model_with_vocab`.
        section: Section text (already substituted in assimilate mode).
        mask: Positions to score; ``None`` scores every position. Must have
            ``len(section) - 1`` entries when given.

    Returns:
        :class:`SectionScore` with summed loss over the scored positions.

    Raises:
        ValueError: If the section is shorter than 2 characters or the mask
            length does not match the section.
    """
    if len(section) < 2:
        msg = f"Section too short for next-char prediction: {len(section)} chars."
        raise ValueError(msg)
    indices = encode(section, loaded["stoi"])
    inputs = torch.tensor([indices[:-1]], dtype=torch.long)
    targets = torch.tensor(indices[1:], dtype=torch.long)
    with torch.no_grad():
        logits, _hidden = loaded["model"](inputs)
    losses = functional.cross_entropy(
        logits.view(-1, loaded["vocab_size"]), targets, reduction="none"
    )
    n_total = int(targets.shape[0])
    if mask is None:
        return {"loss_sum": float(losses.sum().item()), "n_scored": n_total, "n_total": n_total}
    if len(mask) != n_total:
        msg = f"Mask length {len(mask)} != position count {n_total}."
        raise ValueError(msg)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return {
        "loss_sum": float(losses[mask_t].sum().item()),
        "n_scored": int(mask_t.sum().item()),
        "n_total": n_total,
    }


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
    excesses.sort()
    lo_idx = math.floor(0.025 * (n_boot - 1))
    hi_idx = math.ceil(0.975 * (n_boot - 1))
    return excesses[lo_idx], excesses[hi_idx]


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
    diffs.sort()
    lo_idx = math.floor(0.025 * (n_boot - 1))
    hi_idx = math.ceil(0.975 * (n_boot - 1))
    return diffs[lo_idx], diffs[hi_idx]


# ---------------------------------------------------------------------------
# CSV rendering
# ---------------------------------------------------------------------------


CSV_HEADER = (
    "listener_language,text_language,scoring_mode,cross_entropy,native_cross_entropy,"
    "excess_cross_entropy,excess_confidence_interval_low,excess_confidence_interval_high,"
    "fraction_of_positions_scored,number_of_positions_scored"
)


def excess_observations(results: list[PairResult]) -> tuple[Observation, ...]:
    """Name every excess-CE number so two runs can be paired by it.

    One observation per ordered language pair rather than a single mean. The
    mean is not the finding -- "Finnish is the hardest text for every Turkic
    listener" is a statement about the shape of the matrix, and a comparison
    that could only see its average would not be able to check it.

    Args:
        results: One entry per ordered (source, target) pair.

    Returns:
        The observations, named ``excess_ce.<src>.<tgt>``. Sorting is left to
        :func:`~platform_core.run_record.run_record`, which owns the order.
    """
    return tuple(
        Observation(name=f"excess_ce.{r['src']}.{r['tgt']}", value=r["excess_ce"]) for r in results
    )


ASYMMETRY_CSV_HEADER = (
    "language_a,language_b,scoring_mode,excess_a_reading_b,excess_b_reading_a,"
    "difference,difference_confidence_interval_low,difference_confidence_interval_high,"
    "interval_excludes_zero"
)


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
                    "yes" if r["excludes_zero"] else "no",
                ]
            )
        )
    return "\n".join(lines) + "\n"


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
        excess_ab = ce_from_scores(scores[(a, b)]) - ce_from_scores(scores[(b, b)])
        excess_ba = ce_from_scores(scores[(b, a)]) - ce_from_scores(scores[(a, a)])
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


def render_results_csv(results: list[PairResult]) -> str:
    """Render a list of :class:`PairResult` as a CSV string.

    Args:
        results: Evaluation results; output rows preserve the input order.

    Returns:
        CSV text including header and trailing newline.
    """
    lines = [CSV_HEADER]
    for r in results:
        lines.append(
            ",".join(
                [
                    r["src"],
                    r["tgt"],
                    r["mode"],
                    f"{r['ce']:.6f}",
                    f"{r['self_ce']:.6f}",
                    f"{r['excess_ce']:.6f}",
                    f"{r['excess_lo']:.6f}",
                    f"{r['excess_hi']:.6f}",
                    f"{r['support']:.6f}",
                    str(r["n_scored"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# run / main
# ---------------------------------------------------------------------------


def load_sources(checkpoint_dir: Path) -> dict[str, LoadedModel]:
    """Load every available source model from a checkpoint directory.

    Args:
        checkpoint_dir: Directory with ``{lang}_best.pt`` / ``{lang}_vocab.json``.

    Returns:
        Mapping of language code to loaded model, in :data:`LANGS` order.
    """
    sources: dict[str, LoadedModel] = {}
    for lang in LANGS:
        ckpt = checkpoint_dir / f"{lang}_best.pt"
        vocab = checkpoint_dir / f"{lang}_vocab.json"
        if not ckpt.exists() or not vocab.exists():
            print(f"  model missing for {lang} ({ckpt}); skipping as source")
            continue
        sources[lang] = load_model_with_vocab(ckpt, vocab)
    return sources


def _load_targets(args: EvalArgs, sources: dict[str, LoadedModel]) -> dict[str, list[str]]:
    """Load and parse target sections for every scorable language.

    A language is a usable target only if its snippet exists, parses into
    at least one section, and its own model is loaded (excess CE needs the
    self baseline).

    Args:
        args: Validated CLI arguments.
        sources: Loaded source models.

    Returns:
        Mapping of language code to its section list, in :data:`LANGS` order.
    """
    targets: dict[str, list[str]] = {}
    for lang in LANGS:
        path = snippet_path(args["snippet_dir"], args["snippet_template"], lang)
        if not path.exists():
            print(f"  snippet missing for {lang}: {path}; skipping as target")
            continue
        if lang not in sources:
            print(f"  no self model for {lang}; skipping as target (excess CE undefined)")
            continue
        sections = parse_sections(path.read_text(encoding="utf-8"))
        if not sections:
            print(f"  snippet for {lang} parsed to zero sections; skipping as target")
            continue
        targets[lang] = sections
    return targets


def attested_chars(corpus_path: Path, min_count: int) -> set[str]:
    """Characters a corpus contains at least ``min_count`` times.

    Vocabulary membership alone is a broken proxy for what a model has
    learned: a character quoted four times in ten million is in the
    vocabulary, but the model assigns it vanishing probability, and one
    such character unmasked a tenth of another language's text and
    dominated its score. A real phoneme of the training language occurs
    thousands of times, contamination occurs tens, so the floor separates
    them by orders of magnitude either way.

    Args:
        corpus_path: The listener's training corpus.
        min_count: Occurrences required to count as attested.

    Returns:
        The attested character set.
    """
    counts: collections.Counter[str] = collections.Counter(corpus_path.read_text(encoding="utf-8"))
    return {ch for ch, n in counts.items() if n >= min_count}


def _build_masks(
    targets: dict[str, list[str]],
    sources: dict[str, LoadedModel],
    corpus_dir: Path,
    min_attested: int = MIN_ATTESTED,
) -> dict[str, list[list[bool]]]:
    """Build common-support masks per target, dropping zero-support sections.

    Args:
        targets: Section lists per target language. Mutated in place when a
            section has no common-support position.
        sources: Loaded source models whose vocabularies define the support.
        corpus_dir: Training corpora, one per source, which decide
            attestation.
        min_attested: Occurrences a character needs in a corpus to count.

    Returns:
        Masks aligned with each target's (possibly filtered) sections.
    """
    vocabs = [
        set(loaded["stoi"])
        & attested_chars(corpus_dir / CORPUS_TEMPLATE.format(lang=lang), min_attested)
        for lang, loaded in sources.items()
    ]
    masks: dict[str, list[list[bool]]] = {}
    for lang, sections in targets.items():
        section_masks = [common_support_mask(s, vocabs) for s in sections]
        keep = [i for i, m in enumerate(section_masks) if sum(m) > 0]
        if len(keep) < len(sections):
            print(f"  {lang}: dropped {len(sections) - len(keep)} zero-support section(s)")
        targets[lang] = [sections[i] for i in keep]
        masks[lang] = [section_masks[i] for i in keep]
    return masks


def run(args: EvalArgs) -> list[PairResult]:
    """Run zero-shot evaluation across all (src, tgt) pairs and write CSV.

    Args:
        args: Validated CLI arguments.

    Returns:
        List of :class:`PairResult` for every evaluated pair.
    """
    sources = load_sources(args["checkpoint_dir"])
    targets = _load_targets(args, sources)
    masks = _build_masks(targets, sources, args["corpus_dir"]) if args["oov_mode"] == "skip" else {}
    assimilation = (
        load_assimilation_map(args["assimilation_csv"]) if args["oov_mode"] == "assimilate" else {}
    )

    scores: dict[tuple[str, str], list[SectionScore]] = {}
    for src, loaded in sources.items():
        for tgt, sections in targets.items():
            pair_sections = sections
            if args["oov_mode"] == "assimilate":
                subs = assimilation[src]
                pair_sections = [apply_assimilation(s, subs) for s in sections]
            pair_masks: list[list[bool] | None] = []
            if args["oov_mode"] == "skip":
                pair_masks.extend(masks[tgt])
            else:
                pair_masks.extend([None] * len(sections))
            scores[(src, tgt)] = [
                score_section(loaded, section, mask)
                for section, mask in zip(pair_sections, pair_masks, strict=True)
            ]

    results: list[PairResult] = []
    for src in sources:
        for tgt in targets:
            pair = scores[(src, tgt)]
            self_scores = scores[(tgt, tgt)]
            ce = ce_from_scores(pair)
            self_ce = ce_from_scores(self_scores)
            lo, hi = bootstrap_excess(pair, self_scores, args["n_boot"], args["seed"])
            n_scored = sum(s["n_scored"] for s in pair)
            n_total = sum(s["n_total"] for s in pair)
            result: PairResult = {
                "src": src,
                "tgt": tgt,
                "mode": args["oov_mode"],
                "ce": ce,
                "self_ce": self_ce,
                "excess_ce": ce - self_ce,
                "excess_lo": lo,
                "excess_hi": hi,
                "support": n_scored / n_total,
                "n_scored": n_scored,
            }
            results.append(result)
            print(
                f"  {src}->{tgt} [{args['oov_mode']}] ce={ce:.4f} "
                f"excess={ce - self_ce:+.4f} [{lo:+.4f},{hi:+.4f}] "
                f"support={result['support']:.3f}"
            )

    args["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    args["output_csv"].write_text(render_results_csv(results), encoding="utf-8")
    print(f"Wrote {len(results)} pair(s) to {args['output_csv']}")

    # The directional claim is about a DIFFERENCE, so it gets its own
    # interval rather than being read off two others. Written beside the
    # matrix because a reader comparing the matrix's intervals by eye is
    # applying a test that can only fail to detect an asymmetry.
    # Only languages that are BOTH a model and a scored target can be asked
    # the question: the reverse direction needs the other language's own
    # sections, and a target whose sections were all dropped for zero
    # support has none to give.
    paired = tuple(lang for lang in sources if lang in targets)
    asymmetries = asymmetry_results(scores, paired, args["oov_mode"], args["n_boot"], args["seed"])
    asymmetry_csv = args["output_csv"].with_name(
        args["output_csv"].stem + "_asymmetry" + args["output_csv"].suffix
    )
    asymmetry_csv.write_text(render_asymmetry_csv(asymmetries), encoding="utf-8")
    significant = sum(1 for r in asymmetries if r["excludes_zero"])
    print(
        f"Wrote {len(asymmetries)} asymmetry test(s) to {asymmetry_csv}; "
        f"{significant} interval(s) exclude zero"
    )

    # The CSV keeps its shape; the provenance goes beside it. Every number in
    # it is a subtraction, and until this sidecar existed the only record of
    # what produced them was the filename.
    sidecar = write_run_record(
        args["output_csv"],
        EXPERIMENT,
        args["oov_mode"],
        excess_observations(results),
        scoring_fingerprint(),
    )
    print(f"Wrote provenance to {sidecar}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Zero-shot excess-CE evaluation across language pairs.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--snippet-dir", type=str, default=str(DEFAULT_SNIPPET_DIR))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--snippet-template", type=str, default=DEFAULT_SNIPPET_TEMPLATE)
    parser.add_argument("--oov-mode", type=str, choices=OOV_MODES, default="skip")
    parser.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--assimilation-csv", type=str, default=str(DEFAULT_ASSIMILATION_CSV))
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def _extract_args(namespace: argparse.Namespace) -> EvalArgs:
    """Validate and convert an argparse Namespace into typed EvalArgs.

    Args:
        namespace: Parsed argparse namespace from :func:`_build_arg_parser`.

    Returns:
        Validated :class:`EvalArgs`.

    Raises:
        TypeError: If any argument has an unexpected type.
        ValueError: If ``oov_mode`` is unknown or ``n_boot`` is not positive.
    """
    str_fields = {
        "checkpoint_dir": namespace.checkpoint_dir,
        "snippet_dir": namespace.snippet_dir,
        "output_csv": namespace.output_csv,
        "snippet_template": namespace.snippet_template,
        "oov_mode": namespace.oov_mode,
        "corpus_dir": namespace.corpus_dir,
        "assimilation_csv": namespace.assimilation_csv,
    }
    for name, value in str_fields.items():
        if not isinstance(value, str):
            msg = f"Expected str for --{name.replace('_', '-')}, got {type(value).__name__}"
            raise TypeError(msg)
    if not isinstance(namespace.n_boot, int):
        msg = f"Expected int for --n-boot, got {type(namespace.n_boot).__name__}"
        raise TypeError(msg)
    if not isinstance(namespace.seed, int):
        msg = f"Expected int for --seed, got {type(namespace.seed).__name__}"
        raise TypeError(msg)
    if str_fields["oov_mode"] not in OOV_MODES:
        msg = f"Unknown --oov-mode {str_fields['oov_mode']!r}; expected one of {OOV_MODES}"
        raise ValueError(msg)
    if namespace.n_boot < 1:
        msg = f"--n-boot must be >= 1, got {namespace.n_boot}"
        raise ValueError(msg)
    return {
        "checkpoint_dir": Path(str_fields["checkpoint_dir"]),
        "snippet_dir": Path(str_fields["snippet_dir"]),
        "output_csv": Path(str_fields["output_csv"]),
        "snippet_template": str_fields["snippet_template"],
        "oov_mode": str_fields["oov_mode"],
        "corpus_dir": Path(str_fields["corpus_dir"]),
        "assimilation_csv": Path(str_fields["assimilation_csv"]),
        "n_boot": namespace.n_boot,
        "seed": namespace.seed,
    }


def parse_args(argv: list[str] | None = None) -> EvalArgs:
    """Parse CLI arguments into a typed :class:`EvalArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`EvalArgs`.
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
