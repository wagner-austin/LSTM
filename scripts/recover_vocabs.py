"""Reconstruct per-language vocabs lost to a vocab-persistence bug.

train.py historically saved every base's vocab to ``checkpoints/vocab.json``,
so each from-scratch training run overwrote the previous one. Only the last
base trained (Finnish) had its vocab survive. This script regenerates every
base's vocab by re-running :func:`build_vocab_with_unk` on the unchanged
corpus -- a pure deterministic function of input text -- so the reconstructed
vocab is bit-for-bit what training used at the time.

Outputs (under ``--checkpoint-dir``):
    - ``{lang}_vocab.json``       per-language vocab, one per :data:`LANGS`
    - ``union_vocab.json``        union of all language char sets, UNK at end
    - ``vocab_coverage.csv``      per-pair coverage matrix (rows=src, cols=tgt)

Usage::

    poetry run python -m scripts.recover_vocabs
    poetry run python -m scripts.recover_vocabs --corpus-dir corpora_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

from char_lstm.data import UNK, build_vocab_with_unk, save_vocab_json

LANGS: tuple[str, ...] = ("az", "fi", "kk", "ky", "tr", "ug", "uz")
DEFAULT_CORPUS_DIR = Path("corpora_raw")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
MAX_CHARS = 10_000_000


class RecoveryArgs(TypedDict):
    """Parsed and validated CLI arguments for vocabulary recovery.

    Attributes:
        corpus_dir: Directory containing ``oscar_{lang}_ipa.txt`` files.
        checkpoint_dir: Directory to write vocab JSONs and coverage CSV into.
    """

    corpus_dir: Path
    checkpoint_dir: Path


def reconstruct_vocab(
    corpus_path: Path,
    max_chars: int = MAX_CHARS,
) -> tuple[dict[str, int], dict[int, str], int]:
    """Rebuild a single language's vocab from its training corpus.

    Re-runs :func:`build_vocab_with_unk` over the same byte slice the original
    training run consumed, producing a vocab identical to the lost one.

    Args:
        corpus_path: Path to the IPA-transliterated corpus text file.
        max_chars: Maximum number of leading characters to read. Matches the
            training pipeline's truncation in ``train.py``.

    Returns:
        Tuple of ``(stoi, itos, vocab_size)`` matching the contract of
        :func:`build_vocab_with_unk`.
    """
    text = corpus_path.read_text(encoding="utf-8")[:max_chars]
    return build_vocab_with_unk(text)


def build_union_vocab(
    per_lang_stoi: dict[str, dict[str, int]],
    unk: str = UNK,
) -> tuple[dict[str, int], dict[int, str], int]:
    """Union character sets across languages, appending UNK at the end.

    Args:
        per_lang_stoi: Mapping of language code to that language's stoi.
        unk: Token to append at the end of the union vocab. Excluded from
            the union before UNK is appended so it appears exactly once.

    Returns:
        Tuple of ``(union_stoi, union_itos, vocab_size)``.
    """
    chars: set[str] = set()
    for lang_stoi in per_lang_stoi.values():
        for ch in lang_stoi:
            if ch != unk:
                chars.add(ch)
    sorted_chars = sorted(chars)
    sorted_chars.append(unk)
    union_stoi: dict[str, int] = {ch: i for i, ch in enumerate(sorted_chars)}
    union_itos: dict[int, str] = dict(enumerate(sorted_chars))
    return union_stoi, union_itos, len(sorted_chars)


def coverage_of(src_chars: set[str], tgt_chars: set[str], unk: str = UNK) -> float:
    """Compute the fraction of target characters present in the source vocab.

    UNK is excluded from both sides because it is a structural token, not a
    character drawn from the language's character distribution.

    Args:
        src_chars: Character set of the source language vocab.
        tgt_chars: Character set of the target language vocab.
        unk: UNK token to exclude from both sides before computing coverage.

    Returns:
        Fraction in ``[0.0, 1.0]``. Returns ``1.0`` when the target vocab is
        empty after removing UNK (vacuously fully covered).
    """
    src = src_chars - {unk}
    tgt = tgt_chars - {unk}
    if not tgt:
        return 1.0
    return len(src & tgt) / len(tgt)


def render_coverage_csv(per_lang_stoi: dict[str, dict[str, int]], unk: str = UNK) -> str:
    """Render a per-pair coverage matrix as a CSV string.

    Args:
        per_lang_stoi: Mapping of language code to that language's stoi.
        unk: UNK token to exclude from coverage computation.

    Returns:
        CSV text. Header row is ``src,<lang1>,<lang2>,...`` and each
        subsequent row reports coverage of that source against every target.
        Trailing newline included.
    """
    langs = sorted(per_lang_stoi.keys())
    char_sets = {lang: set(per_lang_stoi[lang].keys()) for lang in langs}
    lines: list[str] = ["src," + ",".join(langs)]
    for src in langs:
        cells = [f"{coverage_of(char_sets[src], char_sets[tgt], unk):.4f}" for tgt in langs]
        lines.append(f"{src}," + ",".join(cells))
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` configured with the recovery
        script's options. All ``--corpus-dir`` and ``--checkpoint-dir``
        values are accepted as strings and converted to ``Path`` later.
    """
    parser = argparse.ArgumentParser(
        description="Reconstruct per-language vocabs from training corpora.",
    )
    parser.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    return parser


def _extract_args(namespace: argparse.Namespace) -> RecoveryArgs:
    """Validate and convert an argparse Namespace into a typed RecoveryArgs.

    Args:
        namespace: Parsed argparse namespace produced by
            :func:`_build_arg_parser`.

    Returns:
        Validated :class:`RecoveryArgs` with both paths as :class:`Path`.

    Raises:
        TypeError: If either ``--corpus-dir`` or ``--checkpoint-dir`` is not
            a string. Defensive guard against future ``add_argument`` changes
            that would silently produce non-string values.
    """
    corpus_raw = namespace.corpus_dir
    checkpoint_raw = namespace.checkpoint_dir
    if not isinstance(corpus_raw, str):
        msg = f"Expected str for --corpus-dir, got {type(corpus_raw).__name__}"
        raise TypeError(msg)
    if not isinstance(checkpoint_raw, str):
        msg = f"Expected str for --checkpoint-dir, got {type(checkpoint_raw).__name__}"
        raise TypeError(msg)
    return {"corpus_dir": Path(corpus_raw), "checkpoint_dir": Path(checkpoint_raw)}


def parse_args(argv: list[str] | None = None) -> RecoveryArgs:
    """Parse CLI arguments into a typed :class:`RecoveryArgs`.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Typed :class:`RecoveryArgs` with both paths as :class:`Path`.
    """
    parser = _build_arg_parser()
    namespace = parser.parse_args(argv)
    return _extract_args(namespace)


def run(
    corpus_dir: Path,
    checkpoint_dir: Path,
    langs: tuple[str, ...] = LANGS,
) -> dict[str, int]:
    """Reconstruct vocabs and write per-language, union, and coverage outputs.

    For each language whose corpus exists, this writes
    ``{checkpoint_dir}/{lang}_vocab.json``. After processing all languages,
    if at least one corpus was found, it also writes ``union_vocab.json`` and
    ``vocab_coverage.csv`` into ``checkpoint_dir``.

    Args:
        corpus_dir: Directory containing ``oscar_{lang}_ipa.txt`` files.
        checkpoint_dir: Directory to write outputs into. Created if missing.
        langs: Language codes to process. Defaults to :data:`LANGS`.

    Returns:
        Mapping of ``{lang: vocab_size}`` for each language whose corpus was
        present and processed. Empty if no corpora were found.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    per_lang_stoi: dict[str, dict[str, int]] = {}
    sizes: dict[str, int] = {}

    print(f"Reconstructing vocabs from {corpus_dir}/")
    for lang in langs:
        corpus_path = corpus_dir / f"oscar_{lang}_ipa.txt"
        if not corpus_path.exists():
            print(f"  {lang}: corpus not found at {corpus_path}; skipping")
            continue
        stoi, itos, vocab_size = reconstruct_vocab(corpus_path)
        out_path = checkpoint_dir / f"{lang}_vocab.json"
        save_vocab_json(itos, out_path)
        per_lang_stoi[lang] = stoi
        sizes[lang] = vocab_size
        print(f"  {lang}: {vocab_size:>5} tokens -> {out_path.name}")

    if not per_lang_stoi:
        print("No corpora found; nothing to write.")
        return sizes

    _union_stoi, union_itos, union_size = build_union_vocab(per_lang_stoi)
    union_path = checkpoint_dir / "union_vocab.json"
    save_vocab_json(union_itos, union_path)
    print(f"Union: {union_size} tokens -> {union_path.name}")

    coverage_csv = render_coverage_csv(per_lang_stoi)
    coverage_path = checkpoint_dir / "vocab_coverage.csv"
    coverage_path.write_text(coverage_csv, encoding="utf-8")
    print(f"Coverage matrix -> {coverage_path.name}")
    return sizes


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Optional list of CLI tokens. ``None`` defers to ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success.
    """
    args = parse_args(argv)
    run(args["corpus_dir"], args["checkpoint_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
