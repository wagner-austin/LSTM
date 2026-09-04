"""A model must be evaluated in the notation it was trained on.

The Kazakh column of the zero-shot matrix was unusable because nothing
checked this: the training corpus wrote Cyrillic u as w while the perception
text wrote it as u, so the Kazakh model met its own language's evaluation
text in a spelling it had never seen, and its inflated native
cross-entropy deflated every cell in its column.

Genre differences between web text and read prose are expected and
harmless: punctuation frequencies differ, rare loan phonemes may not
occur in twenty short passages. What genre cannot do is move probability
mass between two letter symbols — one common in training and absent from
evaluation, while another is common in evaluation and absent from
training. That paired swap is the signature of two texts written under
different transliteration conventions, and it is what these tests refuse.
"""

from __future__ import annotations

import collections
import unicodedata as ud
from pathlib import Path

import pytest

from char_lstm.corpora import CORPUS_TEMPLATE, PERCEPTION_LANGS, SNIPPET_TEMPLATE

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "corpora_clean"
EVAL_DIR = REPO / "data" / "perception_clean"

# Per 1,000 characters: a symbol carrying real weight on one side...
COMMON_PER_1K = 5.0
# ...while effectively absent from the other.
ABSENT_PER_1K = 1.0

KNOWN_MISMATCHED: frozenset[str] = frozenset()


def letter_rates(path: Path) -> dict[str, float]:
    """Occurrences per 1,000 characters for every letter in a file.

    Case, digits and punctuation are excluded: headers contribute
    uppercase, and genre legitimately moves punctuation. Letters are
    where a notation difference would live.

    Args:
        path: Text file to profile.

    Returns:
        Letter to rate per 1,000 characters of the file.
    """
    text = path.read_text(encoding="utf-8")
    counts = collections.Counter(ch for ch in text if ch.isalpha() and not ch.isupper())
    scale = len(text) / 1000
    return {ch: n / scale for ch, n in counts.items()}


def swapped_pairs(train: dict[str, float], evaluation: dict[str, float]) -> list[str]:
    """Symbols whose mass sits on one side and is missing from the other.

    Args:
        train: Letter rates of the training corpus.
        evaluation: Letter rates of the evaluation text.

    Returns:
        Descriptions of offending symbols, one direction each; a
        convention mismatch produces at least one in each direction.
    """
    gone_from_eval = [
        f"{ch!r} ({ud.name(ch, ch)}): train {train[ch]:.1f}/1k,"
        f" eval {evaluation.get(ch, 0.0):.1f}/1k"
        for ch, rate in train.items()
        if rate >= COMMON_PER_1K and evaluation.get(ch, 0.0) <= ABSENT_PER_1K
    ]
    gone_from_train = [
        f"{ch!r} ({ud.name(ch, ch)}): eval {evaluation[ch]:.1f}/1k,"
        f" train {train.get(ch, 0.0):.1f}/1k"
        for ch, rate in evaluation.items()
        if rate >= COMMON_PER_1K and train.get(ch, 0.0) <= ABSENT_PER_1K
    ]
    if gone_from_eval and gone_from_train:
        return gone_from_eval + gone_from_train
    return []


def rates_for(lang: str) -> tuple[dict[str, float], dict[str, float]]:
    """Letter rates of a language's training corpus and evaluation text.

    Args:
        lang: Language code.

    Returns:
        Train and evaluation rate tables.
    """
    return (
        letter_rates(TRAIN_DIR / CORPUS_TEMPLATE.format(lang=lang)),
        letter_rates(EVAL_DIR / SNIPPET_TEMPLATE.format(lang=lang)),
    )


@pytest.mark.parametrize("lang", sorted(set(PERCEPTION_LANGS) - KNOWN_MISMATCHED))
def test_training_and_evaluation_share_one_notation(lang: str) -> None:
    """No letter's mass moves wholesale between symbols across the seam.

    Args:
        lang: Language whose train and evaluation texts are compared.
    """
    train, evaluation = rates_for(lang)

    offending = swapped_pairs(train, evaluation)

    assert not offending, f"{lang} train/eval look like different notations: {offending}"
