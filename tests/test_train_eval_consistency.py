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

The training side is read from the committed letter-rate profile
(``data/training_letter_rates.json``, written by ``scripts.letter_rates``),
not from the corpora, which are untracked: a clean checkout has the profile
and not the 104 MB of text it was taken from.
"""

from __future__ import annotations

import unicodedata as ud
from pathlib import Path

import pytest
from scripts.letter_rates import DEFAULT_PROFILE, letter_rates, load_letter_rate_profile

from char_lstm.corpora import PERCEPTION_LANGS, SNIPPET_TEMPLATE

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / DEFAULT_PROFILE
EVAL_DIR = REPO / "data" / "perception_clean"

# Per 1,000 characters: a symbol carrying real weight on one side...
COMMON_PER_1K = 5.0
# ...while effectively absent from the other.
ABSENT_PER_1K = 1.0

KNOWN_MISMATCHED: frozenset[str] = frozenset()


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
        Train rates from the committed profile and evaluation rates
        taken from the perception text.
    """
    profile = {p["lang"]: p for p in load_letter_rate_profile(PROFILE.read_text("utf-8"))}
    evaluation = (EVAL_DIR / SNIPPET_TEMPLATE.format(lang=lang)).read_text(encoding="utf-8")
    return profile[lang]["rates"], letter_rates(evaluation)


def test_the_profile_covers_every_perception_language_from_one_generation() -> None:
    """Every evaluated language has a training profile, all from one corpus set.

    A profile mixing generations would compare some evaluation texts
    against a corpus no current model was trained on.
    """
    profiles = load_letter_rate_profile(PROFILE.read_text(encoding="utf-8"))

    assert tuple(p["lang"] for p in profiles) == PERCEPTION_LANGS
    assert {p["corpus"].split(":")[0] for p in profiles} == {"corpora_clean"}


def test_a_convention_swap_is_detected() -> None:
    """The check refuses the Kazakh w-for-u swap it was written for."""
    train = {"w": 30.0, "a": 80.0}
    evaluation = {"u": 30.0, "a": 80.0}

    offending = swapped_pairs(train, evaluation)

    assert offending == [
        "'w' (LATIN SMALL LETTER W): train 30.0/1k, eval 0.0/1k",
        "'u' (LATIN SMALL LETTER U): eval 30.0/1k, train 0.0/1k",
    ]


@pytest.mark.parametrize("lang", sorted(set(PERCEPTION_LANGS) - KNOWN_MISMATCHED))
def test_training_and_evaluation_share_one_notation(lang: str) -> None:
    """No letter's mass moves wholesale between symbols across the seam.

    Args:
        lang: Language whose train and evaluation texts are compared.
    """
    train, evaluation = rates_for(lang)

    offending = swapped_pairs(train, evaluation)

    assert not offending, f"{lang} train/eval look like different notations: {offending}"
