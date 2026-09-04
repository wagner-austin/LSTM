"""The data layout is pinned, because file names are a cross-repo contract.

The corpora are produced by turkic-clean-corpus in the turkic-translit
repository and consumed by training here; the perception files are named
the same way on both sides of that boundary. A silent change to either
template would break the contract without any import failing, so the
exact values are asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from char_lstm.corpora import (
    CORPUS_TEMPLATE,
    LANGS,
    PERCEPTION_LANGS,
    SNIPPET_TEMPLATE,
    corpus_file,
)


def test_the_language_set_is_the_eight_with_a_corpus() -> None:
    """Six Turkic languages and two non-Turkic controls, in sorted order.

    Seven until 2026-09-03, when Russian was added. Finnish is the
    agglutinative control and Russian the contact-language control.
    """
    assert LANGS == ("az", "fi", "kk", "ky", "ru", "tr", "ug", "uz")


def test_perception_languages_are_the_subset_with_listener_data() -> None:
    """Russian has a corpus and no recording, so the two sets diverged.

    They were the same tuple until Russian arrived, and nothing named the
    difference because nothing needed to. Anything comparing a training
    corpus against evaluation text ranges over this one.
    """
    assert PERCEPTION_LANGS == ("az", "fi", "kk", "ky", "tr", "ug", "uz")
    assert set(PERCEPTION_LANGS) < set(LANGS)
    assert set(LANGS) - set(PERCEPTION_LANGS) == {"ru"}


def test_the_templates_produce_the_published_file_names() -> None:
    """The names match the artifacts already on disk and in manifests."""
    assert CORPUS_TEMPLATE.format(lang="kk") == "oscar_kk_ipa.txt"
    assert SNIPPET_TEMPLATE.format(lang="kk") == "perception_kk.txt"


def test_corpus_file_joins_directory_and_published_name() -> None:
    """corpus_file resolves a language to its file inside a directory."""
    assert corpus_file(Path("corpora_clean"), "ug") == Path("corpora_clean/oscar_ug_ipa.txt")
    assert corpus_file(Path("variants/nopunct"), "fi") == Path("variants/nopunct/oscar_fi_ipa.txt")


def test_corpus_file_rejects_unknown_language() -> None:
    """A typo in the language code fails by name, not as a missing file."""
    with pytest.raises(ValueError, match="Unknown language code 'xx'"):
        corpus_file(Path("corpora_clean"), "xx")


def test_the_trainers_language_table_matches_the_corpus_layout() -> None:
    """Two hand-maintained copies of one set, now held together.

    ``train.py`` keeps ``LANGUAGES`` for display names and builds the
    ``--lang`` choices from its keys, while ``corpus_file`` validates
    against :data:`LANGS`. Nothing compared them, so adding a language to
    one and not the other would have produced either a ``--lang`` the
    corpus layer rejects by name, or a corpus no command line can reach.
    Neither failure names the real cause.
    """
    from char_lstm.train import LANGUAGES

    assert set(LANGUAGES) == set(LANGS)
