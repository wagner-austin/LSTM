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

from char_lstm.corpora import CORPUS_TEMPLATE, LANGS, SNIPPET_TEMPLATE, corpus_file


def test_the_language_set_is_the_seven_of_the_paper() -> None:
    """Six Turkic languages and the Finnish control, in sorted order."""
    assert LANGS == ("az", "fi", "kk", "ky", "tr", "ug", "uz")


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
