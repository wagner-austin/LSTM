"""The data layout is pinned, because file names are a cross-repo contract.

The corpora are produced by turkic-clean-corpus in the turkic-translit
repository and consumed by training here; the perception files are named
the same way on both sides of that boundary. A silent change to either
template would break the contract without any import failing, so the
exact values are asserted.
"""

from __future__ import annotations

from scripts.corpora import CORPUS_TEMPLATE, LANGS, SNIPPET_TEMPLATE


def test_the_language_set_is_the_seven_of_the_paper() -> None:
    """Six Turkic languages and the Finnish control, in sorted order."""
    assert LANGS == ("az", "fi", "kk", "ky", "tr", "ug", "uz")


def test_the_templates_produce_the_published_file_names() -> None:
    """The names match the artifacts already on disk and in manifests."""
    assert CORPUS_TEMPLATE.format(lang="kk") == "oscar_kk_ipa.txt"
    assert SNIPPET_TEMPLATE.format(lang="kk") == "perception_kk.txt"
