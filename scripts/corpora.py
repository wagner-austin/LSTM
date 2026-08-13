"""The experiment's data layout: which languages, and where their files live.

These constants used to live in the corpus-cleaning script, which made
every consumer of a file name import the cleaning pipeline. The pipeline
itself now ships with turkic-translit (``turkic-clean-corpus``), where it
is tested against this project's published corpora byte for byte, so what
remains here is only what is genuinely this experiment's own: the
language set and the naming scheme of its artifacts.
"""

from __future__ import annotations

LANGS: tuple[str, ...] = ("az", "fi", "kk", "ky", "tr", "ug", "uz")

CORPUS_TEMPLATE = "oscar_{lang}_ipa.txt"
SNIPPET_TEMPLATE = "perception_{lang}.txt"
