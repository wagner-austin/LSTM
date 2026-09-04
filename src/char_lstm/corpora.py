"""The experiment's data layout: which languages, and where their files live.

These constants used to live beside the scripts, which meant the training
module in ``src`` could not name a corpus file without hardcoding its own
copy of the layout. The layout now lives with the library so that both the
trainer and the scripts read one definition; scripts import it from here.
"""

from __future__ import annotations

from pathlib import Path

LANGS: tuple[str, ...] = ("az", "fi", "kk", "ky", "ru", "tr", "ug", "uz")
"""Languages with a training corpus.

Russian joined on 2026-09-03. It is not Turkic, and neither is Finnish: both
are here as controls. Finnish is the agglutinative non-Turkic control, and
Russian is the CONTACT language the Cyrillic-script corpora borrow from,
which makes it the control that separates genealogical relatedness from
shared loan vocabulary.
"""

PERCEPTION_LANGS: tuple[str, ...] = ("az", "fi", "kk", "ky", "tr", "ug", "uz")
"""Languages with listener-perception snippets, a strict subset of :data:`LANGS`.

These two were the same tuple until Russian arrived, which is why nothing
named the difference. They are different questions: :data:`LANGS` asks what
a model can be trained on, and this asks what a human listener was actually
played. A corpus can be built from public text; a perception snippet cannot,
because it has to correspond to a recording that participants heard.

Anything comparing a training corpus against evaluation text must range over
this tuple, not over :data:`LANGS`, or it demands a file that no experiment
produced.
"""

CORPUS_TEMPLATE = "oscar_{lang}_ipa.txt"
SNIPPET_TEMPLATE = "perception_{lang}.txt"


def corpus_file(corpus_dir: Path, lang: str) -> Path:
    """Path of one language's corpus inside a corpus directory.

    Args:
        corpus_dir: Directory holding the cleaned corpora.
        lang: Language code, one of :data:`LANGS`.

    Returns:
        The corpus path ``corpus_dir/oscar_{lang}_ipa.txt``.

    Raises:
        ValueError: If ``lang`` is not one of :data:`LANGS`; a typo here
            would otherwise surface only as a missing-file error at read
            time, without naming the argument that caused it.
    """
    if lang not in LANGS:
        msg = f"Unknown language code {lang!r}; expected one of {LANGS}"
        raise ValueError(msg)
    return corpus_dir / CORPUS_TEMPLATE.format(lang=lang)
