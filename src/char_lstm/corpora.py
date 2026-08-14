"""The experiment's data layout: which languages, and where their files live.

These constants used to live beside the scripts, which meant the training
module in ``src`` could not name a corpus file without hardcoding its own
copy of the layout. The layout now lives with the library so that both the
trainer and the scripts read one definition; scripts import it from here.
"""

from __future__ import annotations

from pathlib import Path

LANGS: tuple[str, ...] = ("az", "fi", "kk", "ky", "tr", "ug", "uz")

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
