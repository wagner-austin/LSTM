"""The per-section scores behind an excess-CE matrix, as a JSON document.

A matrix row is a summary: a pooled excess and its interval. Every question
asked of the matrix afterwards -- does this listener differ from that one,
does this pair read differently each way -- needs the SECTION scores the
summary was pooled from, because the answer depends on which sections the
two numbers share. Until 2026-10-06 nothing kept them, so re-testing a
published table meant re-scoring it, and the published skip matrix could
only be re-scored by rebuilding the July pipeline that produced it.

The document is::

    {"mode": "skip",
     "pairs": [{"listener": "az", "text": "kk",
                "sections": [{"loss_sum": 12.3, "n_scored": 7, "n_total": 9}, ...]},
               ...]}

with one entry per (listener, text) cell in matrix row order. Listener and
text order are read back from that order, so a round trip preserves the
matrix layout exactly.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from scripts.zero_shot_bootstrap import SectionScore

SECTION_SCORES_SUFFIX = "_section_scores.json"


class SectionScoreTable(TypedDict):
    """Every section score of one matrix, keyed by (listener, text).

    Attributes:
        mode: OOV regime the sections were scored under.
        listeners: Listener languages in matrix row order.
        texts: Text languages in matrix column order. Every text is also a
            listener, since its own scores are its excess baseline.
        scores: Per-section scores for every (listener, text) cell. All
            listeners of one text carry the same number of sections.
    """

    mode: str
    listeners: tuple[str, ...]
    texts: tuple[str, ...]
    scores: dict[tuple[str, str], list[SectionScore]]


def _decode_section(value: JSONValue) -> SectionScore:
    """Decode one section's score.

    Args:
        value: The JSON object for one section.

    Returns:
        The typed score.

    Raises:
        ValueError: If a count is negative or more positions were scored
            than the section has.
    """
    obj = narrow_json_to_dict(value)
    section = SectionScore(
        loss_sum=require_float(obj, "loss_sum"),
        n_scored=require_int(obj, "n_scored"),
        n_total=require_int(obj, "n_total"),
    )
    if not 0 <= section["n_scored"] <= section["n_total"]:
        msg = (
            f"Section scored {section['n_scored']} of {section['n_total']} positions; "
            "a count must lie between zero and the section's length."
        )
        raise ValueError(msg)
    return section


def _append_once(order: list[str], lang: str) -> None:
    """Record a language the first time it is seen, keeping first-seen order.

    Args:
        order: Languages seen so far; extended in place.
        lang: The language just read.
    """
    if lang not in order:
        order.append(lang)


def decode_section_score_table(value: JSONValue) -> SectionScoreTable:
    """Decode and validate a section-score document.

    Args:
        value: The parsed JSON document.

    Returns:
        The typed table.

    Raises:
        ValueError: If a cell appears twice, a (listener, text) cell is
            missing, a text has no self-scores, or two listeners of one
            text carry different section counts.
    """
    obj: JSONObject = narrow_json_to_dict(value)
    mode = require_str(obj, "mode")
    listeners: list[str] = []
    texts: list[str] = []
    scores: dict[tuple[str, str], list[SectionScore]] = {}
    for entry in require_list(obj, "pairs"):
        pair = narrow_json_to_dict(entry)
        key = (require_str(pair, "listener"), require_str(pair, "text"))
        if key in scores:
            msg = f"Cell {key[0]}->{key[1]} appears twice."
            raise ValueError(msg)
        scores[key] = [_decode_section(s) for s in require_list(pair, "sections")]
        _append_once(listeners, key[0])
        _append_once(texts, key[1])
    for text in texts:
        if text not in listeners:
            msg = f"Text {text} has no self-scores, so no excess can be taken against it."
            raise ValueError(msg)
        counts: set[int] = set()
        for listener in listeners:
            if (listener, text) not in scores:
                msg = f"Cell {listener}->{text} is missing."
                raise ValueError(msg)
            counts.add(len(scores[(listener, text)]))
        if len(counts) != 1:
            msg = f"Listeners of text {text} carry different section counts: {sorted(counts)}."
            raise ValueError(msg)
    return SectionScoreTable(
        mode=mode, listeners=tuple(listeners), texts=tuple(texts), scores=scores
    )


def load_section_score_table(raw: str) -> SectionScoreTable:
    """Parse and decode a section-score document from its text.

    Args:
        raw: The document's JSON text.

    Returns:
        The typed table, validated by :func:`decode_section_score_table`.
    """
    return decode_section_score_table(load_json_str(raw))


def encode_section_score_table(table: SectionScoreTable) -> str:
    """Encode a table as the document :func:`load_section_score_table` reads.

    Args:
        table: The table to write; every (listener, text) cell must be in
            ``scores``.

    Returns:
        The JSON text, indented, with a trailing newline.
    """
    pairs: list[JSONValue] = []
    for listener in table["listeners"]:
        for text in table["texts"]:
            sections: list[JSONValue] = [
                {"loss_sum": s["loss_sum"], "n_scored": s["n_scored"], "n_total": s["n_total"]}
                for s in table["scores"][(listener, text)]
            ]
            pairs.append({"listener": listener, "text": text, "sections": sections})
    document: JSONObject = {"mode": table["mode"], "pairs": pairs}
    return dump_json_str(document, indent=1) + "\n"
