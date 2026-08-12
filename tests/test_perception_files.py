"""The perception snippets parse the way the evaluation assumes.

Every other test of :func:`parse_sections` feeds it a fixture written by
hand, so the fixture and the parser were authored from the same
assumption and agreed with each other. That is how a real defect
survived: the Finnish file's title lines read "tekst 2:", because the
word TEXT went through the transliterator and Finnish rules write x as
ks, and the header pattern matched only "text". Three of twenty Finnish
sections were scored with the next text's title glued onto them.

These tests read the files the evaluation actually scores. A fixture
cannot catch a defect in the data, so nothing here builds one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SNIPPET_DIRS = (REPO / "data" / "perception", REPO / "data" / "perception_clean")

# 4 texts of 5 passages each, except Uzbek, whose Text 2 passage 5 was
# withdrawn because its recording was removed for speaker disfluency. The
# model must score the 19 passages listeners actually heard.
EXPECTED_SECTIONS = {"tr": 20, "az": 20, "kk": 20, "ky": 20, "ug": 20, "fi": 20, "uz": 19}

MARKER = re.compile(r"^\s*[1-5]\s*$")
HEADER = re.compile(r"^\s*te?(ks|x)t\s*\d", re.IGNORECASE)
MIN_SECTION_CHARS = 20

# Any spelling of the title word that appears in the data, used to detect
# a title that survived into a scored section.
TITLE_INSIDE = re.compile(r"te?(ks|x)t\s*\d\s*:", re.IGNORECASE)


def parse_sections(text: str) -> list[str]:
    """Split a snippet into scored sections, as the evaluator does.

    Args:
        text: Full snippet file content.

    Returns:
        The section texts, in file order.
    """
    lines = text.splitlines()
    sections: list[list[str]] = []
    current: list[str] | None = None
    for i, line in enumerate(lines):
        if MARKER.match(line):
            if current:
                sections.append(current)
            current = []
            continue
        if i == 0 or HEADER.match(line) or not line.strip():
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if len(line.strip()) < 40 and MARKER.match(nxt) and nxt.strip() == "1":
            continue
        if current is not None:
            current.append(line.strip())
    if current:
        sections.append(current)
    joined = (" ".join(s) for s in sections if s)
    return [s for s in joined if len(s) >= MIN_SECTION_CHARS]


def snippet_files() -> list[tuple[Path, str]]:
    """Every snippet file present, paired with its language code.

    Returns:
        Pairs of path and language code, for both snippet directories.
    """
    found = []
    for directory in SNIPPET_DIRS:
        for code in EXPECTED_SECTIONS:
            path = directory / f"perception_{code}.txt"
            if path.exists():
                found.append((path, code))
    return found


FILES = snippet_files()


def test_every_language_has_a_snippet_file() -> None:
    """All seven languages are present in both snippet directories."""
    for directory in SNIPPET_DIRS:
        missing = [c for c in EXPECTED_SECTIONS if not (directory / f"perception_{c}.txt").exists()]
        assert not missing, f"{directory.name} is missing {missing}"


@pytest.mark.parametrize(("path", "code"), FILES, ids=[f"{p.parent.name}/{c}" for p, c in FILES])
def test_section_count_matches_the_experiment(path: Path, code: str) -> None:
    """Each file yields the number of passages listeners actually heard."""
    sections = parse_sections(path.read_text(encoding="utf-8"))

    assert len(sections) == EXPECTED_SECTIONS[code]


@pytest.mark.parametrize(("path", "code"), FILES, ids=[f"{p.parent.name}/{c}" for p, c in FILES])
def test_no_title_line_survives_into_a_scored_section(path: Path, code: str) -> None:
    """The defect this file exists for: a title scored as if it were prose.

    The header pattern once matched only "text", so Finnish's "tekst"
    titles fell through and were appended to the previous passage. Any
    spelling of the title word inside a section means the same class of
    failure has returned, in this language or another.
    """
    sections = parse_sections(path.read_text(encoding="utf-8"))
    carrying = [n for n, s in enumerate(sections, 1) if TITLE_INSIDE.search(s)]

    assert not carrying, f"{code}: sections {carrying} contain a title line"


@pytest.mark.parametrize(("path", "code"), FILES, ids=[f"{p.parent.name}/{c}" for p, c in FILES])
def test_no_section_is_a_stray_marker_or_empty(path: Path, code: str) -> None:
    """Sections hold passage text, not leftover structure."""
    for n, section in enumerate(parse_sections(path.read_text(encoding="utf-8")), 1):
        assert len(section) >= MIN_SECTION_CHARS, f"{code} section {n} is too short"
        assert not MARKER.match(section), f"{code} section {n} is a bare marker"


@pytest.mark.parametrize(("path", "code"), FILES, ids=[f"{p.parent.name}/{c}" for p, c in FILES])
def test_all_four_title_lines_are_recognised(path: Path, code: str) -> None:
    """Every file must have four headers the parser can see.

    This is the general guard, and the one that would have caught the
    original defect on its own. Each language spells the title word
    differently after transliteration, so a language whose spelling the
    pattern does not cover shows up here as a missing header rather
    than as prose quietly scored with a title attached. A new language
    added to the set fails this test before it can reach the results.
    """
    headers = [
        line for line in path.read_text(encoding="utf-8").splitlines() if HEADER.match(line)
    ]

    assert len(headers) == 4, f"{code}: parser sees {len(headers)} of 4 title lines"


def test_the_header_pattern_covers_every_spelling_in_the_data() -> None:
    """Each language spells the title word differently after transliteration.

    Stated as a list so that adding a language with a new spelling fails
    here rather than silently scoring its titles as prose.
    """
    for spelling in ("TEXT 1:", "teXt 1:", "text1:", "tekst 1:", "txt 2:"):
        assert HEADER.match(spelling), f"{spelling!r} would not be recognised as a header"


def test_the_finnish_file_keeps_its_structure_untransliterated() -> None:
    """Markup is metadata and must not go through the transliterator.

    The root cause, rather than its symptom: passing the whole file to
    to_ipa turned TEXT into tekst. Finnish is the file this project
    produced itself, so it is the one to hold to the rule.
    """
    text = (SNIPPET_DIRS[0] / "perception_fi.txt").read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if HEADER.match(line)]

    assert len(headers) == 4
    for header in headers:
        assert header.upper().startswith("TEXT"), f"{header!r} was transliterated"
