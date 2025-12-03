"""Guard rules for enforcing code quality standards.

This module provides a modular guard system with reusable rule definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol


class Violation(NamedTuple):
    """A single guard violation."""

    file: Path
    line_no: int
    kind: str
    line: str


class RuleReport(NamedTuple):
    """Summary of violations for a rule."""

    name: str
    violations: int


class Rule(Protocol):
    """Protocol for guard rules."""

    @property
    def name(self) -> str: ...

    def run(self, files: list[Path]) -> list[Violation]: ...


__all__ = ["Rule", "RuleReport", "Violation"]
