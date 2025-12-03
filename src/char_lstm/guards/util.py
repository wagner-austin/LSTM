"""Utility functions for guard rules."""

from __future__ import annotations

from pathlib import Path


def read_lines(path: Path) -> list[str]:
    """Read file contents as a list of lines.

    Uses utf-8-sig to handle optional BOM.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc
    return text.splitlines()


__all__ = ["read_lines"]
