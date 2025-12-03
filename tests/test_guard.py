"""Tests for scripts/guard.py module."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from scripts.guard import main as guard_main


def _write(path: Path, text: str) -> None:
    """Helper to write a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[1]


def test_guard_main_with_root(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test guard main with --root argument on empty project."""
    code = guard_main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Guard checks passed" in out


def test_guard_main_unrecognized_arg(tmp_path: Path) -> None:
    """Test guard main ignores unrecognized arguments."""
    code = guard_main(["--root", str(tmp_path), "--unknown-flag"])
    assert code == 0


def test_guard_run_as_main() -> None:
    """Test guard.py can be run as __main__."""
    guard_path = _project_root() / "scripts" / "guard.py"
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(guard_path), run_name="__main__")
    assert exc.value.code == 0


def test_guard_detects_any_import(tmp_path: Path) -> None:
    """Test guard detects Any import violation."""
    src = tmp_path / "src" / "bad.py"
    any_kw = "An" + "y"
    _write(src, f"from typing import {any_kw}\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_cast_import(tmp_path: Path) -> None:
    """Test guard detects cast import violation."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "from typing import cast\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_typealias_import(tmp_path: Path) -> None:
    """Test guard detects TypeAlias import violation."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "from typing import TypeAlias\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_typing_any_usage(tmp_path: Path) -> None:
    """Test guard detects typing.Any usage."""
    src = tmp_path / "src" / "bad.py"
    any_kw = "An" + "y"
    _write(src, f"import typing\nx: typing.{any_kw} = None\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_typing_cast_usage(tmp_path: Path) -> None:
    """Test guard detects typing.cast usage."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "import typing\nx = typing.cast(int, 1)\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_cast_call(tmp_path: Path) -> None:
    """Test guard detects direct cast() call."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "from typing import cast\nx = cast(int, 1)\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_any_usage(tmp_path: Path) -> None:
    """Test guard detects Any usage as name."""
    src = tmp_path / "src" / "bad.py"
    any_kw = "An" + "y"
    _write(src, f"from typing import {any_kw}\nx: {any_kw} = None\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_object_annotation(tmp_path: Path) -> None:
    """Test guard detects object in type annotation."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "x: object = None\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_object_in_function_arg(tmp_path: Path) -> None:
    """Test guard detects object in function argument annotation."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "def f(x: object) -> None: pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_object_in_return_type(tmp_path: Path) -> None:
    """Test guard detects object in function return type."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "def f() -> object: pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_object_in_async_return(tmp_path: Path) -> None:
    """Test guard detects object in async function return type."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "async def f() -> object: pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_type_ignore(tmp_path: Path) -> None:
    """Test guard detects forbidden type comments."""
    src = tmp_path / "src" / "bad.py"
    ti = "# " + "type" + ": " + "ignore"
    _write(src, f"x = 1  {ti}\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_syntax_error(tmp_path: Path) -> None:
    """Test guard raises RuntimeError on syntax error."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "def f(\n")

    with pytest.raises(RuntimeError, match="Failed to parse"):
        guard_main(["--root", str(tmp_path)])


def test_guard_clean_code(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test guard passes for clean code."""
    src = tmp_path / "src" / "good.py"
    _write(src, "from typing import Protocol\nx: int = 1\n")

    code = guard_main(["--root", str(tmp_path)])

    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip().endswith("Guard checks passed: no violations found.")


def test_guard_scans_tests_dir(tmp_path: Path) -> None:
    """Test guard scans tests directory."""
    test_file = tmp_path / "tests" / "bad.py"
    any_kw = "An" + "y"
    _write(test_file, f"from typing import {any_kw}\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_scans_scripts_dir(tmp_path: Path) -> None:
    """Test guard scans scripts directory."""
    script_file = tmp_path / "scripts" / "bad.py"
    any_kw = "An" + "y"
    _write(script_file, f"from typing import {any_kw}\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_violation_output(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test guard outputs violation details."""
    src = tmp_path / "src" / "bad.py"
    any_kw = "An" + "y"
    _write(src, f"from typing import {any_kw}\n")

    code = guard_main(["--root", str(tmp_path)])

    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("Guard checks failed:")
    assert "src\\bad.py:1:" in captured.err or "src/bad.py:1:" in captured.err


def test_guard_no_pyproject_via_subprocess(tmp_path: Path) -> None:
    """Test guard returns 1 when pyproject.toml not found (subprocess)."""
    # Copy guard.py to temp location without pyproject.toml
    guard_src = _project_root() / "scripts" / "guard.py"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    guard_dest = scripts_dir / "guard.py"
    guard_dest.write_text(guard_src.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(guard_dest)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("ERROR: pyproject.toml not found")


def test_guard_detects_pragma_comment(tmp_path: Path) -> None:
    """Test guard detects pragma comment violation."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "x = 1  # pragma: no cover\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_contextlib_suppress(tmp_path: Path) -> None:
    """Test guard detects contextlib.suppress usage."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "import contextlib\nwith contextlib.suppress(ValueError): pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_bare_suppress(tmp_path: Path) -> None:
    """Test guard detects bare suppress usage after import."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "from contextlib import suppress\nwith suppress(ValueError): pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_silent_except_pass(tmp_path: Path) -> None:
    """Test guard detects silent except: pass."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "try:\n    x = 1\nexcept ValueError:\n    pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_silent_except_ellipsis(tmp_path: Path) -> None:
    """Test guard detects silent except: ..."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "try:\n    x = 1\nexcept ValueError:\n    ...\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_broad_except_without_log_and_raise(tmp_path: Path) -> None:
    """Test guard detects broad Exception without both log and raise."""
    src = tmp_path / "src" / "bad.py"
    # Only has raise, no log
    _write(src, "try:\n    x = 1\nexcept Exception:\n    raise\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_bare_except_without_log_and_raise(tmp_path: Path) -> None:
    """Test guard detects bare except without both log and raise."""
    src = tmp_path / "src" / "bad.py"
    # Only has logging, no raise
    _write(src, "import logging\ntry:\n    x = 1\nexcept:\n    logging.error('err')\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_detects_specific_except_without_log_or_raise(tmp_path: Path) -> None:
    """Test guard detects specific exception without log or raise."""
    src = tmp_path / "src" / "bad.py"
    # Has neither log nor raise
    _write(src, "try:\n    x = 1\nexcept ValueError:\n    y = 2\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_allows_broad_except_with_log_and_raise(tmp_path: Path) -> None:
    """Test guard allows broad Exception with both log and raise."""
    src = tmp_path / "src" / "good.py"
    _write(
        src,
        "import logging\ntry:\n    x = 1\nexcept Exception:\n    logging.error('err')\n    raise\n",
    )

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_allows_specific_except_with_log(tmp_path: Path) -> None:
    """Test guard allows specific exception with just log."""
    src = tmp_path / "src" / "good.py"
    _write(
        src,
        "import logging\ntry:\n    x = 1\nexcept ValueError:\n    logging.warning('warn')\n",
    )

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_allows_specific_except_with_raise(tmp_path: Path) -> None:
    """Test guard allows specific exception with just raise."""
    src = tmp_path / "src" / "good.py"
    _write(src, "try:\n    x = 1\nexcept ValueError:\n    raise\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_skips_exception_rules_in_tests(tmp_path: Path) -> None:
    """Test guard skips exception rules in tests directory."""
    test_file = tmp_path / "tests" / "test_foo.py"
    # This would be a violation in src but allowed in tests
    _write(test_file, "try:\n    x = 1\nexcept ValueError:\n    pass\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_detects_multiple_suppress_same_line(tmp_path: Path) -> None:
    """Test guard detects multiple suppress usages on same line."""
    src = tmp_path / "src" / "bad.py"
    # Two suppress calls on same line in single with statement
    content = (
        "from contextlib import suppress\nwith suppress(ValueError), suppress(KeyError): pass\n"
    )
    _write(src, content)

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_find_body_start_empty_lines() -> None:
    """Test _find_body_start returns total when only empty lines remain."""
    from scripts.guard import _find_body_start

    lines = ["try:", "    x = 1", "except:", "", "  "]
    # Start at index 3, which has only empty lines after
    result = _find_body_start(lines, 3)
    assert result == len(lines)


def test_guard_no_pyproject_direct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """Test guard returns 1 when pyproject.toml not found (direct call for coverage)."""
    import scripts.guard as guard_module

    # Monkeypatch __file__ resolution to point to temp location without pyproject.toml
    fake_guard_path = tmp_path / "scripts" / "guard.py"
    fake_guard_path.parent.mkdir(parents=True)
    fake_guard_path.touch()

    # Patch Path(__file__).resolve() to return our fake path
    original_resolve = Path.resolve

    def patched_resolve(self: Path) -> Path:
        # Check if this is the guard module's __file__
        if str(self).endswith("guard.py") and "scripts" in str(self):
            return fake_guard_path
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", patched_resolve)

    # Call main without --root (no argv)
    code = guard_module.main([])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR: pyproject.toml not found")


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    """Test guard main entry with subprocess on empty project."""
    guard_path = _project_root() / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_guard_detects_print_in_src(tmp_path: Path) -> None:
    """Test guard detects print() usage in src/ files."""
    src = tmp_path / "src" / "bad.py"
    _write(src, "print('hello')\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 2


def test_guard_allows_print_in_tests(tmp_path: Path) -> None:
    """Test guard allows print() usage in tests/ files."""
    src = tmp_path / "tests" / "test_foo.py"
    _write(src, "print('hello')\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_allows_print_in_scripts(tmp_path: Path) -> None:
    """Test guard allows print() usage in scripts/ files."""
    src = tmp_path / "scripts" / "helper.py"
    _write(src, "print('hello')\n")

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0
