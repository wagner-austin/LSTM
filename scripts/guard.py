"""Guard script to enforce strict typing and exception handling rules.

Checks for violations:
- No `Any` usage
- No `cast` usage
- No `object` in type annotations
- No `type: ignore` comments
- No `TypeAlias` usage
- No `contextlib.suppress` usage
- No `# pragma` comments
- No silent exception handling (except: pass)
- Broad exceptions (Exception/BaseException) must log AND re-raise
- Specific exceptions must log OR re-raise
- No `print()` in src/ (use _console module instead)
- Weak test assertions (is not None, isinstance, hasattr, len > 0)
- ML test quality (training tests must verify loss, forward tests must check values)

Run with: python -m scripts.guard
"""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from collections.abc import Generator, Sequence
from io import StringIO
from pathlib import Path

from char_lstm.guards import RuleReport, Violation
from char_lstm.guards.test_quality_rules import MLTestQualityRule, WeakAssertionRule


def _iter_py_files(root: Path) -> Generator[Path, None, None]:
    """Iterate over Python files in src and tests directories."""
    for subdir in ["src", "tests", "scripts"]:
        dir_path = root / subdir
        if dir_path.exists():
            yield from dir_path.rglob("*.py")


def _iter_tokens(text: str) -> Generator[tokenize.TokenInfo, None, None]:
    """Generate tokens from source text."""
    reader = StringIO(text).readline
    yield from tokenize.generate_tokens(reader)


# =============================================================================
# Typing Rules
# =============================================================================


def _contains_object_in_annotation(node: ast.AST) -> bool:
    """Check if AST node contains 'object' as a type annotation."""
    return any(isinstance(child, ast.Name) and child.id == "object" for child in ast.walk(node))


def _check_object_annotations(path: Path, node: ast.AST) -> list[Violation]:
    """Check for object in type annotations."""
    violations: list[Violation] = []
    kind = "object-in-annotation"

    if (
        isinstance(node, ast.AnnAssign)
        and node.annotation is not None
        and _contains_object_in_annotation(node.annotation)
    ):
        violations.append(Violation(file=path, line_no=node.lineno, kind=kind, line=""))

    if (
        isinstance(node, ast.arg)
        and node.annotation is not None
        and _contains_object_in_annotation(node.annotation)
    ):
        violations.append(Violation(file=path, line_no=node.lineno, kind=kind, line=""))

    if (
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.returns is not None
        and _contains_object_in_annotation(node.returns)
    ):
        violations.append(Violation(file=path, line_no=node.lineno, kind=kind, line=""))

    return violations


def _run_typing_rule(path: Path, tree: ast.AST) -> list[Violation]:
    """Check AST for typing violations."""
    violations: list[Violation] = []
    forbidden_imports = {"Any", "cast", "TypeAlias"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name in forbidden_imports:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind=f"typing-import-{alias.name.lower()}",
                            line="",
                        )
                    )

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr in forbidden_imports
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind=f"typing-{node.attr.lower()}-usage",
                    line="",
                )
            )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cast"
        ):
            violations.append(Violation(file=path, line_no=node.lineno, kind="cast-call", line=""))

        if isinstance(node, ast.Name) and node.id == "Any":
            violations.append(Violation(file=path, line_no=node.lineno, kind="any-usage", line=""))

        violations.extend(_check_object_annotations(path, node))

    return violations


def _run_comments_rule(path: Path, text: str) -> list[Violation]:
    """Check for type: ignore and pragma comments."""
    violations: list[Violation] = []
    for tok in _iter_tokens(text):
        if tok.type == tokenize.COMMENT:
            if "type: ignore" in tok.string:
                violations.append(
                    Violation(
                        file=path,
                        line_no=tok.start[0],
                        kind="type-ignore",
                        line=tok.line.rstrip("\n"),
                    )
                )
            if "pragma" in tok.string:
                violations.append(
                    Violation(
                        file=path,
                        line_no=tok.start[0],
                        kind="pragma-comment",
                        line=tok.line.rstrip("\n"),
                    )
                )
    return violations


# =============================================================================
# Suppress Rules
# =============================================================================


def _is_suppress(expr: ast.AST) -> bool:
    """Check if expression is contextlib.suppress."""
    func = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(func, ast.Attribute):
        is_contextlib = isinstance(func.value, ast.Name) and func.value.id == "contextlib"
        return is_contextlib and func.attr == "suppress"
    return isinstance(func, ast.Name) and func.id == "suppress"


def _run_suppress_rule(path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    """Check for contextlib.suppress usage."""
    violations: list[Violation] = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                if _is_suppress(item.context_expr):
                    line_no = item.context_expr.lineno
                    if line_no in seen:
                        continue
                    seen.add(line_no)
                    idx = line_no - 1
                    text = lines[idx] if 0 <= idx < len(lines) else ""
                    violations.append(
                        Violation(
                            file=path,
                            line_no=line_no,
                            kind="suppress-usage",
                            line=text.rstrip("\n"),
                        )
                    )
    return violations


# =============================================================================
# Logging Rules
# =============================================================================


def _run_logging_rule(path: Path, tree: ast.AST) -> list[Violation]:
    """Check for print() usage in src/ files.

    print() is forbidden in src/ - use _console module instead.
    Tests and scripts are allowed to use print().
    """
    # Only check src/ files
    if "src" not in path.parts:
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="print-usage",
                    line="Use _console module instead of print()",
                )
            )
    return violations


# =============================================================================
# Exception Rules
# =============================================================================

_EXCEPT_HEADER = re.compile(r"^(\s*)except(\s+([^:]+))?:\s*$")
_BROAD_TYPES = re.compile(r"\b(Exception|BaseException)\b")
_LOG_CALL = re.compile(r"\b(logging|log|logger)\.(debug|info|warning|error|exception|critical)\(")
_RAISE_RE = re.compile(r"\braise\b")


def _parse_except_header(raw: str) -> tuple[int, str] | None:
    """Parse an except header line, returning (indent, exception_types)."""
    match = _EXCEPT_HEADER.match(raw)
    if match is None:
        return None
    indent_group = match.group(1)
    group3 = match.group(3)
    indent_str = indent_group if indent_group is not None else ""
    types_str = group3 if group3 is not None else ""
    return len(indent_str), types_str.strip()


def _is_broad_exception(types: str) -> bool:
    """Check if exception type is broad (Exception/BaseException or bare except)."""
    return types == "" or _BROAD_TYPES.search(types) is not None


def _first_body_is_trivial(line: str) -> bool:
    """Check if first body line is just pass or ..."""
    return re.match(r"^\s+(pass|\.\.\.)\s*(#.*)?$", line) is not None


def _scan_except_body(
    lines: Sequence[str], start: int, header_indent: int
) -> tuple[bool, bool, int]:
    """Scan except body for log and raise statements."""
    total = len(lines)
    has_log = False
    has_raise = False
    i = start
    while i < total:
        body_line = lines[i]
        if body_line.strip() == "":
            i += 1
            continue
        body_indent = len(body_line) - len(body_line.lstrip(" \t"))
        if body_indent <= header_indent and re.match(
            r"^\s*(except\b|finally\b|else\b|$)", body_line
        ):
            break
        if _RAISE_RE.search(body_line):
            has_raise = True
        if _LOG_CALL.search(body_line):
            has_log = True
        i += 1
    return has_log, has_raise, i


def _find_body_start(lines: Sequence[str], start: int) -> int:
    """Find first non-empty line after except header.

    Returns the index of the first non-empty line, or len(lines) if none found.
    In practice, valid Python always has a body after except, so this returns
    a valid index for syntactically correct files.
    """
    total = len(lines)
    i = start
    while i < total:
        if lines[i].strip() != "":
            return i
        i += 1
    return total


def _run_exceptions_rule(path: Path, lines: list[str]) -> list[Violation]:
    """Check exception handling rules.

    Skips test files since tests legitimately catch exceptions to verify behavior.
    """
    # Skip test files
    if "tests" in path.parts:
        return []

    violations: list[Violation] = []
    total = len(lines)
    idx = 0
    while idx < total:
        raw = lines[idx]
        parsed = _parse_except_header(raw)
        if parsed is None:
            idx += 1
            continue
        indent, types = parsed
        broad = _is_broad_exception(types)

        body_start = _find_body_start(lines, idx + 1)

        if body_start < total and _first_body_is_trivial(lines[body_start]):
            violations.append(
                Violation(
                    file=path,
                    line_no=idx + 1,
                    kind="silent-except-body",
                    line=raw.rstrip("\n"),
                )
            )

        has_log, has_raise, body_end = _scan_except_body(lines, body_start, indent)
        if broad:
            if not (has_log and has_raise):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="broad-except-requires-log-and-raise",
                        line=raw.rstrip("\n"),
                    )
                )
        else:
            if not (has_log or has_raise):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="except-without-log-or-raise",
                        line=raw.rstrip("\n"),
                    )
                )

        idx = body_end if body_end > idx else idx + 1
    return violations


# =============================================================================
# Main Runner
# =============================================================================


def run_guards(root: Path) -> int:
    """Run all guard checks and return exit code."""
    typing_violations: list[Violation] = []
    comments_violations: list[Violation] = []
    suppress_violations: list[Violation] = []
    exceptions_violations: list[Violation] = []
    logging_violations: list[Violation] = []

    all_files: list[Path] = list(_iter_py_files(root))

    for path in all_files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

        typing_violations.extend(_run_typing_rule(path, tree))
        comments_violations.extend(_run_comments_rule(path, source))
        suppress_violations.extend(_run_suppress_rule(path, tree, lines))
        exceptions_violations.extend(_run_exceptions_rule(path, lines))
        logging_violations.extend(_run_logging_rule(path, tree))

    weak_assertion_rule = WeakAssertionRule()
    ml_test_quality_rule = MLTestQualityRule()
    weak_assertion_violations = weak_assertion_rule.run(all_files)
    ml_test_quality_violations = ml_test_quality_rule.run(all_files)

    reports = [
        RuleReport(name="typing", violations=len(typing_violations)),
        RuleReport(name="comments", violations=len(comments_violations)),
        RuleReport(name="suppress", violations=len(suppress_violations)),
        RuleReport(name="exceptions", violations=len(exceptions_violations)),
        RuleReport(name="logging", violations=len(logging_violations)),
        RuleReport(name="test-quality", violations=len(weak_assertion_violations)),
        RuleReport(name="ml-test-quality", violations=len(ml_test_quality_violations)),
    ]

    all_violations = (
        typing_violations
        + comments_violations
        + suppress_violations
        + exceptions_violations
        + logging_violations
        + weak_assertion_violations
        + ml_test_quality_violations
    )

    print("Guard rule summary:")
    for rep in reports:
        print(f"  {rep.name}: {rep.violations} violations")

    if all_violations:
        print("Guard checks failed:", file=sys.stderr)
        for v in all_violations:
            rel_path = v.file.relative_to(root) if v.file.is_relative_to(root) else v.file
            text = v.line[:80] + "..." if len(v.line) > 80 else v.line
            print(f"  {rel_path}:{v.line_no}: {v.kind} {text}", file=sys.stderr)
        return 2

    print("Guard checks passed: no violations found.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for guard script."""
    args = argv if argv is not None else sys.argv[1:]

    root_override: Path | None = None
    idx = 0
    while idx < len(args):
        if args[idx] == "--root" and idx + 1 < len(args):
            root_override = Path(args[idx + 1]).resolve()
            idx += 2
        else:
            idx += 1

    if root_override is not None:
        return run_guards(root_override)

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    if not (project_root / "pyproject.toml").exists():
        print(f"ERROR: pyproject.toml not found in {project_root}", file=sys.stderr)
        return 1

    return run_guards(project_root)


if __name__ == "__main__":
    raise SystemExit(main())
