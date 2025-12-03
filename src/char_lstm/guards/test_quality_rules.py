"""Guard rules for detecting weak or fake tests.

These rules identify test anti-patterns that achieve code coverage
without actually verifying behavior. Coverage shows lines executed,
not correctness proven.

Violations:
- weak-assertion-is-not-none: `assert x is not None` proves existence only
- weak-assertion-isinstance: Type check doesn't verify behavior
- weak-assertion-hasattr: Attribute exists, but what's its value?
- weak-assertion-len-zero: `assert len(x) > 0` checks existence not content
- weak-assertion-in-output: String matching in captured output is fragile
- mock-without-assert-called-with: Mock verified called but not with what args
- excessive-mocking: Test mocks more than 3 things, probably not integration
- ml-train-no-loss-check: ML training test without loss decrease check
- ml-forward-shape-only: Forward pass test only checks shapes
- ml-optimizer-no-weight-check: Optimizer test doesn't verify weights changed
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from char_lstm.guards import Violation
from char_lstm.guards.util import read_lines


def _is_patch_call(func: ast.expr) -> bool:
    """Check if func is a patch() call."""
    if isinstance(func, ast.Attribute) and func.attr == "patch":
        return True
    return isinstance(func, ast.Name) and func.id == "patch"


class _AssertVisitor(ast.NodeVisitor):
    """Visitor to analyze assert statements in test functions."""

    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.violations: list[Violation] = []
        self.current_function: str = ""
        self.function_has_comparison: bool = False
        self.function_mock_count: int = 0
        self.function_start_line: int = 0

    def _get_line(self, line_no: int) -> str:
        """Get source line content by line number (1-indexed)."""
        idx = line_no - 1
        if 0 <= idx < len(self.lines):
            return self.lines[idx].strip()
        return ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def _analyze_test_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.current_function = node.name
        self.function_has_comparison = False
        self.function_mock_count = 0
        self.function_start_line = node.lineno

        for child in ast.walk(node):
            self._check_assert(child)
            self._check_mock_usage(child)
            self._check_comparison(child)

        self._check_function_level_issues()

    def _check_assert(self, node: ast.AST) -> None:
        """Check for weak assertion patterns."""
        if not isinstance(node, ast.Assert):
            return

        test = node.test

        if self._is_identity_check_negated(test, "None"):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-is-not-none",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_isinstance_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-isinstance",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_hasattr_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-hasattr",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_len_existence_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-len-zero",
                    line=self._get_line(node.lineno),
                )
            )

        if self._is_string_in_output(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-in-output",
                    line=self._get_line(node.lineno),
                )
            )

    def _check_mock_usage(self, node: ast.AST) -> None:
        """Check for mock-related issues."""
        if isinstance(node, ast.Call) and _is_patch_call(node.func):
            self.function_mock_count += 1

        if isinstance(node, ast.Assert) and self._is_mock_called_check(node.test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="mock-without-assert-called-with",
                    line=self._get_line(node.lineno),
                )
            )

    def _check_comparison(self, node: ast.AST) -> None:
        """Track if test has meaningful comparisons."""
        if not isinstance(node, ast.Compare):
            return

        comparison_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        for op in node.ops:
            if isinstance(op, comparison_ops) and self._is_variable_comparison(node):
                self.function_has_comparison = True

    def _check_function_level_issues(self) -> None:
        """Check issues that require analyzing the whole function."""
        if self.function_mock_count > 3:
            src_line = self._get_line(self.function_start_line)
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=self.function_start_line,
                    kind="excessive-mocking",
                    line=f"{src_line} ({self.function_mock_count} patches)",
                )
            )

    def _is_identity_check_negated(self, node: ast.expr, const_name: str) -> bool:
        """Check if node is `x is not <const>`."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.IsNot):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Constant):
            return False

        return const_name == "None" and comparator.value is None

    def _is_isinstance_check(self, node: ast.expr) -> bool:
        """Check if node is isinstance(x, Y)."""
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "isinstance"

    def _is_hasattr_check(self, node: ast.expr) -> bool:
        """Check if node is hasattr(x, "y")."""
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "hasattr"

    def _is_len_existence_check(self, node: ast.expr) -> bool:
        """Check if node is len(x) > 0 or len(x) >= 1."""
        if not isinstance(node, ast.Compare):
            return False
        if not isinstance(node.left, ast.Call):
            return False

        func = node.left.func
        if not (isinstance(func, ast.Name) and func.id == "len"):
            return False
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False

        op = node.ops[0]
        comp = node.comparators[0]
        if not isinstance(comp, ast.Constant):
            return False

        if isinstance(op, ast.Gt) and comp.value == 0:
            return True
        return isinstance(op, ast.GtE) and comp.value == 1

    def _is_string_in_output(self, node: ast.expr) -> bool:
        """Check if node is 'string' in x.out or x.err."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Attribute):
            return False

        return comparator.attr in ("out", "err", "stdout", "stderr")

    def _is_mock_called_check(self, node: ast.expr) -> bool:
        """Check if node is mock.called without args check."""
        return isinstance(node, ast.Attribute) and node.attr == "called"

    def _is_variable_comparison(self, node: ast.Compare) -> bool:
        """Check if comparison involves variables (not just constants)."""
        var_types = (ast.Name, ast.Attribute, ast.Subscript)
        left_is_var = isinstance(node.left, var_types)
        right_is_var = any(isinstance(c, var_types) for c in node.comparators)
        return left_is_var and right_is_var


class WeakAssertionRule:
    """Guard rule for detecting weak or fake tests."""

    name = "test-quality"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []

        for path in files:
            if "/tests/" not in path.as_posix() and "\\tests\\" not in str(path):
                continue
            if not path.name.startswith("test_"):
                continue

            lines = read_lines(path)
            source = "\n".join(lines)

            tree = ast.parse(source, filename=str(path))

            visitor = _AssertVisitor(path, lines)
            visitor.visit(tree)
            out.extend(visitor.violations)

        return out


class _MLPatternVisitor(ast.NodeVisitor):
    """Visitor to detect ML patterns in test functions using AST."""

    _HTTP_CLIENT_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"http", "client", "api", "api_client", "http_client", "trainer_client"}
    )

    def __init__(self) -> None:
        self.has_backward: bool = False
        self.has_step: bool = False
        self.has_train_call: bool = False
        self.has_forward_call: bool = False
        self.has_loss_compare: bool = False
        self.has_weight_check: bool = False
        self.has_value_check: bool = False
        self.has_clone: bool = False
        self.has_state_dict: bool = False
        self.has_allclose: bool = False

    _ATTR_FLAGS: ClassVar[dict[str, str]] = {
        "backward": "has_backward",
        "step": "has_step",
        "train": "has_train_call",
        "forward": "has_forward_call",
        "clone": "has_clone",
        "state_dict": "has_state_dict",
        "allclose": "has_allclose",
        "item": "has_value_check",
        "mean": "has_value_check",
        "sum": "has_value_check",
    }

    def _is_http_client_call(self, node: ast.Attribute) -> bool:
        """Check if the attribute call is on an HTTP client object."""
        if isinstance(node.value, ast.Name):
            return node.value.id.lower() in self._HTTP_CLIENT_NAMES
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "train" and self._is_http_client_call(node.func):
                self.generic_visit(node)
                return
            flag = self._ATTR_FLAGS.get(node.func.attr)
            if flag is not None:
                setattr(self, flag, True)
        elif isinstance(node.func, ast.Name) and node.func.id == "model":
            self.has_forward_call = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Detect loss comparisons like loss_after < loss_before."""
        for op in node.ops:
            if isinstance(op, ast.Lt | ast.LtE):
                left_has_loss = self._name_contains(node.left, ("loss", "after", "final"))
                right_has_loss = any(
                    self._name_contains(c, ("loss", "before", "initial")) for c in node.comparators
                )
                if left_has_loss and right_has_loss:
                    self.has_loss_compare = True
                left_has_weight = self._name_contains(node.left, ("weight", "param"))
                right_has_weight = any(
                    self._name_contains(c, ("weight", "param", "before")) for c in node.comparators
                )
                if left_has_weight or right_has_weight:
                    self.has_weight_check = True
        self.generic_visit(node)

    def _name_contains(self, node: ast.expr, keywords: tuple[str, ...]) -> bool:
        if isinstance(node, ast.Name):
            name_lower = node.id.lower()
            return any(kw in name_lower for kw in keywords)
        return False


class MLTestQualityRule:
    """Guard rule specifically for ML project test quality.

    Enforces that ML tests verify actual learning behavior, not just execution.
    """

    name = "ml-test-quality"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []

        for path in files:
            if "/tests/" not in path.as_posix() and "\\tests\\" not in str(path):
                continue
            if not path.name.startswith("test_"):
                continue

            lines = read_lines(path)
            source = "\n".join(lines)

            tree = ast.parse(source, filename=str(path))

            out.extend(self._check_ml_patterns(path, tree, lines))

        return out

    def _get_line(self, lines: list[str], line_no: int) -> str:
        """Get source line content by line number (1-indexed)."""
        idx = line_no - 1
        if 0 <= idx < len(lines):
            return lines[idx].strip()
        return ""

    def _check_ml_patterns(self, path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue

            visitor = _MLPatternVisitor()
            visitor.visit(node)

            violations.extend(self._check_training(path, node, visitor, lines))
            violations.extend(self._check_forward_pass(path, node, visitor, lines))
            violations.extend(self._check_optimizer(path, node, visitor, lines))

        return violations

    def _check_training(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        visitor: _MLPatternVisitor,
        lines: list[str],
    ) -> list[Violation]:
        """Check for training tests without loss comparison."""
        is_training = visitor.has_backward or visitor.has_train_call
        if not is_training:
            return []
        if visitor.has_loss_compare:
            return []

        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="ml-train-no-loss-check",
                line=self._get_line(lines, node.lineno),
            )
        ]

    def _check_forward_pass(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        visitor: _MLPatternVisitor,
        lines: list[str],
    ) -> list[Violation]:
        """Check for forward pass tests that only check shapes."""
        if not visitor.has_forward_call:
            return []
        has_value = visitor.has_value_check or visitor.has_allclose or visitor.has_loss_compare
        if has_value:
            return []

        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="ml-forward-shape-only",
                line=self._get_line(lines, node.lineno),
            )
        ]

    def _check_optimizer(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        visitor: _MLPatternVisitor,
        lines: list[str],
    ) -> list[Violation]:
        """Check for optimizer tests that don't verify weight changes."""
        if not visitor.has_step:
            return []
        has_weight = (
            visitor.has_weight_check
            or visitor.has_clone
            or visitor.has_state_dict
            or visitor.has_allclose
        )
        if has_weight:
            return []

        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="ml-optimizer-no-weight-check",
                line=self._get_line(lines, node.lineno),
            )
        ]


__all__ = ["MLTestQualityRule", "WeakAssertionRule"]
