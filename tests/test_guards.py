"""Tests for char_lstm.guards module."""

from __future__ import annotations

from pathlib import Path

import pytest

from char_lstm.guards import Rule, RuleReport, Violation
from char_lstm.guards.test_quality_rules import MLTestQualityRule, WeakAssertionRule
from char_lstm.guards.util import read_lines


def _write(path: Path, text: str) -> None:
    """Helper to write a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestViolation:
    """Tests for Violation NamedTuple."""

    def test_violation_fields(self, tmp_path: Path) -> None:
        """Test Violation has correct fields."""
        v = Violation(file=tmp_path, line_no=42, kind="test-kind", line="test line")
        assert v.file == tmp_path
        assert v.line_no == 42
        assert v.kind == "test-kind"
        assert v.line == "test line"


class TestRuleReport:
    """Tests for RuleReport NamedTuple."""

    def test_rule_report_fields(self) -> None:
        """Test RuleReport has correct fields."""
        r = RuleReport(name="test-rule", violations=5)
        assert r.name == "test-rule"
        assert r.violations == 5


class TestReadLines:
    """Tests for read_lines utility function."""

    def test_read_lines_basic(self, tmp_path: Path) -> None:
        """Test reading a simple file."""
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3", encoding="utf-8")
        lines = read_lines(f)
        assert lines == ["line1", "line2", "line3"]

    def test_read_lines_with_bom(self, tmp_path: Path) -> None:
        """Test reading a file with BOM marker."""
        f = tmp_path / "test.py"
        f.write_bytes(b"\xef\xbb\xbfline1\nline2")
        lines = read_lines(f)
        assert lines == ["line1", "line2"]

    def test_read_lines_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading a nonexistent file raises RuntimeError."""
        f = tmp_path / "nonexistent.py"
        with pytest.raises(RuntimeError, match="failed to read"):
            read_lines(f)


class TestWeakAssertionRule:
    """Tests for WeakAssertionRule."""

    def test_rule_name(self) -> None:
        """Test rule has correct name."""
        rule = WeakAssertionRule()
        assert rule.name == "test-quality"

    def test_detects_assert_is_not_none(self, tmp_path: Path) -> None:
        """Test detection of assert x is not None."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = 1\n    assert x is not None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-is-not-none"

    def test_detects_isinstance_check(self, tmp_path: Path) -> None:
        """Test detection of isinstance checks."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = 1\n    assert isinstance(x, int)\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-isinstance"

    def test_detects_hasattr_check(self, tmp_path: Path) -> None:
        """Test detection of hasattr checks."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = object()\n    assert hasattr(x, 'a')\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-hasattr"

    def test_detects_len_greater_than_zero(self, tmp_path: Path) -> None:
        """Test detection of len(x) > 0."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = [1]\n    assert len(x) > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-len-zero"

    def test_detects_len_gte_one(self, tmp_path: Path) -> None:
        """Test detection of len(x) >= 1."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = [1]\n    assert len(x) >= 1\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-len-zero"

    def test_detects_string_in_output(self, tmp_path: Path) -> None:
        """Test detection of string in captured output."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = (
            "def test_example(capsys):\n"
            "    print('hello')\n"
            "    captured = capsys.readouterr()\n"
            "    assert 'hello' in captured.out\n"
        )
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-in-output"

    def test_detects_string_in_stderr(self, tmp_path: Path) -> None:
        """Test detection of string in stderr."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'err' in result.stderr\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-in-output"

    def test_detects_string_in_stdout(self, tmp_path: Path) -> None:
        """Test detection of string in stdout."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.stdout\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-in-output"

    def test_detects_string_in_err(self, tmp_path: Path) -> None:
        """Test detection of string in .err attribute."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.err\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-in-output"

    def test_detects_mock_called_without_args(self, tmp_path: Path) -> None:
        """Test detection of mock.called without args verification."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    mock = Mock()\n    mock()\n    assert mock.called\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "mock-without-assert-called-with"

    def test_detects_excessive_mocking(self, tmp_path: Path) -> None:
        """Test detection of excessive mocking (>3 patches)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from unittest.mock import patch

def test_example():
    with patch('a.b'):
        with patch('c.d'):
            with patch('e.f'):
                with patch('g.h'):
                    pass
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert any(v.kind == "excessive-mocking" for v in violations)

    def test_detects_patch_via_mock_module(self, tmp_path: Path) -> None:
        """Test detection of patch via mock.patch."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from unittest import mock

def test_example():
    with mock.patch('a.b'):
        with mock.patch('c.d'):
            with mock.patch('e.f'):
                with mock.patch('g.h'):
                    pass
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert any(v.kind == "excessive-mocking" for v in violations)

    def test_ignores_non_test_files(self, tmp_path: Path) -> None:
        """Test that non-test files are ignored."""
        src_file = tmp_path / "src" / "foo.py"
        code = "def foo():\n    assert x is not None\n"
        _write(src_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([src_file])

        assert len(violations) == 0

    def test_ignores_non_test_functions(self, tmp_path: Path) -> None:
        """Test that non-test functions are ignored."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def helper():\n    x = None\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_ignores_non_test_prefix_files(self, tmp_path: Path) -> None:
        """Test that files not starting with test_ are ignored."""
        test_file = tmp_path / "tests" / "conftest.py"
        code = "def test_example():\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_strong_assertions(self, tmp_path: Path) -> None:
        """Test that strong assertions are allowed."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    before = get_value()
    do_something()
    after = get_value()
    assert after < before
    assert result == expected
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        weak_kinds = {"weak-assertion-is-not-none"}
        weak_violations = [v for v in violations if v.kind in weak_kinds]
        assert len(weak_violations) == 0

    def test_handles_async_test_functions(self, tmp_path: Path) -> None:
        """Test handling of async test functions."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "async def test_example():\n    x = 1\n    assert x is not None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-is-not-none"

    def test_ignores_valid_len_comparisons(self, tmp_path: Path) -> None:
        """Test that len(x) == 5 is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) == 5\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ignores_non_constant_is_not(self, tmp_path: Path) -> None:
        """Test that assert x is not y is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is not y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        not_none_violations = [v for v in violations if v.kind == "weak-assertion-is-not-none"]
        assert len(not_none_violations) == 0

    def test_ignores_is_not_non_none_constant(self, tmp_path: Path) -> None:
        """Test that assert x is not 1 is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is not 1\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        not_none_violations = [v for v in violations if v.kind == "weak-assertion-is-not-none"]
        assert len(not_none_violations) == 0

    def test_ignores_len_with_non_call_left(self, tmp_path: Path) -> None:
        """Test that assert x > 0 is not flagged as len check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ignores_len_with_non_len_func(self, tmp_path: Path) -> None:
        """Test that assert size(x) > 0 is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert size(x) > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ignores_len_with_multiple_ops(self, tmp_path: Path) -> None:
        """Test that chained comparisons are not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 0 < len(x) < 10\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ignores_len_with_non_constant(self, tmp_path: Path) -> None:
        """Test that assert len(x) > y is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) > y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ignores_string_in_non_attribute(self, tmp_path: Path) -> None:
        """Test that assert 'x' in result is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        output_violations = [v for v in violations if v.kind == "weak-assertion-in-output"]
        assert len(output_violations) == 0

    def test_ignores_string_in_non_output_attr(self, tmp_path: Path) -> None:
        """Test that assert 'x' in result.data is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.data\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        output_violations = [v for v in violations if v.kind == "weak-assertion-in-output"]
        assert len(output_violations) == 0

    def test_async_non_test_function_ignored(self, tmp_path: Path) -> None:
        """Test that async helper functions are ignored."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "async def helper():\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0


class TestMLTestQualityRule:
    """Tests for MLTestQualityRule."""

    def test_rule_name(self) -> None:
        """Test rule has correct name."""
        rule = MLTestQualityRule()
        assert rule.name == "ml-test-quality"

    def test_detects_training_without_loss_check(self, tmp_path: Path) -> None:
        """Test detection of training tests without loss comparison."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_epoch():
    model.train()
    optimizer.step()
    assert model is not None
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_allows_training_with_loss_check(self, tmp_path: Path) -> None:
        """Test that training tests with loss comparison are allowed."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_epoch():
    loss_before = get_loss()
    model.train()
    optimizer.step()
    loss_after = get_loss()
    assert loss_after < loss_before
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        loss_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(loss_violations) == 0

    def test_detects_forward_pass_shape_only(self, tmp_path: Path) -> None:
        """Test detection of forward pass tests that only check shapes."""
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = model.forward(input)
    assert output.shape == (batch, seq, vocab)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-forward-shape-only" for v in violations)

    def test_allows_forward_pass_with_value_check(self, tmp_path: Path) -> None:
        """Test that forward pass tests with value checks are allowed."""
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = model.forward(input)
    assert output.shape == (batch, seq, vocab)
    assert output.mean().item() > 0.0
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        shape_violations = [v for v in violations if v.kind == "ml-forward-shape-only"]
        assert len(shape_violations) == 0

    def test_allows_forward_pass_with_sum(self, tmp_path: Path) -> None:
        """Test that forward pass tests with sum() are allowed."""
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = model.forward(input)
    assert output.sum() > 0
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        shape_violations = [v for v in violations if v.kind == "ml-forward-shape-only"]
        assert len(shape_violations) == 0

    def test_detects_optimizer_without_weight_check(self, tmp_path: Path) -> None:
        """Test detection of optimizer tests without weight verification."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    optimizer.step()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-optimizer-no-weight-check" for v in violations)

    def test_allows_optimizer_with_clone(self, tmp_path: Path) -> None:
        """Test that optimizer tests with clone() are allowed."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    weights_before = model.linear.weight.clone()
    optimizer.step()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        weight_violations = [v for v in violations if v.kind == "ml-optimizer-no-weight-check"]
        assert len(weight_violations) == 0

    def test_allows_optimizer_with_state_dict(self, tmp_path: Path) -> None:
        """Test that optimizer tests with state_dict() are allowed."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    state_dict_before = model.state_dict()
    optimizer.step()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        weight_violations = [v for v in violations if v.kind == "ml-optimizer-no-weight-check"]
        assert len(weight_violations) == 0

    def test_allows_optimizer_with_allclose(self, tmp_path: Path) -> None:
        """Test that optimizer tests with allclose() are allowed."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    optimizer.step()
    torch.allclose(w1, w2)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        weight_violations = [v for v in violations if v.kind == "ml-optimizer-no-weight-check"]
        assert len(weight_violations) == 0

    def test_ignores_non_test_files(self, tmp_path: Path) -> None:
        """Test that non-test files are ignored."""
        src_file = tmp_path / "src" / "train.py"
        code = """
def train():
    model.train()
    optimizer.step()
"""
        _write(src_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([src_file])

        assert len(violations) == 0

    def test_ignores_non_test_prefix_files(self, tmp_path: Path) -> None:
        """Test that files not starting with test_ are ignored."""
        test_file = tmp_path / "tests" / "conftest.py"
        code = """
def test_train():
    model.train()
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_ignores_non_test_functions(self, tmp_path: Path) -> None:
        """Test that non-test functions are ignored."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def helper():
    model.train()
    optimizer.step()
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_detects_train_call_usage(self, tmp_path: Path) -> None:
        """Test detection of model.train() calls."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_loop():
    model.train()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_detects_backward_call(self, tmp_path: Path) -> None:
        """Test detection of loss.backward() calls."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_backward():
    loss.backward()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_allows_http_client_train_call(self, tmp_path: Path) -> None:
        """Test that http.train() is not flagged as ML training."""
        test_file = tmp_path / "tests" / "test_http_client.py"
        code = """
async def test_http_client_train_method():
    http = HTTPClient(base_url="url")
    out = await http.train(user_id=1)
    assert out["run_id"] == "r1"
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 0

    def test_allows_client_train_call(self, tmp_path: Path) -> None:
        """Test that client.train() is not flagged as ML training."""
        test_file = tmp_path / "tests" / "test_api_client.py"
        code = """
async def test_api_client_methods():
    client = ModelTrainerClient(base_url="url")
    result = await client.train(params)
    assert result["status"] == "ok"
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 0

    def test_allows_api_client_train_call(self, tmp_path: Path) -> None:
        """Test that api_client.train() is not flagged as ML training."""
        test_file = tmp_path / "tests" / "test_api.py"
        code = """
async def test_api_method():
    api_client = Client()
    result = await api_client.train()
    assert result is not None
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 0

    def test_detects_model_function_call(self, tmp_path: Path) -> None:
        """Test detection of model() function calls as forward pass."""
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_model_call():
    output = model(input)
    assert output.shape == expected
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-forward-shape-only" for v in violations)

    def test_allows_forward_with_allclose(self, tmp_path: Path) -> None:
        """Test that forward pass with allclose is allowed."""
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = model.forward(input)
    torch.allclose(output, expected)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        shape_violations = [v for v in violations if v.kind == "ml-forward-shape-only"]
        assert len(shape_violations) == 0

    def test_detects_weight_comparison_left(self, tmp_path: Path) -> None:
        """Test detection of weight comparison on left side."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_weights():
    loss.backward()
    assert weight_after < weight_before
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 0

    def test_weight_comparison_right_still_needs_loss_check(self, tmp_path: Path) -> None:
        """Test that weight comparison on right side still requires loss check."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_weights():
    loss.backward()
    assert x < param_before
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        # Weight comparisons are for optimizer tests, not training loss checks
        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 1

    def test_chained_train_call_flagged(self, tmp_path: Path) -> None:
        """Test that chained.train() is flagged as ML training."""
        test_file = tmp_path / "tests" / "test_chained.py"
        code = """
def test_chained():
    self.model.train()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        # Chained access is flagged since it's not a simple http/client
        assert any(v.kind == "ml-train-no-loss-check" for v in violations)


class TestBranchCoverage:
    """Additional tests for branch coverage."""

    def test_len_chained_comparison_not_flagged(self, tmp_path: Path) -> None:
        """Test that len(x) > 0 < y (chained) is not flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) > 0 < y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(violations) == 0 or len(len_violations) == 0

    def test_chained_attribute_train_flagged(self, tmp_path: Path) -> None:
        """Test that self.model.train() is flagged (not http client)."""
        test_file = tmp_path / "tests" / "test_chained.py"
        code = """
def test_chained_attr():
    self.model.train()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        # self.model is Attribute, not Name, so _is_http_client_call returns False
        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_get_line_out_of_bounds_weak_assertion(self, tmp_path: Path) -> None:
        """Test _AssertVisitor._get_line returns empty string for out-of-bounds."""
        from char_lstm.guards.test_quality_rules import _AssertVisitor

        # Create visitor with minimal lines
        visitor = _AssertVisitor(tmp_path, ["line1"])

        # Test out-of-bounds line number
        result = visitor._get_line(999)
        assert result == ""

        # Test negative index (also out of bounds)
        result2 = visitor._get_line(-1)
        assert result2 == ""

    def test_get_line_out_of_bounds_ml_rule(self, tmp_path: Path) -> None:
        """Test MLTestQualityRule._get_line returns empty string for out-of-bounds."""
        test_file = tmp_path / "tests" / "test_foo.py"
        # Create a minimal test file
        code = "def test_x(): pass\n"
        _write(test_file, code)

        rule = MLTestQualityRule()
        # Call _get_line directly with out-of-bounds line number
        result = rule._get_line(["line1"], 999)
        assert result == ""

        # Also test with negative index
        result2 = rule._get_line(["line1"], -1)
        assert result2 == ""


class TestRuleProtocol:
    """Tests for Rule Protocol compliance."""

    def test_weak_assertion_rule_conforms_to_protocol(self) -> None:
        """Test that WeakAssertionRule conforms to Rule protocol."""
        rule: Rule = WeakAssertionRule()
        assert rule.name == "test-quality"
        assert callable(rule.run)

    def test_ml_test_quality_rule_conforms_to_protocol(self) -> None:
        """Test that MLTestQualityRule conforms to Rule protocol."""
        rule: Rule = MLTestQualityRule()
        assert rule.name == "ml-test-quality"
        assert callable(rule.run)
