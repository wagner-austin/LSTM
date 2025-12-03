"""Rich console wrapper for styled terminal output.

This module provides typed console functions for training output.
All print statements in the codebase should use these functions instead.
"""

from __future__ import annotations

from typing import Protocol


class _RichConsole(Protocol):
    """Protocol for rich.console.Console interface."""

    def print(
        self,
        *objects: str,
        style: str | None = None,
        highlight: bool = True,
    ) -> None:
        """Print styled output to console."""
        ...


def _get_console() -> _RichConsole:
    """Get rich Console instance with strict typing."""
    rich_console_mod = __import__("rich.console", fromlist=["Console"])
    console_cls = rich_console_mod.Console
    console: _RichConsole = console_cls()
    return console


# Module-level console instance
_console: _RichConsole = _get_console()


# =============================================================================
# Style Constants
# =============================================================================

STYLE_HEADER = "bold cyan"
STYLE_LABEL = "dim white"
STYLE_VALUE = "green"
STYLE_EPOCH = "yellow"
STYLE_LOSS = "magenta"
STYLE_PPL = "blue"
STYLE_SAVED = "bold green"
STYLE_WARNING = "yellow"
STYLE_ERROR = "bold red"
STYLE_SUCCESS = "bold green"
STYLE_INFO = "cyan"


# =============================================================================
# Output Functions
# =============================================================================


def log_header(text: str) -> None:
    """Print a section header with separator lines."""
    separator = "=" * 60
    _console.print(separator, style=STYLE_HEADER)
    _console.print(text, style=STYLE_HEADER)
    _console.print(separator, style=STYLE_HEADER)


def log_subheader(text: str) -> None:
    """Print a subheader."""
    _console.print(f"\n{text}", style=STYLE_HEADER)


def log_config(label: str, value: str | int | float | bool) -> None:
    """Print a configuration key-value pair."""
    _console.print(f"  [dim]{label}:[/dim] [{STYLE_VALUE}]{value}[/{STYLE_VALUE}]")


def log_info(text: str) -> None:
    """Print an informational message."""
    _console.print(text, style=STYLE_INFO)


def log_metrics(loss: float, ppl: float, prefix: str = "") -> None:
    """Print loss and perplexity metrics."""
    loss_str = f"[{STYLE_LOSS}]Loss: {loss:.4f}[/{STYLE_LOSS}]"
    ppl_str = f"[{STYLE_PPL}]PPL: {ppl:.2f}[/{STYLE_PPL}]"
    _console.print(f"{prefix}{loss_str} | {ppl_str}")


def log_progress(
    epoch: int,
    total_epochs: int,
    step: int,
    pct: float,
    loss: float,
    ppl: float,
) -> None:
    """Print training progress with metrics."""
    prefix = f"  [{STYLE_EPOCH}][{pct:5.1f}%][/{STYLE_EPOCH}] Step {step:,} | "
    log_metrics(loss, ppl, prefix=prefix)


def log_epoch_start(epoch: int, total_epochs: int) -> None:
    """Print epoch start message."""
    _console.print(f"[{STYLE_EPOCH}][Epoch {epoch + 1}/{total_epochs}][/{STYLE_EPOCH}] Starting...")


def log_epoch_val(epoch: int, total_epochs: int) -> None:
    """Print epoch validation message."""
    _console.print(
        f"[{STYLE_EPOCH}][Epoch {epoch + 1}/{total_epochs}][/{STYLE_EPOCH}] Validating..."
    )


def log_epoch_result(epoch: int, total_epochs: int, val_loss: float, val_ppl: float) -> None:
    """Print epoch validation results."""
    prefix = f"[{STYLE_EPOCH}][Epoch {epoch + 1}/{total_epochs}][/{STYLE_EPOCH}] "
    log_metrics(val_loss, val_ppl, prefix=prefix)


def log_saved(path: str) -> None:
    """Print checkpoint saved message."""
    _console.print(f"  -> [{STYLE_SAVED}]Saved best model to {path}[/{STYLE_SAVED}]")


def log_no_improvement(epochs_no_improve: int, patience: int) -> None:
    """Print no improvement warning."""
    _console.print(
        f"  -> [{STYLE_WARNING}]No improvement ({epochs_no_improve}/{patience})[/{STYLE_WARNING}]"
    )


def log_early_stopping() -> None:
    """Print early stopping message."""
    _console.print(f"[{STYLE_ERROR}]Early stopping.[/{STYLE_ERROR}]")


def log_final_results(test_loss: float, test_ppl: float) -> None:
    """Print final test results."""
    _console.print(
        f"\n[{STYLE_SUCCESS}]Final Test Loss: {test_loss:.4f} | "
        f"Test PPL: {test_ppl:.2f}[/{STYLE_SUCCESS}]"
    )
    _console.print(f"[{STYLE_SUCCESS}]Done![/{STYLE_SUCCESS}]")


__all__ = [
    "log_config",
    "log_early_stopping",
    "log_epoch_result",
    "log_epoch_start",
    "log_epoch_val",
    "log_final_results",
    "log_header",
    "log_info",
    "log_metrics",
    "log_no_improvement",
    "log_progress",
    "log_saved",
    "log_subheader",
]
