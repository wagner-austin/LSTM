# Each platform keeps its native shell, and every recipe below is written in
# the intersection the two read alike: one plain command per line and make's
# own $(VAR) for substitution. On Windows a PowerShell-launched make finds no
# sh; on Linux there is no PowerShell, so a SHELL of powershell.exe alone
# made this check runnable only on the fleet's one Windows node (API board
# task e35812cc). The block is API scripts/make/shell.mk's; `-eu -c` makes a
# failing command fail its line, as PowerShell's native exit propagation
# does on Windows.
ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
else
SHELL := /bin/sh
.SHELLFLAGS := -eu -c
endif

.PHONY: lint test check train train-bases train-1 train-2 train-3 train-4 train-5 train-6 train-7

# Lint: install, guards (no Any, cast, object, type:ignore), Ruff, Mypy
lint:
	poetry lock
	poetry install --with dev
	poetry run python -m scripts.guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test: install deps, then pytest with branch+statement coverage
test:
	poetry lock
	poetry install --with dev
	poetry run pytest -n auto -v --cov-branch --cov-report=term-missing --cov=src --cov=scripts

# Check: run lint then test. The banner is the last line a passing check
# prints, and the only one the fleet verdict (API
# tools/fleet/src/fleet/core/verdict.py) reads as a pass.
check: lint | test
	@echo "=== ALL CHECKS PASSED ==="

# Train all experiments sequentially
train: train-1 train-2 train-3 train-4 train-5 train-6 train-7

# Base models only (no transfer fine-tunes) -- all that zero-shot eval needs
train-bases:
	poetry run python -m char_lstm.train --lang tr
	poetry run python -m char_lstm.train --lang az
	poetry run python -m char_lstm.train --lang kk
	poetry run python -m char_lstm.train --lang ky
	poetry run python -m char_lstm.train --lang uz
	poetry run python -m char_lstm.train --lang ug
	poetry run python -m char_lstm.train --lang fi

# Experiment 1: Turkish (Oghuz) as base
train-1:
	poetry run python -m char_lstm.train --lang tr
	poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/tr_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/tr_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ky --from-checkpoint checkpoints/tr_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang uz --from-checkpoint checkpoints/tr_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/tr_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang fi --from-checkpoint checkpoints/tr_best.pt --freeze-embed

# Experiment 2: Azerbaijani (Oghuz) as base
train-2:
	poetry run python -m char_lstm.train --lang az
	poetry run python -m char_lstm.train --lang tr --from-checkpoint checkpoints/az_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/az_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ky --from-checkpoint checkpoints/az_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang uz --from-checkpoint checkpoints/az_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/az_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang fi --from-checkpoint checkpoints/az_best.pt --freeze-embed

# Experiment 3: Kazakh (Kipchak) as base
train-3:
	poetry run python -m char_lstm.train --lang kk
	poetry run python -m char_lstm.train --lang tr --from-checkpoint checkpoints/kk_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/kk_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ky --from-checkpoint checkpoints/kk_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang uz --from-checkpoint checkpoints/kk_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/kk_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang fi --from-checkpoint checkpoints/kk_best.pt --freeze-embed

# Experiment 4: Kyrgyz (Kipchak) as base
train-4:
	poetry run python -m char_lstm.train --lang ky
	poetry run python -m char_lstm.train --lang tr --from-checkpoint checkpoints/ky_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/ky_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/ky_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang uz --from-checkpoint checkpoints/ky_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/ky_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang fi --from-checkpoint checkpoints/ky_best.pt --freeze-embed

# Experiment 5: Uzbek (Karluk) as base
train-5:
	poetry run python -m char_lstm.train --lang uz
	poetry run python -m char_lstm.train --lang tr --from-checkpoint checkpoints/uz_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/uz_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/uz_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ky --from-checkpoint checkpoints/uz_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/uz_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang fi --from-checkpoint checkpoints/uz_best.pt --freeze-embed

# Experiment 6: Uyghur (Karluk) as base
train-6:
	poetry run python -m char_lstm.train --lang ug
	poetry run python -m char_lstm.train --lang tr --from-checkpoint checkpoints/ug_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/ug_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/ug_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ky --from-checkpoint checkpoints/ug_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang uz --from-checkpoint checkpoints/ug_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang fi --from-checkpoint checkpoints/ug_best.pt --freeze-embed

# Experiment 7: Finnish (non-Turkic control) as base
train-7:
	poetry run python -m char_lstm.train --lang fi
	poetry run python -m char_lstm.train --lang tr --from-checkpoint checkpoints/fi_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang az --from-checkpoint checkpoints/fi_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang kk --from-checkpoint checkpoints/fi_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ky --from-checkpoint checkpoints/fi_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang uz --from-checkpoint checkpoints/fi_best.pt --freeze-embed
	poetry run python -m char_lstm.train --lang ug --from-checkpoint checkpoints/fi_best.pt --freeze-embed
