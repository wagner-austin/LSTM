SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check train train-bases train-1 train-2 train-3 train-4 train-5 train-6 train-7

# Lint: venv cleanup, guards, Ruff, Mypy
lint:
	# Clean stale venv if mypy not runnable; do not fail
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; poetry env remove --all | Out-Null }; exit 0
	# Ensure dependencies are installed first
	poetry lock
	poetry install --with dev
	# Run guard checks (no Any, cast, object, type:ignore)
	if ((Test-Path ".\scripts\guard.py") -or (Test-Path ".\scripts\guard\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	# Ruff + Mypy
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test: install deps, then pytest with branch+statement coverage
test:
	poetry lock
	poetry install --with dev
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; poetry run pytest -n auto -v @covArgs

# Check: run lint then test
check: lint | test

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
