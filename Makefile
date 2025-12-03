SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

# Lint: venv cleanup, guards, Ruff, Mypy
lint:
	# Clean stale venv if mypy not runnable; do not fail
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; poetry env remove --all | Out-Null }; exit 0
	# Run guard checks (no Any, cast, object, type:ignore)
	if ((Test-Path ".\scripts\guard.py") -or (Test-Path ".\scripts\guard\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	# Ensure dependencies are installed
	poetry lock
	poetry install --with dev
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
