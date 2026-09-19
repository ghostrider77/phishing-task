.PHONY: inspect check lint typecheck

sync:
	uv sync --frozen --no-install-project

lock:
	uv lock

notebook:
	uv run jupyter notebook

format:
	uv run --no-sync isort predict.py
	uv run --no-sync black predict.py --config pyproject.toml

check:
	uv run --no-sync isort predict.py --check-only --diff
	uv run --no-sync black predict.py --check --config pyproject.toml

lint:
	uv run --no-sync flake8 predict.py -v

typecheck:
	uv run --no-sync mypy predict.py

inspect: check lint typecheck
