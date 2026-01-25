set shell := ["bash", "-c"]

default:
    @just --list

test *ARGS:
    uv run pytest {{ARGS}}

type:
    uv run ty check

lint:
    uv run ruff check

fmt-check:
    uv run ruff format --check

qa: fmt-check lint type test

qa-fix:
    uv run ruff format
    uv run ruff check --fix
    uv run ty check
    uv run pytest

clean:
    rm -rf .pytest_cache .ruff_cache .venv dist htmlcov .coverage
    find . -type d -name "__pycache__" -exec rm -rf {} +

build:
    uv build
