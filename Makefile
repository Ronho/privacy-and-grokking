SHELL := /bin/bash

.PHONY: *

# Helper to run pag commands with automatic --help fallback
define pag_cmd
	@if [ -z "$(ARGS)" ]; then \
		uv run pag $(1) --help; \
	else \
		uv run pag $(1) $(ARGS); \
	fi
endef

help:
	@echo "Available targets:"
	@echo "  install    - Sync dependencies with uv (auto-detects GPU)"
	@echo "  lock       - Lock dependencies"
	@echo "  qa         - Run all checks (format, lint, type, test)"
	@echo "  qa-fix     - Run all checks and auto-fix issues"
	@echo "  clean      - Remove cache and build artifacts"
	@echo "  train      - Train models (use ARGS to pass arguments)"
	@echo "  attack     - Run attacks (use ARGS to pass arguments)"
	@echo "  evaluate   - Evaluate results (use ARGS to pass arguments)"
	@echo ""
	@echo "Package specific commands:"
	@uv run pag --help

install:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected - installing with CUDA support..."; \
		uv sync --extra cu130; \
	else \
		echo "No NVIDIA GPU detected - installing CPU-only version..."; \
		uv sync --extra cpu; \
	fi

lock:
	uv lock

qa:
	@echo "Running all QA checks..."
	uv run ruff format --check || true
	uv run ruff check || true
	uv run ty check || true
	uv run pytest || true
	@echo "QA checks complete"

qa-fix:
	@echo "Running all QA checks and fixing issues..."
	uv run ruff format || true
	uv run ruff check --fix || true
	uv run ty check	--fix || true
	uv run pytest || true
	@echo "QA checks and fixes complete"

clean:
	rm -rf .pytest_cache .ruff_cache .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

train:
	$(call pag_cmd,train)

attack:
	$(call pag_cmd,attack)

evaluate:
	$(call pag_cmd,evaluate)
