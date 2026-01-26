.PHONY: *

# Detect OS
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	RM := cmd /C del /F /Q
	RMDIR := cmd /C rmdir /S /Q
	FIND_PYCACHE := cmd /C "for /d /r . %d in (__pycache__) do @if exist "%d" rmdir /s /q "%d""
	NULL := NUL
	SHELL := cmd
else
	DETECTED_OS := $(shell uname -s)
	RM := rm -f
	RMDIR := rm -rf
	FIND_PYCACHE := find . -type d -name "__pycache__" -exec rm -rf {} +
	NULL := /dev/null
	SHELL := /bin/bash
endif

# Helper to run pag commands with automatic --help fallback
ifeq ($(DETECTED_OS),Windows)
define pag_cmd
	@if "$(ARGS)"=="" ( uv run pag $(1) --help ) else ( uv run pag $(1) $(ARGS) )
endef
else
define pag_cmd
	@if [ -z "$(ARGS)" ]; then \
		uv run pag $(1) --help; \
	else \
		uv run pag $(1) $(ARGS); \
	fi
endef
endif

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

ifeq ($(DETECTED_OS),Windows)
install:
	@where nvidia-smi >$(NULL) 2>&1 && ( echo NVIDIA GPU detected - installing with CUDA support... && uv sync --extra cu130 ) || ( echo No NVIDIA GPU detected - installing CPU-only version... && uv sync --extra cpu )
else
install:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected - installing with CUDA support..."; \
		uv sync --extra cu130; \
	else \
		echo "No NVIDIA GPU detected - installing CPU-only version..."; \
		uv sync --extra cpu; \
	fi
endif

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

ifeq ($(DETECTED_OS),Windows)
clean:
	-@if exist .pytest_cache cmd /C rmdir /S /Q .pytest_cache 2>NUL
	-@if exist .ruff_cache cmd /C rmdir /S /Q .ruff_cache 2>NUL
	-@if exist .venv cmd /C rmdir /S /Q .venv 2>NUL
	-@cmd /C "for /d /r . %d in (__pycache__) do @if exist "%d" rmdir /s /q "%d"" 2>NUL
else
clean:
	-@rm -rf .pytest_cache .ruff_cache .venv 2>/dev/null
	-@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
endif

train:
	$(call pag_cmd,train)

attack:
	$(call pag_cmd,attack)

evaluate:
	$(call pag_cmd,evaluate)
