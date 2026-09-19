# Testing and Linting

Ensuring high code quality and reliability is critical. We use `pytest` for testing and `ruff` for linting and formatting.

## Pre-Merge Requirements

> [!IMPORTANT]
> **Strict Rule:** Linters and tests **MUST** be executed and pass before merging any code changes.

## Running Tests

To execute the test suite:
```bash
uv run pytest tests/
```

*(Additional details regarding mocking, fixtures, and specific test coverage requirements will be added here).*

## Running Linters

To check for formatting and linting errors:
```bash
uv run ruff check src/
```

To automatically format the code:
```bash
uv run ruff format src/
```
