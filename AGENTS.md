# Instructions for AI Agents

Welcome! If you are an AI agent working on this repository, you must adhere to the following strict guidelines:

## 1. Documentation Integrity
**CRITICAL RULE:** The documentation must ALWAYS be kept in sync with code changes. 
If you modify, add, or remove features, scripts, or core logic, you MUST update the corresponding documentation files in the `docs/` folder before concluding your task. 
Do not wait for the user to remind you.

## 2. Project Structure
The repository is structured as follows:
- `src/privacy_and_grokking/`: The core python package containing all source code (metrics, models, datasets, etc.).
- `experiments/`: Experiment definitions, SLURM batch scripts, and generated job lists organized by experiment.
- `commands/`: General utility scripts (e.g., for mlflow).
- `docs/`: The centralized documentation hub.
- `tests/`: Pytest test suite.

## 3. Core Commands
- **Environment**: Managed by `uv`. Use `uv sync --extra cpu` (or `cu130`) to sync dependencies.
- **Running Code**: Always use `uv run <command>` (e.g., `uv run pag --help`).
- **MLflow Server**: Can be started locally using `uv run poe mlflow-host`.

## 4. Coding Standards
- All comments, docstrings, and documentation MUST be in English.
- Always explain *why* a particular piece of logic was chosen in comments, not just *what* it does.
- Run linters and tests before concluding your work.

Please read the `docs/developer/implementation_standards.md` for more in-depth technical coding rules.
