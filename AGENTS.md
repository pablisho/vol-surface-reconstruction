# Repository Guidelines

## Project Structure & Module Organization
- `pricing/` contains the core pricing logic (Black-76, greeks, implied vol) and shared types.
- `tests/` holds pytest suites mirroring module names, e.g. `tests/test_black76.py`.
- `pyproject.toml` defines tooling configuration (pytest, ruff, black).
- `requirements-dev.txt` lists dev tools used for linting and formatting.

## Build, Test, and Development Commands
- `python -m pytest` runs the full test suite in `tests/`.
- `python -m ruff check .` performs linting; add `--fix` to apply safe fixes.
- `python -m black .` formats Python files to the repo standards.

## Coding Style & Naming Conventions
- Python 3.11+ only; follow Black formatting with 100-char lines.
- Ruff is the linter; keep imports sorted per Ruff/isort settings.
- Module and function names use `snake_case`; tests follow `test_*.py` and `test_*` functions.

## Testing Guidelines
- Framework: pytest (configured in `pyproject.toml`).
- Tests live under `tests/` and should mirror the module under test.
- Run targeted tests with `python -m pytest tests/test_implied_vol.py`.

## Commit & Pull Request Guidelines
- Commit messages are short, sentence-style summaries (e.g., “Added vega”, “Completed greeks”).
- PRs should describe the change, note testing performed, and link related issues if any.
- Include minimal reproduction or numerical examples for pricing changes when relevant.

## Configuration Tips
- Install dev tooling via `python -m pip install -r requirements-dev.txt`.
- Keep `pricing/` import paths as first-party in Ruff by using the `pricing` package.
