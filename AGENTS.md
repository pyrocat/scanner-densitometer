# Repository Guidelines

## Project Structure & Module Organization

`densitometer/` is the application package. `app.py` and `__main__.py` provide the Tkinter entry points; `logic/` contains image loading, data models, the NumPy-based step-wedge analysis, and the ISO(R) estimate (`sensitometry.py`, see `.docs/ADR/0001-iso-r-from-step-wedge-curve.md`); and `ui/` contains the main window and canvas widgets. Keep numerical and image-processing behavior in `logic/` so it remains testable without starting the GUI. Tests live in `tests/` (`test_analysis.py`, `test_sensitometry.py`). `img/` holds sample/reference imagery, while `pyproject.toml` and `poetry.lock` define the Python 3.10+ Poetry environment.

## Build, Test, and Development Commands

- `poetry install` installs runtime dependencies into the managed environment.
- `poetry run densitometer` launches the desktop application through the configured console script. `poetry run python -m densitometer` is equivalent.
- `poetry run python -m unittest discover -s tests -v` runs the complete test suite with verbose names.
- `poetry build` creates source and wheel distributions under `dist/`.

For meaningful manual checks, open a linear 16-bit TIFF and drag a strip along the complete 21-step wedge end to end; it can be rotated, stretched, widened, and moved with its handles. The wedge geometry is fixed (Stouffer T2115), so steps are divided equally, never detected.

## Coding Style & Naming Conventions

Business logic must be self-contained and testable. Avoid global variables.

Add comments to explain non-obvious behavior in business logic. Provide docstrings in business logic to document the objectives of the function, its parameters and return values.

Follow standard PEP 8 formatting with four-space indentation. Preserve the existing use of `from __future__ import annotations`, explicit type hints, relative imports inside the package, and small focused functions. Use `snake_case` for modules, functions, methods, and variables; `PascalCase` for classes and dataclasses; and `UPPER_SNAKE_CASE` for constants such as `T2115_DENSITIES`. No formatter or linter is configured, so match nearby code and keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

Tests use the standard-library `unittest` framework with NumPy assertions where appropriate. Name files `test_*.py`, classes `*Tests`, and methods `test_<behavior>`. Prefer small synthetic arrays over large image fixtures. Cover horizontal, rotated, and reversed wedges, invalid selections, strip direction, calibration, and both density reference modes when changing analysis code. There is no configured coverage threshold; every bug fix should include a regression test.

## Commit & Pull Request Guidelines

Use a concise, imperative subject such as `Handle vertical wedge selections`, and keep each commit focused. Pull requests should explain the user-visible or analytical change, note assumptions, link related issues, and list the exact test command run. Include screenshots for UI changes and describe any new sample images or generated artifacts.
