# Contributing to ssiv

## Development setup

```bash
git clone https://github.com/iskakovs/ssiv
cd ssiv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

To also run the validation against ShiftShareSE (needs R):

```bash
bash scripts/fetch_adh.sh adh_data
SSIV_ADH_DIR=adh_data pytest tests/test_adh_validation.py
```

## Ground rules

* **Implement from the papers.** Cite the equation or section you implement in the
  docstring. Do not translate code from GPL-licensed packages; use them only as a black
  box to compare results.
* **Every estimator has a test**: an exact identity, a closed form computed by hand, or a
  published result.
* Keep the dependencies to `numpy` and `pandas`.

## Pull requests

1. Open an issue describing the change.
2. Add tests and documentation.
3. Make sure `pytest` and `ruff` pass.
4. Add a line to `CHANGELOG.md`.
