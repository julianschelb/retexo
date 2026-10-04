# Development

## Setup

```bash
git clone https://github.com/julianschelb/retexo.git
cd retexo
pip install -e ".[dev,lexical]"
pre-commit install
```

## Tasks

```bash
poe test          # pytest
poe test-cov      # pytest with coverage
poe lint          # ruff
poe docs          # serve the docs at http://127.0.0.1:8000
poe docs-build    # build the docs to site/
poe webapp        # run the review app in development mode
```

## Tests

`pytest` runs the whole suite. A test that needs something a plain install does not bring is skipped with its reason:
the `lexical` extra (CLTK), a spaCy English model, or the word-level annotation files, which are not part of this
repository. `tests/conftest.py` lists what each skipped test needs.

Some older test modules are scripts: they run their checks at import time and exit with a status. `conftest.py` keeps
pytest from importing them and `tests/test_script_style.py` runs each as a subprocess instead.

## Scope

Retexo is the package: the model, the generator, the active-learning loop, the baselines and the scorer. The experiments
that use it, with their sweeps and run folders, live in the experiment repository, which installs retexo like any other
dependency.

## Documentation

The docs are built with [MkDocs](https://www.mkdocs.org/), the Material theme and mkdocstrings, from `docs/` and the
docstrings (Google style). The review app in `webapp/` is built separately and published under `/demo/` next to them.

## Commits and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/). Versioning is configured for
`python-semantic-release` in `pyproject.toml`; the release workflow needs a `GH_TOKEN` secret and is not enabled yet.
