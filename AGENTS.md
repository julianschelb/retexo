# AGENTS.md

Formatting conventions for retexo. The template is the **locisimiles** package
(sibling repo, `../locisimiles`): its source and its `pyproject.toml` are the
authoritative reference for how code in this repo should be formatted.

The standing task is **formatting-only clean-up**: change how the code reads,
never what it computes. No renames of public APIs, no refactoring, no
behaviour changes unless explicitly asked. Every module in `src/` carries a
prose docstring explaining its design; that prose is content — reflow it to
the format below, do not shorten it.

## Tooling (mirror the locisimiles configuration)

- Ruff lint plus `ruff format`; line length **100**; `target-version = "py310"`.
- Lint select: `E, W, F, I, UP, B, SIM`; ignores: `E501, E702, SIM108, B905,
  UP006, UP007, UP035, UP045`.
- isort with `known-first-party = ["retexo", "retexo_gui"]`; per-file ignores:
  `tests/*` relaxes `B, SIM`; `src/**/__init__.py` allows `F401` re-exports.
- Poe tasks `lint` and `format`/`format-check` run over `src/` and `tests/`.
- Deltas against this repo today: line length is 120 and lint is error-only
  (`E9, F63, F7, F82`); converging to the lists above is part of the task.

## File layout

- First line is a path comment **relative to the package**, then the module
  docstring:
  ```python
  # core/oracle.py
  """One-sentence summary of the module."""
  ```
  (not `# retexo/core/oracle.py` — locisimiles drops the package prefix).
- Imports: `from __future__ import annotations` first (where present), then
  stdlib, third-party, first-party — one blank line between groups.
- Major classes and file sections are separated by full-width `#` banners
  with a title; smaller groups inside a file use
  `# ---------- Name ----------` lines.
- `__init__.py` files re-export the public API and document it in the module
  docstring (bullet lists of the exported names).

## Docstrings

- Every module, class, and public function/method gets one.
- First line: a one-sentence summary (fits in one line). Then short
  paragraphs for the why-and-how.
- Google-style sections where applicable: `Args:`, `Returns:`, `Raises:`,
  `Attributes:` — each parameter on its own indented line as
  `name: description`.
- Runnable examples in fenced code blocks; inline code in double backticks
  (``pipeline.run()``).

## Naming and typing

- Classes in PascalCase; functions and variables in snake_case; private
  helpers take a leading underscore; constants in UPPER_CASE
  (`ID = Union[str, int]` in locisimiles).
- Configuration objects are frozen dataclasses with sensible defaults
  (`@dataclass(frozen=True)`).
- Type hints from `typing` (`Optional`, `Mapping`, `Sequence`, ...);
  `Optional[X]` and `X | None` both occur in locisimiles — keep each file
  internally consistent.
- Contracts are abstract base classes with `@abstractmethod`; concrete
  variants subclass them. Strategy/composition over inheritance for wiring
  (locisimiles' generator + judge pattern); thin preconfigured subclasses
  for convenience variants.

## Comments

- Sparingly; they explain intent and are full sentences in sentence case.
- No commented-out code, no change logs in comments.

## Terminology (from the companion paper's drafting guide)

Keep the paper's object names stable in code and comments: a pair consists
of a **source** S and a **reuse** Q; the model predicts an **edit script**
of word-level links, each with a **label**; reference types are *cit.*
(verbatim reference) and *cf.* (allusion).
