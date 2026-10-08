# AGENTS.md

Formatting conventions for retexo. The template is the **locisimiles** package
(sibling repo, `../locisimiles`): its source and its `pyproject.toml` are the
authoritative reference for how code in this repo should be formatted.

The standing task is a **readability clean-up**: make the code read like
locisimiles. Two levels, both in scope:

1. **Formatting** — banners, spacing, import order, line length. Never
   changes behaviour.
2. **Structural refactoring** — splitting mega-files and mega-classes into
   small modules, breaking up very long methods into helpers. Behaviour
   must not change; public imports are preserved through `__init__.py`
   re-exports, so `from retexo.core.oracle import EditPlan` keeps working
   after the oracle moves. Prefer *module-level* splits (new file per
   concern) over helper-extraction when both are possible.

Never rename a public API. Every module in `src/` carries a prose docstring
explaining its design; that prose is content — move it with the code it
describes, do not shorten it. After any split, the old module path must
either still import (re-export) or the change must be coordinated with all
call sites. Run the tests after each structural change.

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
- **A class with many methods is grouped, not flat.** Group the methods of a
  large class into banner-labelled sections inside the class (e.g.
  encoding, losses, prediction, evaluation) and separate the groups with a
  blank line above the `# ----------` header. A class with 30+ methods and
  no section breaks reads as one block of code — that is the failure mode
  to avoid. Better still, if the sections are independent concerns, split
  them into mixins or modules per rule 2 above.
- One concern per file: when a file passes ~400 lines or holds more than
  one major class plus its helpers, split it.
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
