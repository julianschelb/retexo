# Changelog

## Unreleased

- The paper's operations of 2026-10-09 at every boundary, while the code keeps its own spelling inside:
  - `retexo.operations.unlinked` derives the words without a link (`INS` or `NEW` for a reuse word, `MARK` in a
    citing formula, `DEL` or `OUT` for a source word, split by the reused stretch from the first to the last linked
    word) and the patterns `QUOTE`, `REORDER` and `DISPERSE`; tests reproduce the paper's Table 1 on the annotation.
  - `retexo export` writes `SCHEMA_VERSION` 2: `mark` instead of `frame`, the relation `DERIV` instead of `POS`,
    `insertions` / `new_text`, `deletions` / `not_taken_up`, and `patterns`.
  - The record codec reads both revisions of the annotation (`MARK` spans and `DERIV` relations map to the code's
    `FRAME` and `POS`) and writes a record of the revised annotation back unchanged, with its `unlinked` words and
    `patterns`; the default annotation is `gold_2026-10-12`.
  - The demo app shows MARK, INS/NEW and DEL/OUT; the annotation app shows INFLECT, DERIV and MARK and still saves
    corrections in the code's spelling.

## v0.1.0 (2026-10-04)

- Initial package, taken from the experiment repository: the stretch-and-link pointer, the synthetic-pair generator,
  the active-learning loop, the baselines it is compared with, and the scorer.
- `retexo export` writes the predicted edit scripts in the released format.
- Documentation and an interactive demo page, published with GitHub Pages.
