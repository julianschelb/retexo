# formulations/typed_pointer/__init__.py
"""
E25: a typed pointer -- one softmax over every (source word, type) cell.

E24 answered "which source word?" with a pointer and "what kind of change?" with
a second head that ran afterwards, plus a frame head, an operation head and a
source head, each with its own loss. The evidence about a word pair -- shared
lemma, a wordnet entry, two proper names -- reached only the second head, so it
could never help the pointer *find* the link, even though E23 measured that it
can (alignment 0.941 -> 0.964 when lemma similarity re-ranks the shortlist).

Here the two questions are one decision. For reuse word t the head scores every
cell (s, k) -- source word s, type k -- plus two nulls, and takes one softmax:

    score(t, s, k) = f(h_t).g(h_s)/sqrt(d)                     locate
                   + u_k . MLP[h_t, h_s, |h_t-h_s|, h_t*h_s]   name
                   + v_k . phi(t, s)                          evidence
    score(t, o, INS)   = f(h_t).n_INS
    score(t, o, FRAME) = f(h_t).n_FRAME

The loss is one cross-entropy per reuse word over a *set* of allowed cells, so
a label that names only part of the answer -- a hand-labelled SUBST that says
"lexical, kind unknown" -- supervises exactly what it knows.

Inference marginalises: p(s | t) = sum_k p(s, k | t), and hands those to the
same assignment stack E24 used, so the two are compared on identical terms.
The type is the argmax at the chosen cell, under the same evidence vetoes.

Everything that was a head in E24 and is a definition -- COPY/SUBST/INS from the
cell, DEL from what no cell consumed, REORDER/QUOTE/ADAPT/DISPERSE from the
alignment -- is derived, not learned.

The old single-file module is now a package split by concern; every public
name it exported is re-exported here, so
``from retexo.formulations.typed_pointer import X`` keeps working:

- ``TypedPointer`` -- the model, assembled in ``model.py`` from the
  ``EncodingMixin``, ``CellGridMixin``, ``AuxiliaryLossMixin``, ``LossesMixin``,
  ``PredictionMixin`` and ``TrainingMixin`` on top of ``ChangeDetector``;
- ``cell_index``, ``allowed_cells`` -- module-level aliases of the class's
  pure helpers (unchanged behaviour);
- ``NULL_INS``, ``NULL_FRAME``, ``CELLS_FROM`` -- the flattened cell layout
  (``cells.py``);
- ``STRETCH_MODES``, ``STRETCH_NOMATCH`` -- the stretch head's modes (``cells.py``).
"""

from __future__ import annotations

from retexo.formulations.typed_pointer.cells import (
    CELLS_FROM,
    NULL_FRAME,
    NULL_INS,
    STRETCH_MODES,
    STRETCH_NOMATCH,
)
from retexo.formulations.typed_pointer.model import TypedPointer

#: Backward-compatible module-level aliases.
cell_index = TypedPointer.cell_index
allowed_cells = TypedPointer.allowed_cells

__all__ = [
    "CELLS_FROM",
    "NULL_FRAME",
    "NULL_INS",
    "STRETCH_MODES",
    "STRETCH_NOMATCH",
    "TypedPointer",
    "allowed_cells",
    "cell_index",
]
