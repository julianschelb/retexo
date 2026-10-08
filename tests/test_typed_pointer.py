"""The cell layout and the allowed-set rule, checked without a model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.formulations.change_detector import GROUP_TARGET  # noqa: E402
from retexo.formulations.typed_pointer import (  # noqa: E402
    CELLS_FROM,
    NULL_FRAME,
    NULL_INS,
    TypedPointer,
)

K = 12
LEX = [2, 3, 4, 5, 6, 7, 8, 11]  # SYN … POS, SUBST in FINE_OPERATIONS order


def test_cell_layout_is_dense_and_unique():
    seen = set()
    for c in range(5):
        for k in range(K):
            i = TypedPointer.cell_index(c, k, K)
            assert i >= CELLS_FROM and i not in seen
            seen.add(i)
    assert min(seen) == CELLS_FROM and max(seen) == CELLS_FROM + 5 * K - 1


def test_allowed_cells_follow_the_label():
    # an exact type: one cell
    assert TypedPointer.allowed_cells(3, 1, 0, K, LEX) == [TypedPointer.cell_index(3, 1, K)]
    # a hand-labelled SUBST: the lexical cells of that source, nothing else
    grp = TypedPointer.allowed_cells(3, GROUP_TARGET, 0, K, LEX)
    assert grp == [TypedPointer.cell_index(3, k, K) for k in LEX]
    assert (
        TypedPointer.cell_index(3, 0, K) not in grp and TypedPointer.cell_index(3, 1, K) not in grp
    )
    # a link whose kind is unknown: every type at that source
    assert len(TypedPointer.allowed_cells(3, -100, 0, K, LEX)) == K
    # gold nulls
    assert TypedPointer.allowed_cells(None, -100, 1, K, LEX) == [NULL_FRAME]
    assert TypedPointer.allowed_cells(None, -100, 0, K, LEX) == [NULL_INS]
    assert TypedPointer.allowed_cells(None, -100, -100, K, LEX) == [NULL_INS, NULL_FRAME]
