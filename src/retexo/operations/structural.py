# retexo/operations/structural.py
"""
Structural operations: what the alignment leaves over, plus reordering.

``INS`` and ``DEL`` are inverses of one another and are the most expensive
operations in the inventory. Any pair can be related by deleting the whole
source and inserting the whole reuse, so if that route is cheap the distance
collapses and every pair scores alike.

``REORDER`` is a marker. The movement itself is already encoded in the index
mapping of the operations that write the tokens, so emitting positions here
would cover them twice. It exists to be counted, which is what keeps the cost
recoverable from the operation profile.
"""

from __future__ import annotations

from retexo.operations.base import Level, Operation, Role

# =============================================================================
# Operations
# =============================================================================


class Ins(Operation):
    """A reuse token with no source."""

    tag = "INS"
    level = Level.STRUCTURAL
    role = Role.WRITING
    default_cost = 1.00

    def inverse_tag(self) -> str:
        return "DEL"


class Del(Operation):
    """A source token that the reuse does not carry over."""

    tag = "DEL"
    level = Level.STRUCTURAL
    role = Role.CONSUMING
    default_cost = 1.00

    def inverse_tag(self) -> str:
        return "INS"


class Reorder(Operation):
    """An aligned pair whose position crosses another's."""

    tag = "REORDER"
    level = Level.STRUCTURAL
    role = Role.MARKER
    default_cost = 0.50

    def inverse_tag(self) -> str:
        return "REORDER"
