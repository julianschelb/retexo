# retexo/operations/__init__.py
"""The operation vocabulary, grouped by level."""

from __future__ import annotations

from retexo.operations.base import (
    EditOperation,
    Level,
    Operation,
    OperationRegistry,
    Role,
)
from retexo.operations.cardinality import Merge, Split
from retexo.operations.span import Adapt, Disperse, Frame, Quote
from retexo.operations.structural import Del, Ins, Reorder
from retexo.operations.token import (
    Ant,
    Hyper,
    Hypo,
    Morph,
    NeSub,
    Nop,
    Pos,
    Subst,
    Form,
    Sense,
    Syn,
    SynDist,
)

#: Every operation, in precedence order: span, then token from most to least
#: specific, then cardinality, then structural. SUBST is the token residual and
#: comes last among the token operations, after everything a resource can name.
ALL_OPERATIONS = (
    Quote(),
    Frame(),
    Adapt(),
    Disperse(),
    Nop(),
    Morph(),
    Pos(),
    NeSub(),
    Syn(),
    Hyper(),
    Hypo(),
    Ant(),
    SynDist(),
    Subst(),
    Form(),
    Sense(),
    Split(),
    Merge(),
    Reorder(),
    Ins(),
    Del(),
)

__all__ = [
    "ALL_OPERATIONS",
    "Adapt",
    "Ant",
    "Del",
    "Disperse",
    "EditOperation",
    "Frame",
    "Hyper",
    "Hypo",
    "Subst",
    "Form",
    "Sense",
    "Ins",
    "Level",
    "Merge",
    "Morph",
    "NeSub",
    "Nop",
    "Operation",
    "OperationRegistry",
    "Pos",
    "Quote",
    "Reorder",
    "Role",
    "Split",
    "Syn",
    "SynDist",
]
