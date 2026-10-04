# retexo/operations/cardinality.py
"""
Cardinality operations: one token becomes several, or several become one.

Latin reuse does this constantly — a participle unpacked into a relative
clause, a phrase compressed into a compound, the perfect passive written
analytically. Without these operations every such case is described as a
deletion followed by insertions, which both inflates the residual that is
supposed to measure what the vocabulary cannot name and misnames the act.

They are usable for generation now. Detection is disabled: recognising that a
reuse span is the analytic expansion of one source word needs a periphrasis
model beyond lemmatization, and only the perfect passive is special-caseable.
They are also what forces the aligner beyond one-to-one matching, which is why
they stay switched off in the first run until a diagnostic shows how much of
the residual they would actually recover.
"""

from __future__ import annotations

from retexo.operations.base import Level, Operation, Role

# =============================================================================
# Operations
# =============================================================================


class Split(Operation):
    """One source token expands into several contiguous reuse tokens."""

    tag = "SPLIT"
    level = Level.CARDINALITY
    role = Role.WRITING
    default_cost = 0.75
    requires = ("morphology",)
    detectable = False

    def inverse_tag(self) -> str:
        return "MERGE"


class Merge(Operation):
    """Several contiguous source tokens compress into one reuse token."""

    tag = "MERGE"
    level = Level.CARDINALITY
    role = Role.WRITING
    default_cost = 0.75
    requires = ("morphology",)
    detectable = False

    def inverse_tag(self) -> str:
        return "SPLIT"
