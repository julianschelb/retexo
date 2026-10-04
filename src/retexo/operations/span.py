# retexo/operations/span.py
"""
Span operations: acts that cover a region rather than a single token.

Two of them **cover** their positions, replacing the token operations that
would otherwise be emitted there. Two are **markers**, annotating a region
while the token operations inside it still stand. The distinction is what
keeps every reuse position accounted for exactly once, and so keeps the cost
linear in the operation counts.

``QUOTE`` earns its place by encoding contiguity, which the token level cannot
express. Four consecutive shared words and four scattered ones both yield four
copies at zero cost and are indistinguishable in the profile, yet only the
first is a quotation.

``DISPERSE`` is the one operation that is evidence *against* reuse: shared
words straddling a clause boundary are usually coincidence. It is priced above
insertion and deletion so that it pushes a pair apart. Note that clause
boundaries in Latin critical editions are supplied by modern editors, so this
operation measures the edition as well as the text.
"""

from __future__ import annotations

from retexo.operations.base import Level, Operation, Role

# =============================================================================
# Covering spans
# =============================================================================


class Quote(Operation):
    """A contiguous verbatim span, at least four tokens long."""

    tag = "QUOTE"
    level = Level.SPAN
    role = Role.WRITING
    default_cost = 0.0

    def inverse_tag(self) -> str:
        return "QUOTE"


class Frame(Operation):
    """An attribution formula wrapped around a quotation.

    Priced once rather than as one insertion per token, because introducing a
    quotation is a single act. Generation is straightforward; detection needs a
    rule for where the frame stops and the quotation begins, which is not
    settled, so it is disabled.
    """

    tag = "FRAME"
    level = Level.SPAN
    role = Role.WRITING
    default_cost = 0.50
    requires = ("citation_verbs",)
    detectable = False

    def inverse_tag(self) -> str:
        return "FRAME"


# =============================================================================
# Marker spans
# =============================================================================


class Adapt(Operation):
    """A region reused with substitutions concentrated inside it.

    The counterpart of ``QUOTE`` for non-verbatim reuse: it separates
    adaptation concentrated in one phrase from substitutions scattered across a
    passage. Detection needs a stoplist to identify content words.
    """

    tag = "ADAPT"
    level = Level.SPAN
    role = Role.MARKER
    default_cost = 0.0
    requires = ("stopwords",)
    detectable = False

    def inverse_tag(self) -> str:
        return "ADAPT"


class Disperse(Operation):
    """Shared material split across a clause boundary: evidence against reuse."""

    tag = "DISPERSE"
    level = Level.SPAN
    role = Role.MARKER
    default_cost = 1.50
    requires = ("clauses",)

    def inverse_tag(self) -> str:
        return "DISPERSE"
