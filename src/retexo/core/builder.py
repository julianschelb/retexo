# retexo/core/builder.py
"""
Building a variant by applying operations, so its script is known exactly.

This is the generation direction. Rather than recovering a script from a pair,
the operations are *chosen* and applied to a seed, so the script is correct by
construction and needs no annotation. Stacking more operations moves the
variant further from its seed, which is the difficulty dial.

Because generation is parameterised — to substitute a synonym you say which
synonym — every operation in the inventory is usable here, including those
with no detector yet.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from retexo.core.script import EditScript
from retexo.operations import EditOperation, OperationRegistry

# =============================================================================
# Builder
# =============================================================================


class VariantBuilder:
    """Applies operations to a seed passage, recording the script as it goes.

    Source positions are consumed in order unless stated otherwise; reuse
    positions are assigned in the order operations are applied. Call
    :meth:`build` to obtain the variant and its script.

    Example:
        ```python
        builder = VariantBuilder("uox faucibus haesit")
        builder.keep(0)
        builder.syn(1, "gutture")
        builder.delete(2)
        builder.insert("tacuit")
        variant, script = builder.build()
        ```
    """

    #: The method that applies each operation tag. Every tag in the inventory
    #: has exactly one, so a script can always be written without naming a tag
    #: as a string; the test suite asserts this stays complete as operations are
    #: added. Names are the operation's own verb where it has one, and the tag
    #: itself for the substitution family.
    METHOD_FOR_TAG = {
        "NOP": "keep",
        "MORPH": "inflect",
        "SYN": "syn",
        "SYN-DIST": "syn_dist",
        "HYPER": "hyper",
        "HYPO": "hypo",
        "ANT": "ant",
        "NE-SUB": "ne_sub",
        "POS": "pos",
        "SUBST": "subst",
        "FORM": "form",
        "SENSE": "sense",
        "INS": "insert",
        "DEL": "delete",
        "REORDER": "mark_reordered",
        "SPLIT": "split",
        "MERGE": "merge",
        "QUOTE": "quote",
        "FRAME": "frame",
        "ADAPT": "mark_adapted",
        "DISPERSE": "mark_dispersed",
    }

    def __init__(
        self,
        source: Union[Sequence[str], str],
        registry: Optional[OperationRegistry] = None,
    ):
        self.source_tokens: List[str] = (
            list(source) if not isinstance(source, str) else source.split()
        )
        self.registry = registry or OperationRegistry.default()
        self._operations: List[EditOperation] = []
        self._next_target = 0

    # ---------- Token operations ----------

    def keep(self, source_index: int, detail: str = "") -> VariantBuilder:
        """Copy a source token unchanged."""
        return self._one_to_one("NOP", source_index, self.source_tokens[source_index], detail)

    def substitute(
        self, source_index: int, replacement: str, *, tag: str = "SYN", detail: str = ""
    ) -> VariantBuilder:
        """Replace a source token, recording why under ``tag``."""
        return self._one_to_one(tag, source_index, replacement, detail)

    def inflect(self, source_index: int, form: str, detail: str = "") -> VariantBuilder:
        """Rewrite a source token in a different inflection."""
        return self._one_to_one("MORPH", source_index, form, detail)

    # ---------- Substitution family ----------
    #
    # One named method per substitution tag, all delegating to
    # :meth:`substitute`. Every other tag in the inventory already has its own
    # verb (``keep``, ``delete``, ``quote`` …); these close the gap, so a script
    # can be written without passing an operation tag as a string. ``substitute``
    # stays available for the case where the tag is chosen at runtime.

    def syn(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Substitute an attested synonym."""
        return self._one_to_one("SYN", source_index, replacement, detail)

    def subst(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Replace a word with one no named relation fits: the residual."""
        return self._one_to_one("SUBST", source_index, replacement, detail)

    def form(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """E31 back-off: the same lemma or stem in another form, kind not settled."""
        return self._one_to_one("FORM", source_index, replacement, detail)

    def sense(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """E31 back-off: a different word with a nameable relation, not settled."""
        return self._one_to_one("SENSE", source_index, replacement, detail)

    def syn_dist(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Substitute a distributional near-synonym."""
        return self._one_to_one("SYN-DIST", source_index, replacement, detail)

    def hyper(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Generalize: name a broader category."""
        return self._one_to_one("HYPER", source_index, replacement, detail)

    def hypo(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Specify: name a narrower category."""
        return self._one_to_one("HYPO", source_index, replacement, detail)

    def ant(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Substitute an antonym, usually alongside a negation."""
        return self._one_to_one("ANT", source_index, replacement, detail)

    def ne_sub(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Substitute one named entity for another."""
        return self._one_to_one("NE-SUB", source_index, replacement, detail)

    def pos(self, source_index: int, replacement: str, detail: str = "") -> VariantBuilder:
        """Derivational shift: same root, different word class."""
        return self._one_to_one("POS", source_index, replacement, detail)

    # ---------- Structural ----------

    def insert(self, token: str, detail: str = "") -> VariantBuilder:
        """Add a token with no source."""
        op = EditOperation(
            tag="INS",
            target_indices=(self._next_target,),
            target_tokens=(token,),
            detail=detail,
        )
        self._next_target += 1
        self._operations.append(op)
        return self

    def delete(self, source_index: int, detail: str = "") -> VariantBuilder:
        """Drop a source token from the reuse."""
        self._operations.append(
            EditOperation(
                tag="DEL",
                source_indices=(source_index,),
                source_tokens=(self.source_tokens[source_index],),
                detail=detail,
            )
        )
        return self

    def mark_reordered(self, source_index: int, detail: str = "") -> VariantBuilder:
        """Record that an already-written token crossed another's position."""
        self._operations.append(
            EditOperation(
                tag="REORDER",
                source_indices=(source_index,),
                source_tokens=(self.source_tokens[source_index],),
                detail=detail,
            )
        )
        return self

    # ---------- Cardinality ----------

    def split(self, source_index: int, tokens: Sequence[str], detail: str = "") -> VariantBuilder:
        """Expand one source token into several reuse tokens."""
        indices = tuple(range(self._next_target, self._next_target + len(tokens)))
        self._next_target += len(tokens)
        self._operations.append(
            EditOperation(
                tag="SPLIT",
                source_indices=(source_index,),
                target_indices=indices,
                source_tokens=(self.source_tokens[source_index],),
                target_tokens=tuple(tokens),
                detail=detail,
            )
        )
        return self

    def merge(self, source_indices: Sequence[int], token: str, detail: str = "") -> VariantBuilder:
        """Compress several source tokens into one reuse token."""
        self._operations.append(
            EditOperation(
                tag="MERGE",
                source_indices=tuple(source_indices),
                target_indices=(self._next_target,),
                source_tokens=tuple(self.source_tokens[i] for i in source_indices),
                target_tokens=(token,),
                detail=detail,
            )
        )
        self._next_target += 1
        return self

    # ---------- Spans ----------

    def quote(self, source_indices: Sequence[int], detail: str = "") -> VariantBuilder:
        """Copy a contiguous span verbatim as one act."""
        tokens = tuple(self.source_tokens[i] for i in source_indices)
        indices = tuple(range(self._next_target, self._next_target + len(tokens)))
        self._next_target += len(tokens)
        self._operations.append(
            EditOperation(
                tag="QUOTE",
                source_indices=tuple(source_indices),
                target_indices=indices,
                source_tokens=tokens,
                target_tokens=tokens,
                detail=detail,
            )
        )
        return self

    def frame(self, tokens: Sequence[str], detail: str = "") -> VariantBuilder:
        """Insert an attribution formula as one act rather than several."""
        indices = tuple(range(self._next_target, self._next_target + len(tokens)))
        self._next_target += len(tokens)
        self._operations.append(
            EditOperation(
                tag="FRAME",
                target_indices=indices,
                target_tokens=tuple(tokens),
                detail=detail,
            )
        )
        return self

    def mark_adapted(self, source_indices: Sequence[int], detail: str = "") -> VariantBuilder:
        """Note that a region was reused with substitutions inside it."""
        return self._marker("ADAPT", source_indices, detail)

    def mark_dispersed(self, source_indices: Sequence[int], detail: str = "") -> VariantBuilder:
        """Note that shared material was split across a clause boundary."""
        return self._marker("DISPERSE", source_indices, detail)

    # ---------- Result ----------

    def build(self) -> Tuple[List[str], EditScript]:
        """Return the variant and the script that produces it."""
        from retexo.core.scriba import Scriba

        script = EditScript(
            source_tokens=list(self.source_tokens),
            target_tokens=[],
            operations=list(self._operations),
            registry=self.registry,
        )
        target = Scriba(self.registry).execute(script, self.source_tokens)
        script.target_tokens = target
        return target, script

    # ---------- Internals ----------

    def _one_to_one(self, tag: str, source_index: int, token: str, detail: str) -> VariantBuilder:
        self._operations.append(
            EditOperation(
                tag=tag,
                source_indices=(source_index,),
                target_indices=(self._next_target,),
                source_tokens=(self.source_tokens[source_index],),
                target_tokens=(token,),
                detail=detail,
            )
        )
        self._next_target += 1
        return self

    def _marker(self, tag: str, source_indices: Sequence[int], detail: str) -> VariantBuilder:
        self._operations.append(
            EditOperation(
                tag=tag,
                source_indices=tuple(source_indices),
                source_tokens=tuple(self.source_tokens[i] for i in source_indices),
                detail=detail,
            )
        )
        return self
