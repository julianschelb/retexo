# retexo/operations/base.py
"""
The operation contract: what an operation is, and what every one must support.

An operation is defined by three abilities. It can be **applied**, turning a
source region into a reuse region, which is how synthetic variants are built.
It can be **inverted**, yielding the operation that undoes it, so a script can
be run backwards. And it may be **detected**, recovered from a pair by the
oracle — the only one of the three that some operations cannot yet do.

That asymmetry is deliberate. Generation is *parameterised*: to substitute a
synonym you are told which synonym. Detection is *inferred*: you must work out
that two words stand in a synonym relation, which needs a resource. So an
operation can be fully usable for building variants while still having no
reliable detector.

Operations fall into three roles, which determine how a script replays:

- **writing** — occupies reuse positions and puts tokens there;
- **consuming** — occupies source positions and writes nothing (``DEL``);
- **marker** — occupies nothing, and exists to be counted (``REORDER``,
  ``ADAPT``, ``DISPERSE``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

# =============================================================================
# Roles and levels
# =============================================================================


class Role(str, Enum):
    """How an operation participates in replaying a script."""

    WRITING = "writing"
    CONSUMING = "consuming"
    MARKER = "marker"


class Level(str, Enum):
    """Where an operation sits in the vocabulary."""

    TOKEN = "token"
    CARDINALITY = "cardinality"
    STRUCTURAL = "structural"
    SPAN = "span"


# =============================================================================
# Operation instances
# =============================================================================


@dataclass(frozen=True)
class EditOperation:
    """One operation as it appears in a script.

    Indices and tokens are tuples throughout so that one-to-one, one-to-many
    and many-to-one operations share a representation and replay by the same
    rule.

    Attributes:
        tag: Operation type, e.g. ``"NOP"``.
        source_indices: Source positions consumed.
        target_indices: Reuse positions written.
        source_tokens: Surface forms consumed.
        target_tokens: Surface forms written.
        detail: Why this operation was chosen, for a reader.
    """

    tag: str
    source_indices: Tuple[int, ...] = ()
    target_indices: Tuple[int, ...] = ()
    source_tokens: Tuple[str, ...] = ()
    target_tokens: Tuple[str, ...] = ()
    detail: str = ""

    def human(self) -> str:
        """One-line rendering for a philologist to read."""
        src = " ".join(self.source_tokens) if self.source_tokens else "—"
        tgt = " ".join(self.target_tokens) if self.target_tokens else "—"
        note = f"  [{self.detail}]" if self.detail else ""
        return f"{self.tag:9} {src:>24} -> {tgt:<24}{note}"

    def serialize(self) -> str:
        """Compact form: ``TAG src_indices>target_indices:tokens``."""
        src = ",".join(str(i) for i in self.source_indices) or "-"
        tgt = ",".join(str(i) for i in self.target_indices) or "-"
        toks = " ".join(self.target_tokens)
        body = f"{self.tag} {src}>{tgt}"
        return f"{body} {toks}" if toks else body


# =============================================================================
# Operation types
# =============================================================================


class Operation(ABC):
    """A kind of operation: its cost, its role, and how it applies and inverts."""

    tag: str = ""
    level: Level = Level.TOKEN
    role: Role = Role.WRITING
    default_cost: float = 0.0

    #: Resources a detector would need. Empty means detection needs nothing.
    requires: Tuple[str, ...] = ()

    #: Whether a detector exists. Operations usable only for generation set
    #: this to False; see the module docstring.
    detectable: bool = True

    @abstractmethod
    def inverse_tag(self) -> str:
        """Tag of the operation that undoes this one."""

    def invert(self, op: EditOperation) -> EditOperation:
        """Return the operation undoing ``op``.

        Source and reuse swap roles, and the tag becomes its own inverse.
        Overridden only where swapping is not enough.
        """
        return EditOperation(
            tag=self.inverse_tag(),
            source_indices=op.target_indices,
            target_indices=op.source_indices,
            source_tokens=op.target_tokens,
            target_tokens=op.source_tokens,
            detail=op.detail,
        )

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        """Whether this operation relates the two tokens, and why.

        Returns a short justification, or ``None``. Operations that cannot yet
        be detected leave this returning ``None`` and set ``detectable``.
        """
        return None


# =============================================================================
# Registry
# =============================================================================


class OperationRegistry:
    """The operations available to an oracle or a generator.

    Example:
        ```python
        registry = OperationRegistry.default()
        registry["SYN"].default_cost      # 0.5
        registry.detectable()             # only those with a working detector
        ```
    """

    def __init__(self, operations: Sequence[Operation]):
        self._by_tag: Dict[str, Operation] = {op.tag: op for op in operations}

    # ---------- Access ----------

    def __getitem__(self, tag: str) -> Operation:
        """The operation registered under a tag.

        Raises:
            KeyError: If the tag is not registered.
        """
        if tag not in self._by_tag:
            raise KeyError(f"unknown operation tag: {tag!r}")
        return self._by_tag[tag]

    def __contains__(self, tag: str) -> bool:
        return tag in self._by_tag

    def __iter__(self):
        return iter(self._by_tag.values())

    def __len__(self) -> int:
        return len(self._by_tag)

    # ---------- Introspection ----------

    def tags(self) -> Tuple[str, ...]:
        """Every registered tag, in registration order."""
        return tuple(self._by_tag)

    def detectable(self) -> Tuple[Operation, ...]:
        """Operations that have a working detector."""
        return tuple(op for op in self._by_tag.values() if op.detectable)

    def costs(self) -> Dict[str, float]:
        """Default cost per tag."""
        return {op.tag: op.default_cost for op in self._by_tag.values()}

    def rows(self) -> Tuple[Dict[str, object], ...]:
        """One record per operation, in registration order.

        The tabular view of the inventory, kept free of pandas so it can be
        asserted against in tests and printed anywhere.
        """
        return tuple(
            {
                "tag": op.tag,
                "level": op.level.value,
                "role": op.role.value,
                "cost": op.default_cost,
                "detectable": op.detectable,
                "requires": ", ".join(op.requires) or "—",
            }
            for op in self._by_tag.values()
        )

    def to_frame(self):
        """The inventory as a DataFrame indexed by tag.

        Example:
            ```python
            OperationRegistry.default().to_frame()
            ```
        """
        import pandas as pd

        return pd.DataFrame(self.rows()).set_index("tag")

    # ---------- Construction ----------

    @classmethod
    def default(cls) -> OperationRegistry:
        """Every operation in the inventory."""
        from retexo.operations import ALL_OPERATIONS

        return cls(ALL_OPERATIONS)
