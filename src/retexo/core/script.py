# retexo/core/script.py
"""
The edit script, its cost model, and its inverse.

A script is the paper's central object: an ordered, typed, replayable account
of how a source passage becomes a reuse. It is an explanation, a distance, and
a profile of operation types, all at once.

The distance is linear in the operation counts by construction:

    d(S, Q) = sum over tags of  cost[tag] * count[tag]

so every costed operation must appear in :meth:`EditScript.op_counts`,
``REORDER`` included. Folding a cost into a neighbouring operation would break
the linearity that cost learning depends on, and a unit test enforces it.

Scripts are invertible. Every operation records both the tokens it consumes
and the tokens it writes, so swapping those roles and mapping each tag to its
inverse yields a script that turns the reuse back into the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from retexo.operations import EditOperation, OperationRegistry, Role

# =============================================================================
# Cost model
# =============================================================================


@dataclass
class CostModel:
    """Maps operation tags to costs.

    Example:
        ```python
        costs = CostModel()          # inventory defaults
        costs.score(script)          # -> 1.25
        ```
    """

    costs: Dict[str, float] = field(default_factory=dict)
    registry: Optional[OperationRegistry] = None

    def __post_init__(self) -> None:
        if not self.costs:
            registry = self.registry or OperationRegistry.default()
            self.costs = registry.costs()

    def score(self, script: EditScript) -> float:
        """Total cost, linear in the operation counts."""
        return sum(self.costs.get(tag, 0.0) * n for tag, n in script.op_counts().items())


# =============================================================================
# Edit script
# =============================================================================


@dataclass
class EditScript:
    """An ordered, typed transformation from a source passage into a reuse.

    Example:
        ```python
        script.op_counts()        # {'NOP': 3, 'REORDER': 1}
        script.cost()             # 0.5
        script.human()            # a readable account
        script.invert()           # the script back to the source
        ```
    """

    source_tokens: List[str]
    target_tokens: List[str]
    operations: List[EditOperation] = field(default_factory=list)
    registry: Optional[OperationRegistry] = None

    def _registry(self) -> OperationRegistry:
        if self.registry is None:
            self.registry = OperationRegistry.default()
        return self.registry

    # ---------- Measurement ----------

    def op_counts(self) -> Dict[str, int]:
        """Count of each operation tag: the feature vector for the cost model."""
        counts: Dict[str, int] = {}
        for op in self.operations:
            counts[op.tag] = counts.get(op.tag, 0) + 1
        return counts

    def cost(self, costs: Optional[CostModel] = None) -> float:
        """Total cost under a cost model, defaulting to the inventory costs."""
        return (costs or CostModel(registry=self._registry())).score(self)

    def normalized_cost(self, costs: Optional[CostModel] = None) -> float:
        """Cost divided by the longer passage, so pairs are comparable."""
        denominator = max(len(self.source_tokens), len(self.target_tokens), 1)
        return self.cost(costs) / denominator

    # ---------- Inversion ----------

    def invert(self) -> EditScript:
        """The script that turns the reuse back into the source."""
        registry = self._registry()
        inverted = [registry[op.tag].invert(op) for op in self.operations]
        return EditScript(
            source_tokens=list(self.target_tokens),
            target_tokens=list(self.source_tokens),
            operations=inverted,
            registry=registry,
        )

    # ---------- Rendering ----------

    def human(self) -> str:
        """Multi-line rendering for a philologist to read and contest."""
        lines = [op.human() for op in self.operations]
        lines.append("")
        lines.append(f"cost = {self.cost():.2f}   normalized = {self.normalized_cost():.3f}")
        lines.append(f"operations = {self.op_counts()}")
        return "\n".join(lines)

    def serialize(self) -> str:
        """Compact one-line form: operations separated by ``|``."""
        return " | ".join(op.serialize() for op in self.operations)

    # ---------- Introspection ----------

    def writing_operations(self) -> List[EditOperation]:
        """Operations that put tokens into the reuse, in target order."""
        registry = self._registry()
        writing = [op for op in self.operations if registry[op.tag].role is Role.WRITING]
        return sorted(writing, key=lambda op: op.target_indices or (0,))
