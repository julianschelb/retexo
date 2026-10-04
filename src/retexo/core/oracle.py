# retexo/core/oracle.py
"""
The EditPlan oracle: derives a typed script for a pair without annotation.

Four stages, kept separate so each can be tested and swapped independently.

1. **Relation search** — ask every available relation about every token pair.
   Answers are memoised on the normalized pair, since a relation between two
   lemmas does not depend on the passage they came from. Without that, scoring
   a candidate pool means repeating the same lexical lookups thousands of
   times.
2. **Alignment** — choose a one-to-one assignment. The distance is defined as a
   *minimum*, so this is minimum-cost bipartite matching by default rather than
   the greedy assignment the earlier prototype used.
3. **Reorder detection** — aligned pairs outside a longest increasing
   subsequence of source indices are crossing, and are marked.
4. **Emission** — unaligned source tokens become deletions, unaligned reuse
   tokens insertions, and ``REORDER`` is emitted as an operation in its own
   right so the cost stays recoverable from the operation counts.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from retexo.core.normalize import normalize
from retexo.operations import EditOperation, Level, OperationRegistry
from retexo.resources import Resources
from retexo.core.script import CostModel, EditScript

# =============================================================================
# Alignment
# =============================================================================


@dataclass(frozen=True)
class AlignmentCandidate:
    """A possible pairing of one source token with one reuse token."""

    source_index: int
    target_index: int
    tag: str
    detail: str
    cost: float


class Aligner(ABC):
    """Chooses a one-to-one assignment from the scored candidate pairings."""

    @abstractmethod
    def align(self, candidates: Sequence[AlignmentCandidate]) -> List[AlignmentCandidate]:
        """Select a mutually exclusive subset."""


class OptimalAligner(Aligner):
    """Minimum-cost bipartite matching.

    What the definition of the distance assumes: the script is the cheapest one
    relating the two passages. Falls back to the greedy assignment if SciPy is
    unavailable, reporting the substitution rather than failing.
    """

    def align(self, candidates: Sequence[AlignmentCandidate]) -> List[AlignmentCandidate]:
        if not candidates:
            return []
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return GreedyAligner().align(candidates)

        sources = sorted({c.source_index for c in candidates})
        targets = sorted({c.target_index for c in candidates})
        source_at = {s: i for i, s in enumerate(sources)}
        target_at = {t: j for j, t in enumerate(targets)}

        # Unpaired cells must never be chosen, so they cost more than any real
        # assignment could total.
        blocked = sum(c.cost for c in candidates) + len(candidates) + 1.0
        matrix = np.full((len(sources), len(targets)), blocked)
        best: Dict[Tuple[int, int], AlignmentCandidate] = {}
        for candidate in candidates:
            key = (candidate.source_index, candidate.target_index)
            if key not in best or candidate.cost < best[key].cost:
                best[key] = candidate
        for (source, target), candidate in best.items():
            matrix[source_at[source], target_at[target]] = candidate.cost

        rows, columns = linear_sum_assignment(matrix)
        chosen = []
        for row, column in zip(rows, columns):
            if matrix[row, column] >= blocked:
                continue
            chosen.append(best[(sources[row], targets[column])])
        return sorted(chosen, key=lambda c: c.target_index)


class GreedyAligner(Aligner):
    """Lowest-cost-first assignment, skipping already-used indices.

    Does not minimise total cost, so it is not what the definition assumes.
    Retained so the two can be compared: if the difference is negligible on
    real pairs, the cheaper aligner is defensible for large candidate pools.
    """

    def align(self, candidates: Sequence[AlignmentCandidate]) -> List[AlignmentCandidate]:
        used_source: Set[int] = set()
        used_target: Set[int] = set()
        chosen: List[AlignmentCandidate] = []
        for candidate in sorted(candidates, key=lambda c: (c.cost, c.source_index, c.target_index)):
            if candidate.source_index in used_source or candidate.target_index in used_target:
                continue
            used_source.add(candidate.source_index)
            used_target.add(candidate.target_index)
            chosen.append(candidate)
        return sorted(chosen, key=lambda c: c.target_index)


# =============================================================================
# Oracle
# =============================================================================


@dataclass
class OracleConfig:
    """Knobs for the oracle. Defaults reproduce the configuration in the paper."""

    detect_reorder: bool = True
    cache_relations: bool = True


class EditPlanOracle:
    """Derives an :class:`EditScript` for a (source, reuse) pair.

    Usable with no arguments; everything is injectable for experiments.

    Example:
        ```python
        oracle = EditPlanOracle()
        script = oracle.plan("uox faucibus haesit", "haesit uox faucibus")
        print(script.human())
        ```

    To ablate the vocabulary, drop operations from the registry:

        ```python
        oracle = EditPlanOracle(exclude={"SYN", "SYN-DIST"})
        ```
    """

    def __init__(
        self,
        registry: Optional[OperationRegistry] = None,
        resources: Optional[Resources] = None,
        aligner: Optional[Aligner] = None,
        costs: Optional[CostModel] = None,
        config: Optional[OracleConfig] = None,
        exclude: Optional[Set[str]] = None,
    ):
        self.registry = registry or OperationRegistry.default()
        self.resources = resources or Resources()
        self.aligner = aligner or OptimalAligner()
        self.costs = costs or CostModel(registry=self.registry)
        self.config = config or OracleConfig()
        self.exclude = set(exclude or ())
        self._cache: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}

    # ---------- Public API ----------

    def plan(self, source: str, reuse: str) -> EditScript:
        """Derive the script transforming ``source`` into ``reuse``."""
        return self.plan_tokens(source.split(), reuse.split())

    def plan_tokens(
        self, source_tokens: Sequence[str], target_tokens: Sequence[str]
    ) -> EditScript:
        """Token-level entry point, for callers that have already tokenized."""
        source_tokens, target_tokens = list(source_tokens), list(target_tokens)
        candidates = self._find_candidates(source_tokens, target_tokens)
        aligned = self.aligner.align(candidates)
        reordered = self._mark_reordered(aligned) if self.config.detect_reorder else set()
        return self._emit(source_tokens, target_tokens, aligned, reordered)

    # ---------- Stages ----------

    def _relations(self):
        """Detectable token-level operations, in precedence order."""
        return [
            op for op in self.registry
            if op.level is Level.TOKEN and op.detectable and op.tag not in self.exclude
        ]

    def _classify(self, source: str, target: str) -> Optional[Tuple[str, str]]:
        """The most specific relation holding between two tokens, if any."""
        key = (normalize(source), normalize(target))
        if self.config.cache_relations and key in self._cache:
            return self._cache[key]
        found = None
        for operation in self._relations():
            detail = operation.detect(source, target, self.resources)
            if detail is not None:
                found = (operation.tag, detail)
                break
        if self.config.cache_relations:
            self._cache[key] = found
        return found

    def _find_candidates(
        self, source_tokens: Sequence[str], target_tokens: Sequence[str]
    ) -> List[AlignmentCandidate]:
        """Ask every available relation about every token pair."""
        candidates: List[AlignmentCandidate] = []
        costs = self.costs.costs
        for i, source in enumerate(source_tokens):
            for j, target in enumerate(target_tokens):
                relation = self._classify(source, target)
                if relation is None:
                    continue
                tag, detail = relation
                candidates.append(
                    AlignmentCandidate(i, j, tag, detail, costs.get(tag, 1.0))
                )
        return candidates

    @staticmethod
    def _lis_indices(sequence: Sequence[int]) -> Set[int]:
        """Positions belonging to a longest strictly increasing subsequence."""
        if not sequence:
            return set()
        tails: List[int] = []
        tail_positions: List[int] = []
        previous = [-1] * len(sequence)
        for position, value in enumerate(sequence):
            slot = bisect.bisect_left(tails, value)
            if slot == len(tails):
                tails.append(value)
                tail_positions.append(position)
            else:
                tails[slot] = value
                tail_positions[slot] = position
            previous[position] = tail_positions[slot - 1] if slot > 0 else -1
        keep: Set[int] = set()
        position = tail_positions[-1]
        while position != -1:
            keep.add(position)
            position = previous[position]
        return keep

    def _mark_reordered(self, aligned: Sequence[AlignmentCandidate]) -> Set[int]:
        """Reuse positions whose alignment crosses another's."""
        by_target = sorted(aligned, key=lambda c: c.target_index)
        keep = self._lis_indices([c.source_index for c in by_target])
        return {c.target_index for k, c in enumerate(by_target) if k not in keep}

    def _emit(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        aligned: Sequence[AlignmentCandidate],
        reordered: Set[int],
    ) -> EditScript:
        """Turn an alignment into an ordered operation sequence."""
        operations: List[EditOperation] = []
        by_target = {c.target_index: c for c in aligned}
        used_source = {c.source_index for c in aligned}

        for j, token in enumerate(target_tokens):
            candidate = by_target.get(j)
            if candidate is None:
                operations.append(EditOperation("INS", (), (j,), (), (token,)))
                continue
            operations.append(
                EditOperation(
                    candidate.tag, (candidate.source_index,), (j,),
                    (source_tokens[candidate.source_index],), (token,),
                    candidate.detail,
                )
            )
        for i, token in enumerate(source_tokens):
            if i not in used_source:
                operations.append(EditOperation("DEL", (i,), (), (token,), ()))
        for j in sorted(reordered):
            candidate = by_target[j]
            operations.append(
                EditOperation(
                    "REORDER", (candidate.source_index,), (),
                    (source_tokens[candidate.source_index],), (), "crossing",
                )
            )
        return EditScript(list(source_tokens), list(target_tokens), operations, self.registry)
