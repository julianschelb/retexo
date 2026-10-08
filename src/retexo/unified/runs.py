# retexo/unified/runs.py
"""Runs: the unit the structured decoder decides in.

A run is a stretch of the reuse that comes from a stretch of the source, in
order -- a quotation or a fragment spliced into the citing author's own words
-- or a stretch that comes from nowhere: a citing formula (FRAME) or free text
(INS). A ``Segmentation`` is a partition of the reuse into runs; it is what the
decoder returns, what the gold is read into for the structured margin, and
what turns back into per-word links and frame flags for the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: The three kinds of run.
QUOTE, FRAME, INS = "quote", "frame", "ins"


@dataclass(frozen=True)
class Run:
    """Reuse words ``start .. end - 1``; a QUOTE run maps them, in order, onto
    source words ``source_start .. source_start + length - 1``.

    Attributes:
        start, end: Reuse indices, ``end`` exclusive.
        kind: ``quote`` | ``frame`` | ``ins``.
        source_start: The first source word of a QUOTE run, ``None`` otherwise.
    """

    start: int
    end: int
    kind: str
    source_start: Optional[int] = None

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def source_end(self) -> Optional[int]:
        return None if self.source_start is None else self.source_start + self.length

    def source_of(self, t: int) -> int:
        """The source word reuse word ``t`` maps to, ``-1`` for a null run."""
        if self.kind != QUOTE or not (self.start <= t < self.end):
            return -1
        return self.source_start + (t - self.start)


class Segmentation:
    """A partition of the reuse into runs, in reuse order, covering every word once.

    Example:
        ```python
        seg = Segmentation.from_links([-1, -1, 3, 4, 5, -1], frame=[1, 1, 0, 0, 0, 0])
        seg.runs          # [Run(0, 2, frame), Run(2, 5, quote, source_start=3), Run(5, 6, ins)]
        seg.links()       # [-1, -1, 3, 4, 5, -1]
        ```
    """

    def __init__(self, runs: Sequence[Run], n_reuse: int):
        self.runs = list(runs)
        self.n_reuse = n_reuse
        self._check()

    def _check(self) -> None:
        position = 0
        for run in self.runs:
            if run.start != position or run.end <= run.start:
                raise ValueError(
                    f"runs must tile the reuse in order; got {run} at position {position}"
                )
            if run.kind == QUOTE and run.source_start is None:
                raise ValueError(f"a quote run needs a source_start: {run}")
            position = run.end
        if position != self.n_reuse:
            raise ValueError(f"runs cover {position} of {self.n_reuse} reuse words")

    # ---------- to and from the per-word view ----------

    def links(self) -> List[int]:
        out = [-1] * self.n_reuse
        for run in self.runs:
            for t in range(run.start, run.end):
                out[t] = run.source_of(t)
        return out

    def frame(self) -> List[int]:
        out = [0] * self.n_reuse
        for run in self.runs:
            if run.kind == FRAME:
                for t in range(run.start, run.end):
                    out[t] = 1
        return out

    @classmethod
    def from_links(
        cls, links: Sequence[int], frame: Optional[Sequence[int]] = None
    ) -> Segmentation:
        """The gold read as runs: maximal stretches whose links are consecutive
        source words, unlinked stretches split by the frame flag."""
        n = len(links)
        frame = list(frame) if frame is not None else [0] * n
        runs: List[Run] = []
        t = 0
        while t < n:
            s = links[t]
            if s is None or s < 0:
                kind = FRAME if frame[t] else INS
                end = t + 1
                while (
                    end < n
                    and (links[end] is None or links[end] < 0)
                    and bool(frame[end]) == (kind == FRAME)
                ):
                    end += 1
                runs.append(Run(t, end, kind))
            else:
                end = t + 1
                while end < n and links[end] is not None and links[end] == s + (end - t):
                    end += 1
                runs.append(Run(t, end, QUOTE, source_start=int(s)))
            t = end
        return cls(runs, n)

    # ---------- comparison ----------

    def cells(self) -> List[Tuple[int, int]]:
        """Every (reuse, source) pair a QUOTE run asserts."""
        return [
            (t, run.source_of(t))
            for run in self.runs
            if run.kind == QUOTE
            for t in range(run.start, run.end)
        ]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Segmentation)
            and self.runs == other.runs
            and self.n_reuse == other.n_reuse
        )

    def __repr__(self) -> str:
        return f"Segmentation({self.runs})"
