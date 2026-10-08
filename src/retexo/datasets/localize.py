# retexo/datasets/localize.py
"""
Finding the reused span inside a passage pair.

A benchmark pair is two passages, and the reuse frequently occupies a fragment
of the longer one. One annotated citation pairs twelve tokens of Virgil against
eighty tokens of Ambrose's prose: an oracle planning over the whole pair emits
seventy-six insertions, none of which say anything about the reuse. Measured
over annotated citations, planning over the passage puts insertions and
deletions at 77.6% of operations and copies at 16.8%; planning over the
best-matching window moves those to 65.1% and 27.8%.

That difference is not a property of reuse. It is a property of the unit, and
supervision derived from whole passages would teach a model that reuse is
mostly insertion. Localizing first is therefore part of building the training
data rather than a preprocessing convenience.

This module implements the cheap version: a fixed-length window chosen by
shared vocabulary. It is deliberately not the eventual answer, which is for the
oracle to emit ``FRAME`` for the surrounding material so that it is described
rather than discarded. What is discarded here is recorded on the result, so the
cost of the shortcut stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from retexo.core.normalize import normalize

#: Slack around the shorter passage's length, so a window can cover a reuse
#: that expanded slightly on being quoted.
WINDOW_SLACK = 4


@dataclass(frozen=True)
class Localized:
    """A pair narrowed to its reused span, and what that cost.

    Example:
        ```python
        span = Localized.find(source_tokens, target_tokens)
        script = oracle.plan_tokens(span.source, span.target)
        ```
    """

    source: List[str]
    target: List[str]
    source_offset: int = 0
    target_offset: int = 0
    dropped: int = 0

    @property
    def trimmed(self) -> bool:
        """Whether anything was removed."""
        return self.dropped > 0

    @staticmethod
    def _best_window(long: Sequence[str], short: Sequence[str]) -> Tuple[int, int]:
        """Start and end in ``long`` of the window sharing most words with ``short``.

        Ties are broken toward the earlier window, so the choice is deterministic.
        """
        width = min(len(long), len(short) + WINDOW_SLACK)
        wanted = {normalize(token) for token in short}
        best_start, best_score = 0, -1
        for start in range(len(long) - width + 1):
            score = sum(1 for token in long[start : start + width] if normalize(token) in wanted)
            if score > best_score:
                best_start, best_score = start, score
        return best_start, best_start + width

    @classmethod
    def find(
        cls, source: Sequence[str], target: Sequence[str], *, enabled: bool = True
    ) -> Localized:
        """Narrow the longer passage to the window that best matches the shorter.

        The shorter passage is left whole: it is the better estimate of the extent
        of the reuse, and trimming both sides would let the window drift onto an
        incidental overlap.
        """
        source, target = list(source), list(target)
        if not enabled or not source or not target:
            return cls(source, target)

        if len(target) > len(source) + WINDOW_SLACK:
            start, end = cls._best_window(target, source)
            return cls(source, target[start:end], 0, start, len(target) - (end - start))
        if len(source) > len(target) + WINDOW_SLACK:
            start, end = cls._best_window(source, target)
            return cls(source[start:end], target, start, 0, len(source) - (end - start))
        return cls(source, target)
