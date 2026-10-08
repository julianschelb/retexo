# baselines/span_pair/candidates.py
"""Enumeration of the spans of one record and the two-tier pruning of candidate pairs."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from retexo.baselines.span_pair.constants import Span
from retexo.core.normalize import normalize

# =============================================================================
# SpanEnumerator
# =============================================================================


class SpanEnumerator:
    """Spans and the pruned pairs of one record.

    Example:
        ```python
        spans = SpanEnumerator.enumerate_spans(5, 3)             # 12 spans
        pairs = SpanEnumerator.candidate_pairs(grid, reuse, source, L_max=6)
        ```
    """

    @staticmethod
    def enumerate_spans(n: int, max_length: int) -> List[Span]:
        return [(a, b) for a in range(n) for b in range(a, min(n, a + max_length))]

    @classmethod
    def candidate_pairs(
        cls,
        grid,
        reuse_tokens: Sequence[str],
        source_tokens: Sequence[str],
        *,
        L_max: int = 6,
        eps: float = 0.01,
        K_pairs: int = 2000,
        gold_pairs: Sequence[Tuple[Span, Span]] = (),
    ) -> List[Tuple[Span, Span]]:
        """Two tiers. First, for every cell above ``eps`` and every equal-form
        cell, the 1x1 pair and every equal-length diagonal pair through it (the
        shape of almost every gold run), kept whole. Then the unequal-length
        rectangles (a difference of at most two) holding a cell above ``eps``,
        ranked by their best cell and the smaller area, up to ``K_pairs`` in
        all; the gold pairs always included (training)."""
        import numpy as np

        grid = np.asarray(grid, dtype=np.float32)
        T, S = grid.shape
        if T == 0 or S == 0:
            return list(gold_pairs)
        forms_r = [normalize(w) for w in reuse_tokens]
        forms_s = [normalize(w) for w in source_tokens]
        anchors = {
            (t, u)
            for t in range(T)
            for u in range(S)
            if grid[t, u] > eps or (forms_r[t] and forms_r[t] == forms_s[u])
        }
        kept: List[Tuple[Span, Span]] = []
        seen = set()
        for t, u in sorted(anchors, key=lambda c: -grid[c]):
            for length in range(1, L_max + 1):
                for offset in range(length):
                    a, c = t - offset, u - offset
                    b, d = a + length - 1, c + length - 1
                    if a < 0 or c < 0 or b >= T or d >= S:
                        continue
                    pair = ((a, b), (c, d))
                    if pair not in seen:
                        seen.add(pair)
                        kept.append(pair)
        above = (grid > eps).astype(np.int32)
        scored: List[Tuple[float, int, Span, Span]] = []
        for a, b in cls.enumerate_spans(T, L_max):
            col_any = np.concatenate(
                [[0], np.cumsum(above[a : b + 1].any(axis=0).astype(np.int32))]
            )
            col_max = grid[a : b + 1].max(axis=0)
            length_r = b - a + 1
            for length_s in range(max(1, length_r - 2), min(L_max + 2, S, length_r + 2) + 1):
                if length_s == length_r:
                    continue
                starts = np.arange(0, S - length_s + 1)
                keep = (col_any[starts + length_s] - col_any[starts]) > 0
                peak = np.max(
                    np.stack([col_max[k : k + len(starts)] for k in range(length_s)]), axis=0
                )
                for c in starts[keep]:
                    pair = ((a, b), (int(c), int(c) + length_s - 1))
                    if pair not in seen:
                        scored.append((float(peak[c]), -(length_r * length_s), pair[0], pair[1]))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        for _, _, r, s_ in scored:
            if len(kept) >= K_pairs:
                break
            kept.append((r, s_))
            seen.add((r, s_))
        for pair in gold_pairs:
            if pair not in seen:
                kept.append(pair)
                seen.add(pair)
        return kept
