# baselines/span_pair/decoding.py
"""The segmentation dynamic programme and the expansion of chosen span pairs to word links."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.baselines.base import Rows
from retexo.baselines.record import Edge
from retexo.baselines.span_pair.constants import ENCLITICS, FRAME, INS, NONE, QUOTE, Span
from retexo.core.normalize import normalize

# =============================================================================
# SpanChoice
# =============================================================================


@dataclass
class SpanChoice:
    """One scored option of a reuse span: a source span with a tag, or a null."""

    reuse: Span
    source: Optional[Span]
    tag: str
    log_p: float
    p: float


# =============================================================================
# Segmenter
# =============================================================================


class Segmenter:
    """The span-level dynamic programme and the source-side repair.

    Example:
        ```python
        chosen = Segmenter.segment(options, n_reuse, theta=0.3)
        ```
    """

    @staticmethod
    def segment(
        options: Sequence[SpanChoice], n_reuse: int, *, theta: float = 0.0
    ) -> List[SpanChoice]:
        """The partition of ``[0, n)`` into reuse spans maximising the summed log
        probability of each span's best option (a pair only when its probability
        is at least ``theta``, else its best null); a length-one INS span is
        always available; then two chosen pairs whose source spans overlap
        resolve to the higher-scoring one, the other falling to its null."""
        import math

        best_pair: Dict[Span, SpanChoice] = {}
        best_null: Dict[Span, SpanChoice] = {}
        for option in options:
            if option.tag == NONE:
                continue  # "not a run" is never a choice
            table = best_null if option.source is None else best_pair
            if option.reuse not in table or option.log_p > table[option.reuse].log_p:
                table[option.reuse] = option
        choice: Dict[Span, SpanChoice] = {}
        for span in set(best_pair) | set(best_null):
            pair, null = best_pair.get(span), best_null.get(span)
            if pair is not None and pair.p >= theta and (null is None or pair.log_p >= null.log_p):
                choice[span] = pair
            elif null is not None:
                choice[span] = null
            elif pair is not None and pair.p >= theta:
                choice[span] = pair
        floor = math.log(1e-6)
        score = [float("-inf")] * (n_reuse + 1)
        back: List[Optional[Span]] = [None] * (n_reuse + 1)
        score[0] = 0.0
        for end in range(1, n_reuse + 1):
            for span, option in choice.items():
                a, b = span
                if b + 1 != end:
                    continue
                cand = score[a] + option.log_p
                if cand > score[end]:
                    score[end], back[end] = cand, span
            if back[end] is None:  # a length-one null is always available
                score[end], back[end] = score[end - 1] + floor, (end - 1, end - 1)
                choice.setdefault(
                    (end - 1, end - 1), SpanChoice((end - 1, end - 1), None, INS, floor, 0.0)
                )
        chosen: List[SpanChoice] = []
        end = n_reuse
        while end > 0:
            span = back[end]
            chosen.append(choice[span])
            end = span[0]
        chosen.reverse()
        return Segmenter.repair(chosen, best_null)

    @staticmethod
    def repair(chosen: List[SpanChoice], best_null: Dict[Span, SpanChoice]) -> List[SpanChoice]:
        """Greedy one-to-one on the source side: of two pairs whose source spans
        overlap, the lower-scoring one becomes its null."""
        out = list(chosen)
        order = sorted(range(len(out)), key=lambda i: -out[i].log_p)
        used: List[Span] = []
        for i in order:
            option = out[i]
            if option.source is None:
                continue
            c, d = option.source
            if any(c <= d2 and c2 <= d for c2, d2 in used):
                null = best_null.get(option.reuse)
                out[i] = (
                    null
                    if null is not None
                    else SpanChoice(option.reuse, None, INS, float("-inf"), 0.0)
                )
            else:
                used.append((c, d))
        return out


# =============================================================================
# Expander
# =============================================================================


class Expander:
    """Chosen span pairs to word links, tags, extra edges and interface (I) rows.

    Example:
        ```python
        links, tags, frame, extra, rows = Expander.expand(chosen, grid, reuse, source, type_logits)
        ```
    """

    @staticmethod
    def _enclitic_split(word: str) -> bool:
        key = normalize(word)
        return any(key.endswith(e) and len(key) - len(e) >= 3 for e in ENCLITICS)

    @classmethod
    def word_pairs(cls, choice: SpanChoice, grid) -> Tuple[List[Tuple[int, int]], List[Edge]]:
        """The word links of one pair: position-wise on equal lengths, the
        Hungarian inside the rectangle otherwise; returns ``(links, extra)``."""
        a, b = choice.reuse
        c, d = choice.source
        n_r, n_s = b - a + 1, d - c + 1
        if n_r == n_s:
            return [(a + k, c + k) for k in range(n_r)], []
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        block = np.asarray(grid)[a : b + 1, c : d + 1]
        rows_i, cols_i = linear_sum_assignment(-block)
        return [(a + int(i), c + int(j)) for i, j in zip(rows_i, cols_i)], []

    @classmethod
    def expand(
        cls,
        chosen: Sequence[SpanChoice],
        grid,
        reuse_tokens: Sequence[str],
        source_tokens: Sequence[str],
        type_of,
    ) -> Tuple[List[int], List[str], List[int], List[Edge], Rows]:
        """``type_of(t, s) -> str`` names a word pair inside an ADAPT span."""
        n = len(reuse_tokens)
        links, tags, frame = [-1] * n, [""] * n, [0] * n
        extra: List[Edge] = []
        rows: Rows = [[(-1, 1.0)] for _ in range(n)]
        for choice in chosen:
            a, b = choice.reuse
            if choice.source is None:
                for t in range(a, b + 1):
                    frame[t] = int(choice.tag == FRAME)
                    rows[t] = [(-1, round(max(choice.p, 1e-6), 6))]
                continue
            pairs, more = cls.word_pairs(choice, grid)
            for t, s in pairs:
                links[t] = s
                tags[t] = "COPY" if choice.tag == QUOTE else type_of(t, s)
                rows[t] = [(s, round(choice.p, 6)), (-1, round(1.0 - choice.p, 6))]
            c, d = choice.source
            n_r, n_s = b - a + 1, d - c + 1
            linked_t = {t for t, _ in pairs}
            linked_s = {s for _, s in pairs}
            if n_r == n_s + 1:  # two reuse words render one source word: SPLIT
                lone = next((t for t in range(a, b + 1) if t not in linked_t), None)
                if lone is not None:
                    neighbour = (
                        lone - 1
                        if lone - 1 >= a and links[lone - 1] >= 0
                        else lone + 1
                        if lone + 1 <= b
                        else None
                    )
                    if (
                        neighbour is not None
                        and links[neighbour] >= 0
                        and cls._enclitic_split(source_tokens[links[neighbour]])
                    ):
                        links[lone] = links[neighbour]
                        tags[lone] = "SPLIT"
                        tags[neighbour] = "SPLIT"
                        rows[lone] = [
                            (links[lone], round(choice.p, 6)),
                            (-1, round(1.0 - choice.p, 6)),
                        ]
            elif n_s == n_r + 1:  # one reuse word renders two source words: MERGE
                lone_s = next((s for s in range(c, d + 1) if s not in linked_s), None)
                if lone_s is not None:
                    t_near = next(
                        (t for t in range(a, b + 1) if links[t] in (lone_s - 1, lone_s + 1)), None
                    )
                    if t_near is not None and cls._enclitic_split(reuse_tokens[t_near]):
                        tags[t_near] = "MERGE"
                        extra.append(Edge(t_near, lone_s, "MERGE"))
        return links, tags, frame, extra, rows
