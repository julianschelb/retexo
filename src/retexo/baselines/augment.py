# retexo/baselines/augment.py
"""Augmentations of the gold records for the failure-mode dry run (chain 1, row 5).

``GoldReorder`` swaps the two halves of a record's reuse side -- the case the pointer
misses most (13 of the 19 missed substitutions of fold 4 sit in a reordered or
rewritten clause): the cut falls on a clause boundary near the middle (a token ending
in ``, ; : . ? !``), else on the middle, the second half comes out first, and every
link, span and cached per-token annotation is remapped with it. The source side is
untouched, so the reordered copy teaches "the same links under a crossing".
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from retexo.baselines.record import Record, Span

#: A token that ends a clause: the cut prefers to fall after one of these.
CLAUSE_END = (",", ";", ":", ".", "?", "!")


class GoldReorder:
    """The reuse side of a record with its halves swapped, labels remapped.

    Example:
        ```python
        rec = Record(id="p1", level="gold", fold=4, source_work="", source_tokens=["a", "b", "c"],
                     reuse_work="", reuse_tokens=["a", "x,", "c"], pair_label="cit",
                     links=[Edge(0, 0, "COPY"), Edge(2, 2, "COPY")])
        out = GoldReorder.swap_halves(rec)         # reuse ["c", "a", "x,"], links c<-2 at 0, a<-0 at 1
        GoldReorder.augment(records, rate=0.5, seed=1)   # records + a reordered copy of every second one
        ```
    """

    #: Only records with at least this many reuse words are reordered.
    MIN_WORDS = 6

    @classmethod
    def cut_point(cls, tokens: Sequence[str]) -> int:
        """The index the second half starts at: the clause end nearest the middle, else the middle."""
        n = len(tokens)
        middle = n // 2
        best: Optional[int] = None
        for i, tok in enumerate(tokens[:-1]):
            if tok.endswith(CLAUSE_END):
                cut = i + 1
                if 2 <= cut <= n - 2 and (best is None or abs(cut - middle) < abs(best - middle)):
                    best = cut
        return best if best is not None else middle

    @staticmethod
    def permutation(n: int, cut: int) -> List[int]:
        """``new_index[old_index]`` for the swap of ``tokens[:cut]`` and ``tokens[cut:]``."""
        tail = n - cut
        return [r + tail if r < cut else r - cut for r in range(n)]

    @classmethod
    def swap_halves(cls, record: Record, cut: Optional[int] = None) -> Record:
        """The record with the reuse halves swapped; links, spans and per-token annotation follow."""
        n = record.n_reuse
        cut = cls.cut_point(record.reuse_tokens) if cut is None else cut
        perm = cls.permutation(n, cut)
        tokens = list(record.reuse_tokens[cut:]) + list(record.reuse_tokens[:cut])
        links = sorted((replace(e, r=perm[e.r]) for e in record.links if 0 <= e.r < n), key=lambda e: (e.r, e.s))
        spans: List[Span] = []
        for sp in record.spans:
            if sp.start < cut <= sp.end - 1:          # a span across the cut is split in two
                spans.append(replace(sp, start=perm[sp.start], end=perm[cut - 1] + 1))
                spans.append(replace(sp, start=perm[cut], end=perm[sp.end - 1] + 1))
            elif sp.end > sp.start:
                spans.append(replace(sp, start=perm[sp.start], end=perm[sp.end - 1] + 1))
        spans.sort(key=lambda sp: sp.start)
        annotation: Dict[str, Any] = {}
        for key, value in record.annotation.items():
            if isinstance(value, list) and len(value) == n:
                moved = [None] * n
                for old, new in enumerate(perm):
                    moved[new] = value[old]
                annotation[key] = moved
            else:
                annotation[key] = value
        provenance = dict(record.provenance)
        provenance["reorder"] = {"cut": cut, "of": record.id}
        return replace(record, id=f"{record.id}#reorder", reuse_tokens=tokens, links=links, spans=spans,
                       annotation=annotation, provenance=provenance)

    @classmethod
    def augment(cls, records: Sequence[Record], *, rate: float, seed: int = 1) -> List[Record]:
        """The records plus a reordered copy of a ``rate`` share of them (those long enough)."""
        if rate <= 0:
            return list(records)
        rng = random.Random(4242 + seed)
        out = list(records)
        for record in records:
            if record.n_reuse >= cls.MIN_WORDS and record.links and rng.random() < rate:
                out.append(cls.swap_halves(record))
        return out
