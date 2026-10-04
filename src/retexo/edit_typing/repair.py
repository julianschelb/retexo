# retexo/edit_typing/repair.py
"""Post-decoding repairs that follow from the fold-4 error analysis (2026-09-09).

Three rules, each on one decision the decoder gets wrong for a structural reason:

* ``spelling`` -- NOP vs MORPH: the hand labels call an enclitic (-que, -ue, -ne)
  added or dropped, and orthographic variants (temptare/tentare, quicquid/quidquid,
  haud/haut, urgetur/urguetur, quoties/quotiens, Grai/Graii, hostis/hostes) NOP. The
  typer calls them MORPH/SPLIT/MERGE. Same spelling key -> NOP.
* ``frame_adjacency`` -- a citing formula introduces *this* quotation: in 28 of 29
  gold frames the next linked word follows within one token. Predicted frame spans
  farther than ``max_gap`` from the next link are dropped (they are the formulas of
  other quotations in the same passage), and at most one span is kept.
* ``subst_geometry`` -- a substitution sits where the earlier word stood: gold
  SUBST links lie between their neighbouring links' sources 62 of 78 times; false
  SUBST-type links lie outside that gap 32 of 69 times. A SUBST-type link whose source
  falls outside the gap of its neighbours is declined.

Every rule below is a staticmethod on :class:`Repairer`; :meth:`Repairer.apply`
runs all three in the order they were validated.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

ENCLITICS = ("que", "ue", "ne")
SUBST_LIKE = {"SUBST", "SYN", "SYN-DIST", "HYPER", "HYPO", "ANT", "NE-SUB", "POS"}


class Repairer:
    """Post-decoding repair rules from the fold-4 error analysis.

    Example:
        ```python
        links, tags, frame = Repairer.apply(source, target, links, tags, frame)
        ```
    """

    ENCLITICS = ENCLITICS
    SUBST_LIKE = SUBST_LIKE
    STRONG = (".", ";", ":", "?", "!")

    @staticmethod
    def _key(w: str) -> str:
        return re.sub(r"[^0-9a-z]", "", w.lower().replace("v", "u").replace("j", "i"))

    @classmethod
    def spelling_key(cls, w: str) -> str:
        """Orthographic variants of one form fold onto one key; inflection does not."""
        k = cls._key(w)
        for e in cls.ENCLITICS:                   # enclitic added or dropped
            if k.endswith(e) and len(k) - len(e) >= 3:
                k = k[: -len(e)]; break
        k = k.replace("ae", "e").replace("oe", "e").replace("y", "i")
        k = k.replace("mpt", "nt").replace("cq", "q").replace("dq", "q").replace("gue", "ge").replace("ph", "f")
        k = re.sub(r"h", "", k)                    # reprendas/reprehendas, haut/haud (below)
        k = re.sub(r"(.)\1", r"\1", k)             # double consonants, Graii/Grai
        k = re.sub(r"d$", "t", k)                  # haud/haut, aput/apud
        k = re.sub(r"ns$", "s", k); k = re.sub(r"nst", "st", k); k = re.sub(r"ens$", "es", k)   # quotiens/quoties
        k = re.sub(r"n$", "m", k)                  # Xenocraten/Xenocratem
        return k

    @classmethod
    def spelling(cls, source: Sequence[str], target: Sequence[str], links: Sequence[int], tags: List[str]) -> List[str]:
        out = list(tags)
        for t, s in enumerate(links):
            if s is not None and s >= 0 and out[t] in ("MORPH", "SPLIT", "MERGE") \
                    and cls.spelling_key(source[s]) == cls.spelling_key(target[t]):
                out[t] = "NOP"
        return out

    @staticmethod
    def _spans(mask: Sequence[int]) -> List[Tuple[int, int]]:
        out, start = [], None
        for i, x in enumerate(list(mask) + [0]):
            if x and start is None:
                start = i
            elif not x and start is not None:
                out.append((start, i - 1)); start = None
        return out

    @classmethod
    def frame_adjacency(cls, links: Sequence[int], frame: Sequence[int], *, max_gap: int = 2,
                        one_span: bool = True) -> List[int]:
        out = [0] * len(frame)
        kept = []
        for a, b in cls._spans(frame):
            nxt = next((u for u in range(b + 1, len(links)) if links[u] is not None and links[u] >= 0), None)
            if nxt is not None and nxt - b <= max_gap:
                kept.append((a, b, nxt))
        if one_span and kept:
            kept = [min(kept, key=lambda x: x[2])]          # the formula before the first linked run
        for a, b, _ in kept:
            for t in range(a, b + 1):
                out[t] = 1
        return out

    @classmethod
    def _sentence_start(cls, tokens: Sequence[str], t: int, links: Sequence[int]) -> int:
        """Walk left from t to the token after the previous strong punctuation, never
        crossing a linked token."""
        u = t
        while u > 0 and not tokens[u - 1].endswith(cls.STRONG) and not (links[u - 1] is not None and links[u - 1] >= 0):
            u -= 1
        return u

    @classmethod
    def frame_extend(cls, tokens: Sequence[str], links: Sequence[int], frame: Sequence[int], *,
                     colon_rule: bool = True, extend_left: bool = True, max_len: int = 40) -> List[int]:
        """Citing formulas in the hand labels run from the start of the introducing clause
        to the colon before the quotation. Extend a predicted span leftwards to the
        previous sentence boundary; when no span was predicted but the first link is
        preceded by a token ending in ':', frame the clause before that colon."""
        out = list(frame)
        first = next((u for u in range(len(links)) if links[u] is not None and links[u] >= 0), None)
        spans = cls._spans(out)
        if extend_left:
            for a, b in spans:
                start = cls._sentence_start(tokens, a, links)
                if b - start + 1 <= max_len:
                    for t in range(start, a):
                        out[t] = 1
        if colon_rule and not spans and first is not None and first >= 2:
            end = first - 1
            if tokens[end].endswith(":"):
                start = cls._sentence_start(tokens, end, links)
                if end - start + 1 <= max_len:
                    for t in range(start, end + 1):
                        out[t] = 1
        return out

    @classmethod
    def subst_geometry(cls, links: Sequence[int], tags: Sequence[str], *, allow_size_mismatch: bool = True):
        """Decline SUBST-type links whose source is not between the sources of the
        neighbouring links (the slot rule)."""
        L = list(links); T = list(tags)
        linked = [t for t, s in enumerate(L) if s is not None and s >= 0]
        for t in linked:
            if T[t] not in cls.SUBST_LIKE:
                continue
            s = L[t]
            lt = next((u for u in range(t - 1, -1, -1) if L[u] is not None and L[u] >= 0 and T[u] not in cls.SUBST_LIKE), None)
            rt = next((u for u in range(t + 1, len(L)) if L[u] is not None and L[u] >= 0 and T[u] not in cls.SUBST_LIKE), None)
            if lt is None and rt is None:
                continue
            ok = True
            if lt is not None:
                ok &= s > L[lt] and (allow_size_mismatch or s - L[lt] == t - lt)
            if rt is not None:
                ok &= s < L[rt] and (allow_size_mismatch or L[rt] - s == rt - t)
            if not ok:
                L[t] = -1; T[t] = "INS"
        return L, T

    @classmethod
    def apply(cls, source, target, links, tags, frame, *, frame_p=None, frame_threshold=0.2):
        """All three repairs in the order they were validated (fold 4, 2026-09-09):
        spelling -> NOP, substitution geometry, frame adjacency (on the frame head's
        probabilities at a lower threshold when they are given)."""
        L = list(links); T = list(tags); F = list(frame)
        T = cls.spelling(source, target, L, T)
        L, T = cls.subst_geometry(L, T)
        if frame_p is not None:
            F = [1 if (L[t] < 0 and frame_p[t] is not None and frame_p[t] > frame_threshold) else 0 for t in range(len(L))]
        F = cls.frame_adjacency(L, F)
        return L, T, F
