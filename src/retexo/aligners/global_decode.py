# retexo/aligners/global_decode.py
"""E44: decode the whole script at once instead of one word at a time.

The per-word decision takes the null whenever no single cell beats it, which loses
every substitution whose only evidence is that it *continues a reused stretch*
(fold-4 analysis: the gold source of a declined substitution sits in the model's
top-4 at p ~= 0.03, never at rank 1). This decoder scores a whole assignment,

    sum_t  log p_t(s_t)  +  contiguity bonus  -  crossing penalty  -  stretch opening cost

by dynamic programming over target positions with the last consumed source index as
the state, so a low-probability cell is taken when it extends a stretch and refused
when it stands alone. ``null_scale`` moves the operating point exactly as before, but
now the trade is made per stretch rather than per word.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

NEG = -1e9


class GlobalDecoder:
    """Whole-script dynamic-programming decoder: scores a full assignment
    against a contiguity bonus and a crossing penalty, instead of deciding
    each word alone.

    Example:
        ```python
        links = GlobalDecoder.decode(top, n_source=len(source_tokens), contiguity=1.0)
        ```
    """

    @staticmethod
    def decode(top: Sequence[Sequence[Tuple[int, float]]], n_source: int, *,
              contiguity: float = 1.0, monotone: float = 0.0, crossing: float = 1.0,
              open_cost: float = 0.0, null_scale: float = 1.0, min_p: float = 1e-4) -> List[int]:
        """``top``: per target word, [(source, p), ...] with source -1 for the null.
        Returns the chosen source per target word (-1 = none)."""
        n = len(top)
        if n == 0:
            return []
        lp = []                                  # per word: {source: log p}, and the null's log p
        nulls = []
        for w in top:
            d = {}
            p_null = min_p
            for s, p in w:
                if s < 0:
                    p_null = max(p, min_p)
                elif s < n_source:
                    d[s] = math.log(max(p, min_p))
            nulls.append(math.log(max(p_null, min_p)) + math.log(max(null_scale, 1e-6)))
            lp.append(d)
        # state: -1 = nothing consumed yet / after a null run, else the last consumed source
        states = {-1: (0.0, None)}               # state -> (score, backpointer (t, prev_state, chosen))
        trace: List[dict] = []
        for t in range(n):
            nxt = {}
            for st, (score, _) in states.items():
                # null for this word: the state (last source) is kept, so a stretch can resume
                cand = score + nulls[t]
                if st not in nxt or cand > nxt[st][0]:
                    nxt[st] = (cand, (st, -1))
                for s, l in lp[t].items():
                    if s == st:                                   # a source is consumed once
                        continue
                    bonus = 0.0
                    if st >= 0:
                        if s == st + 1:
                            bonus += contiguity
                        elif s > st:
                            bonus += monotone
                        else:
                            bonus -= crossing
                    else:
                        bonus -= open_cost
                    cand = score + l + bonus
                    if s not in nxt or cand > nxt[s][0]:
                        nxt[s] = (cand, (st, s))
            states = nxt
            trace.append({k: v[1] for k, v in states.items()})
        # walk back from the best final state
        st = max(states, key=lambda k: states[k][0])
        links = [-1] * n
        for t in range(n - 1, -1, -1):
            prev, chosen = trace[t][st]
            links[t] = chosen
            st = prev
        return links
