# retexo/operations/unlinked.py
"""
The words without a link, and the patterns read off a finished script.

The paper's operations of 2026-10-09 split the words that no link touches by
where they stand. The **reused stretch** of a passage runs from its first to its
last linked word. Inside it, a reuse word without a source is an insertion
(``INS``) and a source word that no link claims a deletion (``DEL``); outside
it they are the later author's new text (``NEW``) and source text not taken up
(``OUT``). A reuse word in a citing formula is ``MARK`` wherever it stands.
None of these is annotated or predicted on its own: all follow from the links,
so this module is the single place that derives them.

Three patterns are read off the links in the same way. ``QUOTE`` is a run of at
least ``QUOTE_MIN`` copies whose source words follow one another; ``REORDER``
marks the reuse words whose link lies outside the longest chain of links in
source order (the moved words); ``DISPERSE`` lists the reuse words that end a
clause inside the linked range, so a pair is dispersed when the list is not
empty.

The rules are those of the statistics behind the paper's Table 1 (the
experiment repository's ``pair_view``), which the tests check against the
annotation's totals: INS 2,480 and NEW 28,646 reuse words, MARK 872, DEL 2,824
and OUT 29,700 source words, QUOTE in 324 pairs, 533 REORDER links and
DISPERSE in 71 pairs.

Links are dictionaries with ``r`` (reuse index), ``s`` (source index) and
``op``, in the code's spelling (``NOP``) or the paper's (``COPY``); spans are
dictionaries with ``start`` and ``end`` (half-open), whatever their label.
"""

from __future__ import annotations

import bisect
import re
from typing import Dict, List, Mapping, Sequence

#: The shortest run of in-order copies that counts as a quotation.
QUOTE_MIN = 4

#: A word that ends a clause, as the editions punctuate it.
CLAUSE_END = re.compile(r"[.;:?!]$")

#: The spellings of a copy: the code's and the paper's.
COPY_OPS = frozenset({"NOP", "COPY"})

#: The labels of words without a link.
REUSE_UNLINKED = ("MARK", "INS", "NEW")
SOURCE_UNLINKED = ("DEL", "OUT")


# =============================================================================
# Helpers
# =============================================================================


def lis_positions(values: Sequence[int]) -> set:
    """The positions of one longest strictly increasing subsequence of ``values``."""
    tails: List[int] = []
    tail_pos: List[int] = []
    prev = [-1] * len(values)
    for i, v in enumerate(values):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tail_pos.append(i)
        else:
            tails[k] = v
            tail_pos[k] = i
        prev[i] = tail_pos[k - 1] if k > 0 else -1
    keep = set()
    i = tail_pos[-1] if tail_pos else -1
    while i != -1:
        keep.add(i)
        i = prev[i]
    return keep


def first_links(n_reuse: int, links: Sequence[Mapping]) -> List[Mapping]:
    """Per reuse word its first link (by reuse, then source index), or ``None``."""
    first: List = [None] * n_reuse
    for link in sorted(links, key=lambda x: (int(x["r"]), int(x["s"]))):
        r = int(link["r"])
        if 0 <= r < n_reuse and first[r] is None:
            first[r] = link
    return first


# =============================================================================
# Unlinked words
# =============================================================================


def unlinked(
    n_reuse: int, n_source: int, links: Sequence[Mapping], spans: Sequence[Mapping] = ()
) -> Dict[str, list]:
    """The words without a link: ``{"reuse": [{"i", "op"}], "source": [{"i", "op"}]}``.

    A reuse word is ``MARK`` in a citing formula, ``INS`` strictly between the
    first and the last linked reuse word, ``NEW`` otherwise; a source word that
    no link claims is ``DEL`` strictly between the first and the last claimed
    source word, ``OUT`` otherwise.
    """
    first = first_links(n_reuse, links)
    linked = [t for t in range(n_reuse) if first[t] is not None]
    mark = set()
    for span in spans:
        mark.update(
            t for t in range(int(span["start"]), min(int(span["end"]), n_reuse)) if first[t] is None
        )
    reuse = []
    for t in range(n_reuse):
        if first[t] is not None:
            continue
        if t in mark:
            op = "MARK"
        else:
            op = "INS" if linked and linked[0] < t < linked[-1] else "NEW"
        reuse.append({"i": t, "op": op})
    claimed = {int(link["s"]) for link in links}
    lo, hi = (min(claimed), max(claimed)) if claimed else (0, -1)
    source = [
        {"i": s, "op": "DEL" if lo < s < hi else "OUT"} for s in range(n_source) if s not in claimed
    ]
    return {"reuse": reuse, "source": source}


# =============================================================================
# Patterns
# =============================================================================


def patterns(reuse_tokens: Sequence[str], links: Sequence[Mapping]) -> Dict[str, list]:
    """``{"QUOTE": [{"start", "end"}], "REORDER": [t], "DISPERSE": [t]}`` over reuse word indices."""
    n_reuse = len(reuse_tokens)
    first = first_links(n_reuse, links)
    source = [int(link["s"]) if link is not None else -1 for link in first]
    is_copy = [link is not None and str(link["op"]) in COPY_OPS for link in first]

    quotes, start, run = [], 0, 0
    for t in range(n_reuse + 1):
        copy = t < n_reuse and is_copy[t]
        if copy and (run == 0 or source[t] == source[t - 1] + 1):
            if run == 0:
                start = t
            run += 1
            continue
        if run >= QUOTE_MIN:
            quotes.append({"start": start, "end": t})
        start, run = t, (1 if copy else 0)

    linked = [(t, source[t]) for t in range(n_reuse) if first[t] is not None]
    reorder, disperse = [], []
    if len(linked) >= 2:
        keep = lis_positions([s for _, s in linked])
        reorder = [t for k, (t, _) in enumerate(linked) if k not in keep]
        # clause breaks before the last linked word: a reuse that straddles one is dispersed
        disperse = [
            t for t in range(linked[0][0], linked[-1][0]) if CLAUSE_END.search(str(reuse_tokens[t]))
        ]
    return {"QUOTE": quotes, "REORDER": reorder, "DISPERSE": disperse}
