# retexo/baselines/decoder.py
"""From a score matrix to links, the same way for every score-matrix row.

Most rows end in interface (I): per reuse word a list of ``(source_index, p)``
best first, with ``-1`` as the null. The step from those rows to edges is a
design decision of its own in the alignment literature, so the paper fixes one
default for every such row (definition 2.3): average both directions where the
model has two (Nagata et al. 2020), drop candidates below a null threshold
theta chosen on the dev fold, and assign one-to-one by the Hungarian method
with a private null column per reuse word (Kuhn 1955; the E8 result). The
alternatives of Table 2 live beside it: argmax per row, SimAlign's mutual
argmax, threshold only, awesome-align's intersection, grow-diag-final
(Och and Ney 2003), the raw argmax with the null, and the full system's stack
(identity bonus, lemma re-ranking, null scale, rater bonus).

Every function here is pair-level: one pair's rows in, one pair's links out.
The corpus-level originals, now ``retexo.aligners.assignment`` (ported from the runners ``run_e8.py``, ``run_e9.py`` and ``run_e18.py`` in ``attic/scripts/``),
stay untouched and are not imported; the unit tests pin each rule to its
original on the same rows, which is a stronger tie than an import. The
agreement rules of ``retexo/e29.py`` are already pair-level and are
called directly.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from retexo.aligners.agreement import AgreementDecoder
from retexo.core.normalize import normalize

Row = List[Tuple[int, float]]
Rows = List[Row]

DEFAULT_THETA = 0.45  # the preliminary champion's null threshold, fold 4
BIDI_AVERAGE_THRESHOLD = 0.4  # Nagata et al. 2020, section 2.3


class BaselineDecoder:
    """From a score matrix to links, pair by pair.

    Every method is pair-level: one pair's rows in, one pair's links out.
    The corpus-level originals, ``retexo.aligners.assignment`` (ported
    from the runners ``run_e8.py``, ``run_e9.py`` and ``run_e18.py`` in
    ``attic/scripts/``), stay untouched and are not imported; the unit tests
    pin each rule to its original on the same rows, which is a stronger tie
    than an import.

    Example:
        ```python
        links, extra = BaselineDecoder.decode("default", rows, theta=0.45, rev_rows=rev_rows)
        ```
    """

    DECODERS = (
        "default",
        "argmax",
        "mutual",
        "mutual_threshold",
        "threshold",
        "intersect",
        "gdf",
        "raw",
        "stack",
        "fragment",
        "none",
    )

    # ---------- row arithmetic ----------

    @staticmethod
    def null_probability(row: Row) -> float:
        for s, p in row:
            if s < 0:
                return p
        return 0.0

    @staticmethod
    def _sorted(row: Sequence[Tuple[int, float]]) -> Row:
        return sorted(((int(s), float(p)) for s, p in row), key=lambda x: -x[1])

    @classmethod
    def symmetrise_average(
        cls, rows: Rows, rev_rows: Optional[Rows], n_source: Optional[int] = None
    ) -> Rows:
        """Nagata's bidirectional average: ``p = (p_fwd(s | t) + p_rev(t | s)) / 2``.

        The union of the forward candidates of ``t`` and the reverse
        candidates naming ``t`` is averaged, a missing entry counting as 0;
        the null keeps the forward null probability (the reverse direction
        has no reuse-side null). Rows come back sorted best first.
        """
        if not rev_rows:
            return [cls._sorted(r) for r in rows]
        rev: Dict[int, Dict[int, float]] = {}
        for s, row in enumerate(rev_rows):
            for t, p in row:
                if t >= 0:
                    rev.setdefault(t, {})[s] = p
        out = []
        for t, row in enumerate(rows):
            fwd = {s: p for s, p in row if s >= 0}
            cand = set(fwd) | set(rev.get(t, {}))
            merged = [(s, 0.5 * (fwd.get(s, 0.0) + rev.get(t, {}).get(s, 0.0))) for s in cand]
            merged.append((-1, cls.null_probability(row)))
            out.append(cls._sorted(merged))
        return out

    @classmethod
    def with_threshold(cls, rows: Rows, theta: float) -> Rows:
        """Drop real candidates below ``theta``; a row without a null gains one at ``theta``."""
        out = []
        for row in rows:
            kept = [(s, p) for s, p in row if s >= 0 and p >= theta]
            has_null = any(s < 0 for s, _ in row)
            kept.append((-1, cls.null_probability(row) if has_null else theta))
            out.append(cls._sorted(kept))
        return out

    # ---------- assignment rules (pair-level) ----------

    @classmethod
    def hungarian(cls, rows: Rows) -> List[int]:
        """The assignment of ``run_e8.links_hungarian`` for one pair.

        Cost ``-p`` per real candidate, one private null column per reuse word
        at ``-p_null``, blocked cells at 1e6; the sum of probabilities is
        maximised.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        words = len(rows)
        sources = sorted({s for row in rows for s, _ in row if s >= 0})
        if not words or not sources:
            return [-1] * words
        column = {s: j for j, s in enumerate(sources)}
        blocked = 1e6
        cost = np.full((words, len(sources) + words), blocked)
        for t, row in enumerate(rows):
            for s, p in row:
                if s >= 0:
                    cost[t, column[s]] = -p
            cost[t, len(sources) + t] = -cls.null_probability(row)
        assigned = [-1] * words
        for t, j in zip(*linear_sum_assignment(cost)):
            if j < len(sources) and cost[t, j] < blocked:
                assigned[t] = sources[j]
        return assigned

    @classmethod
    def decode_argmax(cls, rows: Rows) -> List[int]:
        """Each reuse word takes its best real candidate when that beats its null."""
        out = []
        for row in rows:
            best = [(p, s) for s, p in row if s >= 0]
            if best and max(best)[0] > cls.null_probability(row):
                out.append(max(best)[1])
            else:
                out.append(-1)
        return out

    @classmethod
    def decode_raw(cls, rows: Rows) -> List[int]:
        """The head of each row, the null included (``run_e8.links_argmax``)."""
        return [(row[0][0] if row else -1) for row in [cls._sorted(r) for r in rows]]

    @staticmethod
    def decode_threshold(rows: Rows, theta: float) -> List[int]:
        """Best real candidate if ``p >= theta``, else ``-1``; no one-to-one step."""
        out = []
        for row in rows:
            best = [(p, s) for s, p in row if s >= 0]
            out.append(max(best)[1] if best and max(best)[0] >= theta else -1)
        return out

    @classmethod
    def decode_mutual(cls, rows: Rows, rev_rows: Optional[Rows]) -> List[int]:
        """SimAlign's Argmax: keep ``(t, s)`` iff each is the other's best."""
        if not rev_rows:
            return cls.decode_argmax(rows)
        return cls.decode_argmax(AgreementDecoder.mutual_argmax(rows, rev_rows))

    @classmethod
    def decode_mutual_threshold(
        cls, rows: Rows, rev_rows: Optional[Rows], theta: float
    ) -> List[int]:
        """SimAlign's Argmax with a null: a mutual-best link is kept iff its score is at least ``theta``."""
        links = cls.decode_mutual(rows, rev_rows)
        for t, s in enumerate(links):
            if s >= 0 and cls.probability_of(rows[t], s) < theta:
                links[t] = -1
        return links

    @staticmethod
    def probability_of(row: Row, s: int) -> float:
        for cand, p in row:
            if cand == s:
                return p
        return 0.0

    @classmethod
    def decode_intersect(
        cls, rows: Rows, rev_rows: Optional[Rows], c: float = 0.001
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """awesome-align: keep ``(t, s)`` iff ``p(s | t) > c`` and ``p(t | s) > c``.

        Many-to-many; the primary link of ``t`` is its best kept candidate and
        the rest come back as extra ``(t, s)`` edges.
        """
        if not rev_rows:
            links = cls.decode_threshold(rows, c)
            return links, []
        kept = AgreementDecoder.intersect(rows, rev_rows, c)
        links, extra = [], []
        for t, row in enumerate(kept):
            real = [(p, s) for s, p in row if s >= 0]
            if not real:
                links.append(-1)
                continue
            real.sort(reverse=True)
            links.append(real[0][1])
            extra.extend((t, s) for _, s in real[1:])
        return links, extra

    @classmethod
    def decode_gdf(
        cls,
        rows: Rows,
        rev_rows: Optional[Rows],
        n_source: Optional[int] = None,
        *,
        final: str = "and",
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Grow-diag-final(-and) over the two argmax alignments (Och and Ney 2003, section 4).

        Start from the intersection of the forward argmax and the reverse
        argmax; grow with points of the union that neighbour an aligned point
        (8-neighbourhood) while their reuse word or source word is still free;
        then ``final`` adds union points whose reuse word or source word
        (``or``) or both (``and``) are unaligned. Many-to-many; the primary
        link of ``t`` is its highest-probability edge, the rest come back as
        extra edges.
        """
        n_t = len(rows)
        fwd = {(t, s) for t, s in enumerate(cls.decode_argmax(rows)) if s >= 0}
        if not rev_rows:
            links = [-1] * n_t
            for t, s in fwd:
                links[t] = s
            return links, []
        rev_links = cls.decode_argmax(rev_rows)
        rev_set = {(t, s) for s, t in enumerate(rev_links) if 0 <= t < n_t}
        union = fwd | rev_set
        aligned = fwd & rev_set
        t_done = {t for t, _ in aligned}
        s_done = {s for _, s in aligned}
        neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        changed = True
        while changed:
            changed = False
            for t, s in sorted(aligned):
                for dt, ds in neighbours:
                    u, v = t + dt, s + ds
                    if (
                        (u, v) in union
                        and (u, v) not in aligned
                        and (u not in t_done or v not in s_done)
                    ):
                        aligned.add((u, v))
                        t_done.add(u)
                        s_done.add(v)
                        changed = True
        for t, s in sorted(union - aligned):
            free_t, free_s = t not in t_done, s not in s_done
            if (final == "and" and free_t and free_s) or (final == "or" and (free_t or free_s)):
                aligned.add((t, s))
                t_done.add(t)
                s_done.add(s)
        prob = [dict(row) for row in rows]
        links = [-1] * n_t
        extra = []
        for t in range(n_t):
            mine = sorted(((prob[t].get(s, 0.0), s) for u, s in aligned if u == t), reverse=True)
            if mine:
                links[t] = mine[0][1]
                extra.extend((t, s) for _, s in mine[1:])
        return links, extra

    # ---------- the full system's stack ----------

    @classmethod
    def identity_bonus(
        cls, rows: Rows, source_tokens: Sequence[str], reuse_tokens: Sequence[str], w: float
    ) -> Rows:
        """``run_e9.rerank`` for one pair: ``+ w`` in logit space on identical forms."""
        if w == 0.0:
            return [cls._sorted(r) for r in rows]
        source = [normalize(x) for x in source_tokens]
        out = []
        for t, row in enumerate(rows):
            if not row:
                out.append([])
                continue
            target = normalize(reuse_tokens[t]) if t < len(reuse_tokens) else None
            bumped = [
                (
                    s,
                    math.log(max(p, 1e-12))
                    + (w if s >= 0 and s < len(source) and source[s] == target else 0.0),
                )
                for s, p in row
            ]
            top = max(v for _, v in bumped)
            weights = [(s, math.exp(v - top)) for s, v in bumped]
            total = sum(v for _, v in weights) or 1.0
            out.append(cls._sorted((s, v / total) for s, v in weights))
        return out

    @classmethod
    def lemma_rerank(
        cls,
        rows: Rows,
        source_lemmas: Sequence[str],
        reuse_lemmas: Sequence[str],
        similarity: Callable[[str, str], Optional[float]],
        weight: float,
        top_k: int,
    ) -> Rows:
        """``run_e23.similarity_rerank`` for one pair: ``+ weight * sim`` on the top-k real candidates."""
        if weight == 0.0:
            return [cls._sorted(r) for r in rows]
        out = []
        for t, row in enumerate(rows):
            row = cls._sorted(row)
            real = [s for s, _ in row if s >= 0][:top_k]
            bumped = []
            for s, p in row:
                bonus = 0.0
                if s in real and t < len(reuse_lemmas) and s < len(source_lemmas):
                    sim = similarity(source_lemmas[s], reuse_lemmas[t])
                    bonus = weight * (sim or 0.0)
                bumped.append((s, p + bonus))
            total = sum(p for _, p in bumped) or 1.0
            out.append(cls._sorted((s, p / total) for s, p in bumped))
        return out

    @classmethod
    def decode_stack(
        cls,
        rows: Rows,
        record,
        *,
        w: float = 0.0,
        weight: float = 0.0,
        top_k: int = 5,
        k: float = 1.0,
        theta: float = DEFAULT_THETA,
        morph=None,
        vectors=None,
        beta: float = 0.0,
        llm_links: Optional[Sequence[int]] = None,
        rev_rows: Optional[Rows] = None,
    ) -> List[int]:
        """Identity bonus, lemma re-rank, null scale, rater bonus, then the default decoder."""
        rows = cls.identity_bonus(rows, record.source_tokens, record.reuse_tokens, w)
        if weight and vectors is not None:
            lemmas_s = record.annotation.get("lemma_source") or [
                morph.lemma(x) if morph else x for x in record.source_tokens
            ]
            lemmas_t = record.annotation.get("lemma") or [
                morph.lemma(x) if morph else x for x in record.reuse_tokens
            ]
            rows = cls.lemma_rerank(rows, lemmas_s, lemmas_t, vectors.similarity, weight, top_k)
        if k != 1.0:
            rows = AgreementDecoder.null_scale(rows, k)
        if beta and llm_links is not None:
            bumped = []
            for t, row in enumerate(rows):
                cell = llm_links[t] if t < len(llm_links) else -1
                bumped.append(
                    cls._sorted((s, p + (beta if s == cell and s >= 0 else 0.0)) for s, p in row)
                )
            rows = bumped
        return cls.decode_default(rows, theta=theta, rev_rows=rev_rows)

    # ---------- the default and the dispatch table ----------

    @classmethod
    def decode_default(
        cls,
        rows: Rows,
        *,
        theta: float,
        rev_rows: Optional[Rows] = None,
        n_source: Optional[int] = None,
    ) -> List[int]:
        """Bidirectional average where two directions exist, threshold, Hungarian."""
        rows = cls.symmetrise_average(rows, rev_rows, n_source)
        return cls.hungarian(cls.with_threshold(rows, theta))

    # ---------- fragment-then-word (failure-mode dry run, chain 1b, row 1b.2) ----------

    @classmethod
    def fragments(cls, links: Sequence[int], *, gap: int = 3) -> List[List[int]]:
        """The matched fragments of a decoded pair: maximal runs of linked reuse words whose
        source indices move forward by at most ``gap + 1`` (a deletion) or back by one (a two-word
        inversion); a larger backward jump starts a new fragment (a reorder). Each is
        ``[t_first, t_last, s_low, s_high]``."""
        blocks: List[List[int]] = []
        cur: Optional[List[int]] = None
        last_s = -1
        for t, s in enumerate(links):
            if s < 0:
                continue
            if (
                cur is not None and -1 <= s - last_s <= gap + 1
            ):  # a forward gap is a deletion, a backward jump a reorder
                cur[1] = t
                cur[2] = min(cur[2], s)
                cur[3] = max(cur[3], s)
            else:
                if cur is not None:
                    blocks.append(cur)
                cur = [t, t, s, s]
            last_s = s
        if cur is not None:
            blocks.append(cur)
        return blocks

    @classmethod
    def residual_region(
        cls, t: int, blocks: Sequence[Sequence[int]], n_source: int
    ) -> Optional[Tuple[int, int]]:
        """The source words an unlinked reuse word ``t`` may come from: inside a fragment, its
        source span; between two fragments, the source words between their spans (in either
        order); before the first or after the last, the source words before or after it."""
        for a, b, lo, hi in blocks:
            if a < t < b:
                return (lo, hi)
        left = max((blk for blk in blocks if blk[1] < t), key=lambda blk: blk[1], default=None)
        right = min((blk for blk in blocks if blk[0] > t), key=lambda blk: blk[0], default=None)
        if left is not None and right is not None:
            if left[3] < right[2]:
                return (left[3] + 1, right[2] - 1)
            if right[3] < left[2]:
                return (right[3] + 1, left[2] - 1)
            return None
        if left is not None:
            return (left[3] + 1, n_source - 1)
        if right is not None:
            return (0, right[2] - 1)
        return None

    @classmethod
    def split_repair(
        cls,
        links: List[int],
        rows: Rows,
        record,
        theta: float,
        *,
        rev_rows: Optional[Rows] = None,
        n_source: Optional[int] = None,
    ) -> List[int]:
        """SPLIT under a one-to-one decoder (failure-mode row 15, 2026-09-19): three of the eleven sure
        substitution misses of fold 4 were *siqua -> si qua* and *iamdudum -> dudum*, where the model put
        .98 on the right source word and the Hungarian step, which lets no source word be taken twice,
        gave it to the neighbour. An unlinked reuse word keeps its best source word when that word scores
        at least ``theta``, is held by the adjacent reuse word, and the two reuse words together spell it
        (punctuation and case folded)."""
        sym = rows  # the forward view: the reverse row of a split word belongs to its neighbour
        src = [normalize(w) for w in record.source_tokens]
        reu = [normalize(w) for w in record.reuse_tokens]
        holder_of = {s: t for t, s in enumerate(links) if s >= 0}
        out = list(links)
        for t, s in enumerate(links):
            if s >= 0 or t >= len(sym) or t >= len(reu):
                continue
            best = max(((p, s2) for s2, p in sym[t] if s2 >= 0), default=None)
            if best is None or best[0] < theta:
                continue
            s2 = best[1]
            holder = holder_of.get(s2)
            if holder is None or abs(holder - t) != 1 or s2 >= len(src) or holder >= len(reu):
                continue
            pair = reu[min(t, holder)] + reu[max(t, holder)]
            if pair and (pair == src[s2] or src[s2].startswith(pair) or pair.startswith(src[s2])):
                out[t] = s2
        return out

    @classmethod
    def decode_fragment(
        cls,
        rows: Rows,
        *,
        theta: float,
        rev_rows: Optional[Rows] = None,
        n_source: Optional[int] = None,
        frag_scale: float = 0.5,
        gap: int = 3,
    ) -> List[int]:
        """The default decode, then a second pass inside the matched fragments: an unlinked reuse
        word may take an unlinked source word of its fragment's region at the lower threshold
        ``theta * frag_scale`` -- the annotator's slot convention ("everything around it matches")
        as a decoding rule; one-to-one by the Hungarian method among the residual words."""
        sym = cls.symmetrise_average(rows, rev_rows, n_source)
        links = cls.hungarian(cls.with_threshold(sym, theta))
        n_src = (
            n_source
            if n_source is not None
            else max([s for row in sym for s, _ in row if s >= 0] + [-1]) + 1
        )
        blocks = cls.fragments(links, gap=gap)
        if not blocks:
            return links
        used = {s for s in links if s >= 0}
        low = theta * frag_scale
        residual: Dict[int, Row] = {}
        for t, s in enumerate(links):
            if s >= 0:
                continue
            region = cls.residual_region(t, blocks, n_src)
            if region is None or region[0] > region[1]:
                continue
            prob = {s2: p for s2, p in sym[t] if s2 >= 0}
            cands = [
                (s2, prob.get(s2, 0.0)) for s2 in range(region[0], region[1] + 1) if s2 not in used
            ]
            cands = [(s2, p) for s2, p in cands if p >= low]
            if cands:
                residual[t] = cands + [(-1, low)]
        if residual:
            order = sorted(residual)
            for t, s in zip(order, cls.hungarian([residual[t] for t in order])):
                if s >= 0:
                    links[t] = s
        return links

    @classmethod
    def decode(
        cls,
        name: str,
        rows: Rows,
        *,
        theta: float = DEFAULT_THETA,
        rev_rows: Optional[Rows] = None,
        n_source: Optional[int] = None,
        record=None,
        dials: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Dispatch by name; returns ``(links, extra_edges)``."""
        dials = dials or {}
        if name == "default":
            links = cls.decode_default(rows, theta=theta, rev_rows=rev_rows, n_source=n_source)
            if dials.get("split_repair") and record is not None:
                links = cls.split_repair(
                    links, rows, record, theta, rev_rows=rev_rows, n_source=n_source
                )
            return links, []
        if name == "argmax":
            return cls.decode_argmax(cls.symmetrise_average(rows, rev_rows)), []
        if name == "mutual":
            return cls.decode_mutual(rows, rev_rows), []
        if name == "mutual_threshold":
            return cls.decode_mutual_threshold(rows, rev_rows, theta), []
        if name == "threshold":
            return cls.decode_threshold(cls.symmetrise_average(rows, rev_rows), theta), []
        if name == "intersect":
            return cls.decode_intersect(rows, rev_rows, float(dials.get("c", 0.001)))
        if name == "gdf":
            return cls.decode_gdf(rows, rev_rows, n_source, final=str(dials.get("final", "and")))
        if name == "raw":
            return cls.decode_raw(rows), []
        if name == "stack":
            return cls.decode_stack(
                rows,
                record,
                w=float(dials.get("w", 0.0)),
                weight=float(dials.get("weight", 0.0)),
                top_k=int(dials.get("top_k", 5)),
                k=float(dials.get("k", 1.0)),
                theta=theta,
                morph=dials.get("morph"),
                vectors=dials.get("vectors"),
                beta=float(dials.get("beta", 0.0)),
                llm_links=dials.get("llm_links"),
                rev_rows=rev_rows,
            ), []
        if name == "fragment":
            return cls.decode_fragment(
                rows,
                theta=theta,
                rev_rows=rev_rows,
                n_source=n_source,
                frag_scale=float(dials.get("frag_scale", 0.5)),
                gap=int(dials.get("gap", 3)),
            ), []
        if name == "none":
            return cls.decode_raw(rows), []
        raise ValueError(f"unknown decoder {name!r}; expected one of {cls.DECODERS}")

    # ---------- dev-fold tuning of the null threshold ----------

    @classmethod
    def tune_null_threshold(
        cls,
        rows_per_pair: Sequence[Rows],
        gold_links_per_pair: Sequence[Sequence[int]],
        *,
        rev_per_pair: Optional[Sequence[Optional[Rows]]] = None,
        n_source_per_pair: Optional[Sequence[int]] = None,
        grid: Optional[Sequence[float]] = None,
        criterion: str = "token_accuracy",
        decoder: str = "default",
        records=None,
        dials=None,
    ) -> float:
        """The theta that maximises ``criterion`` on the dev pairs; ties go to the larger theta."""
        from retexo.baselines.scorer import link_prf_from_links, token_accuracy_from_links

        grid = list(grid) if grid is not None else [round(0.05 * i, 2) for i in range(1, 20)]
        best_theta, best_value = grid[0], -1.0
        for theta in grid:
            decoded = []
            for i, rows in enumerate(rows_per_pair):
                rev = rev_per_pair[i] if rev_per_pair is not None else None
                n_s = n_source_per_pair[i] if n_source_per_pair is not None else None
                rec = records[i] if records is not None else None
                links, _ = cls.decode(
                    decoder, rows, theta=theta, rev_rows=rev, n_source=n_s, record=rec, dials=dials
                )
                decoded.append(links)
            if criterion == "link_f1":
                value = link_prf_from_links(decoded, gold_links_per_pair)["f1"]
            else:
                value = token_accuracy_from_links(decoded, gold_links_per_pair)
            if value >= best_value:
                best_theta, best_value = theta, value
        return best_theta


#: Backward-compatible module-level aliases; ``dec.decode(...)`` etc. still work.
DECODERS = BaselineDecoder.DECODERS
null_probability = BaselineDecoder.null_probability
symmetrise_average = BaselineDecoder.symmetrise_average
with_threshold = BaselineDecoder.with_threshold
hungarian = BaselineDecoder.hungarian
decode_argmax = BaselineDecoder.decode_argmax
decode_raw = BaselineDecoder.decode_raw
decode_threshold = BaselineDecoder.decode_threshold
decode_mutual = BaselineDecoder.decode_mutual
decode_intersect = BaselineDecoder.decode_intersect
decode_gdf = BaselineDecoder.decode_gdf
identity_bonus = BaselineDecoder.identity_bonus
lemma_rerank = BaselineDecoder.lemma_rerank
decode_stack = BaselineDecoder.decode_stack
decode_default = BaselineDecoder.decode_default
decode = BaselineDecoder.decode
tune_null_threshold = BaselineDecoder.tune_null_threshold
