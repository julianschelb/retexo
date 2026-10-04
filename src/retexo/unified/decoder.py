# retexo/unified/decoder.py
"""The structured decoder: a semi-Markov dynamic programme over the cell grid.

Where the per-word decoder picks, for every reuse word, the best cell or the
null and lets a one-to-one assignment and three rule repairs clean up
afterwards, this one chooses a whole ``Segmentation`` at once: the reuse is
cut into runs, every QUOTE run maps onto a contiguous, in-order source
stretch, and the score of a segmentation is the sum of its cells' log
probabilities plus a contiguity bonus for every cell that continues a run. Contiguity is therefore a property of
the object being scored, not a repair on top of it.

The same programme, with a per-cell cost added wherever a cell disagrees with
the gold, finds the most damaging wrong segmentation -- the loss-augmented
decode of the structured hinge (``margin_loss``), so the grid can be trained
to be right *as runs* and not only cell by cell.

One-to-one on the source side is enforced against the immediately preceding
QUOTE run inside the programme (no overlap with it, a penalty for going
backwards) and against every earlier run by a resolution pass afterwards
(``resolve_overlaps``): an exact constraint over all earlier runs would put
the whole set of consumed intervals into the state, which the pair lengths
here do not justify. The decision this decoder does *not* make is whether a
word is reused at all: that stays in the cell scores and the nulls (E37).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.unified.runs import FRAME, INS, QUOTE, Run, Segmentation

NEG = float("-inf")
#: Stands in for -inf inside the programme so prefix sums stay finite.
NEG_FINITE = -1e9

# =============================================================================
# The grid
# =============================================================================


@dataclass
class CellGrid:
    """One pair's scores in the shape the decoder reads.

    Attributes:
        log_p: ``[n_reuse, n_source]`` log p(s | t); ``-inf`` where ``s`` was not a candidate.
        log_ins: ``[n_reuse]`` log p(INS | t).
        log_frame: ``[n_reuse]`` log p(FRAME | t).

    Any array-like with ``[t, s]`` / ``[t]`` indexing works: lists of lists for
    decoding, torch tensors for the margin loss.
    """

    log_p: object
    log_ins: object
    log_frame: object
    n_reuse: int
    n_source: int

    @classmethod
    def from_word_rows(cls, words: Sequence, n_source: int, *, frame_from_head: bool = False,
                       min_p: float = 1e-6) -> "CellGrid":
        """From ``TypedPointer.predict_cells`` output for one example: per reuse
        word ``(cands, cells, nulls, loc, frame)``, or ``None`` for a word lost
        to truncation.

        ``frame_from_head`` splits the null mass between INS and FRAME by the
        frame head's probability instead of the location row's two nulls: the
        plain pointer never trains that split (its loss allows either null for
        an unlinked word and leaves frames to the head), so without the mode
        gate the head is the only trained source of the decision.
        """
        import torch

        m = len(words)
        log_p = [[NEG] * n_source for _ in range(m)]
        log_ins = [NEG] * m
        log_frame = [NEG] * m
        for t, w in enumerate(words):
            if w is None:
                log_ins[t] = 0.0
                continue
            cands, _, nulls, loc, frame_logits = w
            if loc is None:
                raise ValueError("the structured decoder needs the factorised location logits")
            p = torch.softmax(torch.cat([nulls, loc]), dim=0).tolist()
            p_ins, p_frame = p[0], p[1]
            if frame_from_head and frame_logits is not None:
                p_null = p_ins + p_frame
                head = torch.softmax(frame_logits, dim=0).tolist()
                p_ins, p_frame = p_null * head[0], p_null * head[1]
            log_ins[t] = math.log(max(p_ins, min_p))
            log_frame[t] = math.log(max(p_frame, min_p))
            for s, v in zip(cands, p[2:]):
                if 0 <= s < n_source:
                    log_p[t][s] = math.log(max(v, min_p))
        return cls(log_p, log_ins, log_frame, m, n_source)


# =============================================================================
# The decoder
# =============================================================================


@dataclass(frozen=True)
class DecoderConfig:
    """The decoder's dials.

    Attributes:
        contiguity: Bonus per *continued* cell inside a QUOTE run (a run of
            length L collects it L - 1 times), so one long quotation strictly
            beats the same cells as scattered links. A per-run bonus would
            reward fragmentation; this is the opposite.
        open_cost: Cost per run of any kind; 0 leaves the number of runs to
            the cells and the contiguity bonus.
        crossing: Penalty on a QUOTE run that starts before the previous QUOTE
            run's source end (a REORDER).
        max_run: Longest QUOTE run considered.
        cost: The structured hinge's per-disagreement cost (the Hamming
            augmentation of the loss-augmented decode).
    """

    contiguity: float = 0.5
    open_cost: float = 0.0
    crossing: float = 1.0
    max_run: int = 12
    cost: float = 1.0


class StructuredDecoder:
    """Semi-Markov decoding over a ``CellGrid``, and the structured hinge over the same programme.

    Example:
        ```python
        decoder = StructuredDecoder(DecoderConfig(contiguity=0.5))
        seg = decoder.decode(grid)                    # a Segmentation
        links, frame = seg.links(), seg.frame()
        wrong = decoder.loss_augmented(grid, gold_seg)
        loss = decoder.margin_loss(grid_torch, gold_seg, wrong)
        ```
    """

    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config or DecoderConfig()

    # ---------- scoring ----------

    def run_score(self, grid: CellGrid, run: Run, augment: Optional["Augment"] = None):
        """The score of one run: its cells' log probabilities plus the run bonus,
        plus the augmentation cost where one is given. Works on floats or tensors."""
        c = self.config
        if run.kind == QUOTE:
            total = 0.0
            for k in range(run.length):
                t, s = run.start + k, run.source_start + k
                total = total + grid.log_p[t][s]
                if augment is not None:
                    total = total + augment.cell(t, s)
            return total + c.contiguity * (run.length - 1) - c.open_cost
        source = grid.log_frame if run.kind == FRAME else grid.log_ins
        total = 0.0
        for t in range(run.start, run.end):
            total = total + source[t]
            if augment is not None:
                total = total + augment.null(t, run.kind)
        return total - c.open_cost

    def score(self, grid: CellGrid, segmentation: Segmentation, augment: Optional["Augment"] = None):
        """The score of a whole segmentation, crossing penalties included."""
        total = 0.0
        prev_end = None
        for run in segmentation.runs:
            total = total + self.run_score(grid, run, augment)
            if run.kind == QUOTE:
                if prev_end is not None and run.source_start < prev_end:
                    total = total - self.config.crossing
                prev_end = run.source_end
        return total

    # ---------- the programme ----------

    def decode(self, grid: CellGrid, augment: Optional["Augment"] = None) -> Segmentation:
        """The best segmentation of the reuse under the grid (plus ``augment``, if given).

        Dynamic programme over reuse positions; the state is the previous QUOTE
        run's source end. The set of source words the path has already used is
        carried as a value, not a key: a quote run may not touch any of them
        (one-to-one against *every* earlier run, not only the last one -- with
        the last one alone, a cheap junk link could reset the state and let a
        crossing run land on words an older run holds), and two paths meeting
        at the same end keep only the better one, so the state count stays at
        ``n + 1`` at the price of optimality only in that merge. Run scores come
        from prefix sums, so a run costs O(1) and the whole decode
        O(m * n * max_run) numpy work plus as many relaxations.
        """
        import numpy as np

        m, n = grid.n_reuse, grid.n_source
        if m == 0:
            return Segmentation([], 0)
        c = self.config
        longest = min(c.max_run, m, n)
        cell = np.full((m, n), NEG_FINITE)
        for t in range(m):
            row = grid.log_p[t]
            for s in range(n):
                v = float(row[s])
                if v > NEG_FINITE:
                    cell[t, s] = v + (augment.cell(t, s) if augment is not None else 0.0)
        ins = np.array([max(float(grid.log_ins[t]), NEG_FINITE)
                        + (augment.null(t, INS) if augment is not None else 0.0) for t in range(m)])
        frm = np.array([max(float(grid.log_frame[t]), NEG_FINITE)
                        + (augment.null(t, FRAME) if augment is not None else 0.0) for t in range(m)])
        prefix = {INS: np.concatenate([[0.0], np.cumsum(ins)]), FRAME: np.concatenate([[0.0], np.cumsum(frm)])}
        diagonal = np.zeros((m + 1, n + 1))                  # diagonal[t+1, s+1] = cell[t, s] + diagonal[t, s]
        for t in range(m):
            diagonal[t + 1, 1:] = cell[t, :] + diagonal[t, :-1]

        # best[j][prev_end] = (score, back, used); prev_end -1 before any quote run,
        # used = the source words the path has linked so far, a boolean mask
        best: List[Dict[int, Tuple[float, Optional[Tuple[int, int, Run]], "np.ndarray"]]] = [dict() for _ in range(m + 1)]
        best[0][-1] = (0.0, None, np.zeros(n, dtype=bool))
        for i in range(m):
            if not best[i]:
                continue
            states = [(state, entry) for state, entry in best[i].items() if entry[0] > NEG_FINITE / 2]
            if not states:
                continue
            for state, (score, _, used) in states:
                for kind in (FRAME, INS):
                    pre = prefix[kind]
                    for j in range(i + 1, m + 1):
                        cand = score + pre[j] - pre[i] - c.open_cost
                        self._relax(best[j], state, cand, (i, state, Run(i, j, kind)), used)
            # taken[k, s] = how many of the path's used words lie in source[:s]
            taken = [np.concatenate([[0], np.cumsum(used)]) for _, (_, _, used) in states]
            for length in range(1, min(longest, m - i) + 1):
                a = np.arange(0, n - length + 1)
                base = diagonal[i + length, a + length] - diagonal[i, a] + c.contiguity * (length - 1) - c.open_cost
                table = np.full((len(states), len(a)), NEG_FINITE)
                for k, (state, (score, _, used)) in enumerate(states):
                    cand = score + base
                    if state >= 0:
                        cand = cand - c.crossing * (a < state)
                    overlap = taken[k][a + length] - taken[k][a] > 0
                    table[k] = np.where(overlap, NEG_FINITE, cand)
                winner = table.argmax(axis=0)
                value = table[winner, np.arange(len(a))]
                for idx in np.nonzero(value > NEG_FINITE / 2)[0]:
                    start = int(a[idx])
                    state, (_, _, used) = states[int(winner[idx])]
                    run = Run(i, i + length, QUOTE, source_start=start)
                    now = used.copy()
                    now[start:start + length] = True
                    self._relax(best[i + length], start + length, float(value[idx]), (i, state, run), now)
        if not best[m]:
            return Segmentation([Run(0, m, INS)], m)
        state = max(best[m], key=lambda k: best[m][k][0])
        runs: List[Run] = []
        position = m
        while position > 0:
            _, back, _ = best[position][state]
            i, state, run = back
            runs.append(run)
            position = i
        runs.reverse()
        return self.resolve_overlaps(grid, Segmentation(runs, m))

    @staticmethod
    def _relax(table: dict, state: int, cand: float, back, used) -> None:
        if cand <= NEG_FINITE / 2:
            return
        if state not in table or cand > table[state][0]:
            table[state] = (cand, back, used)

    def resolve_overlaps(self, grid: CellGrid, segmentation: Segmentation) -> Segmentation:
        """A safety net behind ``decode``, which already keeps its runs disjoint:
        where two QUOTE runs overlap in the source, the lower-scoring one loses
        its overlapping words to an INS run."""
        runs = list(segmentation.runs)
        changed = True
        while changed:
            changed = False
            quotes = [(idx, r) for idx, r in enumerate(runs) if r.kind == QUOTE]
            for x in range(len(quotes)):
                for y in range(x + 1, len(quotes)):
                    (ix, rx), (iy, ry) = quotes[x], quotes[y]
                    if rx.source_start < ry.source_end and ry.source_start < rx.source_end:
                        loser_idx, loser = (ix, rx) if self.run_score(grid, rx) < self.run_score(grid, ry) else (iy, ry)
                        runs[loser_idx] = Run(loser.start, loser.end, INS)
                        changed = True
                        break
                if changed:
                    break
        merged = self._merge_adjacent_nulls(runs)
        return Segmentation(merged, segmentation.n_reuse)

    @staticmethod
    def _merge_adjacent_nulls(runs: Sequence[Run]) -> List[Run]:
        out: List[Run] = []
        for run in runs:
            if out and run.kind != QUOTE and out[-1].kind == run.kind and out[-1].end == run.start:
                out[-1] = Run(out[-1].start, run.end, run.kind)
            else:
                out.append(run)
        return out

    # ---------- the structured hinge ----------

    def loss_augmented(self, grid: CellGrid, gold: Segmentation) -> Segmentation:
        """The most damaging wrong segmentation: the decode with ``cost`` added on
        every cell and null that disagrees with the gold."""
        return self.decode(grid, augment=Augment.against(gold, self.config.cost))

    def margin_loss(self, grid: CellGrid, gold: Segmentation, wrong: Segmentation):
        """The margin-rescaled hinge ``max(0, score(wrong) + delta(wrong, gold) - score(gold))``
        with ``delta`` the Hamming cost the augmented decode maximised; exactly 0
        when ``wrong`` is the gold. On a grid of tensors the gradient flows into the
        cells both segmentations touch."""
        import torch

        augment = Augment.against(gold, self.config.cost)
        delta = sum(augment.cell(t, s) for t, s in wrong.cells()) + sum(
            augment.null(t, run.kind) for run in wrong.runs if run.kind != QUOTE for t in range(run.start, run.end))
        gap = self.score(grid, wrong) + delta - self.score(grid, gold)
        if torch.is_tensor(gap):
            return torch.maximum(torch.zeros_like(gap), gap)
        return max(0.0, gap)


# =============================================================================
# Loss augmentation
# =============================================================================


class Augment:
    """A per-cell and per-null cost added to the decode: +1 wherever a choice
    disagrees with the gold, so the augmented decode finds the wrong
    segmentation the hinge should push against.

    Example:
        ```python
        augment = Augment.against(gold_segmentation)
        augment.cell(t, s)         # 1.0 unless the gold links t to s
        augment.null(t, "frame")   # 1.0 unless the gold has t in a frame run
        ```
    """

    def __init__(self, gold_links: Sequence[int], gold_frame: Sequence[int], cost: float = 1.0):
        self.gold_links = list(gold_links)
        self.gold_frame = list(gold_frame)
        self.cost = cost

    @classmethod
    def against(cls, gold: Segmentation, cost: float = 1.0) -> "Augment":
        return cls(gold.links(), gold.frame(), cost)

    def cell(self, t: int, s: int) -> float:
        return 0.0 if self.gold_links[t] == s else self.cost

    def null(self, t: int, kind: str) -> float:
        if self.gold_links[t] >= 0:
            return self.cost
        gold_kind = FRAME if self.gold_frame[t] else INS
        return 0.0 if gold_kind == kind else self.cost
