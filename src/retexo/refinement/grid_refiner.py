# retexo/refinement/grid_refiner.py
"""
E36b: the grid refiner -- a second look that can see the neighbourhood.

The first pass (the composed aligner) is frozen. Its location logits, the two
null logits and the 23 evidence features form a grid [reuse words x source
words]; the previous pass's script is drawn onto that grid as a link map, a
per-row state, a per-column state and per-row confidence scalars. A small 2-D
convolution reads the grid and emits a residual correction to the location
logits and to the two null logits (zero-initialised: pass two starts as pass
one). An isolated false link is a lone dot with empty diagonal neighbours; a
verbatim run is a diagonal -- both visible to a 3x3 kernel, invisible to the
per-word heads.

Training states are the base model's own first pass on pairs it never trained
on (held-out synthetic) and on the gold training pairs, with state dropout
(a share of tokens reset to undecided) and balanced flips (links dropped and
false links added at the same rate), the gold being the target throughout.

Example:
    ```python
    grids = [PairGrid.from_cells(words, n_source, phi) for words, n_source, phi in pairs]
    links, frames, scores = PairGrid.first_pass(grids)

    refiner = GridRefiner(n_evidence=23, hidden=32, layers=2, device="cpu")
    refiner.fit(grids, golds, device="cpu", epochs=3)
    history, settled_at = refiner.iterate(grids, passes=4, until_stable=True, device="cpu")
    ```
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from retexo.refinement.refine import (DECLINED, LINKED, S_CONSUMED, S_FREE, S_UNDECIDED,
                               UNDECIDED, State)

NEG = -30.0            # logit of a cell the first pass did not score (outside its window)
N_ROW_STATE, N_COL_STATE, N_CONF = 4, 3, 3


# =============================================================================
# The grid of one pair
# =============================================================================

@dataclass
class PairGrid:
    """The frozen first pass of one pair, dense. ``loc`` [T, S] logits (NEG where
    unscored), ``mask`` [T, S] bool, ``nulls`` [T, 2] (INS, FRAME), ``frame``
    [T, 2] frame-head logits or None, ``phi`` [T, S, F] evidence, ``scored`` [T]
    bool (False for a reuse word the first pass skipped entirely)."""
    loc: np.ndarray
    mask: np.ndarray
    nulls: np.ndarray
    frame: Optional[np.ndarray]
    phi: np.ndarray
    scored: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.loc.shape

    @classmethod
    def from_cells(cls, words, n_source: int, phi) -> "PairGrid":
        """From one entry of ``TypedPointer.predict_cells`` (per reuse word:
        (cands, [n, K] type logits, [2] nulls, [n] loc logits, [2] frame) or None)."""
        T = len(words)
        loc = np.full((T, n_source), NEG, dtype=np.float32)
        mask = np.zeros((T, n_source), dtype=bool)
        nulls = np.zeros((T, 2), dtype=np.float32)
        frame = np.zeros((T, 2), dtype=np.float32)
        has_frame = False
        scored = np.zeros(T, dtype=bool)
        for t, w in enumerate(words):
            if w is None:
                continue
            cands, _cells, nl, lc, fr = w
            if lc is None:
                raise ValueError("the grid refiner needs the factorized pointer (loc logits per cell)")
            scored[t] = True
            nulls[t] = np.asarray(nl, dtype=np.float32)
            for c, s in enumerate(cands):
                if 0 <= s < n_source:
                    loc[t, s] = float(lc[c]); mask[t, s] = True
            if fr is not None:
                frame[t] = np.asarray(fr, dtype=np.float32); has_frame = True
        phi = np.asarray(phi, dtype=np.float32) if phi is not None else np.zeros((T, n_source, 0), np.float32)
        if phi.shape[0] != T or phi.shape[1] != n_source:
            raise ValueError(f"evidence grid {phi.shape[:2]} does not match the pair ({T}, {n_source})")
        return cls(loc, mask, nulls, frame if has_frame else None, phi, scored)

    # ---------- reading the grid ----------

    def scores(self, loc: Optional[np.ndarray] = None, nulls: Optional[np.ndarray] = None
               ) -> List[List[Tuple[int, float]]]:
        """The ``predict_alignment_scores`` format from this grid's logits, or the
        given (possibly refiner-corrected) ``loc``/``nulls`` in its place: per
        reuse word [(source, p) ...] best first, source -1 for the null."""
        loc = self.loc if loc is None else loc
        nulls = self.nulls if nulls is None else nulls
        out = []
        T, S = loc.shape
        for t in range(T):
            if not self.scored[t]:
                out.append([]); continue
            cols = np.flatnonzero(self.mask[t])
            z = np.concatenate([nulls[t], loc[t, cols]])
            z = z - z.max()
            p = np.exp(z); p /= p.sum()
            pairs = [(-1, float(p[0] + p[1]))] + [(int(s), float(v)) for s, v in zip(cols, p[2:])]
            pairs.sort(key=lambda x: -x[1])
            out.append(pairs)
        return out

    def frames(self, links: Sequence[int]) -> List[int]:
        """The frame head's decision for every unlinked word (the same rule as
        ``TypedPointer.predict_frames``)."""
        T = self.shape[0]
        flags = [0] * T
        for t in range(T):
            if not self.scored[t] or (t < len(links) and links[t] is not None and links[t] >= 0):
                continue
            if self.frame is not None:
                flags[t] = int(self.frame[t, 1] > self.frame[t, 0])
            else:
                flags[t] = int(self.nulls[t, 1] > self.nulls[t, 0])
        return flags

    @staticmethod
    def confidence(scores: List[List[Tuple[int, float]]]) -> np.ndarray:
        """Per reuse word: (p of the best choice, margin to the second, entropy / log n)."""
        T = len(scores)
        out = np.zeros((T, N_CONF), dtype=np.float32)
        for t, pairs in enumerate(scores):
            if not pairs:
                continue
            ps = np.asarray([p for _, p in pairs], dtype=np.float64)
            top = ps[0]; second = ps[1] if len(ps) > 1 else 0.0
            ent = float(-(ps[ps > 0] * np.log(ps[ps > 0])).sum()) / max(math.log(len(ps)), 1e-6) if len(ps) > 1 else 0.0
            out[t] = (top, top - second, ent)
        return out

    @staticmethod
    def first_pass(grids: Sequence["PairGrid"]):
        """Pass one for a list of grids: (links, frames, scores) from the frozen base."""
        from retexo.aligners.assignment import AssignmentPolicy

        scores = [g.scores() for g in grids]
        links = AssignmentPolicy.links_hungarian(scores)
        frames = [g.frames(l) for g, l in zip(grids, links)]
        return links, frames, scores

    # ---------- drawing a state onto the grid ----------

    def channels(self, state: State, conf: np.ndarray) -> np.ndarray:
        """[C, T, S] float32: loc, mask, evidence, link map, row state (4), column
        state (3), row confidence (3), the two null logits broadcast."""
        T, S = self.shape
        F = self.phi.shape[2]
        C = self.n_channels(F)
        x = np.zeros((C, T, S), dtype=np.float32)
        x[0] = np.where(self.mask, self.loc, 0.0)
        x[1] = self.mask
        x[2:2 + F] = np.transpose(self.phi, (2, 0, 1))
        c = 2 + F
        for t, s in enumerate(state.link[:T]):
            if s >= 0 and s < S:
                x[c, t, s] = 1.0
        c += 1
        for t, r in enumerate(state.reuse[:T]):
            x[c + r, t, :] = 1.0
        c += N_ROW_STATE
        for s, r in enumerate(state.source[:S]):
            x[c + r, :, s] = 1.0
        c += N_COL_STATE
        for k in range(N_CONF):
            x[c + k] = conf[:T, k][:, None]
        c += N_CONF
        x[c] = self.nulls[:, 0][:, None]; x[c + 1] = self.nulls[:, 1][:, None]
        return x

    def row_features(self, state: State, conf: np.ndarray) -> np.ndarray:
        T = self.shape[0]
        f = np.zeros((T, N_ROW_STATE + N_CONF + 2), dtype=np.float32)
        for t, r in enumerate(state.reuse[:T]):
            f[t, r] = 1.0
        f[:, N_ROW_STATE:N_ROW_STATE + N_CONF] = conf[:T]
        f[:, -2:] = self.nulls
        return f

    def perturb(self, state: State, rng: random.Random, *, dropout: float = 0.2,
                flip: float = 0.1) -> State:
        """Training-time noise on ``state``: ``dropout`` resets a token to undecided
        (its link removed), ``flip`` drops a link to declined *and*, at the same
        rate, turns a declined word into a false link to a free, scored source word.
        Source states are recomputed from the surviving links."""
        T, S = self.shape
        reuse, link = list(state.reuse), list(state.link)
        used = {s for s in link if s >= 0}
        free = [s for s in range(S) if s not in used]
        for t in range(T):
            r = rng.random()
            if r < dropout:
                reuse[t], link[t] = UNDECIDED, -1
            elif r < dropout + flip:
                if reuse[t] == LINKED:
                    reuse[t], link[t] = DECLINED, -1
                elif reuse[t] == DECLINED and free:
                    cands = [s for s in free if self.mask[t, s]]
                    if cands:
                        s = rng.choice(cands); free.remove(s)
                        reuse[t], link[t] = LINKED, s
        used = {s for s in link if s >= 0}
        undecided_rows = any(r == UNDECIDED for r in reuse)
        source = [S_CONSUMED if s in used else (S_UNDECIDED if undecided_rows else S_FREE) for s in range(S)]
        return State(reuse, link, source)

    @staticmethod
    def n_channels(n_evidence: int) -> int:
        return 2 + n_evidence + 1 + N_ROW_STATE + N_COL_STATE + N_CONF + 2


# =============================================================================
# The refiner model
# =============================================================================


class GridRefiner:
    """A small 2-D convolution that reads a :class:`PairGrid` plus a drawn-on
    :class:`~retexo.refinement.refine.State` and emits a residual
    correction to the location and null logits.

    Zero-initialised: an untrained refiner is the identity, so pass two starts
    exactly as pass one. Holds its layers directly (not a ``torch.nn.Module``
    subclass), so importing this module never requires torch; only
    constructing a refiner does.

    Example:
        ```python
        refiner = GridRefiner(n_evidence=23, hidden=32, layers=2, device="cpu")
        refiner.fit(grids, golds, device="cpu", epochs=3)
        history, settled_at = refiner.iterate(grids, passes=4, until_stable=True, device="cpu")
        ```
    """

    def __init__(self, n_evidence: int, hidden: int = 32, layers: int = 2, device: str = "cpu"):
        self.n_evidence = n_evidence
        self.hidden = hidden
        self.layers = layers
        self.device = device
        self._build()

    def _build(self) -> None:
        import torch

        C = PairGrid.n_channels(self.n_evidence)
        convs, c_in = [], C
        for _ in range(self.layers):
            convs += [torch.nn.Conv2d(c_in, self.hidden, 3, padding=1), torch.nn.ReLU()]
            c_in = self.hidden
        self._body = torch.nn.Sequential(*convs)
        self._loc_out = torch.nn.Conv2d(self.hidden, 1, 1)
        self._null_in = torch.nn.Linear(2 * self.hidden + N_ROW_STATE + N_CONF + 2, self.hidden)
        self._null_out = torch.nn.Linear(self.hidden, 2)
        torch.nn.init.zeros_(self._loc_out.weight); torch.nn.init.zeros_(self._loc_out.bias)
        torch.nn.init.zeros_(self._null_out.weight); torch.nn.init.zeros_(self._null_out.bias)
        for module in (self._body, self._loc_out, self._null_in, self._null_out):
            module.to(self.device)

    def _submodules(self):
        return (self._body, self._loc_out, self._null_in, self._null_out)

    def parameters(self):
        """Every learnable parameter, for one optimizer over the whole refiner."""
        for module in self._submodules():
            yield from module.parameters()

    def _forward(self, x, mask, row_feats):
        """x [C, T, S], mask [T, S] bool, row_feats [T, 4 + 3 + 2] -> (delta_loc [T, S], delta_null [T, 2])."""
        import torch
        h = self._body(x.unsqueeze(0))                       # [1, hidden, T, S]
        d_loc = self._loc_out(h)[0, 0]                       # [T, S]
        m = mask.unsqueeze(0).unsqueeze(0).float()
        h_max = (h * m + (m - 1) * 1e4).amax(dim=3)[0].T    # [T, hidden]
        h_mean = ((h * m).sum(dim=3) / m.sum(dim=3).clamp(min=1))[0].T
        z = torch.relu(self._null_in(torch.cat([h_max, h_mean, row_feats], dim=1)))
        d_null = self._null_out(z)                           # [T, 2]
        return d_loc, d_null

    # ---------- inference ----------

    def correct(self, grid: PairGrid, state: State, conf: np.ndarray, device: str):
        """Corrected (loc, nulls) as numpy, no grad."""
        import torch

        x = torch.from_numpy(grid.channels(state, conf)).to(device)
        mask = torch.from_numpy(grid.mask).to(device)
        rf = torch.from_numpy(grid.row_features(state, conf)).to(device)
        with torch.no_grad():
            d_loc, d_null = self._forward(x, mask, rf)
        loc = grid.loc + d_loc.cpu().numpy() * grid.mask
        nulls = grid.nulls + d_null.cpu().numpy()
        return loc, nulls

    def _second_pass(self, grids: Sequence[PairGrid], links_prev, frames_prev, scores_prev,
                      device: str, active=None):
        """One refined pass for the ``active`` grids (all by default); the others are
        carried forward unchanged."""
        from retexo.aligners.assignment import AssignmentPolicy

        idx = list(range(len(grids))) if active is None else list(active)
        scores_new = []
        for i in idx:
            g = grids[i]
            st = State.from_script(links_prev[i], frames_prev[i], g.shape[1])
            loc, nulls = self.correct(g, st, PairGrid.confidence(scores_prev[i]), device)
            scores_new.append(g.scores(loc, nulls))
        links_new = AssignmentPolicy.links_hungarian(scores_new) if scores_new else []
        links = list(links_prev); frames = list(frames_prev); scores = list(scores_prev)
        for i, l, sc in zip(idx, links_new, scores_new):
            links[i] = l; scores[i] = sc; frames[i] = grids[i].frames(l)
        return links, frames, scores

    def iterate(self, grids: Sequence[PairGrid], *, passes: int, until_stable: bool, device: str):
        """Pass one from the frozen base, then refined passes. Returns the history
        [(links, frames)] (every pass complete for every pair) and ``settled_at``
        per pair (0 = still changing at the cap)."""
        links, frames, scores = PairGrid.first_pass(grids)
        history = [(links, frames)]
        active = list(range(len(grids)))
        settled = [0] * len(grids)
        for k in range(2, passes + 1):
            if not active:
                break
            new_links, new_frames, new_scores = self._second_pass(grids, links, frames, scores, device, active)
            if until_stable:
                still = []
                for i in active:
                    same = list(new_links[i]) == list(links[i]) and list(new_frames[i]) == list(frames[i])
                    if same:
                        settled[i] = k
                    else:
                        still.append(i)
                active = still
            links, frames, scores = new_links, new_frames, new_scores
            history.append((links, frames))
        return history, settled

    # ---------- training ----------

    def pair_loss(self, grid: PairGrid, state: State, conf: np.ndarray, gold_links, gold_frames,
                  device: str, null_weight: float = 0.2):
        """Cross-entropy of the corrected location softmax against the gold, per
        scored reuse word; null-target words weighted by ``null_weight``."""
        import torch

        T, S = grid.shape
        x = torch.from_numpy(grid.channels(state, conf)).to(device)
        mask = torch.from_numpy(grid.mask).to(device)
        rf = torch.from_numpy(grid.row_features(state, conf)).to(device)
        d_loc, d_null = self._forward(x, mask, rf)
        loc = torch.from_numpy(grid.loc).to(device) + d_loc
        loc = loc.masked_fill(~mask, float("-inf"))
        nulls = torch.from_numpy(grid.nulls).to(device) + d_null
        logits = torch.cat([nulls, loc], dim=1)                     # [T, 2 + S]
        targets, weights, rows = [], [], []
        for t in range(T):
            if not grid.scored[t]:
                continue
            s = gold_links[t] if t < len(gold_links) else -1
            if s is not None and s >= 0 and s < S and grid.mask[t, s]:
                targets.append(2 + s); weights.append(1.0)
            else:
                fr = bool(gold_frames[t]) if gold_frames is not None and t < len(gold_frames) else False
                targets.append(1 if fr else 0); weights.append(null_weight)
            rows.append(t)
        if not rows:
            return None
        lp = torch.log_softmax(logits[rows], dim=1)
        tgt = torch.tensor(targets, device=device)
        w = torch.tensor(weights, device=device)
        nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return (nll * w).sum() / w.sum()

    def fit(self, grids: Sequence[PairGrid], golds: Sequence[Tuple[Sequence[int], Optional[Sequence[int]]]],
            *, device: str, epochs: int = 3, lr: float = 1e-3, seed: int = 1,
            dropout: float = 0.2, flip: float = 0.1, null_weight: float = 0.2,
            rollin_rounds: int = 0, log=None) -> "GridRefiner":
        """States are the base's first pass on each grid (perturbed); with
        ``rollin_rounds`` > 0, after each round the refiner's own pass-two scripts
        replace the states for the next round (DAgger-style aggregation: both kept).
        Returns ``self``."""
        import torch

        rng = random.Random(seed)
        links1, frames1, scores1 = PairGrid.first_pass(grids)
        pool = [(i, State.from_script(links1[i], frames1[i], grids[i].shape[1]), PairGrid.confidence(scores1[i]))
                for i in range(len(grids))]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for round_ in range(1 + rollin_rounds):
            for ep in range(epochs):
                order = list(range(len(pool))); rng.shuffle(order)
                total, n = 0.0, 0
                for module in self._submodules():
                    module.train()
                for j in order:
                    i, st, cf = pool[j]
                    st_p = grids[i].perturb(st, rng, dropout=dropout, flip=flip)
                    loss = self.pair_loss(grids[i], st_p, cf, golds[i][0], golds[i][1], device, null_weight)
                    if loss is None:
                        continue
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += float(loss.detach()); n += 1
                if log:
                    log(f"  refiner round {round_ + 1} epoch {ep + 1}/{epochs}: loss {total / max(n, 1):.4f} over {n} pairs")
            if round_ < rollin_rounds:
                for module in self._submodules():
                    module.eval()
                links2, frames2, scores2 = self._second_pass(grids, links1, frames1, scores1, device)
                pool += [(i, State.from_script(links2[i], frames2[i], grids[i].shape[1]), PairGrid.confidence(scores2[i]))
                         for i in range(len(grids))]
        for module in self._submodules():
            module.eval()
        return self
