# retexo/aligners/sinkhorn.py
"""
E37: the one-to-one constraint inside the loss (train-through-the-stack).

Today the pointer's location softmax is per reuse word; the one-to-one constraint
enters only at inference, in the Hungarian assignment. Training and inference
geometry differ (the E11 / E23 lesson). Sinkhorn normalisation with dustbins
(SuperGlue-style) makes a doubly-stochastic assignment differentiable: rows are
reuse words, columns source words, plus one dustbin column (the reuse word is
inserted or frame; its entry is the row's own null logit) and one dustbin row
(the source word is deleted; a learned scalar). The loss is the negative log of
the balanced assignment at the gold cell.

At inference the same balancing can replace the raw softmax before the
Hungarian (:meth:`SinkhornBalancer.balanced_scores`), which is testable on a
saved base in minutes.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class SinkhornBalancer:
    """Sinkhorn normalisation with dustbins, for training through the
    one-to-one constraint and for balancing scores at decode time.

    Example:
        ```python
        balanced = SinkhornBalancer.balanced_scores(scores, n_source=len(source_tokens))
        ```
    """

    @staticmethod
    def log_sinkhorn_torch(scores, row_bins, col_bins, iters: int = 10, corner=None):
        """Log-domain Sinkhorn with dustbins. ``scores`` [T, S] (may hold -inf),
        ``row_bins`` [T] the per-row dustbin entry (the reuse word is inserted / frame),
        ``col_bins`` a scalar tensor or [S] (the source word is deleted), ``corner`` a
        scalar (default 0). Returns log P [T + 1, S + 1]: rows 0..T-1 and columns 0..S-1
        each sum to one in probability, the dustbin row to S, the dustbin column to T."""
        import torch

        T, S = scores.shape
        dev = scores.device
        col_bins = col_bins.expand(S) if col_bins.dim() == 0 else col_bins
        corner = torch.zeros((), device=dev) if corner is None else corner
        Z = torch.cat(
            [
                torch.cat([scores, row_bins.view(T, 1)], dim=1),
                torch.cat([col_bins.view(1, S), corner.view(1, 1)], dim=1),
            ],
            dim=0,
        )
        norm = -torch.log(torch.tensor(float(T + S), device=dev))
        log_mu = torch.cat(
            [norm.expand(T), (torch.log(torch.tensor(float(S), device=dev)) + norm).view(1)]
        )
        log_nu = torch.cat(
            [norm.expand(S), (torch.log(torch.tensor(float(T), device=dev)) + norm).view(1)]
        )
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
        return Z + u.unsqueeze(1) + v.unsqueeze(0) - norm

    @staticmethod
    def log_sinkhorn_np(
        scores: np.ndarray, row_bins: np.ndarray, col_bins, iters: int = 10, corner: float = 0.0
    ) -> np.ndarray:
        """The same in numpy, for decoding; ``col_bins`` a float or an [S] array."""
        T, S = scores.shape
        Z = np.full((T + 1, S + 1), corner, dtype=np.float64)
        Z[:T, :S] = scores
        Z[:T, S] = row_bins
        Z[T, :S] = col_bins
        norm = -np.log(T + S)
        log_mu = np.concatenate([np.full(T, norm), [np.log(S) + norm]])
        log_nu = np.concatenate([np.full(S, norm), [np.log(T) + norm]])
        u = np.zeros(T + 1)
        v = np.zeros(S + 1)

        def lse(a, axis):
            m = a.max(axis=axis, keepdims=True)
            return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

        for _ in range(iters):
            u = log_mu - lse(Z + v[None, :], 1)
            v = log_nu - lse(Z + u[:, None], 0)
        return Z + u[:, None] + v[None, :] - norm

    @classmethod
    def balanced_scores(
        cls,
        scores: Sequence[Sequence[Tuple[int, float]]],
        n_source: int,
        *,
        col_bins=0.0,
        iters: int = 10,
        temperature: float = 1.0,
    ):
        """Decoding: turn one pair's per-word score lists (the predict_alignment_scores
        format, probabilities) into Sinkhorn-balanced ones. Words with no scores stay
        empty and are left out of the balancing."""
        rows = [t for t, pairs in enumerate(scores) if pairs]
        if not rows or n_source == 0:
            return [list(p) for p in scores]
        T = len(rows)
        M = np.full((T, n_source), -30.0)
        null = np.full(T, -30.0)
        for i, t in enumerate(rows):
            for s, p in scores[t]:
                lp = np.log(max(p, 1e-12)) / temperature
                if s < 0:
                    null[i] = lp
                elif s < n_source:
                    M[i, s] = lp
        logP = cls.log_sinkhorn_np(M, null, col_bins, iters)
        out: List[List[Tuple[int, float]]] = [list(p) for p in scores]
        for i, t in enumerate(rows):
            cands = [s for s, _ in scores[t] if 0 <= s < n_source]
            pairs = [(-1, float(np.exp(logP[i, n_source])))] + [
                (s, float(np.exp(logP[i, s]))) for s in cands
            ]
            pairs.sort(key=lambda x: -x[1])
            out[t] = pairs
        return out
