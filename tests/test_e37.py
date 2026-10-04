"""Sinkhorn with dustbins: marginals, torch/numpy agreement, gradients, and the
balanced decoding splitting a contested source word."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.aligners.sinkhorn import SinkhornBalancer  # noqa: E402

torch = pytest.importorskip("torch")


def test_marginals_and_agreement():
    rng = np.random.RandomState(0)
    T, S = 5, 4
    M = rng.randn(T, S); bins = rng.randn(T); cb = 0.3
    P = np.exp(SinkhornBalancer.log_sinkhorn_np(M, bins, cb, iters=100))
    Pv = np.exp(SinkhornBalancer.log_sinkhorn_np(M, bins, np.full(S, cb), iters=100))
    assert np.allclose(P, Pv)
    assert np.allclose(P[:T].sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(P[:, :S].sum(axis=0), 1.0, atol=1e-5)
    assert abs(P[T].sum() - S) < 1e-4 and abs(P[:, S].sum() - T) < 1e-4
    Pt = SinkhornBalancer.log_sinkhorn_torch(torch.tensor(M), torch.tensor(bins), torch.tensor(cb), iters=100).exp().numpy()
    assert np.allclose(P, Pt, atol=1e-6)


def test_gradient_flows_and_minus_inf_is_allowed():
    M = torch.randn(3, 3, requires_grad=True)
    M2 = M.clone(); M2[0, 1] = float("-inf")
    bins = torch.zeros(3); cb = torch.tensor(0.0, requires_grad=True)
    logP = SinkhornBalancer.log_sinkhorn_torch(M2, bins, cb, iters=10)
    loss = -logP[0, 0] - logP[1, 1] - logP[2, 3]
    loss.backward()
    assert torch.isfinite(M.grad).all() and cb.grad is not None and torch.isfinite(logP[0, 1]).item() is False


def test_balanced_decoding_splits_a_contested_source():
    # two reuse words both prefer source 0 (0.6); word 0 is more willing to be null (0.3 vs 0.05)
    scores = [[(0, 0.6), (-1, 0.3), (1, 0.1)], [(0, 0.6), (1, 0.35), (-1, 0.05)], []]
    out = SinkhornBalancer.balanced_scores(scores, n_source=2, col_bins=0.0, iters=50)
    d0, d1 = dict(out[0]), dict(out[1])
    assert out[2] == []
    assert d0[0] < 0.6 and d1[0] < 0.6                     # the contested source is split
    assert d0[-1] > d1[-1]                                  # the null-willing word goes null
    assert abs(d0[0] + d0[1] + d0[-1] - 1) < 1e-6 and abs(d1[0] + d1[1] + d1[-1] - 1) < 1e-6
    assert out[1][0][0] in (0, 1) and out[0][0][0] == -1    # sorted best first


def test_per_source_dustbin_removes_a_deleted_source_from_the_rows():
    scores = [[(0, 0.7), (-1, 0.2), (1, 0.1)], [(2, 0.7), (-1, 0.2), (1, 0.1)], []]
    shared = SinkhornBalancer.balanced_scores(scores, n_source=3, col_bins=0.0, iters=100)
    per_src = SinkhornBalancer.balanced_scores(scores, n_source=3, col_bins=np.array([0.0, 8.0, 0.0]), iters=100)
    assert dict(per_src[0])[1] < dict(shared[0])[1] and dict(per_src[0])[1] < 1e-3   # source 1 is deleted
    assert dict(per_src[0])[0] > dict(shared[0])[0]                                  # its mass goes to the true link
    sharp = SinkhornBalancer.balanced_scores(scores, n_source=3, col_bins=0.0, iters=100, temperature=0.3)
    assert dict(sharp[0])[-1] < dict(shared[0])[-1]                                  # a lower temperature spreads less
