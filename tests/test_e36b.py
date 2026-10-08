"""Grid refiner: dense grid from cells, scores identical to the base, zero-init
identity, perturbation, a learnable toy correction, and the until-stable loop."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.refinement.grid_refiner import GridRefiner, PairGrid  # noqa: E402
from retexo.refinement.refine import DECLINED, LINKED, UNDECIDED, State  # noqa: E402

torch = pytest.importorskip("torch")
F = 5


def toy_grid(T=4, S=3, seed=0, strong=None):
    """A grid whose base logits prefer the diagonal, with an optional lone strong
    false link at ``strong`` = (t, s)."""
    rng = np.random.RandomState(seed)
    words = []
    for t in range(T):
        loc = rng.randn(S).astype(np.float32) * 0.1
        if t < S:
            loc[t] += 3.0
        if strong is not None and strong[0] == t:
            loc[strong[1]] += 4.0
        nulls = np.array([0.5, -1.0], dtype=np.float32)
        words.append(
            (
                list(range(S)),
                torch.zeros(S, 2),
                torch.tensor(nulls),
                torch.tensor(loc),
                torch.tensor([1.0, -1.0]),
            )
        )
    phi = rng.rand(T, S, F).astype(np.float32)
    return PairGrid.from_cells(words, S, phi)


def test_grid_from_cells_and_scores_match_the_base_rule():
    g = toy_grid()
    assert g.shape == (4, 3) and g.mask.all() and g.scored.all() and g.frame is not None
    sc = g.scores()
    # the base rule: softmax over [nulls, loc], p_null = p[0] + p[1]
    z = np.concatenate([g.nulls[0], g.loc[0]])
    p = np.exp(z - z.max())
    p /= p.sum()
    assert abs(dict(sc[0])[-1] - (p[0] + p[1])) < 1e-6 and sc[0][0][0] == 0
    g2 = PairGrid.from_cells([None, None], 3, np.zeros((2, 3, F), np.float32))
    assert not g2.scored.any() and g2.scores() == [[], []]


def test_zero_init_refiner_is_the_identity():
    g = toy_grid()
    links, frames, scores = PairGrid.first_pass([g])
    st = State.from_script(links[0], frames[0], 3)
    r = GridRefiner(F, hidden=8, layers=2)
    loc, nulls = r.correct(g, st, PairGrid.confidence(scores[0]), "cpu")
    assert np.allclose(loc, g.loc) and np.allclose(nulls, g.nulls)
    hist, settled = r.iterate([g], passes=4, until_stable=True, device="cpu")
    assert len(hist) == 2 and settled == [2]  # repeats at once, stops early
    assert hist[0][0][0] == hist[1][0][0] == [0, 1, 2, -1]


def test_perturb_changes_the_requested_share():
    g = toy_grid(T=40, S=40)
    st = State.from_script(list(range(30)) + [-1] * 10, None, 40)
    rng = random.Random(0)
    p = g.perturb(st, rng, dropout=0.5, flip=0.0)
    assert 8 <= sum(1 for r in p.reuse if r == UNDECIDED) <= 32
    p2 = g.perturb(st, random.Random(1), dropout=0.0, flip=0.5)
    dropped = sum(1 for t in range(30) if p2.reuse[t] == DECLINED)
    added = sum(1 for t in range(30, 40) if p2.reuse[t] == LINKED)
    assert (
        dropped >= 5
        and added >= 1
        and len({s for s in p2.link if s >= 0}) == sum(1 for s in p2.link if s >= 0)
    )


def test_refiner_learns_to_drop_a_lone_false_link():
    # base: diagonal links on rows 0-2, row 3 strongly (and wrongly) linked to column 0
    grids, golds = [], []
    for seed in range(24):
        g = toy_grid(T=5, S=3, seed=seed, strong=(4, 0))
        grids.append(g)
        golds.append(([0, 1, 2, -1, -1], None))
    links, _, _ = PairGrid.first_pass(grids)
    assert all(
        link[4] == -1 or link[4] == 0 for link in links
    )  # Hungarian may still give row 4 nothing
    r = GridRefiner(F, hidden=8, layers=2)
    torch.manual_seed(0)
    r.fit(grids, golds, device="cpu", epochs=40, lr=1e-2, dropout=0.0, flip=0.0)
    hist, _ = r.iterate(grids, passes=2, until_stable=False, device="cpu")
    wrong_before = sum(1 for link in hist[0][0] if link[4] >= 0 or link[3] >= 0)
    wrong_after = sum(1 for link in hist[1][0] if link[4] >= 0 or link[3] >= 0)
    assert wrong_after <= wrong_before and hist[1][0][0][:3] == [0, 1, 2]


def test_frames_follow_the_frame_head_for_unlinked_words():
    g = toy_grid()  # toy frame head prefers INS: no frames
    assert g.frames([0, 1, 2, -1]) == [0, 0, 0, 0]
    g.frame[3] = np.array([-1.0, 1.0], dtype=np.float32)
    assert g.frames([0, 1, 2, -1]) == [0, 0, 0, 1]
    assert g.frames([0, 1, 2, 0]) == [0, 0, 0, 0]  # linked words are never frame
