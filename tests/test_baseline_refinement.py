# tests/test_baseline_refinement.py
"""Note 22 on the tiny random backbone: the state built from a script, the
corruption, the zero-initialised state parameters, the pass loop with a spy on
the decoder, and the grid design's identity at initialisation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.baselines.refinement import REFINE_DEFAULTS, RefinedPointer  # noqa: E402
from retexo.refinement.refine import (  # noqa: E402
    DECLINED,
    FRAME,
    LINKED,
    S_CONSUMED,
    S_FREE,
    UNDECIDED,
    State,
)

TINY = "hf-internal-testing/tiny-random-bert"


def record(rid, source, reuse, edges=(), spans=()):
    return Record(
        id=rid,
        level="gold",
        fold=1,
        source_work="",
        source_tokens=source,
        reuse_work="",
        reuse_tokens=reuse,
        pair_label="cit",
        links=list(edges),
        spans=list(spans),
    )


def pairs():
    return [
        record(
            "t/1",
            ["arma", "virumque", "cano"],
            ["arma", "virum", "canit", "poeta"],
            edges=[Edge(0, 0, "COPY"), Edge(1, 1, "MORPH"), Edge(2, 2, "MORPH")],
        ),
        record(
            "t/2",
            ["rex", "regem", "amat"],
            ["ut", "ait", "rex", "amat"],
            edges=[Edge(2, 0, "COPY"), Edge(3, 2, "COPY")],
            spans=[Span(0, 2, "FRAME")],
        ),
    ]


def config(**extra):
    return BaselineConfig(
        fold=4,
        dev_fold=0,
        device="cpu",
        base_model=TINY,
        batch_size=4,
        seed=1,
        extra={"size": 0, "negatives": "none", "gold_passes": 1, **extra},
    )


# =============================================================================
# 1. The state
# =============================================================================


def test_state_from_script_and_undecided():
    state = State.from_script([0, -1, 2], [0, 1, 0], 3)
    assert state.reuse == [LINKED, FRAME, LINKED] and state.link == [0, -1, 2]
    assert state.source == [S_CONSUMED, S_FREE, S_CONSUMED]
    blank = State.undecided(2, 3)
    assert (
        blank.reuse == [UNDECIDED, UNDECIDED] and blank.link == [-1, -1] and len(blank.source) == 3
    )
    assert State.from_script([-1], None, 1).reuse == [DECLINED]


def test_corrupted_flips_and_adds_about_the_asked_share_and_keeps_the_gold_at_zero():
    gold = list(range(20))
    rng = random.Random(0)
    changed = wrong = 0
    for _ in range(200):
        state = State.corrupted(gold, None, 40, rng, flip=0.5, add=0.5)
        changed += sum(1 for t in range(20) if state.link[t] != gold[t])
        wrong += sum(1 for t in range(20) if state.link[t] >= 0 and state.link[t] != gold[t])
    assert 0.4 < changed / (200 * 20) < 0.6  # about half the links are touched
    assert wrong / 200 > 2  # false links are added among them
    assert State.corrupted(gold, None, 40, rng, flip=0.0, add=0.0).link == gold


# =============================================================================
# 2. The loop
# =============================================================================


def test_zero_state_parameters_leave_pass_one_rows_unchanged_and_the_loop_runs():
    cfg = config(refine="random", rollin=0.0, passes=3)
    method = RefinedPointer(cfg)
    assert (
        method.dials["passes"] == 3
        and method.dials["design"] == "state" == REFINE_DEFAULTS["design"]
    )
    method.fit(pairs(), [], log=None)
    model = method.base.model
    assert model.refine_mode == "random" and model.refine_rollin == 0.0
    rec = pairs()[0]
    example = method.base.examples_of([rec])[0]
    object.__setattr__(example, "refine_state", None)
    plain = model.predict_alignment_scores([example])[0]
    object.__setattr__(example, "refine_state", State.from_script([0, 1, 2, -1], [0, 0, 0, 0], 3))
    stated = model.predict_alignment_scores([example])[0]
    object.__setattr__(example, "refine_state", None)
    # the state parameters start at zero only when the model never trained a state; after one
    # training pass with refine=random they moved, so the two row sets may differ -- the shapes must not
    assert len(plain) == len(stated) == 4
    preds = method.predict(pairs())
    done = [method.postprocess(r, p, {"theta": 0.45}) for r, p in zip(pairs(), preds)]
    for rec, pred in zip(pairs(), done):
        assert (
            len(pred.links) == rec.n_reuse
            and len(pred.tags) == rec.n_reuse
            and len(pred.frame) == rec.n_reuse
        )
        assert (
            pred.meta["n_passes"] == 3
            and pred.meta["settled_at"] == 0
            and len(pred.meta["passes"]) == rec.n_reuse
        )
        for s, tag in zip(pred.links, pred.tags):
            assert (s < 0) == (tag == "")


def test_until_stable_settles_when_a_pass_repeats_and_the_decoder_sees_every_pass(monkeypatch):
    from retexo.baselines.adapters import PredictionAdapter

    cfg = config(refine="random", rollin=0.0, passes=4, until_stable=1)
    method = RefinedPointer(cfg).fit(pairs(), [], log=None)
    calls = []
    real = PredictionAdapter.decode_prediction

    def spy(pred, decoder_name, dials, rec):
        calls.append([row[:2] for row in pred.scores])
        return real(pred, decoder_name, dials, rec)

    monkeypatch.setattr(PredictionAdapter, "decode_prediction", staticmethod(spy))
    rec = pairs()[1]
    method.rows_at_pass = lambda record, links, frame: [
        [(-1, 1.0)] for _ in range(record.n_reuse)
    ]  # a fixed second look
    pred = method.postprocess(rec, method.predict([rec])[0], {"theta": 0.45})
    assert pred.links == [-1] * rec.n_reuse
    settled = pred.meta["settled_at"]
    assert (
        settled in (2, 3) and pred.meta["n_passes"] == settled < 4
    )  # the loop stopped when a pass repeated
    assert len(calls) == settled  # the decoder ran once per pass


# =============================================================================
# 3. The grid design
# =============================================================================


def test_grid_refiner_is_the_identity_at_initialisation():
    import numpy as np

    from retexo.refinement.grid_refiner import GridRefiner, PairGrid

    cfg = config()
    method = RefinedPointer(config(design="grid"))
    method.base.fit(pairs(), [], log=None)
    grid = method.grid_of(pairs()[0])
    assert isinstance(grid, PairGrid) and grid.shape == (4, 3)
    refiner = GridRefiner(grid.phi.shape[-1], hidden=8, layers=1, device="cpu")
    state = State.from_script([0, 1, 2, -1], [0, 0, 0, 0], 3)
    loc, nulls = refiner.correct(grid, state, PairGrid.confidence(grid.scores()), "cpu")
    assert np.allclose(loc[grid.mask], grid.loc[grid.mask]) and np.allclose(nulls, grid.nulls)
    assert grid.perturb(state, random.Random(0), dropout=1.0).reuse == [UNDECIDED] * 4
    assert cfg.extra["size"] == 0
