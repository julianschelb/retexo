# tests/test_span_view.py
"""Row 9: the span view -- a second location scorer on the same encoder, averaged with the pointer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record  # noqa: E402

TINY = "hf-internal-testing/tiny-random-bert"


def test_span_scores_marginals_sum_to_one_with_the_null_and_respect_validity():
    from retexo.formulations.change_detector import ChangeDetectorConfig
    from retexo.formulations.typed_pointer import TypedPointer

    cfg = ChangeDetectorConfig(
        base_model=TINY,
        device="cpu",
        pointer=True,
        operations=(),
        fine_operations=("SUBST",),
        typed_pointer=True,
        span_view=True,
        span_max_len=3,
        frame_head=False,
    )
    model = TypedPointer(cfg)
    hidden = model._encoder.config.hidden_size
    T, width = 2, 5
    vectors = torch.randn(T, hidden)
    h_s = torch.randn(T, width, hidden)
    valid = torch.ones(T, width, dtype=torch.bool)
    valid[1, 3:] = False  # row 1 has three source words
    logits, marginal, p_null = model._span_scores(vectors, h_s, valid)
    assert logits.shape == (T, 1 + 3 * width) and marginal.shape == (T, width)
    assert torch.all(marginal[1, 3:] == 0)  # no mass on invalid columns
    # the covered mass of every span lands on the columns it covers, then the row is renormalised with
    # the null into a distribution the decoder can read
    assert torch.allclose(marginal.sum(dim=1) + p_null, torch.ones(T), atol=1e-5)
    prob = torch.softmax(logits, dim=1)
    p_spans = prob[:, 1:].reshape(T, 3, width)
    covered = sum((length + 1) * p_spans[:, length, :].sum(dim=1) for length in range(3))
    assert torch.allclose(
        marginal.sum(dim=1) / p_null, covered / prob[:, 0], atol=1e-4
    )  # the same ratio as before renormalising
    # the loss: gold column 2 for word 0, the null for word 1
    model._last_span = (logits, marginal, p_null)
    word_at = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, -1, -1]])
    loss = model._span_loss(word_at, torch.tensor([12, -1]))
    assert loss is not None and float(loss.detach()) > 0


def test_pointer_trains_and_predicts_with_the_span_view_on_the_tiny_backbone():
    from retexo.baselines.typed_pointer import TypedPointerBaseline

    cfg = BaselineConfig(
        fold=4,
        dev_fold=0,
        device="cpu",
        base_model=TINY,
        batch_size=4,
        seed=1,
        extra={
            "size": 0,
            "gold_passes": 1,
            "negatives": "none",
            "span_view": 1,
            "span_weight": 0.5,
        },
    )
    recs = [
        Record(
            id=f"p{i}",
            level="gold",
            fold=1,
            source_work="",
            source_tokens=["arma", "uirum", "cano", "troiae"],
            reuse_work="",
            reuse_tokens=["armis", "cano", "uirum"],
            pair_label="cit",
            links=[Edge(0, 0, "MORPH"), Edge(1, 2, "COPY"), Edge(2, 1, "COPY")],
        )
        for i in range(3)
    ]
    method = TypedPointerBaseline(cfg)
    assert method.recipe.span_view and method.recipe.span_weight == 0.5
    method.fit(recs, [], log=None)
    assert method.model._span_q is not None
    preds = method.predict(recs)
    rows = preds[0].scores
    assert len(rows) == 3 and all(
        abs(sum(p for _, p in row) - 1.0) < 1e-4 for row in rows
    )  # mixed rows still sum to 1
    assert len(method.model._span_marginals) >= 3
    done = method.postprocess(recs[0], preds[0], {"theta": 0.45})
    assert len(done.links) == 3
