# tests/test_baseline_tagger.py
"""Label derivation, class weights and the keep-bias/min-change-p gate on hand-built
data; Tagger.predict with a stubbed prediction pass (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.baselines.tagger import (  # noqa: E402
    LABEL_SETS,
    TaggerMode,
    TaggerV1,
    _class_weights,
    examples_from_records,
)


def record(source, reuse, edges=(), spans=(), annotation=None):
    return Record(
        id="t/1",
        level="gold",
        fold=4,
        source_work="",
        source_tokens=source,
        reuse_work="",
        reuse_tokens=reuse,
        pair_label="cit",
        links=list(edges),
        spans=list(spans),
        annotation=annotation or {},
    )


# =============================================================================
# 1. Label derivation
# =============================================================================


def test_v1_labels_keep_frame_distinct_and_derive_source_del():
    # source: rex regem amat pater; reuse: dominus amat frater (FRAME on word 2)
    source = ["rex", "regem", "amat", "pater"]
    reuse = ["dominus", "amat", "frater"]
    edges = [Edge(r=0, s=0, op="MORPH"), Edge(r=1, s=2, op="COPY")]
    spans = [Span(start=2, end=3, label="FRAME")]
    rec = record(source, reuse, edges=edges, spans=spans)
    v1 = LABEL_SETS["V1"]
    reuse_labels = v1.token_labels(rec)
    assert reuse_labels == ["MORPH", "COPY", "FRAME"]
    examples = examples_from_records([rec], v1)
    ex = examples[0]
    assert ex.operations == ["MORPH", "COPY", "FRAME"]
    # source words 1 (regem) and 3 (pater) are never pointed at: DEL
    assert ex.source_operations == ["KEEP", "DEL", "KEEP", "DEL"]


def test_mode_labels_verbatim_allusion_frame_nomatch():
    source = ["rex", "regem", "amat", "pater"]
    reuse = ["rex", "dominus", "frater"]
    edges = [Edge(r=0, s=0, op="COPY"), Edge(r=1, s=1, op="SUBST")]
    spans = [Span(start=2, end=3, label="FRAME")]
    rec = record(source, reuse, edges=edges, spans=spans)
    mode = LABEL_SETS["mode"]
    assert mode.token_labels(rec) == ["VERBATIM", "ALLUSION", "FRAME"]


def test_ists_labels_come_from_the_annotation_field_not_edges():
    rec = record(["a", "b"], ["x", "y", "z"], annotation={"ists_type": ["EQUI", "SPE1", "NOALI"]})
    ists = LABEL_SETS["ists"]
    assert ists.token_labels(rec) == ["EQUI", "SPE1", "NOALI"]
    # short of n_reuse: padded with NOALI, not truncated silently past it
    short = record(["a"], ["x", "y"], annotation={"ists_type": ["OPPO"]})
    assert ists.token_labels(short) == ["OPPO", "NOALI"]


# =============================================================================
# 2. LabelSet definitions
# =============================================================================


def test_label_set_shapes():
    assert LABEL_SETS["V1"].classes == ("COPY", "MORPH", "SUBST", "INS", "FRAME")
    assert LABEL_SETS["V1"].source_head is True
    assert LABEL_SETS["mode"].classes == ("VERBATIM", "ALLUSION", "FRAME", "NOMATCH")
    assert LABEL_SETS["mode"].source_head is False
    assert LABEL_SETS["ists"].score_level is None


# =============================================================================
# 3. Class weights
# =============================================================================


def test_class_weights_inverse_frequency_capped_and_normalised():
    v1 = LABEL_SETS["V1"]
    recs = [record(["a"], ["x"], edges=[Edge(r=0, s=0, op="COPY")])] * 100
    recs += [record(["a"], ["x"], edges=[Edge(r=0, s=0, op="MORPH")])] * 5
    examples = examples_from_records(recs, v1)
    weights = dict(_class_weights(examples, v1.classes, cap=20.0))
    assert weights["COPY"] == 1.0  # the most frequent class
    assert weights["MORPH"] == 20.0  # 100/5 = 20, exactly at the cap
    assert weights["SUBST"] == 20.0  # never seen: capped, not infinite
    assert weights["FRAME"] == 20.0


def test_class_weights_uncapped_case():
    v1 = LABEL_SETS["V1"]
    recs = [record(["a"], ["x"], edges=[Edge(r=0, s=0, op="COPY")])] * 10
    recs += [record(["a"], ["x"], edges=[Edge(r=0, s=0, op="MORPH")])] * 5
    examples = examples_from_records(recs, v1)
    weights = dict(_class_weights(examples, v1.classes, cap=20.0))
    assert weights["MORPH"] == 2.0  # 10/5, under the cap


# =============================================================================
# 4. The keep-bias / min-change-p gate
# =============================================================================


def test_gate_keep_bias_forces_every_token_to_the_keep_class():
    import torch

    logits = torch.tensor([[0.0, 5.0], [0.0, 5.0]])  # class 1 wins on raw logits
    chosen = TaggerV1.gate(logits, keep_index=0, keep_bias=10.0, min_change_p=0.0)
    assert chosen == [0, 0]


def test_gate_min_change_p_one_keeps_every_pair():
    import torch

    logits = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    chosen = TaggerV1.gate(logits, keep_index=0, keep_bias=0.0, min_change_p=1.0)
    assert chosen == [0, 0]  # no probability can exceed 1.0, so the pair stays all-keep


def test_gate_default_dials_recovers_the_argmax():
    import torch

    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    chosen = TaggerV1.gate(logits, keep_index=0, keep_bias=0.0, min_change_p=0.0)
    assert chosen == [1, 0]


# =============================================================================
# 5. predict() with a stubbed prediction pass
# =============================================================================


class _FakeModel:
    def predict_source(self, examples):
        return [[0, 1, 0] for _ in examples]  # DEL on the middle source word


def test_predict_wires_links_tags_frame_and_dels():
    method = TaggerV1(BaselineConfig(device="cpu"))
    method.model = _FakeModel()
    method._predict_one_pass = lambda examples: [["MORPH", "FRAME", "COPY"]]
    rec = record(["a", "b", "c"], ["x", "y", "z"])
    pred = method.predict([rec])[0]
    assert pred.links == [-1, -1, -1]
    assert pred.tags == ["MORPH", "FRAME", "COPY"]
    assert pred.frame == [0, 1, 0]
    assert pred.dels == [0, 1, 0]


def test_predict_no_source_head_leaves_dels_none():
    method = TaggerMode(BaselineConfig(device="cpu"))
    method.model = _FakeModel()
    method._predict_one_pass = lambda examples: [["VERBATIM", "NOMATCH"]]
    pred = method.predict([record(["a"], ["x", "y"])])[0]
    assert pred.dels is None


def test_predict_before_fit_returns_empty_predictions():
    method = TaggerV1(BaselineConfig(device="cpu"))
    pred = method.predict([record(["a", "b"], ["x", "y"])])[0]
    assert pred.links == [-1, -1]
    assert pred.tags == ["", ""]


def test_postprocess_returns_prediction_unchanged():
    method = TaggerV1(BaselineConfig(device="cpu"))
    from retexo.baselines.base import Prediction

    pred = Prediction.empty(2)
    pred.tags = ["COPY", "INS"]
    assert method.postprocess(record(["a"], ["x", "y"]), pred, {}) is pred
