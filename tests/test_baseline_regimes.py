# tests/test_baseline_regimes.py
"""Note 33's registry on hand-made records: the resolved overrides, the nested
stratified learning-curve sample, the self-training filter, the distillation
records, and the link-only example the pointer trains on."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import Prediction  # noqa: E402
from retexo.baselines.record import Edge, Record  # noqa: E402
from retexo.baselines.regimes import (CURVE_SIZES, Distillation, ExtraPairs, RegimeApplier, SelfTraining,  # noqa: E402
                                          curve_sample, link_only_example, parse_regime, resolve)
from retexo.baselines.typed_pointer import V1_TYPES, PointerFactory, Recipe  # noqa: E402
from retexo.baselines.base import BaselineConfig  # noqa: E402


def record(rid, source, reuse, edges=(), label="cit"):
    return Record(id=rid, level="gold", fold=1, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label=label, links=list(edges))


# =============================================================================
# 1. The registry
# =============================================================================


def test_resolve_fills_the_rows_switches_and_the_fold():
    assert resolve("gold_only", 4)["size"] == 0
    assert resolve("synthetic_only", 4)["gold_passes"] == 0
    assert resolve("no_negatives", 4)["neg_ratio"] == 0
    assert resolve("dense", 4)["pool_file"].endswith("synthetic_dense_f4.jsonl")
    assert resolve("v1_types", 4)["fine_operations"] == "v1"
    assert resolve("curve", 4, n=300, pretrain=0)["size"] == 0 and resolve("curve", 4, n=300)["n"] == 300
    assert resolve("self_train", 4, rounds=2)["extra_records"] == "runs/self_train_f4/round2.jsonl"
    assert parse_regime("curve,n=300,pretrain=0") == ("curve", {"n": 300, "pretrain": 0})
    try:
        resolve("nonsense", 4)
        assert False
    except KeyError:
        pass


def test_the_v1_types_row_builds_a_pointer_with_three_types():
    cfg = BaselineConfig(device="cpu", base_model="hf-internal-testing/tiny-random-bert",
                         extra={"fine_operations": "v1", "size": 0})
    model = PointerFactory.build(Recipe.from_config(cfg), cfg)
    assert tuple(model.config.fine_operations) == V1_TYPES and model.K == 3


# =============================================================================
# 2. The learning curve
# =============================================================================


def test_curve_samples_are_nested_stratified_and_deterministic():
    records = [record(f"c/{i}", ["a"], ["b"], label="cit") for i in range(700)] + \
              [record(f"f/{i}", ["a"], ["b"], label="cf") for i in range(500)]
    samples = {n: curve_sample(records, n, seed=0) for n in CURVE_SIZES}
    assert len(samples[100]) == 100 and len(samples[300]) == 300 and len(samples[600]) == 600 and len(samples["all"]) == 1200
    ids = {n: {r.id for r in samples[n]} for n in CURVE_SIZES}
    assert ids[100] <= ids[300] <= ids[600] <= ids["all"]
    for n in (100, 300, 600):
        cit = sum(1 for r in samples[n] if r.pair_label == "cit")
        assert abs(cit - round(n * 700 / 1200)) <= 2
    assert [r.id for r in curve_sample(records, 100, seed=0)] == [r.id for r in samples[100]]
    assert [r.id for r in curve_sample(records, 100, seed=1)] != [r.id for r in samples[100]]


# =============================================================================
# 3. Self-training and distillation
# =============================================================================


def rows_of(links, p=0.95, n_source=3):
    out = []
    for s in links:
        if s < 0:
            out.append([(-1, 1.0)])
        else:
            out.append([(s, p), (-1, round(1 - p, 4))])
    return out


def test_self_training_keeps_mutual_confident_replaying_edges_only():
    rec = record("r/1", ["arma", "virum", "cano"], ["arma", "virum", "canit"])
    forward = Prediction(links=[-1] * 3, tags=[""] * 3, frame=[0] * 3, scores=rows_of([0, 1, 2]))
    reverse = Prediction(links=[-1] * 3, tags=[""] * 3, frame=[0] * 3, scores=rows_of([0, 1, -1]))   # word 2 not mutual
    kept = SelfTraining.filter([rec], [forward], [reverse], tau=0.9)
    assert len(kept) == 1
    assert [(e.r, e.s, e.op) for e in kept[0].links] == [(0, 0, "LINK"), (1, 1, "LINK")]
    assert kept[0].provenance["links"] == "self-training" and kept[0].level == "real_pairs"
    # below tau nothing is kept; a pair without agreed edges contributes nothing
    assert SelfTraining.filter([rec], [Prediction(links=[-1] * 3, tags=[""] * 3, frame=[0] * 3, scores=rows_of([0, 1, 2], p=0.6))],
                               [reverse], tau=0.9) == []
    # the replay check: a link that makes the script not reproduce the reuse is dropped with its pair
    crossed = record("r/2", ["a", "b"], ["b", "a"])
    fwd = Prediction(links=[-1] * 2, tags=[""] * 2, frame=[0] * 2, scores=rows_of([1, 0], n_source=2))
    rev = Prediction(links=[-1] * 2, tags=[""] * 2, frame=[0] * 2, scores=rows_of([1, 0], n_source=2))
    replayed = SelfTraining.filter([crossed], [fwd], [rev], tau=0.9)
    assert all(isinstance(r, Record) for r in replayed)
    assert SelfTraining.tune_tau([rec], [forward], [reverse]) in (0.8, 0.85, 0.9, 0.95)


def test_distillation_copies_the_full_systems_edges_and_marks_the_provenance():
    from dataclasses import replace

    rec = replace(record("r/3", ["rex", "amat"], ["regem", "amat", "ut"]),
                  pred={"edges": [{"r": 0, "s": 0, "op": "MORPH"}, {"r": 1, "s": 1, "op": "COPY"}], "frame": [0, 0, 1]})
    out = Distillation.records([rec])
    assert [(e.r, e.s, e.op) for e in out[0].links] == [(0, 0, "MORPH"), (1, 1, "COPY")]
    assert out[0].spans and out[0].spans[0].start == 2 and out[0].provenance["links"] == "full_system"
    assert out[0].pred is None and out[0].id == "distill/r/3"


def test_link_only_records_train_the_location_and_no_type():
    rec = record("r/4", ["rex", "amat", "hostem"], ["regem", "amat", "ense"],
                 edges=[Edge(0, 0, "LINK"), Edge(1, 1, "COPY"), Edge(2, 2, "LINK")])
    example = link_only_example(rec)
    assert example.alignments == [0, 1, 2]
    assert example.fine_operations == ["LINK", "NOP", "LINK"]     # masked at every type level, not the lexical group
    assert example.operations == ["SUBST", "COPY", "SUBST"]      # coarse: a change, the copy stays a copy
    same = link_only_example(record("r/5", ["amat"], ["amat"], edges=[Edge(0, 0, "LINK")]))
    assert same.operations == ["COPY"] and same.fine_operations == ["LINK"]


def test_extra_pairs_and_the_applier(tmp_path):
    import json

    rows = [{"id": "x1", "ref_type": "llm", "source_tokens": ["a", "b"], "target_tokens": ["ut", "a", "c"],
             "target_ops": ["FRAME", "NOP", "SYN"], "target_align": [-1, 0, 1]}]
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(rows))
    plain = ExtraPairs.load(path)
    assert [(e.r, e.s, e.op) for e in plain[0].links] == [(1, 0, "COPY"), (2, 1, "SYN")] and plain[0].spans[0].label == "FRAME"
    stripped = ExtraPairs.load(path, links_only=True)
    assert [e.op for e in stripped[0].links] == ["COPY", "LINK"]
    train = [record(f"g/{i}", ["a"], ["b"]) for i in range(10)]
    overrides, records = RegimeApplier.apply("gold_only", fold=4, train=train)
    assert overrides["size"] == 0 and len(records) == 10
    overrides, records = RegimeApplier.apply("curve,n=4", fold=4, train=train)
    assert len(records) == 4 and overrides["regime"] == "curve"
