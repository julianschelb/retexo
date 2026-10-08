# tests/test_baseline_typed_pointer.py
"""Note 15's adapter on a tiny random-weight backbone: the recipe's defaults, the
typed and untyped builds, predict/postprocess shapes, one fit step, persistence."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import labels  # noqa: E402
from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.baselines.typed_pointer import (  # noqa: E402
    RECIPE_DEFAULTS,
    PointerFactory,
    Recipe,
    TrainingData,
    TypedPointerBaseline,
)
from retexo.datasets import synthetic as syn  # noqa: E402

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
        record(
            "t/3",
            ["gladio", "ferit"],
            ["ense", "ferit", "hostem"],
            edges=[Edge(0, 0, "SUBST", sure=False), Edge(1, 1, "COPY")],
        ),
        record(
            "t/4",
            ["nox", "erat"],
            ["nox", "erat", "et", "silentium"],
            edges=[Edge(0, 0, "COPY"), Edge(1, 1, "COPY")],
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
# 1. The recipe
# =============================================================================


def test_recipe_carries_the_published_defaults_when_nothing_is_overridden():
    recipe = Recipe.from_config(BaselineConfig())
    assert (
        recipe.swap,
        recipe.negatives,
        recipe.neg_ratio,
        recipe.null_weight,
        recipe.size,
        recipe.gold_passes,
    ) == ("double", "mixed", 0.5, 0.2, 25000, 8)
    assert recipe.typed and not recipe.evidence and recipe.directions == "both"
    assert set(recipe.as_dict()) == set(RECIPE_DEFAULTS)


def test_recipe_reads_extra_and_the_config_train_on():
    recipe = Recipe.from_config(
        BaselineConfig(train_on="sure", extra={"typed": 0, "size": 0, "directions": "one"})
    )
    assert (
        not recipe.typed
        and recipe.size == 0
        and recipe.directions == "one"
        and recipe.train_on == "sure"
    )


# =============================================================================
# 2. The builds
# =============================================================================


def test_build_without_evidence_has_no_evidence_modules_and_with_it_23_features():
    from retexo.formulations.typed_pointer import TypedPointer

    off = PointerFactory.build(Recipe.from_config(config()), config())
    assert isinstance(off, TypedPointer)
    assert off.config.feature_dim == 0 and not off.config.use_link_features
    assert (
        getattr(off, "_typer_evidence", None) is None
        and getattr(off, "_loc_evidence", None) is None
    )
    on = PointerFactory.build(Recipe.from_config(config(evidence=1)), config(evidence=1))
    assert on.config.feature_dim == 23 and on.config.use_link_features
    assert on._typer_evidence is not None


def test_untyped_build_is_a_change_detector_with_a_pointer_and_a_frame_head_but_no_typer():
    from retexo.formulations.change_detector import ChangeDetector
    from retexo.formulations.typed_pointer import TypedPointer

    model = PointerFactory.build(Recipe.from_config(config(typed=0)), config(typed=0))
    assert isinstance(model, ChangeDetector) and not isinstance(model, TypedPointer)
    assert (
        model.config.pointer
        and model._frame_head is not None
        and getattr(model, "_typer", None) is None
    )


# =============================================================================
# 3. Training data
# =============================================================================


def test_gold_examples_double_the_records_and_sure_only_drops_possible_links():
    data = TrainingData(Recipe.from_config(config()), config())
    both = data.gold_examples(pairs())
    assert len(both) == 8
    swapped = both[1]
    assert swapped.source_tokens == ["arma", "virum", "canit", "poeta"]
    assert swapped.alignments[:3] == [0, 1, 2]  # the twin's links invert exactly
    one = data.gold_examples(pairs(), swap="none")
    assert len(one) == 4 and one[2].alignments == [0, 1, -1]
    assert one[0].fine_operations == [
        "NOP",
        "MORPH",
        "MORPH",
        "INS",
    ]  # the typer's targets exist without evidence
    assert one[1].frame_labels == [1, 1, 0, 0]
    sure = TrainingData(Recipe.from_config(config(train_on="sure")), config(train_on="sure"))
    assert sure.gold_examples(pairs(), swap="none")[2].alignments == [-1, 1, -1]


def test_v3_annotated_records_supervise_the_fine_typer_and_silver_ones_do_not():
    v3 = record(
        "t/5",
        ["gladio", "ferit", "virumque"],
        ["ense", "ferit", "virum", "que"],
        edges=[
            Edge(0, 0, "SYN", detail="CASE"),
            Edge(1, 1, "COPY"),
            Edge(2, 2, "SPLIT"),
            Edge(3, 2, "SPLIT"),
        ],
    )
    v3.provenance = {"links": "llm-blind", "fine_ops": "llm-blind-V3"}
    silver = record(
        "t/6",
        ["gladio", "ferit"],
        ["ense", "ferit"],
        edges=[Edge(0, 0, "SUBST"), Edge(1, 1, "COPY")],
    )
    silver.provenance = {"links": "llm-silver", "fine_ops": "llm-silver-V1"}
    data = TrainingData(Recipe.from_config(config()), config())
    fine_v3, fine_silver = (
        e.fine_operations for e in data.gold_examples([v3, silver], swap="none")
    )
    assert fine_v3 == ["SYN", "NOP", "SPLIT", "SPLIT"]  # the annotated V3 operation is the target
    assert fine_silver == ["?", "NOP"]  # a silver SUBST stays unsupervised
    off = TrainingData(Recipe.from_config(config(gold_fine=0)), config(gold_fine=0))
    assert off.gold_examples([v3], swap="none")[0].fine_operations == ["?", "NOP", "?", "?"]
    v1 = TrainingData(
        Recipe.from_config(config(fine_operations="v1")), config(fine_operations="v1")
    )
    assert v1.gold_examples([v3], swap="none")[0].fine_operations == [
        "SUBST",
        "NOP",
        "SUBST",
        "SUBST",
    ]


def test_frames_come_from_the_records_frame_spans():
    assert TrainingData._frames(pairs()) == [["ut", "ait"]]


def test_canonical_maps_every_fine_operation_to_an_edge_op():
    for tag in syn.FINE_OPERATIONS:
        op, _ = labels.canonical(tag)
        assert op in labels.EDGE_OPS, tag


# =============================================================================
# 4. Predict and postprocess
# =============================================================================


def test_fit_predict_postprocess_end_to_end_on_the_tiny_backbone(tmp_path):
    import torch

    cfg = config()
    method = TypedPointerBaseline(cfg)
    fresh = (
        next(iter(PointerFactory.build(method.recipe, cfg)._encoder.parameters())).detach().clone()
    )
    method.fit(pairs(), [], log=None)
    assert method.model is not None
    assert not torch.equal(
        fresh, next(iter(method.model._encoder.parameters()))
    )  # one fit step moved the encoder
    preds = method.predict(pairs())
    for rec, pred in zip(pairs(), preds):
        assert len(pred.scores) == rec.n_reuse and len(pred.rev_scores) == rec.n_source
        for row in pred.scores:
            assert abs(sum(p for _, p in row) - 1.0) < 1e-3 and any(s == -1 for s, _ in row)
    done = method.postprocess(pairs()[1], preds[1], {"theta": 0.45})
    assert len(done.links) == 4 and len(done.tags) == 4 and len(done.frame) == 4
    for t, (s, tag) in enumerate(zip(done.links, done.tags)):
        assert (s < 0) == (tag == "")
        if s >= 0:
            assert tag in syn.FINE_OPERATIONS
            assert done.frame[t] == 0
    assert done.dels == [0 if s in set(done.links) else 1 for s in range(3)]
    assert len(done.frame_p) == 4
    # persistence: the same weights come back
    method.save(tmp_path / "m")
    again = TypedPointerBaseline.load(tmp_path / "m", cfg)
    a = next(iter(method.model._encoder.parameters()))
    b = next(iter(again.model._encoder.parameters()))
    assert torch.equal(a, b)


def test_one_direction_leaves_rev_scores_empty_and_untyped_mode_names_with_the_rule_typer():
    cfg = config(typed=0, directions="one")
    method = TypedPointerBaseline(cfg)
    assert method.typer == "rule"
    method.fit(pairs(), [], log=None)
    preds = method.predict(pairs())
    assert all(p.rev_scores is None for p in preds) and all(p.scores is not None for p in preds)
    done = method.postprocess(pairs()[0], preds[0], {"theta": 0.45})
    for s, tag in zip(done.links, done.tags):
        assert (s < 0) == (tag == "")
        if s >= 0:
            assert tag in labels.EDGE_OPS
    assert len(done.dels) == 3
