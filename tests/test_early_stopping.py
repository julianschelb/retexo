"""Early stopping: the grouped validation split, the stopping rule, and the restore of the best weights."""

import torch

from retexo.baselines.early_stopping import EarlyStopping, ValidationSplit
from retexo.baselines.record import Record


def _record(i: int, citation: str) -> Record:
    return Record(id=f"p{i}", level="gold", fold=1, source_work="", source_tokens=["a"], reuse_work="",
                  reuse_tokens=[f"w{i}"], pair_label="cit", reuse_meta={"citation": citation})


def test_split_keeps_every_reusing_passage_on_one_side():
    records = [_record(i, f"<q {i // 3}>") for i in range(300)]          # 100 passages, three pairs each
    fit, valid = ValidationSplit.split(records, share=0.2, seed=1)
    assert len(fit) + len(valid) == 300 and 60 <= len(valid) <= 63
    assert not {r.reuse_meta["citation"] for r in fit} & {r.reuse_meta["citation"] for r in valid}
    again = ValidationSplit.split(records, share=0.2, seed=1)[1]
    assert [r.id for r in again] == [r.id for r in valid]                  # deterministic per seed
    assert ValidationSplit.split(records, share=0.0)[1] == []


def _run(scores, *, lo=2, patience=2, hi=10):
    layer = torch.nn.Linear(1, 1, bias=False)
    values = iter(scores)
    stopper = EarlyStopping(min_epochs=lo, patience=patience, max_epochs=hi, score=lambda: next(values))
    for epoch in range(1, hi + 1):
        with torch.no_grad():
            layer.weight.fill_(float(epoch))                               # the weight remembers its epoch
        if not stopper.step(epoch, {"layer": layer}):
            break
    stopper.restore({"layer": layer})
    return stopper, float(layer.weight)


def test_stops_after_patience_and_restores_the_best_epoch():
    stopper, weight = _run([0.5, 0.7, 0.6, 0.65, 0.9])
    assert stopper.best_epoch == 2 and len(stopper.history) == 4 and weight == 2.0


def test_never_stops_before_the_minimum():
    stopper, weight = _run([0.9, 0.1, 0.1, 0.1, 0.1, 0.1], lo=4, patience=1)
    assert len(stopper.history) == 4 and stopper.best_epoch == 1 and weight == 1.0


def test_stops_at_the_maximum_and_minimises_a_loss():
    layer = torch.nn.Linear(1, 1, bias=False)
    losses = iter([3.0, 2.0, 1.5, 1.2])
    stopper = EarlyStopping(min_epochs=1, patience=1, max_epochs=4, score=lambda: next(losses), higher_is_better=False)
    steps = [stopper.step(e, {"layer": layer}) for e in range(1, 5)]
    assert steps == [True, True, True, False] and stopper.best_epoch == 4
    assert stopper.summary()["history"][-1] == {"epoch": 4, "valid": 1.2}


def test_tune_theta_l3_keeps_the_low_similarity_substitution_token_accuracy_drops():
    # the fine-tuned similarity aligner's failure (2026-09-28): a SUBST link at .70 and a stray candidate at .72
    # tie on token accuracy, whose tie goes to the larger theta; L3 macro F1 keeps the substitution
    from dataclasses import replace

    from retexo.baselines.base import Baseline, BaselineConfig, Prediction
    from retexo.baselines.record import Edge
    from retexo.baselines.early_stopping import ValidationScorer
    from retexo.baselines.scorer import BaselineScorer

    class Stub(Baseline):
        name, emits, decoder, typer = "stub", "scores", "default", "rule"

    rest = 6                                                # correctly unlinked words on both sides
    record = Record(id="s/1", level="gold", fold=1, source_work="", reuse_work="", pair_label="cit",
                    source_tokens=["arma", "gladius", "q"] + [f"v{i}" for i in range(rest)],
                    reuse_tokens=["arma", "ensis", "x"] + [f"u{i}" for i in range(rest)],
                    links=[Edge(0, 0, "COPY"), Edge(1, 1, "SUBST")])
    pred = Prediction.empty(record.n_reuse)
    pred.scores = [[(0, 0.95)], [(1, 0.70)], [(2, 0.72)]] + [[(3 + i, 0.3)] for i in range(rest)]
    method, grid = Stub(BaselineConfig()), [0.7, 0.8]
    by_tokens = ValidationScorer.tune_theta(method, [record], [pred], {"theta": 0.45}, grid=grid)
    by_l3 = ValidationScorer.tune_theta(method, [record], [pred], {"theta": 0.45}, grid=grid, criterion="l3")
    assert (by_tokens, by_l3) == (0.8, 0.7)
    assert pred.links == [-1] * record.n_reuse and pred.tags == [""] * record.n_reuse      # untouched

    def l3(theta):
        done = method.postprocess(record, replace(pred, links=list(pred.links), tags=list(pred.tags),
                                                  frame=list(pred.frame)), {"theta": theta})
        return BaselineScorer.op_scores([record], [done], "V1")["macro_f1"]

    assert l3(by_l3) > l3(by_tokens)
