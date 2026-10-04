"""The note's metrics over training time: the flat record, the re-pairings, the evaluation grid, the monitor."""

from types import SimpleNamespace

import torch

from retexo.baselines.base import Prediction
from retexo.baselines.early_stopping import NoteMetrics, TrainingMonitor
from retexo.baselines.record import Edge, Record
from retexo.baselines.scorer import BaselineScorer


def _record(i: int) -> Record:
    return Record(id=f"p{i}", level="gold", fold=1, source_work=f"s{i}", source_tokens=["arma", "virum", f"x{i}"],
                  reuse_work="", reuse_tokens=["arma", "viros", "canit"], pair_label="cit",
                  links=[Edge(0, 0, "COPY"), Edge(1, 1, "MORPH")])


def _perfect(record: Record) -> Prediction:
    pred = Prediction.empty(record.n_reuse)
    for edge in record.links:
        pred.links[edge.r] = edge.s
        pred.tags[edge.r] = edge.op
    return pred


def test_the_flat_record_names_every_note_metric():
    records = [_record(i) for i in range(3)]
    negatives = TrainingMonitor.repairings(records)
    invented = [_perfect(r) for r in records]                          # each re-pairing gets two invented links
    result = BaselineScorer.score(records, [_perfect(r) for r in records], negatives=(negatives, invented))
    flat = NoteMetrics.flat(result)
    for key in ("token_accuracy", "link_precision", "link_recall", "link_f1", "invented_links_per_pair",
                "L3_macro_f1", "L3_micro_f1", "L3_f1_INFLECT", "L3_f1_COPY", "L4_macro_f1", "L4_f1_SUBST",
                "sure_recall_SUBST", "sure_recall_INFLECT", "G1_mode_macro_f1", "G2_group_macro_f1", "frame_span_f1",
                "quote_span_f1", "reorder_token_f1", "disperse_pair_f1", "resource_silent_link_accuracy",
                "replay_rate", "invalid_output_rate"):
        assert key in flat, key
    assert flat["link_f1"] == 1.0 and flat["L3_f1_INFLECT"] == 1.0 and flat["invented_links_per_pair"] == 2.0


def test_the_repairings_pair_no_passage_with_its_own_source():
    records = [_record(i) for i in range(5)]
    pairs = TrainingMonitor.repairings(records)
    assert len(pairs) == 5 and all(not p.links and p.pair_label == "no_match" for p in pairs)
    assert all(p.source_tokens[-1] != f"x{p.id.split('/')[1][1:]}" for p in pairs)       # never its own source
    assert [p.id for p in TrainingMonitor.repairings(records)] == [p.id for p in pairs]    # seeded


class _FakeMethod:
    """A method whose prediction is perfect and whose predict switches its module to eval mode."""

    name, emits, decoder = "fake", "tags", None

    def __init__(self, valid):
        self.validation, self.cfg = valid, SimpleNamespace(extra={}, smoke=0)
        self.layer = torch.nn.Linear(1, 1)
        self.validation_level = "V1"

    def modules(self):
        return {"layer": self.layer}

    def predict(self, records):
        self.layer.eval()
        return [_perfect(r) for r in records]

    def postprocess(self, record, pred, dials):
        return pred

    def validation_loss(self, records):
        assert not self.layer.training                                  # the validation loss runs without dropout
        return 0.25 * len(records)


def test_the_monitor_scores_inside_the_epoch_and_leaves_training_untouched():
    method = _FakeMethod([_record(i) for i in range(4)])
    monitor = TrainingMonitor.for_method(method)
    assert TrainingMonitor.for_method(method) is monitor                # one monitor per run
    method.layer.train()
    on_batch = monitor.progress("synthetic", 4)
    for done in range(10, 101, 10):                                     # ten batches of one epoch of 100
        on_batch(done, 100)
        assert method.layer.training                                    # evaluation restored the train flag
    monitor.evaluate("synthetic", fraction=1.0)
    monitor.evaluate("epoch", epoch=1)
    assert [p["fraction"] for p in monitor.points] == [0.25, 0.5, 0.75, 1.0, None]
    assert monitor.points[-1]["epoch"] == 1 and monitor.points[-1]["L3_macro_f1"] == 1.0
    assert monitor.points[-1]["headline"] == 1.0 and "seconds" in monitor.points[-1]
    assert monitor.points[-1]["validation_loss"] == 1.0                   # 0.25 x four validation pairs
    assert TrainingMonitor(method, method.validation).progress("x", 1) is None       # one evaluation: the epoch's own
