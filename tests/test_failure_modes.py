# tests/test_failure_modes.py
"""The failure-mode dry run's small parts: the gold reorder, the merged views, the stored rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.augment import GoldReorder  # noqa: E402
from retexo.baselines.base import BaselineConfig, Prediction  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402


def _record() -> Record:
    return Record(id="p1", level="gold", fold=4, source_work="", source_tokens=["a", "b", "c", "d", "e", "f"],
                  reuse_work="", reuse_tokens=["a", "b,", "x", "d", "e", "y"], pair_label="cit",
                  links=[Edge(0, 0, "COPY"), Edge(1, 1, "COPY"), Edge(3, 3, "COPY"), Edge(4, 4, "MORPH")],
                  spans=[Span(0, 2, "FRAME"), Span(2, 5, "FRAME")],
                  annotation={"regime": ["form", "form", "none", "form", "lemma", "none"], "note": "x"})


def test_gold_reorder_swaps_halves_and_remaps_everything():
    rec = _record()
    assert GoldReorder.cut_point(rec.reuse_tokens) == 2          # after "b,"
    out = GoldReorder.swap_halves(rec)
    assert out.reuse_tokens == ["x", "d", "e", "y", "a", "b,"]
    assert [(e.r, e.s, e.op) for e in out.links] == [(1, 3, "COPY"), (2, 4, "MORPH"), (4, 0, "COPY"), (5, 1, "COPY")]
    assert out.annotation["regime"] == ["none", "form", "lemma", "none", "form", "form"]
    assert out.annotation["note"] == "x"
    assert [(s.start, s.end) for s in out.spans] == [(0, 3), (4, 6)]      # both spans moved, none crosses the cut
    assert out.id == "p1#reorder" and out.provenance["reorder"]["cut"] == 2
    assert rec.reuse_tokens[0] == "a"                                      # the original untouched


def test_gold_reorder_splits_a_span_across_the_cut():
    rec = _record()
    rec.spans = [Span(1, 4, "FRAME")]
    out = GoldReorder.swap_halves(rec, cut=2)
    assert [(s.start, s.end) for s in out.spans] == [(0, 2), (5, 6)]


def test_gold_reorder_augment_rate():
    recs = [_record() for _ in range(20)]
    assert len(GoldReorder.augment(recs, rate=0.0)) == 20
    out = GoldReorder.augment(recs, rate=1.0)
    assert len(out) == 40 and sum(r.id.endswith("#reorder") for r in out) == 20




def test_stored_rows_reads_a_dump(tmp_path):
    from retexo.baselines import BaselineRegistry
    from retexo.baselines.adapters import PredictionAdapter, write_dump

    BaselineRegistry.load_all()
    rec = _record()
    pred = Prediction.empty(rec.n_reuse)
    pred.scores = [[(t, 0.8), (-1, 0.2)] for t in range(rec.n_reuse)]
    PredictionAdapter.TOP_K_IN_DUMP = 10 ** 6
    write_dump([rec], [pred], tmp_path / "predictions.jsonl")
    cls = BaselineRegistry.get("stored_rows")
    method = cls(BaselineConfig(fold=4, dev_fold=0, extra={"test": str(tmp_path / "predictions.jsonl")}))
    out = method.predict([rec, Record(**{**rec.__dict__, "id": "p2"})])
    assert out[0].scores[2][0] == (2, 0.8)
    assert out[1].scores == [[(-1, 1.0)]] * rec.n_reuse


def test_fragment_decoder_links_the_residual_word_inside_a_matched_fragment():
    from retexo.baselines.decoder import BaselineDecoder as D

    # reuse a b X d  <-  source a b Y d: X is a substitution of Y (p .3, under theta .45) inside the fragment
    rows = [[(0, 0.95), (-1, 0.05)], [(1, 0.9), (-1, 0.1)], [(2, 0.3), (-1, 0.6), (5, 0.1)], [(3, 0.9), (-1, 0.1)]]
    assert D.decode_default(rows, theta=0.45) == [0, 1, -1, 3]
    assert D.decode_fragment(rows, theta=0.45, n_source=6) == [0, 1, 2, 3]
    assert D.fragments([0, 1, -1, 3]) == [[0, 3, 0, 3]]
    # below the lower threshold (theta * .5 = .225) the word stays unlinked
    rows[2] = [(2, 0.2), (-1, 0.8)]
    assert D.decode_fragment(rows, theta=0.45, n_source=6) == [0, 1, -1, 3]


def test_fragment_decoder_between_two_crossed_fragments():
    from retexo.baselines.decoder import BaselineDecoder as D

    # reuse: [c d] X [a b]  <-  source: a b Y c d  (the halves swapped; X between the fragments comes from Y)
    rows = [[(3, 0.9), (-1, 0.1)], [(4, 0.9), (-1, 0.1)], [(2, 0.3), (-1, 0.7)], [(0, 0.9), (-1, 0.1)], [(1, 0.9), (-1, 0.1)]]
    assert D.decode_default(rows, theta=0.45) == [3, 4, -1, 0, 1]
    blocks = D.fragments([3, 4, -1, 0, 1])
    assert blocks == [[0, 1, 3, 4], [3, 4, 0, 1]]
    assert D.residual_region(2, blocks, 5) == (2, 2)
    assert D.decode_fragment(rows, theta=0.45, n_source=5) == [3, 4, 2, 0, 1]
    # a source word already used by a link is never a residual candidate
    rows[2] = [(0, 0.4), (-1, 0.6)]
    assert D.decode_fragment(rows, theta=0.45, n_source=5) == [3, 4, -1, 0, 1]
