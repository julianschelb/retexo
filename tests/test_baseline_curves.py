# tests/test_baseline_curves.py
"""The curve recorder's parser on the trainers' real log shapes, and the sample selector on toy rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import Prediction  # noqa: E402
from retexo.baselines.curves import CurvePlot, CurveRecorder  # noqa: E402
from retexo.baselines.record import Edge, Record  # noqa: E402
from retexo.baselines.samples import DifficultyTable, LeakageSplit, MarkdownRenderer, SampleSelector  # noqa: E402


def test_recorder_parses_every_trainer_shape_and_ignores_the_rest():
    rec = CurveRecorder()
    for line in ("    epoch 3/8  loss 0.4123", "[span_aligner] epoch 2/3: loss 0.51", "[span_pair] pass 4/6: loss 0.20 (12 records)",
                 "[nmt_aligner] layer fwd update 500/2000: guided loss 1.2e-2", "    typer refine 1/3  loss 0.9  (1,200 links)",
                 "[baseline x] theta tuned on dev: 0.45", "[baseline x] fold 4: train 12 dev 3 test 3 (gold)"):
        rec(line)
    stages = [(p.tag, p.stage, p.step, p.total, p.loss) for p in rec.points]
    assert stages == [("", "epoch", 3, 8, 0.4123), ("span_aligner", "epoch", 2, 3, 0.51), ("span_pair", "pass", 4, 6, 0.2),
                      ("nmt_aligner", "layer fwd update", 500, 2000, 0.012), ("", "typer refine", 1, 3, 0.9)]
    series = CurvePlot.series(rec.points_as_dicts())
    assert set(series) == {"epoch", "span_aligner epoch", "span_pair pass", "nmt_aligner layer fwd update", "typer refine"}


def _rows():
    def rec(rid, fold, src, reuse, edges):
        return Record(id=rid, level="gold", fold=fold, source_work="", source_tokens=src, reuse_work="", reuse_tokens=reuse,
                      pair_label="cit", links=edges, spans=[], provenance={}, annotation={"note": "n"})
    a = rec("a", 4, ["arma", "virum", "cano"], ["arma", "virum", "canit", "x"], [Edge(0, 0, "COPY"), Edge(1, 1, "COPY"), Edge(2, 2, "MORPH")])
    b = rec("b", 4, ["nox", "erat", "et"], ["nox", "erat", "et"], [Edge(0, 0, "COPY"), Edge(1, 1, "COPY"), Edge(2, 2, "COPY")])
    c = rec("c", 4, ["rex", "amat"], ["rex", "odit", "y"], [Edge(0, 0, "COPY"), Edge(1, 1, "SUBST")])
    pa = Prediction.empty(4); pa.links = [0, 1, 2, -1]; pa.tags = ["COPY", "COPY", "MORPH", ""]
    pb = Prediction.empty(3); pb.links = [0, 1, 2]; pb.tags = ["COPY"] * 3
    pc = Prediction.empty(3); pc.links = [0, -1, -1]; pc.tags = ["COPY", "", ""]
    train = rec("t", 1, ["nox", "erat", "et"], ["z"], [])
    return [(a, pa), (b, pb), (c, pc)], [a, b, c, train]


def test_selector_prefers_informative_correct_pairs_and_renders_differences():
    rows, records = _rows()
    best, worst = SampleSelector(min_links=2).select(rows, k=1)
    assert best[0].record.id == "a"                     # perfect, and it has a MORPH link; the verbatim pair ranks below
    assert worst[0].record.id == "c" and abs(worst[0].accuracy - 2 / 3) < 1e-9
    text = MarkdownRenderer.render(worst[0])
    assert "| 1:odit | 1:amat SUBST | \u2014 | ! |" in text and "annotator's note: n" in text
    leaky, clean = LeakageSplit(records, fold=4).split(rows)
    assert [r.id for r, _ in leaky] == ["b"] and [r.id for r, _ in clean] == ["a", "c"]
    table = DifficultyTable({"m1": rows, "m2": rows})
    assert table.hardest(1)[0][0] == "c" and table.failed_by_all() == ["c"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_curves] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
