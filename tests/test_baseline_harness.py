# tests/test_baseline_harness.py
"""The harness on the gold and on a hand-made 3 x 3 pair (no model, no resources)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import BaselineRegistry, labels  # noqa: E402
from retexo.baselines.adapters import from_script, read_dump, to_script, write_dump  # noqa: E402
from retexo.baselines.base import BaselineConfig, Prediction  # noqa: E402
from retexo.baselines.record import (  # noqa: E402
    Edge,
    Record,
    edges_from,
    extra_edges,
    gold_to_records,
    links_of,
    record_to_example,
)
from retexo.baselines.scorer import token_accuracy  # noqa: E402

GOLD = Path(__file__).resolve().parents[1] / "data" / "gold_full"


def small_record() -> Record:
    # source: arma uirum cano ; reuse: arma cano uirumque  (a MORPH and a crossing)
    return Record(
        id="t/1",
        level="gold",
        fold=4,
        source_work="s",
        source_tokens=["arma", "uirum", "cano"],
        reuse_work="r",
        reuse_tokens=["arma", "cano", "uirumque"],
        pair_label="cit",
        links=[Edge(0, 0, "COPY"), Edge(1, 2, "COPY"), Edge(2, 1, "MORPH")],
    )


def merge_split_record() -> Record:
    # reuse word 1 comes from source 1 and 2 (MERGE); reuse words 3 and 4 both from source 4 (SPLIT)
    return Record(
        id="t/2",
        level="gold",
        fold=4,
        source_work="s",
        source_tokens=["a", "necesse", "est", "b", "armaque"],
        reuse_work="r",
        reuse_tokens=["a", "necessest", "b", "arma", "que"],
        pair_label="cit",
        links=[
            Edge(0, 0, "COPY"),
            Edge(1, 1, "MERGE"),
            Edge(1, 2, "MERGE"),
            Edge(2, 3, "COPY"),
            Edge(3, 4, "SPLIT"),
            Edge(4, 4, "SPLIT"),
        ],
    )


def test_gold_to_records():
    records = gold_to_records(GOLD)
    assert len(records) == 1490
    counts = {k: sum(1 for r in records if r.fold == k) for k in range(5)}
    assert counts == {0: 279, 1: 275, 2: 336, 3: 317, 4: 283}, counts
    assert sum(len(r.links) for r in records) == 6175
    assert sum(1 for r in records if r.spans) == 153
    from retexo.datasets.gold import GoldPair

    for record, pair in zip(records, GoldPair.load(GOLD)):
        example = record_to_example(record)
        reference = pair.as_example()
        assert example.alignments == reference.alignments, record.id
        assert example.operations == reference.operations, record.id


def test_labels():
    assert labels.canonical("NOP") == ("COPY", "")
    assert labels.canonical("HYPER") == ("SUBST", "HYPER")
    assert labels.to_v1("SPLIT") == "SUBST"
    assert labels.to_mode("MORPH", False) == "VERBATIM"
    assert labels.to_mode("INS", True) == "FRAME"
    assert labels.to_group("NE-SUB") == "lexical-semantic"
    # section 3.1b: one lexical value at most, MORPH features after it, fixed order
    assert labels.parse_detail("SUBST", "HYPER+CASE") == ("HYPER", ("CASE",))
    assert labels.parse_detail("SYN", "NUMBER+CASE") == ("", ("CASE", "NUMBER"))
    assert labels.parse_detail("MORPH", "") == ("", ())
    assert labels.parse_detail("MERGE", "+12") == ("+12", ())
    assert labels.join_detail("HYPER", ("NUMBER", "CASE")) == "HYPER+CASE+NUMBER"
    for op, detail in (
        ("SUBST", "HYPER+HYPO"),
        ("COPY", "CASE"),
        ("SYN", "HYPER"),
        ("SUBST", "FOO"),
        ("MERGE", "12"),
    ):
        try:
            labels.parse_detail(op, detail)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{op} {detail} accepted")
    for op in labels.V3:
        for level in labels.LEVELS:
            assert labels.collapse(level, op) in labels.classes(level), (level, op)
    try:
        labels.collapse("V9", "COPY")
    except ValueError:
        pass
    else:
        raise AssertionError("collapse accepted an unknown level")
    assert labels.token_labels([0, -1, 2], ["COPY", "", "MORPH"], [0, 1, 0], "mode") == [
        "VERBATIM",
        "FRAME",
        "VERBATIM",
    ]
    assert labels.source_labels([0, -1, 2], 3) == ["KEEP", "DEL", "KEEP"]


def test_round_trip_edges():
    record = merge_split_record()
    links, tags, frame, sure = links_of(record)
    assert links == [0, 1, 3, 4, 4]  # the SPLIT keeps both links, the MERGE keeps its first
    assert tags == ["COPY", "MERGE", "COPY", "SPLIT", "SPLIT"]
    extra = extra_edges(record)
    assert [(e.r, e.s) for e in extra] == [(1, 2)]
    back = {(e.r, e.s, e.op) for e in edges_from(links, tags, frame, extra)}
    assert back == {(e.r, e.s, e.op) for e in record.links}


def test_to_script_replays():
    from retexo.core.scriba import Scriba
    from retexo.metrics import ScriptScorer

    record = small_record()
    links, tags, frame, _ = links_of(record)
    pred = Prediction(links=links, tags=tags, frame=frame)
    script = to_script(record, pred)
    assert Scriba().verify(script, record.source_tokens, record.reuse_tokens)
    gold_script = to_script(record, Prediction(links=links, tags=tags, frame=frame))
    metrics = ScriptScorer.evaluate([script], [gold_script], generative=False)
    assert metrics.alignment.f1 == 1.0


def test_floors():
    BaselineRegistry.load_all()
    record = Record(
        id="t/3",
        level="gold",
        fold=4,
        source_work="s",
        source_tokens=["et", "arma", "et", "uirum"],
        reuse_work="r",
        reuse_tokens=["et", "et", "cano", "uirum"],
        pair_label="cit",
        links=[Edge(0, 0, "COPY"), Edge(1, 2, "COPY"), Edge(3, 3, "COPY")],
    )
    cfg = BaselineConfig(device="cpu")
    nothing = BaselineRegistry.get("do_nothing")(cfg)
    pred = nothing.postprocess(record, nothing.predict([record])[0], {})
    assert pred.links == [-1, -1, -1, -1] and pred.tags == ["INS"] * 4
    assert (
        abs(token_accuracy([record], [pred]) - 0.25) < 1e-9
    )  # one of four reuse words is unlinked
    copy = BaselineRegistry.get("copy_input")(cfg)
    pred = copy.postprocess(record, copy.predict([record])[0], {})
    assert pred.links == [0, 2, -1, 3], pred.links  # the second *et* goes to the next unused *et*
    assert pred.tags == ["COPY", "COPY", "INS", "COPY"]


def test_dump_round_trip():
    from retexo.baselines.adapters import rescore_legacy

    record = small_record()
    links, tags, frame, _ = links_of(record)
    pred = Prediction(
        links=links,
        tags=tags,
        frame=frame,
        scores=[[(0, 0.9), (-1, 0.1)], [(2, 0.7), (-1, 0.3)], [(1, 0.6), (-1, 0.4)]],
        meta={"passes": 2},
        raw=None,
    )
    invalid = Prediction(links=[], tags=[], frame=[0, 0, 0], raw="no <- lines here")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.jsonl"
        write_dump([record, record], [pred, invalid], path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for key in (
            "source",
            "target",
            "gold_ops",
            "gold_align",
            "gold_del",
            "ref_type",
            "regime",
            "links",
            "gated_tags",
            "view",
            "top",
            "frames",
            "frame_p",
            "record",
            "pred",
        ):
            assert key in rows[0], key
        rescore_legacy(rows, verbose_name="test")
        back = read_dump(path)
    r0, p0 = back[0]
    assert p0.links == links and p0.tags == tags and p0.frame == frame
    assert p0.scores[0][0] == (0, 0.9) and p0.meta == {"passes": 2} and p0.raw is None
    assert back[1][1].raw == "no <- lines here" and back[1][1].invalid
    assert r0.source_tokens == record.source_tokens and [(e.r, e.s) for e in r0.links] == [
        (0, 0),
        (1, 2),
        (2, 1),
    ]


def test_driver_split():
    from retexo.baselines.splits import split_gold

    records = [
        Record(
            id=f"g/{i}",
            level="gold",
            fold=i % 5,
            source_work="",
            source_tokens=["a"],
            reuse_work="",
            reuse_tokens=["a"],
            pair_label="cit",
        )
        for i in range(25)
    ]
    for k in range(5):
        train, dev, test = split_gold(records, k, (k + 1) % 5)
        assert {r.fold for r in dev} == {(k + 1) % 5}
        assert not ({r.id for r in test} & {r.id for r in train})
        assert {r.id for r in dev} <= {r.id for r in train}


def test_tag_boundary():
    record = small_record()
    links, tags, frame, _ = links_of(record)
    tags = ["COPY", "COPY", "SUBST"]  # uirumque / uirum differ, so SUBST is not forced to NOP
    script = to_script(record, Prediction(links=links, tags=tags, frame=frame))
    back_links, back_tags, back_frame = from_script(script)
    assert "NOP" not in back_tags and back_links == links
    assert back_tags[2] == "SUBST" and back_tags[0] == "COPY"


def test_empty_and_invalid():
    empty = Prediction.empty(3)
    assert empty.links == [-1, -1, -1] and empty.tags == ["", "", ""] and empty.frame == [0, 0, 0]
    invalid = Prediction(links=[], tags=[], frame=[], raw="garbage")
    assert to_script(small_record(), invalid) is None
    BaselineRegistry.load_all()
    nothing = BaselineRegistry.get("do_nothing")(BaselineConfig(device="cpu"))
    pred = nothing.predict([small_record()])[0]
    assert nothing.postprocess(small_record(), pred, {}) is pred


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_harness] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
