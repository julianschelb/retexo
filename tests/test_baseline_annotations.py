# tests/test_baseline_annotations.py
"""The annotation reader, checker and agreement on toy records; no data, no models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.annotations import (  # noqa: E402
    AnnotationAgreement,
    AnnotationChecker,
    AnnotationReader,
)
from retexo.baselines.record import Edge, Record, Span  # noqa: E402


def _gold() -> Record:
    return Record(
        id="p1",
        level="gold",
        fold=0,
        source_work="verg",
        source_tokens="Arma virumque cano, Troiae qui".split(),
        reuse_work="hier",
        reuse_tokens="ut ait poeta: arma virum canit qui venit".split(),
        pair_label="cit",
        links=[Edge(3, 0, "COPY"), Edge(4, 1, "SPLIT"), Edge(5, 2, "MORPH"), Edge(6, 4, "COPY")],
        spans=[Span(0, 3)],
        provenance={"links": "llm-silver"},
        annotation={},
    )


def test_reader_merges_and_normalises():
    reader = AnnotationReader({"p1": _gold()})
    record = reader.merge(
        _gold(),
        {
            "id": "p1",
            "links": [
                {"r": 3, "s": 0, "op": "copy"},
                {"r": 4, "s": 1, "op": "SUBST", "detail": "SYN-DIST"},
                {"r": 5, "s": 2, "op": "NOP", "sure": False},
            ],
            "spans": [{"start": 0, "end": 3, "label": "FRAME"}],
            "note": "x",
        },
    )
    assert [e.op for e in record.links] == ["COPY", "SYN", "COPY"]
    assert record.links[1].detail == "SYN-DIST" and record.links[2].sure is False
    assert record.reuse_tokens == _gold().reuse_tokens and record.provenance["links"] == "llm-blind"
    app = reader.merge(
        _gold(),
        {
            "id": "p1",
            "words": [
                {"t": 0, "op": "FRAME", "source": -1},
                {"t": 1, "op": "FRAME", "source": -1},
                {"t": 3, "op": "HYPER", "source": 0},
                {"t": 7, "op": "INS", "source": -1},
            ],
        },
    )
    assert [(e.r, e.s, e.op, e.detail) for e in app.links] == [(3, 0, "SUBST", "HYPER")]
    assert [(s.start, s.end) for s in app.spans] == [(0, 2)]


def test_checker():
    record = _gold()
    assert [p for p in AnnotationChecker().check(record) if p.severity == "error"] == []
    bad = _gold()
    bad.links += [
        Edge(3, 4, "COPY"),
        Edge(7, 0, "MORPH"),
        Edge(2, 2, "COPY"),
        Edge(1, 3, "SUBST", detail="HYPER+HYPO"),
    ]
    messages = [p.message for p in AnnotationChecker().check(bad) if p.severity == "error"]
    assert any("linked twice" in m for m in messages)
    assert any("s=0 linked by" in m for m in messages)
    assert any("inside a FRAME span" in m for m in messages)
    assert any("at most one lexical detail" in m for m in messages)
    warns = [p.message for p in AnnotationChecker().check(bad) if p.severity == "warn"]
    assert any("MORPH 'Arma' / 'venit' share no stem" in m for m in warns)


def test_agreement():
    system = _gold()
    system.links = [
        Edge(3, 0, "COPY"),
        Edge(4, 1, "SPLIT"),
        Edge(5, 2, "SUBST"),
        Edge(7, 3, "SUBST"),
    ]
    system.spans = [Span(0, 2)]
    a = AnnotationAgreement().score([system], [_gold()])
    assert a.pairs == 1 and a.shared_links == 3
    assert abs(a.link_precision - 3 / 4) < 1e-9 and abs(a.link_recall - 3 / 4) < 1e-9
    assert abs(a.op_accuracy_v3 - 2 / 3) < 1e-9 and abs(a.frame_recall - 2 / 3) < 1e-9
    assert (
        a.per_op["COPY"]["P"] == 1.0
        and a.per_op["COPY"]["R"] == 0.5
        and a.confusion[("SUBST", "MORPH")] == 1
    )


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_annotations] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
