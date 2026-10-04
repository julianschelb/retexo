"""The paper format of the record: label conversion both ways and the codec round trip."""

from retexo.baselines.labels import Labels
from retexo.baselines.record import Edge, Record, RecordCodec, Span


def test_to_paper_and_back():
    cases = [("MORPH", "MOOD+TENSE", ("INFLECT", "TENSE+MOOD")),
             ("SYN", "CASE", ("SUBST", "SYN+CASE")),
             ("POS", "", ("SUBST", "POS")),
             ("NE-SUB", "CASE", ("SUBST", "NE-SUB+CASE")),
             ("SYN", "SYN-DIST", ("SUBST", "SYN-DIST")),
             ("SUBST", "HYPER+CASE", ("SUBST", "HYPER+CASE")),
             ("SUBST", "", ("SUBST", "")),
             ("MERGE", "+7", ("MERGE", "+7")),
             ("COPY", "", ("COPY", ""))]
    for op, detail, paper in cases:
        assert Labels.to_paper(op, detail) == paper
        assert Labels.from_paper(*paper) == (op, Labels.normalise_detail(op, detail))


def test_code_spelling_passes_through_from_paper():
    assert Labels.from_paper("MORPH", "CASE") == ("MORPH", "CASE")
    assert Labels.from_paper("SYN", "CASE") == ("SYN", "CASE")


def _record(fmt: str) -> Record:
    prov = {"links": "llm-blind", "fine_ops": "llm-blind-V3"}
    if fmt:
        prov["format"] = fmt
    return Record(id="p1", level="gold", fold=4, source_work="verg. georg.", source_tokens=["a", "b", "c"],
                  reuse_work="hier. epist.", reuse_tokens=["a", "x", "c", "y"], pair_label="cit",
                  links=[Edge(0, 0, "COPY"), Edge(1, 1, "SYN", False, "CASE", None, "llm-blind"),
                         Edge(2, 2, "MORPH", True, "CASE+NUMBER")],
                  spans=[Span(3, 4, "FRAME")], provenance=prov,
                  annotation={"note": "a note", "regime": ["form", "none", "lemma", "none"]},
                  benchmark_id=9, source_meta={"author": "verg", "citation": "<verg. georg. 2.483.1>"},
                  reuse_meta={"author": "hier"})


def test_paper_format_on_disk_code_spelling_in_memory():
    obj = RecordCodec.to_json(_record("2026-09-24"))
    assert [(e["op"], e.get("detail", "")) for e in obj["links"]] == \
        [("COPY", ""), ("SUBST", "SYN+CASE"), ("INFLECT", "CASE+NUMBER")]
    assert obj["links"][1]["sure_by"] == "llm-blind"
    assert obj["note"] == "a note" and obj["lookup"] == {"regime": ["form", "none", "lemma", "none"]}
    assert "annotation" not in obj and obj["benchmark_id"] == 9
    assert list(obj["source"]) == ["author", "work", "citation", "tokens"]
    back = RecordCodec.from_json(obj)
    assert [(e.op, e.detail, e.sure_by) for e in back.links] == \
        [("COPY", "", ""), ("SYN", "CASE", "llm-blind"), ("MORPH", "CASE+NUMBER", "")]
    assert back.annotation["note"] == "a note" and back.source_author == "verg" and back.benchmark_id == 9
    assert RecordCodec.to_json(back) == obj


def test_code_format_unchanged_without_a_format():
    obj = RecordCodec.to_json(_record(""))
    assert [e["op"] for e in obj["links"]] == ["COPY", "SYN", "MORPH"]
    assert obj["annotation"]["note"] == "a note" and "note" not in obj
