# tests/test_unlinked.py
"""The words without a link and the patterns (retexo.operations.unlinked).

Hand-made pairs pin each rule; with the word-level annotation present, the totals of the paper's Table 1 are
reproduced on both revisions of the annotation, and the record codec writes the revised file back unchanged.
"""

import json

import pytest

from retexo.baselines.record import RecordCodec
from retexo.operations.unlinked import QUOTE_MIN, lis_positions, patterns, unlinked
from retexo.paths import home

GOLD = home() / "data" / "gold_full"


def link(r, s, op="NOP"):
    return {"r": r, "s": s, "op": op}


def ops(words):
    return [(w["i"], w["op"]) for w in words]


# =============================================================================
# Unlinked words
# =============================================================================


def test_words_between_the_first_and_last_link_are_inside_the_stretch():
    words = unlinked(6, 6, [link(1, 1), link(4, 4)])
    assert ops(words["reuse"]) == [(0, "NEW"), (2, "INS"), (3, "INS"), (5, "NEW")]
    assert ops(words["source"]) == [(0, "OUT"), (2, "DEL"), (3, "DEL"), (5, "OUT")]


def test_a_citing_formula_is_mark_wherever_it_stands():
    words = unlinked(6, 3, [link(4, 0), link(5, 1)], [{"start": 0, "end": 2, "label": "MARK"}])
    assert ops(words["reuse"]) == [(0, "MARK"), (1, "MARK"), (2, "NEW"), (3, "NEW")]


def test_a_linked_word_inside_a_citing_formula_keeps_its_link():
    words = unlinked(3, 3, [link(1, 1)], [{"start": 0, "end": 3, "label": "FRAME"}])
    assert ops(words["reuse"]) == [(0, "MARK"), (2, "MARK")]


def test_a_merge_leaves_its_second_source_word_unclaimed():
    words = unlinked(3, 4, [link(0, 0), link(1, 1, "MERGE"), link(2, 3)])
    assert ops(words["source"]) == [(2, "DEL")]


def test_a_split_claims_its_source_word_once():
    words = unlinked(3, 2, [link(0, 0), link(1, 1, "SPLIT"), link(2, 1, "SPLIT")])
    assert words == {"reuse": [], "source": []}


def test_two_separate_stretches_count_the_gap_as_inside():
    words = unlinked(7, 7, [link(0, 0), link(1, 1), link(5, 5), link(6, 6)])
    assert [op for _, op in ops(words["reuse"])] == ["INS"] * 3


def test_without_links_every_word_is_new_text_or_not_taken_up():
    words = unlinked(3, 2, [])
    assert {op for _, op in ops(words["reuse"])} == {"NEW"}
    assert {op for _, op in ops(words["source"])} == {"OUT"}


# =============================================================================
# Patterns
# =============================================================================


def test_a_quote_needs_quote_min_in_order_copies():
    tokens = [f"w{i}" for i in range(8)]
    four = [link(i, i) for i in range(QUOTE_MIN)]
    assert patterns(tokens, four)["QUOTE"] == [{"start": 0, "end": QUOTE_MIN}]
    assert patterns(tokens, four[:-1])["QUOTE"] == []


def test_an_inflection_or_a_jump_in_the_source_breaks_a_quote():
    tokens = [f"w{i}" for i in range(8)]
    assert (
        patterns(tokens, [link(0, 0), link(1, 1), link(2, 2, "MORPH"), link(3, 3), link(4, 4)])[
            "QUOTE"
        ]
        == []
    )
    assert patterns(tokens, [link(0, 0), link(1, 1), link(2, 5), link(3, 6)])["QUOTE"] == []


def test_paper_spelling_counts_as_a_copy():
    tokens = [f"w{i}" for i in range(4)]
    assert patterns(tokens, [link(i, i, "COPY") for i in range(4)])["QUOTE"] == [
        {"start": 0, "end": 4}
    ]


def test_reorder_marks_the_words_outside_the_longest_chain():
    # Ecl. 4.61 -> Ep. 21.2.5: decem mensum fastidia sustineret against decem tulerunt fastidia menses
    found = patterns(
        ["decem", "mensum", "fastidia", "sustineret"],
        [link(0, 8), link(1, 11, "MORPH"), link(2, 10), link(3, 9, "SUBST")],
    )
    assert len(found["REORDER"]) == 2
    assert lis_positions([8, 11, 10, 9]) in ({0, 1}, {0, 2}, {0, 3})


def test_disperse_lists_clause_breaks_before_the_last_linked_word():
    tokens = ["arma", "virumque.", "cano", "Troiae."]
    assert patterns(tokens, [link(0, 0), link(2, 1)])["DISPERSE"] == [1]
    # a clause break on the last linked word itself does not disperse the reuse
    assert patterns(tokens, [link(2, 0), link(3, 1)])["DISPERSE"] == []


# =============================================================================
# The annotation: Table 1 and the codec
# =============================================================================

TABLE_1 = {
    "INS": 2480,
    "NEW": 28646,
    "MARK": 872,
    "DEL": 2824,
    "OUT": 29700,
    "QUOTE pairs": 324,
    "REORDER": 533,
    "DISPERSE pairs": 71,
}


def _totals(path):
    totals = dict.fromkeys(TABLE_1, 0)
    for line in path.open(encoding="utf-8"):
        record = RecordCodec.from_json(json.loads(line))
        script = [{"r": e.r, "s": e.s, "op": e.op} for e in record.links]
        spans = [{"start": sp.start, "end": sp.end} for sp in record.spans]
        words = unlinked(len(record.reuse_tokens), len(record.source_tokens), script, spans)
        for w in words["reuse"] + words["source"]:
            totals[w["op"]] += 1
        found = patterns(record.reuse_tokens, script)
        totals["QUOTE pairs"] += bool(found["QUOTE"])
        totals["REORDER"] += len(found["REORDER"])
        totals["DISPERSE pairs"] += bool(found["DISPERSE"])
    return totals


@pytest.mark.parametrize("name", ["gold_2026-09-24.records.jsonl", "gold_2026-10-12.records.jsonl"])
def test_the_annotation_reproduces_table_1(name):
    path = GOLD / name
    if not path.exists():
        pytest.skip(f"needs {name} under $RETEXO_HOME/data/gold_full")
    assert _totals(path) == TABLE_1


def test_the_revised_annotation_is_written_back_unchanged():
    path = GOLD / "gold_2026-10-12.records.jsonl"
    if not path.exists():
        pytest.skip("needs gold_2026-10-12.records.jsonl under $RETEXO_HOME/data/gold_full")
    for line in path.open(encoding="utf-8"):
        obj = json.loads(line)
        assert RecordCodec.to_json(RecordCodec.from_json(obj)) == obj
