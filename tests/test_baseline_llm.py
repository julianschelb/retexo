# tests/test_baseline_llm.py
"""Note 23's line format on hand-made replies: the parser, the gold script, the
prediction it becomes, the prompt variants, the cost tally; no model is called."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.llm import (  # noqa: E402
    PROMPT_DEFAULTS,
    CostTally,
    LLMPromptOnly,
    PassageWriter,
    ScriptFormat,
    ScriptParser,
    ScriptRater,
)
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.llm.script_prompts import (  # noqa: E402
    COARSE_OPS,
    FINE_OPS,
    RATER_SYSTEM,
    PromptVariants,
)


def record(source, reuse, edges=(), spans=()):
    return Record(
        id="t/1",
        level="gold",
        fold=4,
        source_work="",
        source_tokens=source,
        reuse_work="",
        reuse_tokens=reuse,
        pair_label="cit",
        links=list(edges),
        spans=list(spans),
    )


SOURCE = ["Platonis", "vita,", "mortis", "sapienti"]
REUSE = ["Ut", "ait", "Plato,", "uitam", "mortis.", "sapienti", "mortis"]


# =============================================================================
# 1. The parser
# =============================================================================


def test_parse_words_reads_the_three_line_reply_of_the_prompt_example():
    text = "Platonis <- -  FRAME\nuitam <- vita,  MORPH\nmortis. <- mortis  NOP\nsapienti <- -  INS"
    links, ops, seen = ScriptFormat.parse_words(
        text, ["vita,", "mortis"], ["Platonis", "uitam", "mortis.", "sapienti"]
    )
    assert seen
    assert links == [-1, 0, 1, -1]
    assert ops == ["FRAME", "MORPH", "NOP", "INS"]


def test_parse_words_matches_a_repeated_word_to_its_next_free_occurrence_and_ignores_prose():
    text = (
        "Here is the script:\n"
        "* Ut <- -  FRAME\n- ait <- -  FRAME\nPlato, <- Platonis  MORPH\n"
        "uitam <- vita,  MORPH\nmortis. <- mortis  NOP\nsapienti <- sapienti  NOP\nmortis <- mortis  NOP\n"
    )
    links, ops, seen = ScriptFormat.parse_words(text, SOURCE, REUSE)
    assert links == [-1, -1, 0, 1, 2, 3, -1]  # the second "mortis" finds no free source word
    assert ops[:3] == ["FRAME", "FRAME", "MORPH"] and ops[6] == "NOP"


def test_parse_words_without_an_arrow_is_not_seen():
    links, ops, seen = ScriptFormat.parse_words("I cannot do this.", SOURCE, REUSE)
    assert not seen and ops is None and links == [-1] * len(REUSE)


def test_gold_script_round_trips_through_the_parser():
    links, ops = [-1, -1, 0, 1, 2, 3, -1], ["FRAME", "FRAME", "MORPH", "MORPH", "NOP", "NOP", "INS"]
    text = ScriptFormat.gold_script(SOURCE, REUSE, links, ops)
    assert text.splitlines()[2] == "Plato, <- Platonis  MORPH"
    back_links, back_ops, seen = ScriptFormat.parse_words(text, SOURCE, REUSE)
    assert seen and back_links == links and back_ops == ops


# =============================================================================
# 2. Reply to prediction
# =============================================================================


def test_to_prediction_canonicalises_types_and_reads_frame_lines():
    rec = record(SOURCE, REUSE)
    text = "Ut <- -  FRAME\nait <- -  FRAME\nPlato, <- Platonis  MORPH\nuitam <- vita,  HYPER\nmortis. <- mortis  NOP\nsapienti <- -  INS\nmortis <- -  INS"
    pred = ScriptParser.to_prediction(text, rec)
    assert pred.meta["parsed"] and not pred.invalid
    assert pred.links == [-1, -1, 0, 1, 2, -1, -1]
    assert pred.tags == ["", "", "MORPH", "SUBST", "COPY", "", ""]
    assert pred.frame == [1, 1, 0, 0, 0, 0, 0]
    stripped = ScriptParser.to_prediction(text, rec, links_only=True)
    assert stripped.tags[3] == "SUBST" and stripped.tags[2] == "MORPH"


def test_an_unparsable_reply_is_an_invalid_prediction_that_keeps_the_text():
    pred = ScriptParser.to_prediction("no script here", record(SOURCE, REUSE))
    assert pred.invalid and pred.raw == "no script here" and pred.meta["parsed"] is False


# =============================================================================
# 3. Prompts and the rater's completion
# =============================================================================


def test_prompt_variants_carry_the_gate_and_the_inventory_they_are_named_for():
    system_p4, user = PromptVariants.get("P4")
    assert (
        "IMPORTANT -- there may be no reuse at all" in system_p4
        and "Two identical words are not a stretch" in system_p4
    )
    assert "HYPER" in system_p4 and "{source_annotated}" in user and "{n_lines_note}" in user
    system_p3, _ = PromptVariants.get("P3")
    assert (
        "REUSED:" in system_p3
        and "HYPER" not in system_p3.split("The operations")[1].split("Each word")[0]
    )
    assert COARSE_OPS in RATER_SYSTEM and FINE_OPS not in RATER_SYSTEM
    try:
        PromptVariants.get("P9")
        assert False, "unknown variant accepted"
    except KeyError:
        pass


def test_the_prompt_only_row_builds_a_user_turn_with_both_passages_and_the_line_count():
    method = LLMPromptOnly(BaselineConfig(extra={"annotate": "none"}))
    assert method.dials["prompt"] == PROMPT_DEFAULTS["prompt"] == "P4"
    system, user = method.prompt_for(record(SOURCE, REUSE))
    assert "  1: vita," in user and "  3: uitam" in user and "That is 7 lines" in user
    assert system.startswith("You are a Latin philologist")


def test_the_rater_writes_the_gold_completion_in_the_coarse_inventory():
    rec = record(
        SOURCE,
        REUSE,
        edges=[Edge(2, 0, "MORPH"), Edge(3, 1, "SYN"), Edge(4, 2, "COPY")],
        spans=[Span(0, 2, "FRAME")],
    )
    lines = ScriptRater.completion(rec).splitlines()
    assert lines[0] == "Ut <- -  FRAME" and lines[2] == "Plato, <- Platonis  MORPH"
    assert (
        lines[3] == "uitam <- vita,  SUBST"
        and lines[4] == "mortis. <- mortis  NOP"
        and lines[6] == "mortis <- -  INS"
    )
    assert ScriptRater.completion(rec, "fine").splitlines()[3] == "uitam <- vita,  SYN"


def test_plain_passages_number_the_words():
    assert PassageWriter("none")(["a", "b"]) == "  0: a\n  1: b"


# =============================================================================
# 4. The cost tally
# =============================================================================


def test_cost_tally_prices_cache_reads_at_a_tenth():
    class Usage:
        input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = (
            1_000_000,
            0,
            1_000_000,
            0,
        )

    tally = CostTally("claude-sonnet-5")
    tally.add(Usage(), parsed=True)
    assert abs(tally.dollars() - (3.0 + 0.3)) < 1e-9 and tally.parsed == 1 and tally.n == 1
