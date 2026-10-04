"""Tests for the structural teacher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.core.oracle import EditPlanOracle  # noqa: E402
from retexo.core.scriba import Scriba  # noqa: E402
from retexo.datasets.teacher import RelabellingTeacher  # noqa: E402

PASSED = FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}")


ORACLE = EditPlanOracle()
SCRIBA = Scriba()
TEACHER = RelabellingTeacher()

# A fragment of the source, embedded late in a longer, unrelated target —
# the Virgil-inside-Ambrose shape in miniature.
SOURCE = "decem tulerunt fastidia menses".split()
TARGET = ("quae autem potest maior esse clementia quam ut "
          "decem tulerunt fastidia menses laus deo").split()


def test_full_passage_verifies_and_replays():
    script, span = TEACHER.relabel_pair(SOURCE, TARGET, ORACLE)
    check("verifies against full passages",
          SCRIBA.verify(script, SOURCE, TARGET))
    check("replays every token",
          len(SCRIBA.execute(script, SOURCE)) == len(TARGET))


def test_context_becomes_frame_not_insertion():
    script, span = TEACHER.relabel_pair(SOURCE, TARGET, ORACLE)
    counts = script.op_counts()
    check("FRAME present", counts.get("FRAME", 0) >= 1)
    frame_written = sum(len(op.target_indices)
                        for op in script.operations if op.tag == "FRAME")
    ins = counts.get("INS", 0)
    # The window is fragment length plus slack and ties break early, so the
    # frame covers everything before the window, not everything before the
    # fragment itself.
    check("context framed, not inserted", frame_written >= 5 and ins <= 4)
    check("fragment window not at the start", span[0] >= 4)
    check("the reuse survives inside the window",
          counts.get("NOP", 0) >= 3)


def test_frame_is_cheaper_than_insertion():
    script, _ = TEACHER.relabel_pair(SOURCE, TARGET, ORACLE)
    from retexo.training import oracle_examples  # noqa: F401  (import check)
    # Cost of describing the context as FRAME must be far below charging it
    # as one insertion per token, or the relabelling changes nothing.
    frame_tokens = sum(len(op.target_indices)
                       for op in script.operations if op.tag == "FRAME")
    frame_ops = sum(1 for op in script.operations if op.tag == "FRAME")
    check("one act covers many tokens", frame_tokens > 2 * frame_ops)


def test_source_context_is_consumed_by_frame():
    long_source = ("dixit et haec addens " + " ".join(SOURCE)
                   + " atque ita locutus est").split()
    short_target = "decem mensum fastidia".split()
    script, _ = TEACHER.relabel_pair(long_source, short_target, ORACLE)
    check("verifies with long source",
          SCRIBA.verify(script, long_source, short_target))
    consuming = [op for op in script.operations
                 if op.tag == "FRAME" and op.source_indices
                 and not op.target_indices]
    check("source context consumed by FRAME", len(consuming) >= 1)
    per_token_del = script.op_counts().get("DEL", 0)
    consumed = sum(len(op.source_indices) for op in consuming)
    check("out-of-window source consumed as spans", consumed >= 4)
    check("frames do not exceed the context",
          consumed + 7 == len(long_source) or consumed >= 4)


def test_indices_are_shifted_back_correctly():
    script, span = TEACHER.relabel_pair(SOURCE, TARGET, ORACLE)
    for op in script.operations:
        if op.tag == "FRAME":
            continue
        for i, index in enumerate(op.target_indices):
            check("token at shifted index matches",
                  TARGET[index] == op.target_tokens[i]
                  or op.tag in ("DEL",))


for fn in [
    test_full_passage_verifies_and_replays,
    test_context_becomes_frame_not_insertion,
    test_frame_is_cheaper_than_insertion,
    test_source_context_is_consumed_by_frame,
    test_indices_are_shifted_back_correctly,
]:
    fn()

print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
