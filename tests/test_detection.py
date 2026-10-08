"""Tests for the windowed detection score."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.core.oracle import EditPlanOracle  # noqa: E402
from retexo.datasets.detection import DetectionScore  # noqa: E402
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
TEACHER = RelabellingTeacher()
SOURCE = "decem tulerunt fastidia menses".split()
FRAMED = (
    "quae autem potest maior esse clementia quam ut decem tulerunt fastidia menses laus deo"
).split()


def test_frames_do_not_count_as_distance():
    script, span = TEACHER.relabel_pair(SOURCE, FRAMED, ORACLE)
    bare = ORACLE.plan_tokens(SOURCE, SOURCE)
    dense_framed = DetectionScore.window_density(script, span)
    dense_bare = DetectionScore.window_density(bare)
    check("framing adds little distance", dense_framed < dense_bare + 1.5)
    check(
        "windows read off the frames",
        DetectionScore.window_widths(script, span)[1] == span[1] - span[0],
    )


def test_unrelated_pair_scores_higher():
    related, span_r = TEACHER.relabel_pair(SOURCE, FRAMED, ORACLE)
    unrelated_target = (
        "arma uirumque cano Troiae qui primus ab oris Italiam fato profugus Lauiniaque uenit"
    ).split()
    unrelated, span_u = TEACHER.relabel_pair(SOURCE, unrelated_target, ORACLE)
    check(
        "related is cheaper per token",
        DetectionScore.window_density(related, span_r)
        < DetectionScore.window_density(unrelated, span_u),
    )


def test_no_frames_falls_back_to_full_lengths():
    bare = ORACLE.plan_tokens(SOURCE, "decem mensum fastidia".split())
    source_window, target_window = DetectionScore.window_widths(bare)
    check("no frames -> full lengths", source_window == len(SOURCE) and target_window == 3)


for fn in [
    test_frames_do_not_count_as_distance,
    test_unrelated_pair_scores_higher,
    test_no_frames_falls_back_to_full_lengths,
]:
    fn()

print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
