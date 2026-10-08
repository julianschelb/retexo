# tests/test_curriculum.py
"""Row 8, version 1: the error-driven resampling of the pool."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.curriculum import ErrorCurriculum  # noqa: E402
from retexo.formulations.change_detector import ChangeExample  # noqa: E402


def example(aligns, fine):
    n = len(aligns)
    return ChangeExample(
        source_tokens=["s"] * 4,
        target_tokens=["t"] * n,
        labels=[0] * n,
        operations=["NOP"] * n,
        n_operations=1,
        alignments=list(aligns),
        fine_operations=list(fine),
    )


class StubModel:
    """Scores that put the argmax on the gold link for every word except the ones named."""

    def __init__(self, wrong: dict):
        self.wrong = wrong  # example index -> set of reuse positions answered wrongly
        self.calls = 0

    def predict_alignment_scores(self, examples):
        self.calls += 1
        out = []
        for ex in examples:
            rows = []
            bad = self.wrong.get(id(ex), set())
            for t, gold in enumerate(ex.alignments):
                best = (gold + 1) % 4 if t in bad else gold
                rows.append([(best, 0.9), (-1, 0.1)] if best >= 0 else [(-1, 0.9), (0, 0.1)])
            out.append(rows)
        return out


def test_example_error_weights_the_focus_operations_twice():
    ex = example([0, 1, -1, 3], ["NOP", "SUBST", "INS", "MORPH"])
    rows = [
        [(0, 0.9)],
        [(2, 0.9), (1, 0.1)],
        [(-1, 0.9)],
        [(3, 0.9)],
    ]  # only the SUBST word is wrong
    assert abs(ErrorCurriculum.example_error(ex, rows) - 2 / 6) < 1e-9
    rows[0] = [(1, 0.9)]  # a NOP word wrong too
    assert abs(ErrorCurriculum.example_error(ex, rows) - 3 / 6) < 1e-9


def test_draw_prefers_the_pairs_the_model_gets_wrong_and_keeps_a_random_floor():
    pool = [example([0, 1, 2], ["NOP", "SUBST", "NOP"]) for _ in range(40)]
    hard = {id(pool[i]): {1} for i in range(5)}  # five pairs with a wrong SUBST word
    model = StubModel(hard)
    cur = ErrorCurriculum(floor=0.5, candidates=40, seed=1)
    batch = cur.draw(model, pool, 10)
    assert len(batch) == 10 and model.calls >= 1
    by_error = [ex for ex in batch if id(ex) in hard]
    assert len(by_error) >= 4  # the error share lands on the hard five
    assert cur.history[-1]["with_error"] == 5 and cur.history[-1]["drawn"] == 5
    cur0 = ErrorCurriculum(floor=1.0, candidates=40, seed=1)
    assert len(cur0.draw(model, pool, 10)) == 10 and not cur0.history  # all random: no scoring
