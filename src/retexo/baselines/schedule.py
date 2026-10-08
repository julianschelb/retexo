# retexo/baselines/schedule.py
"""The training schedule that every trained system follows: the same examples, in the same stages."""

from __future__ import annotations

import random
from typing import Dict, List

from retexo.baselines.record import Record, RecordCodec


class SharedSchedule:
    """Ours' training schedule, for any method: the same examples, in the same stages.

    Counted in oriented examples (one pair shown in one direction): the synthetic stage is
    one epoch over the 25,000 synthetic pairs in both orientations (50,000); every real pass
    holds the real pairs in both orientations, one negative per real pair, and a fresh draw
    of 600 synthetic examples. A method that already shows each record in both directions
    itself (the span aligner asks every word from both sides) gets one orientation and half
    the negatives and the per-pass draw as records, which is the same count of oriented
    examples.

    Example:
        ```python
        schedule = SharedSchedule(method, train)
        stage_1 = schedule.synthetic_epoch()
        for each pass: batch = schedule.real_pass()
        ```
    """

    PER_PASS = 600

    def __init__(self, method, real, *, both_directions_built_in: bool = False):
        self.built_in = both_directions_built_in
        synthetic = list(method.shared_synthetic)
        both = lambda records: list(records) + [RecordCodec.swapped(r) for r in records]  # noqa: E731
        self.synthetic = synthetic if self.built_in else both(synthetic)
        self.real = list(real) if self.built_in else both(real)
        self.negatives = list(
            method.shared_negatives[: len(real) // 2 if self.built_in else len(real)]
        )
        self.per_pass = self.PER_PASS // 2 if self.built_in else self.PER_PASS
        self.rng = random.Random(99 + method.cfg.seed)

    @staticmethod
    def applies(method) -> bool:
        return bool(method.shared_synthetic or method.shared_negatives)

    def synthetic_epoch(self) -> List[Record]:
        out = list(self.synthetic)
        self.rng.shuffle(out)
        return out

    def real_pass(self) -> List[Record]:
        draw = (
            self.rng.sample(self.synthetic, min(self.per_pass, len(self.synthetic)))
            if self.synthetic
            else []
        )
        return self.real + self.negatives + draw

    def summary(self) -> Dict[str, int]:
        """The counts in oriented examples, identical for every method that follows the schedule."""
        factor = 2 if self.built_in else 1
        return {
            "synthetic_epoch": factor * len(self.synthetic),
            "real_per_pass": factor * len(self.real),
            "negatives_per_pass": factor * len(self.negatives),
            "synthetic_per_pass": factor * self.per_pass,
        }


__all__ = ["SharedSchedule"]
