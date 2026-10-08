# retexo/baselines/curriculum.py
"""Error-driven resampling of the synthetic pool (failure-mode dry run, row 8, version 1).

Each gold pass of the typed pointer adds ``pool_per_pass`` synthetic pairs drawn at random from
the pool. ``ErrorCurriculum`` draws them by the model's *current* errors instead: a random
candidate slice of the pool is scored, every pair gets an error -- the share of its reuse words
whose argmax source is wrong (a linked word pointing elsewhere or at the null; an inserted word
pointing at a source word), with the substitution-type and MORPH words counted twice so the
classes the run cares about weigh more -- and the pass's pairs are drawn with probability
``floor / n + (1 - floor) * error / sum(error)``. The floor keeps a random share so the
distribution never collapses onto a few pairs, and the error is capped so a pair the generator
labelled badly cannot dominate.

    curriculum = ErrorCurriculum(floor=0.5, candidates=6000, cap=0.6, seed=1)
    batch = curriculum.draw(model, pool, k=600, log=log)      # k pairs for the next pass
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

#: The fine operations whose words count twice in a pair's error.
FOCUS_OPS = {"SUBST", "SYN", "POS", "NE-SUB", "HYPER", "HYPO", "ANT", "MORPH", "SPLIT", "MERGE"}


class ErrorCurriculum:
    """Draw the next pass's synthetic pairs where the model is wrong now.

    Example:
        ```python
        cur = ErrorCurriculum(floor=0.5, candidates=200, seed=1)
        errors = cur.errors(model, pool[:200])          # one float per example, 0 = all argmaxes right
        batch = cur.draw(model, pool, k=50)
        ```
    """

    def __init__(
        self, *, floor: float = 0.5, candidates: int = 6000, cap: float = 0.6, seed: int = 1
    ):
        self.floor = min(max(float(floor), 0.0), 1.0)
        self.candidates = int(candidates)
        self.cap = float(cap)
        self.rng = random.Random(777 + seed)
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def example_error(example, rows: Sequence[Sequence]) -> float:
        """The weighted share of wrong argmaxes over the reuse words the rows cover."""
        aligns = list(getattr(example, "alignments", None) or [])
        fine = list(getattr(example, "fine_operations", None) or [])
        wrong = total = 0.0
        for t, row in enumerate(rows):
            if t >= len(aligns) or not row:
                continue
            best = max(row, key=lambda x: x[1])[0]
            gold = aligns[t]
            weight = 2.0 if t < len(fine) and fine[t] in FOCUS_OPS else 1.0
            total += weight
            if best != gold:
                wrong += weight
        return wrong / total if total else 0.0

    def errors(self, model, examples: Sequence, batch_size: int = 64) -> List[float]:
        out: List[float] = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            rows = model.predict_alignment_scores(chunk)
            out.extend(min(self.example_error(ex, r), self.cap) for ex, r in zip(chunk, rows))
        return out

    def draw(self, model, pool: Sequence, k: int, *, log=None) -> List:
        """``k`` pairs of the pool: a random floor share plus an error-weighted share drawn
        from a scored candidate slice."""
        k = min(k, len(pool))
        if k <= 0:
            return []
        n_random = int(round(self.floor * k))
        n_error = k - n_random
        chosen = self.rng.sample(list(pool), n_random) if n_random else []
        if n_error > 0:
            cands = self.rng.sample(list(pool), min(self.candidates, len(pool)))
            errs = self.errors(model, cands)
            total = sum(errs)
            if total > 0:
                weights = [e / total for e in errs]
                picked = set()
                # weighted draws without replacement, capped at the candidates that carry error
                order = self.rng.choices(range(len(cands)), weights=weights, k=n_error * 4)
                for i in order:
                    if errs[i] > 0 and i not in picked:
                        picked.add(i)
                        if len(picked) >= n_error:
                            break
                chosen += [cands[i] for i in picked]
                mean = total / len(errs)
                self.history.append(
                    {
                        "candidates": len(cands),
                        "mean_error": round(mean, 4),
                        "with_error": sum(1 for e in errs if e > 0),
                        "drawn": len(picked),
                    }
                )
                if log:
                    log(
                        f"[curriculum] {len(cands)} candidates scored, mean error {mean:.3f}, "
                        f"{self.history[-1]['with_error']} with an error, {len(picked)} drawn by error + {n_random} at random"
                    )
            else:
                chosen += self.rng.sample(cands, min(n_error, len(cands)))
        return chosen
