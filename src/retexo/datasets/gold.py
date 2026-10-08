# retexo/datasets/gold.py
"""
The hand-labelled gold: loading it, and scoring predictions against it.

Extracted from the preliminary experiment module E3 (``retexo/e3.py``, now
``attic/retexo/e3.py``), whose parallel-generation half is superseded by
``retexo.datasets.synthetic`` and is not carried forward. ``GoldPair`` and
:class:`TypedScorer` are load-bearing for the baselines harness
(``retexo.baselines.record``, ``retexo.baselines.adapters``) and are
kept verbatim under a name that says what they are rather than which
experiment first needed them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from retexo.formulations.change_detector import ChangeExample

# =============================================================================
# The hand-labelled pairs
# =============================================================================


@dataclass
class GoldPair:
    """A hand-labelled real pair, in the same shape as a generated example.

    Example:
        ```python
        pairs = GoldPair.load("data/gold_full")
        example = pairs[0].as_example()
        ```
    """

    id: str
    ref_type: str
    source_tokens: List[str]
    target_tokens: List[str]
    target_ops: List[str]
    source_del: List[int]
    #: Per target token, the source token it was aligned to by hand, -1 for
    #: none. The labels always carried this -- they are triples of
    #: [source, target, operation] -- and it was being dropped on load, which
    #: is why the pointer looked as though it could only be scored on generated
    #: data. It can be scored on Jerome.
    target_align: Optional[List[int]] = None
    note: str = ""

    def as_example(self) -> ChangeExample:
        return ChangeExample(
            source_tokens=self.source_tokens,
            target_tokens=self.target_tokens,
            labels=[0 if op == "NOP" else 1 for op in self.target_ops],
            operations=self.target_ops,
            n_operations=sum(1 for op in self.target_ops if op != "NOP"),
            source_labels=self.source_del,
            source_operations=["DEL" if d else "NOP" for d in self.source_del],
            alignments=self.target_align,
        )

    @classmethod
    def load(cls, directory: Path) -> List[GoldPair]:
        """Expand the exception-format hand labels into per-token operations.

        Unlisted target tokens are ``INS``, unlisted source tokens are ``DEL`` --
        which is what makes the format compact, and is also the honest default:
        in these pairs most tokens really are unaligned.
        """
        directory = Path(directory)
        pairs = {p["id"]: p for p in json.loads((directory / "sample_pairs.json").read_text())}
        labels = json.loads((directory / "labels.json").read_text())

        out: List[GoldPair] = []
        for key, label in labels.items():
            pair = pairs[key]
            source, target = pair["source"].split(), pair["target"].split()
            target_ops = ["INS"] * len(target)
            target_align = [-1] * len(target)
            source_del = [1] * len(source)
            for start, end in label.get("frame", []):
                for j in range(start, min(end + 1, len(target))):
                    target_ops[j] = "FRAME"
            for src, tgt, op in label["alignments"]:
                if 0 <= tgt < len(target):
                    target_ops[tgt] = op
                    if 0 <= src < len(source):
                        target_align[tgt] = src
                if 0 <= src < len(source):
                    source_del[src] = 0
            out.append(
                cls(
                    id=key,
                    ref_type=pair["ref_type"],
                    source_tokens=source,
                    target_tokens=target,
                    target_ops=target_ops,
                    source_del=source_del,
                    target_align=target_align,
                    note=label.get("note", ""),
                )
            )
        return out


# =============================================================================
# Scoring
# =============================================================================


class TypedScorer:
    """Per-operation and macro scores of predicted operation sequences against
    the gold's.

    Example:
        ```python
        scores = TypedScorer.score_typed(gold_ops, predicted_ops, classes)
        ```
    """

    @staticmethod
    def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "support": tp + fn,
        }

    @classmethod
    def score_typed(cls, gold_ops, predicted_ops, classes) -> Dict[str, object]:
        """Per-operation and macro scores over aligned token sequences."""
        counts = {c: [0, 0, 0] for c in classes}
        correct = total = 0
        confusion = {g: {p: 0 for p in classes} for g in classes}
        for gold, predicted in zip(gold_ops, predicted_ops):
            for g, p in zip(gold, predicted):
                if g not in counts or p not in counts:
                    continue
                total += 1
                confusion[g][p] += 1
                if g == p:
                    correct += 1
                    counts[g][0] += 1
                else:
                    counts[g][2] += 1
                    counts[p][1] += 1
        per = {c: cls.prf(*counts[c]) for c in classes}
        present = [c for c in classes if per[c]["support"] > 0]
        return {
            "accuracy": correct / total if total else 0.0,
            "macro_f1": sum(per[c]["f1"] for c in present) / len(present) if present else 0.0,
            "per_operation": per,
            "confusion": confusion,
            "tokens": total,
        }


__all__ = ["GoldPair", "TypedScorer"]
