# datasets/synthetic/pools.py
"""Pool material: attribution frames from the gold labels, and selection with a rare-tag floor."""

from __future__ import annotations

from typing import Dict, List

from retexo.datasets.synthetic.inventory import RARE_TAGS

# =============================================================================
# Frame templates
# =============================================================================


def frame_pool(gold_dir, folds_by_id: Dict[str, int], exclude_fold: int) -> List[List[str]]:
    """Attribution formulas from the hand labels, minus the held-out fold.

    The frames are the citing author's own words -- *quod et illustris poeta
    testatur dicens:* -- and the only honest source of their shape. Taking them
    from the folds the model trains on anyway keeps the held-out frames unseen.
    """
    import json
    from pathlib import Path

    gold_dir = Path(gold_dir)
    pairs = {p["id"]: p for p in json.loads((gold_dir / "sample_pairs.json").read_text())}
    labels = json.loads((gold_dir / "labels.json").read_text())
    out = []
    for key, label in labels.items():
        if folds_by_id.get(key) == exclude_fold:
            continue
        target = pairs[key]["target"].split()
        for a, b in label.get("frame", []):
            span = target[a : b + 1]
            if 1 <= len(span) <= 14:
                out.append(span)
    return out


# =============================================================================
# Pool selection
# =============================================================================


def select_pool(pool, size: int, rng, rare_min: int = 1200):
    """The training pairs: a random sample, with rare relations guaranteed.

    For each rare tag, up to ``rare_min`` pairs that contain it are kept
    first; the rest of the budget is a plain random sample. This moves the
    pool's substitution share above the calibrated 0.228 -- the realism report
    prints where it lands -- and it is the price of a typer that has seen a
    few hundred antonyms rather than fifty.
    """
    pool = list(pool)
    rng.shuffle(pool)
    chosen, taken = [], set()
    for tag in RARE_TAGS:
        n = 0
        for k, ex in enumerate(pool):
            if n >= rare_min or len(chosen) >= size:
                break
            if k in taken or tag not in ex.fine_operations:
                continue
            chosen.append(ex)
            taken.add(k)
            n += 1
    for k, ex in enumerate(pool):
        if len(chosen) >= size:
            break
        if k not in taken:
            chosen.append(ex)
            taken.add(k)
    rng.shuffle(chosen)
    return chosen


def balanced_subset(held, size: int, rng, per_rare: int = 60):
    """A held-out set where every fine class has enough links to score."""
    return select_pool(held, size, rng, rare_min=per_rare)


def fine_mix(examples) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        for op in ex.fine_operations or ():
            if op != "INS":
                counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
