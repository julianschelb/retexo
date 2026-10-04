# retexo/baselines/splits.py
"""The nested-fold rule: which folds a system trains on, tunes on, and is scored on."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from retexo.baselines.record import Record


def split_gold(records: Sequence[Record], fold: int, dev_fold: int) -> Tuple[List[Record], List[Record], List[Record]]:
    """``train`` = every fold but K (the dev fold inside it), ``dev`` = fold (K + 1) mod 5, ``test`` = fold K.

    Args:
        records: Annotated pairs, each carrying its ``fold``.
        fold: The test fold K.
        dev_fold: The fold that tunes a system's dials, ``(K + 1) mod 5`` in the protocol.

    Returns:
        The train, dev and test records, in this order.
    """
    test = [r for r in records if r.fold == fold]
    train = [r for r in records if r.fold != fold]
    dev = [r for r in train if r.fold == dev_fold]
    return train, dev, test
