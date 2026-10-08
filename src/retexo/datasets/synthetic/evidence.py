# datasets/synthetic/evidence.py
"""The per-pair evidence grid (E25): shared parser state, the workers, and ``featurize_pairs``."""

from __future__ import annotations

import sys
from typing import Optional

from retexo.edit_typing.dep_features import DEP_FEATURES, DependencyParser
from retexo.edit_typing.link_features import LinkFeaturizer

# =============================================================================
# Evidence for every pair (E25)
# =============================================================================

_FEAT: dict = {}
#: E41: when True, featurize_pairs parses every passage with Stanza (parent, GPU) and
#: appends retexo.edit_typing.dep_features.DEP_FEATURES to every cell
DEP_FEATURES_ON = False
#: The Frame Support dry run (2026-09-27): when True, featurize_pairs appends one channel to every cell, the
#: in-order anchor support (``frame_support_channel``)
FRAME_CHANNEL_ON = False
#: One DependencyParser per process, so its parse cache survives repeat calls.
_DEP_PARSER: Optional[DependencyParser] = None


def _dep_parser() -> DependencyParser:
    global _DEP_PARSER
    if _DEP_PARSER is None:
        _DEP_PARSER = DependencyParser()
    return _DEP_PARSER


def _feat_init(vectors_path):
    from retexo.resources import Resources

    _FEAT["fz"] = LinkFeaturizer(Resources(offline=True, vectors_path=vectors_path))


def _feat_chunk(items):
    import numpy as np

    from retexo.edit_typing.link_features import N_FEATURES

    fz = _FEAT["fz"]
    out = []
    for item in items:
        source, target = item[0], item[1]
        dep = item[2] if len(item) > 2 else None  # E41: (dep_s, dep_t)
        width = N_FEATURES + (len(DEP_FEATURES) if dep is not None else 0)
        arr = np.zeros((len(target), len(source), width), dtype=np.float16)
        n_s, n_t = len(source), len(target)
        for t, tw in enumerate(target):
            for s, sw in enumerate(source):
                row = fz(sw, tw, s, t, n_s, n_t)
                if dep is not None:
                    row = list(row) + DependencyParser.cell_features(
                        source, target, dep[0], dep[1], s, t, lemma=fz.lemma
                    )
                arr[t, s] = row
        out.append(arr)
    return out


def frame_support_channel(arr, window: int = 2, slack: int = 3):
    """The kept-frame support of every cell, from the evidence alone (no labels): the number of anchor cells
    (same form or same lemma) at reuse positions within ``window`` of the cell's reuse word, in the same order on the
    source side (the offsets agree in sign) and within ``slack`` of the cell's source offset, divided by
    ``2 * window``. The error analysis of 2026-09-26 measured the same count on links: sure substitutions sit in a
    kept frame (mean support 2.1), invented links do not (0.8)."""
    import numpy as np

    from retexo.edit_typing.link_features import FEATURE_NAMES

    anchor = (arr[..., FEATURE_NAMES.index("same_form")] > 0) | (
        arr[..., FEATURE_NAMES.index("same_lemma")] > 0
    )
    n_t, n_s = anchor.shape
    # a reuse neighbour counts once, however many source offsets match
    per_neighbour = np.zeros((n_t, n_s), dtype=np.float32)
    for dt in range(-window, window + 1):
        if dt == 0:
            continue
        hit = np.zeros((n_t, n_s), dtype=bool)
        for ds in range(-(abs(dt) + slack), abs(dt) + slack + 1):
            if ds == 0 or (ds > 0) != (dt > 0) or abs(ds - dt) > slack:
                continue
            t0, t1 = max(0, -dt), min(n_t, n_t - dt)
            s0, s1 = max(0, -ds), min(n_s, n_s - ds)
            if t0 < t1 and s0 < s1:
                hit[t0:t1, s0:s1] |= anchor[t0 + dt : t1 + dt, s0 + ds : s1 + ds]
        per_neighbour += hit
    return (per_neighbour / (2 * window)).astype(arr.dtype)


def featurize_pairs(examples, *, vectors_path=None, workers=24, chunk_size=48, log=None):
    """Attach ``pair_features`` -- the evidence for every (reuse, source) pair.

    The typed pointer needs the evidence inside the alignment decision, which
    means every candidate pair, not only the gold link. Per-word lookups are
    cached inside each worker, so the pair cost is string work; a 20k pool is
    ~8M pairs and takes well under a minute on forty workers.
    """
    import multiprocessing as mp

    items = [(list(e.source_tokens), list(e.target_tokens)) for e in examples]
    if DEP_FEATURES_ON:
        parses = _dep_parser().parse_all(
            [e.source_tokens for e in examples] + [e.target_tokens for e in examples], log=log
        )
        items = [(s, t, (parses[" ".join(s)], parses[" ".join(t)])) for s, t in items]
    tasks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    done = 0
    with context.Pool(workers, initializer=_feat_init, initargs=(vectors_path,)) as pool:
        for i, arrays in enumerate(pool.imap(_feat_chunk, tasks)):
            for e, a in zip(examples[i * chunk_size : (i + 1) * chunk_size], arrays):
                if FRAME_CHANNEL_ON:
                    import numpy as np

                    a = np.concatenate([a, frame_support_channel(a)[..., None]], axis=-1)
                # ChangeExample is frozen; this is the one field filled in after
                # construction, and it is filled exactly once
                object.__setattr__(e, "pair_features", a)
            done += len(arrays)
    if log:
        cells = sum(
            e.pair_features.shape[0] * e.pair_features.shape[1]
            for e in examples
            if e.pair_features is not None
        )
        log(f"evidence for {done:,} pairs, {cells:,} cells")
    return examples
