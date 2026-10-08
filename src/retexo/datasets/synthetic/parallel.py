# datasets/synthetic/parallel.py
"""Multiprocess generation: the worker state, the chunk loop, and ``generate_typed``."""

from __future__ import annotations

import random
import sys

from retexo.datasets.synthetic.inventory import ENCLITIC_RATE, FRAME_RATE
from retexo.datasets.synthetic.one_pair import TypedReport, _make_one_typed
from retexo.datasets.synthetic.substitution import _make_typed_substitute, calibrate_acceptance
from retexo.edit_typing.link_features import LinkFeaturizer

# =============================================================================
# Parallel generation
# =============================================================================

_WORKER: dict = {}
_MLM_CACHE: dict = {}


def _init_worker(
    vectors_path,
    attested,
    accept,
    frames,
    frame_rate,
    enclitic_rate,
    reorder_inter,
    reorder_intra,
    mlm_subst=False,
    subst_weight=None,
    mlm_model=None,
    cohypo=False,
    dense_rate=0.0,
    norel_share=0.0,
    stem_subst=0.0,
    ins_slot=0.0,
    lexical_share=None,
    rel_share=None,
    copy_variants=0.0,
    frameless_fill="",
):
    mlm = None
    if mlm_subst:
        from retexo.datasets.mlm_subst import ContextualSubstituter

        mlm = ContextualSubstituter(mlm_model) if mlm_model else ContextualSubstituter()
        mlm.cache = _MLM_CACHE.get("cache")  # inherited from the parent through fork
    substitute, resources = _make_typed_substitute(
        vectors_path,
        attested,
        accept,
        mlm=mlm,
        subst_weight=subst_weight,
        cohypo=cohypo,
        lexical_share=lexical_share,
        norel_share=norel_share,
        stem_subst=stem_subst,
        rel_share=rel_share,
    )
    slot_words = sorted(w for w in attested if len(w) >= 5) if (ins_slot > 0 and attested) else None
    _WORKER.update(
        substitute=substitute,
        featurizer=LinkFeaturizer(resources),
        frames=frames,
        frame_rate=frame_rate,
        enclitic_rate=enclitic_rate,
        reorder_inter=reorder_inter,
        reorder_intra=reorder_intra,
        attested=attested,
        dense_rate=dense_rate,
        ins_slot=ins_slot,
        slot_words=slot_words,
        copy_variants=copy_variants,
        frameless_fill=frameless_fill,
    )


def _chunk(args):
    seeds, context_pool, seed, per_seed = args
    rng = random.Random(seed)
    report = TypedReport()
    out = []
    w = _WORKER
    for tokens in seeds:
        for _ in range(per_seed):
            report.attempted += 1
            example, stats = _make_one_typed(
                tokens,
                context_pool,
                w["substitute"],
                w["featurizer"],
                rng,
                frames=w["frames"],
                frame_rate=w["frame_rate"],
                enclitic_rate=w["enclitic_rate"],
                reorder_inter=w["reorder_inter"],
                reorder_intra=w["reorder_intra"],
                attested=w["attested"],
                dense_rate=w.get("dense_rate", 0.0),
                ins_slot=w.get("ins_slot", 0.0),
                slot_words=w.get("slot_words"),
                copy_variants=w.get("copy_variants", 0.0),
                frameless_fill=w.get("frameless_fill", ""),
            )
            if example is None:
                report.no_substitute += 1
                continue
            out.append(example)
            report.kept += 1
            for op in example.operations:
                report.op_counts[op] = report.op_counts.get(op, 0) + 1
            for op in example.fine_operations:
                if op != "INS":
                    report.fine_counts[op] = report.fine_counts.get(op, 0) + 1
            report.frames += stats["frame"]
            report.reorders += stats["reorder"]
            report.enclitics += stats["enclitic"]
            report.spelling += stats["spelling"]
            report.frameless += stats["frameless"]
    return out, report.__dict__


def generate_typed(
    seeds,
    context_pool,
    *,
    workers=24,
    per_seed=4,
    seed=1,
    chunk_size=200,
    vectors_path=None,
    attested=None,
    accept=None,
    frames=(),
    frame_rate=FRAME_RATE,
    enclitic_rate=ENCLITIC_RATE,
    reorder_inter=0.0,
    reorder_intra=0.0,
    log=None,
    mlm_subst=False,
    subst_weight=None,
    mlm_model=None,
    cohypo=False,
    dense_rate=0.0,
    norel_share=0.0,
    stem_subst=0.0,
    ins_slot=0.0,
    lexical_share=None,
    rel_share=None,
    copy_variants=0.0,
    frameless_fill="",
):
    """Generate in parallel, offline. Returns examples and a report."""
    import multiprocessing as mp

    chunks = [list(seeds[i : i + chunk_size]) for i in range(0, len(seeds), chunk_size)]
    context_pool = [list(p) for p in context_pool]
    frames = [list(f) for f in frames]
    tasks = [(c, context_pool, seed + i, per_seed) for i, c in enumerate(chunks)]
    if accept is None:
        # E4's calibration measures untyped coverage; the typed source realises
        # the same tags plus two rarer ones, so the same acceptance is close
        # enough, and the realism report says where it lands.
        accept, coverage = calibrate_acceptance(seeds, vectors_path, attested)
        if log:
            log(f"substitute coverage {coverage:.1%} -> accept {accept:.1%}")
    if mlm_subst:
        from retexo.datasets.mlm_subst import ContextualSubstituter

        _MLM_CACHE["cache"] = ContextualSubstituter.precompute(
            seeds, mlm_model or "bowphs/LaBerta", log=log
        )
    context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    out, totals = [], TypedReport()
    with context.Pool(
        workers,
        initializer=_init_worker,
        initargs=(
            vectors_path,
            attested,
            accept,
            frames,
            frame_rate,
            enclitic_rate,
            reorder_inter,
            reorder_intra,
            mlm_subst,
            subst_weight,
            mlm_model,
            cohypo,
            dense_rate,
            norel_share,
            stem_subst,
            ins_slot,
            lexical_share,
            rel_share,
            copy_variants,
            frameless_fill,
        ),
    ) as pool:
        for examples, report in pool.imap_unordered(_chunk, tasks):
            out.extend(examples)
            totals.attempted += report["attempted"]
            totals.kept += report["kept"]
            totals.no_substitute += report["no_substitute"]
            totals.frames += report["frames"]
            totals.reorders += report["reorders"]
            totals.enclitics += report["enclitics"]
            totals.spelling += report["spelling"]
            totals.frameless += report["frameless"]
            for k, v in report["op_counts"].items():
                totals.op_counts[k] = totals.op_counts.get(k, 0) + v
            for k, v in report["fine_counts"].items():
                totals.fine_counts[k] = totals.fine_counts.get(k, 0) + v
    return out, totals
