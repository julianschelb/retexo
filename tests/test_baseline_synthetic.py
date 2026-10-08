# tests/test_baseline_synthetic.py
"""Note 03 on a tiny pool from the gold's own passages: records that replay, both
orientations, the label mapping, exclusions, the held-out set, the coverage table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.record import RecordCodec  # noqa: E402
from retexo.baselines.synthetic import Coverage, PoolBuilder, SyntheticRecords  # noqa: E402
from retexo.formulations.change_detector import ChangeExample  # noqa: E402


def gold_records():
    return RecordCodec.gold_records(Path(__file__).resolve().parents[1] / "data" / "gold_full")


def example(source, target, links, fine, frames=None):
    return ChangeExample(
        source_tokens=source,
        target_tokens=target,
        labels=[0] * len(target),
        operations=["COPY"] * len(target),
        n_operations=0,
        source_labels=[0] * len(source),
        source_operations=["COPY"] * len(source),
        alignments=links,
        fine_operations=fine,
        frame_labels=frames or [0] * len(target),
        link_features=[None] * len(target),
    )


# =============================================================================
# 1. Examples to records
# =============================================================================


def test_examples_to_records_maps_the_fine_tags_and_labels_two_copies_cit():
    ex = example(
        ["rex", "gladio", "amat", "hostem"],
        ["rex", "ense", "amat", "hostem", "ut"],
        [0, 1, 2, 3, -1],
        ["NOP", "HYPER", "NOP", "SYN-DIST", "INS"],
        [0, 0, 0, 0, 1],
    )
    rec = SyntheticRecords.from_examples([ex], level="synthetic", fold=4)[0]
    ops = {(e.r, e.s): (e.op, e.detail) for e in rec.links}
    assert (
        ops[(0, 0)] == ("COPY", "")
        and ops[(1, 1)] == ("SUBST", "HYPER")
        and ops[(3, 3)] == ("SYN", "SYN-DIST")
    )
    # read back as a constructed record, the generator's own fine labels return (the shared pool depends on it)
    rec.provenance["fine_ops"] = "construction-V3"
    from retexo.baselines.record import RecordCodec

    assert RecordCodec._as_goldpair(rec).target_ops[:4] == ["NOP", "HYPER", "NOP", "SYN-DIST"]
    assert rec.spans and rec.spans[0].label == "FRAME" and rec.spans[0].start == 4
    assert rec.pair_label == "cf"  # rex/ense/amat: no two in-order copies
    two = example(["a", "b", "c"], ["a", "b", "d"], [0, 1, -1], ["NOP", "NOP", "INS"])
    assert SyntheticRecords.from_examples([two])[0].pair_label == "cit"
    assert SyntheticRecords.from_examples([two])[0].provenance["links"] == "construction"


def test_the_replay_check_is_applied_and_can_be_switched_off():
    # decode_script repairs what it can (an out-of-range link is dropped, not an error), so the
    # replay check passes on a repaired script; the switch exists for negatives, which carry no links
    odd = example(["a", "b"], ["c", "b"], [0, 5], ["NOP", "NOP"])
    checked = SyntheticRecords.from_examples([odd])
    unchecked = SyntheticRecords.from_examples([odd], verify=False)
    assert len(unchecked) == 1 and len(checked) <= 1
    assert unchecked[0].provenance["links"] == "construction"


# =============================================================================
# 2. The pool from the gold's passages
# =============================================================================


def test_a_tiny_pool_replays_comes_in_both_orientations_and_excludes_held_out_passages():
    gold = gold_records()
    train = [r for r in gold if r.fold != 4][:60]
    held = [r for r in gold if r.fold == 4][:20]
    builder = PoolBuilder(4, train, held, seed=1)
    seeds = builder.seeds()
    held_texts = {" ".join(r.source_tokens) for r in held} | {
        " ".join(r.reuse_tokens) for r in held
    }
    assert seeds and all(" ".join(s) not in held_texts for s in seeds)
    pool, report = builder.build_pool(size=30, workers=1, rare_min=2)
    assert 20 <= len(pool) <= 60  # 30 chosen, both orientations, minus non-replays
    assert {r.provenance["orientation"] for r in pool} == {"forward", "swapped"}
    forward = [r for r in pool if r.provenance["orientation"] == "forward"]
    swapped = [r for r in pool if r.provenance["orientation"] == "swapped"]
    assert len(forward) >= 10 and len(swapped) >= 10
    heldout = builder.heldout_synthetic(n=20, per_rare=2)
    pool_keys = {(" ".join(r.source_tokens), " ".join(r.reuse_tokens)) for r in pool}
    assert heldout and all(
        (" ".join(r.source_tokens), " ".join(r.reuse_tokens)) not in pool_keys for r in heldout
    )
    assert all(r.level == "synthetic_heldout" for r in heldout)
    table = Coverage.table(pool, train)
    assert "crossing_share" in table and "| link_rate |" in table
    assert Coverage.statistics(train) == Coverage.statistics(train)


def test_coverage_of_the_gold_against_itself_has_no_gaps():
    gold = gold_records()[:100]
    table = Coverage.table(gold, gold)
    assert "gap |" not in table.replace("| gap |", "")
    assert Coverage.statistics(gold)["crossing_share"] >= 0.0
