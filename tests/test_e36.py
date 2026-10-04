"""Refinement states: round trips, the chain's stages, corruption, isolation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.refinement.refine import (DECLINED, FRAME, LINKED, S_CONSUMED, S_FREE, S_UNDECIDED,  # noqa: E402
                               UNDECIDED, State)


def test_state_from_script_round_trips_the_gold():
    st = State.from_script([2, -1, 0, -1], [0, 1, 0, 0], n_source=4)
    assert st.reuse == [LINKED, FRAME, LINKED, DECLINED] and st.link == [2, -1, 0, -1]
    assert st.source == [S_CONSUMED, S_FREE, S_CONSUMED, S_FREE]
    assert State.undecided(3, 2).reuse == [UNDECIDED] * 3 and State.undecided(3, 2).source == [S_UNDECIDED] * 2


def test_chain_stages_grow_from_skeleton_to_gold():
    gold = [0, 1, 2, -1, 3]
    # tiers [n_reuse, n_source]: word 0 same-form unique (4), word 1 same-form but two candidates,
    # word 2 lemma (3), word 4 nothing (0)
    tiers = np.zeros((5, 4), dtype=int)
    tiers[0, 0] = 4; tiers[1, 1] = 4; tiers[1, 3] = 4; tiers[2, 2] = 3
    s1 = State.chain(gold, None, 4, tiers, 1)
    assert s1.link == [0, -1, -1, -1, -1] and s1.reuse[1] == UNDECIDED
    s2 = State.chain(gold, None, 4, tiers, 2)
    assert s2.link == [0, 1, 2, -1, -1] and s2.reuse[4] == UNDECIDED and s2.reuse[3] == UNDECIDED
    s3 = State.chain(gold, None, 4, tiers, 3)
    assert s3.link == gold and s3.reuse[3] == DECLINED


def test_corruption_changes_the_requested_share():
    rng = random.Random(0)
    gold = list(range(20)) + [-1] * 10
    st = State.corrupted(gold, None, 40, rng, flip=0.5, add=0.5)
    flipped = sum(1 for t in range(20) if st.link[t] < 0)
    added = sum(1 for t in range(20, 30) if st.link[t] >= 0)
    assert 4 <= flipped <= 16 and added == 10 or added <= 10
    assert all(st.link[t] < 40 for t in range(30))


def test_sample_modes_and_isolation():
    class Ex:
        target_tokens = ["a", "b", "c"]; source_tokens = ["x", "y"]
        alignments = [0, -1, 1]; frame_labels = [0, 1, 0]
    rng = random.Random(1)
    assert State.sample_for_training(Ex, "gold", rng).link == [0, -1, 1]
    assert State.sample_for_training(Ex, "none", rng).n_decided() == 0
    assert State.sample_for_training(Ex, "chain", rng, tiers=None).n_decided() in (0, 3)
    assert State.isolated_false_links([0, -1, 5, -1, 1], [0, -1, -1, -1, 1]) == (1, 1)
    assert State.isolated_false_links([0, 5, 1], [0, -1, 1]) == (1, 0)


def test_until_stable_freezes_settled_pairs_and_keeps_every_pass_complete():
    from retexo.formulations.typed_pointer import TypedPointer

    class Ex:
        def __init__(self):
            self.source_tokens = ["x", "y"]

    # pair 0 settles at pass 2 (same script twice), pair 1 keeps changing until the cap
    scripts = {0: [[0, -1], [0, -1], [0, -1], [0, -1]], 1: [[0, 1], [1, 0], [0, 1], [1, 0]]}

    class Stub:
        calls = 0
        def predict_links(self, sub):
            k = self.calls; self.calls += 1
            return [scripts[id_of[id(ex)]][k] for ex in sub]
        def predict_frames(self, sub, links):
            return [[0, 0] for _ in sub]

    exs = [Ex(), Ex()]; id_of = {id(e): i for i, e in enumerate(exs)}
    stub = Stub()
    hist = TypedPointer.predict_iterative(stub, exs, passes=4, until_stable=True)
    assert len(hist) == 4 and stub.calls == 4
    assert [h[0][0] for h in hist] == [[0, -1]] * 4          # frozen pair carried forward
    assert [h[0][1] for h in hist] == scripts[1]             # active pair re-decided each pass
    assert stub.settled_at == [2, 0]
    stub2 = Stub(); exs2 = [Ex()]; id_of.update({id(exs2[0]): 0})
    hist2 = TypedPointer.predict_iterative(stub2, exs2, passes=4, until_stable=True)
    assert len(hist2) == 2 and stub2.calls == 2              # all settled: stops before the cap
    stub3 = Stub(); hist3 = TypedPointer.predict_iterative(stub3, exs, passes=4)   # default: fixed passes
    assert len(hist3) == 4 and stub3.calls == 4 and stub3.settled_at == [0, 0]
