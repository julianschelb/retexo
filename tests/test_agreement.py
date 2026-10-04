"""The agreement rules on a hand-made 3 x 3 pair."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.aligners.agreement import AgreementDecoder, PairSwap  # noqa: E402

# reuse words t0..t2 over source words s0..s2; null is -1
FWD = [sorted([(0, 0.8), (1, 0.1), (-1, 0.1)], key=lambda x: -x[1]),      # t0 -> s0, sure
       sorted([(1, 0.45), (2, 0.35), (-1, 0.2)], key=lambda x: -x[1]),    # t1 -> s1, shaky
       sorted([(2, 0.6), (-1, 0.4)], key=lambda x: -x[1])]                # t2 -> s2, but s2 wants t1
REV = [sorted([(0, 0.9), (-1, 0.1)], key=lambda x: -x[1]),                # s0 -> t0
       sorted([(1, 0.5), (-1, 0.5)], key=lambda x: -x[1]),                # s1 -> t1 (tie with null, t1 listed first)
       sorted([(1, 0.7), (2, 0.2), (-1, 0.1)], key=lambda x: -x[1])]      # s2 -> t1


def links_of(rows):
    return [r[0][0] for r in rows]


def test_intersection_needs_both_directions():
    kept = AgreementDecoder.intersect(FWD, REV, c=0.3)
    assert links_of(kept) == [0, 1, -1]        # t2 -> s2 fails: p(t2 | s2) = 0.2
    kept = AgreementDecoder.intersect(FWD, REV, c=0.6)
    assert links_of(kept) == [0, -1, -1]       # t1 -> s1 fails at 0.45 / 0.5


def test_mutual_argmax_is_thresholdless():
    kept = AgreementDecoder.mutual_argmax(FWD, REV)
    assert links_of(kept) == [0, 1, -1]        # s2's best is t1, not t2


def test_entropy_filter_drops_flat_rows():
    # t1's row is the flattest; s1's column is flat too -> dropped at a strict tau
    kept = AgreementDecoder.entropy_filter(FWD, REV, tau=0.5)
    assert kept[1][0][0] == -1 and kept[0][0][0] == 0


def test_null_scale_and_compose():
    scaled = AgreementDecoder.null_scale(FWD, 3.0)
    assert scaled[2][0][0] == -1               # 0.4 * 3 beats 0.6
    both = AgreementDecoder.null_scale(AgreementDecoder.intersect(FWD, REV, 0.3), 3.0)   # A then D: scale the survivors' null
    assert links_of(both) == [0, -1, -1]       # t1 survives the intersection but not the scale
    ac = AgreementDecoder.compose(AgreementDecoder.intersect(FWD, REV, 0.3), AgreementDecoder.entropy_filter(FWD, REV, 0.5))
    assert links_of(ac) == [0, -1, -1]         # t1 survives A but not C


def test_reverse_and_swap():
    assert AgreementDecoder.reverse_links(REV, 3) == [0, 1, -1]  # s2 wants t1, already taken by s1

    class Ex:
        source_tokens = ["a", "b"]; target_tokens = ["x", "y", "z"]
    sw = PairSwap.example(Ex)
    assert sw.source_tokens == ["x", "y", "z"] and sw.target_tokens == ["a", "b"]
    assert sw.alignments == [-1, -1]


def test_swap_labelled_inverts_links_and_relations():
    from retexo.formulations.change_detector import ChangeExample
    ex = ChangeExample(source_tokens=["arma", "uirum", "cano"], target_tokens=["ut", "arma", "uiri"],
                       labels=[1, 0, 1], operations=["INS", "COPY", "SUBST"], n_operations=2,
                       source_labels=[0, 0, 1], source_operations=["COPY", "COPY", "DEL"],
                       alignments=[-1, 0, 1], fine_operations=["INS", "NOP", "HYPO"],
                       frame_labels=[1, 0, 0], link_features=[None] * 3)
    sw = PairSwap.labelled(ex)
    assert sw.source_tokens == ["ut", "arma", "uiri"] and sw.target_tokens == ["arma", "uirum", "cano"]
    assert sw.alignments == [1, 2, -1]                  # arma<-arma, uirum<-uiri, cano inserted
    assert sw.fine_operations == ["NOP", "HYPER", "INS"]  # HYPO seen from the other side
    assert sw.source_labels == [1, 0, 0]                # the old frame word "ut" is now a dropped source word
    assert sw.frame_labels == [0, 0, 0] and sw.operations == ["COPY", "SUBST", "INS"]
