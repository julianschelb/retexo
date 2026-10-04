# tests/test_baseline_decoder.py
"""The pair-level decoders on the FWD / REV rows of test_e29.py and a 4 x 4 collision."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import decoder as dec  # noqa: E402
from retexo.baselines.record import Edge, Record  # noqa: E402

# reuse words t0..t2 over source words s0..s2; null is -1
FWD = [sorted([(0, 0.8), (1, 0.1), (-1, 0.1)], key=lambda x: -x[1]),
       sorted([(1, 0.45), (2, 0.35), (-1, 0.2)], key=lambda x: -x[1]),
       sorted([(2, 0.6), (-1, 0.4)], key=lambda x: -x[1])]
REV = [sorted([(0, 0.9), (-1, 0.1)], key=lambda x: -x[1]),
       sorted([(1, 0.5), (-1, 0.5)], key=lambda x: -x[1]),
       sorted([(1, 0.7), (2, 0.2), (-1, 0.1)], key=lambda x: -x[1])]


def prob(rows, t, s):
    return dict(rows[t]).get(s, 0.0)


def test_symmetrise_average():
    rows = dec.symmetrise_average(FWD, REV, 3)
    assert abs(prob(rows, 0, 0) - 0.85) < 1e-9
    assert abs(prob(rows, 1, 1) - 0.475) < 1e-9
    assert abs(prob(rows, 2, 2) - 0.4) < 1e-9
    assert abs(prob(rows, 1, 2) - 0.525) < 1e-9          # 0.35 forward, 0.7 reverse
    assert abs(prob(rows, 1, -1) - 0.2) < 1e-9            # nulls unchanged
    for row in rows:
        assert [p for _, p in row] == sorted([p for _, p in row], reverse=True)


def test_with_threshold():
    rows = dec.with_threshold(dec.symmetrise_average(FWD, REV, 3), 0.5)
    assert (1, 0.475) not in rows[1]
    assert dict(dec.with_threshold([[(0, 0.9)]], 0.5)[0])[-1] == 0.5   # a row without a null gains one at theta


def test_decode_default():
    assert dec.decode_default(FWD, theta=0.3, rev_rows=REV, n_source=3) == [0, 2, -1]
    assert dec.decode_default(FWD, theta=0.6, rev_rows=REV, n_source=3) == [0, -1, -1]


def test_hungarian_one_to_one():
    rows = [[(0, 0.9), (1, 0.5), (-1, 0.05)], [(0, 0.8), (1, 0.6), (-1, 0.1)],
            [(2, 0.7), (-1, 0.3)], [(3, 0.2), (-1, 0.8)]]
    links = dec.hungarian(rows)
    assert len(set(s for s in links if s >= 0)) == len([s for s in links if s >= 0])
    assert links == [0, 1, 2, -1]        # the weaker word moves to its second choice, which beats its null


def test_variants():
    assert dec.decode_mutual(FWD, REV) == [0, 1, -1]
    assert dec.decode_threshold(FWD, 0.5) == [0, -1, 2]
    assert dec.decode_raw(FWD) == [0, 1, 2]


def test_gdf():
    fwd = [[(0, 0.9), (-1, 0.1)], [(1, 0.8), (-1, 0.2)], [(2, 0.6), (-1, 0.4)]]
    rev = [[(0, 0.9), (-1, 0.1)], [(1, 0.7), (-1, 0.3)], [(1, 0.6), (-1, 0.4)]]
    # intersection {(0,0), (1,1)}; union adds (2,2) from forward and (1,2) from reverse
    links, extra = dec.decode_gdf(fwd, rev, 3, final="and")
    assert links[0] == 0 and links[1] == 1
    assert (2, 2) in {(t, s) for t, s in enumerate(links)} | set(extra)   # a neighbour of (1,1): grown
    links_or, extra_or = dec.decode_gdf(fwd, rev, 3, final="or")
    assert len([1 for s in links_or if s >= 0]) + len(extra_or) >= len([1 for s in links if s >= 0]) + len(extra)
    isolated_fwd = [[(0, 0.9), (-1, 0.1)], [(-1, 1.0)], [(2, 0.6), (-1, 0.4)]]
    isolated_rev = [[(0, 0.9), (-1, 0.1)], [(-1, 1.0)], [(-1, 1.0)]]
    links_and, extra_and = dec.decode_gdf(isolated_fwd, isolated_rev, 3, final="and")
    assert links_and == [0, -1, 2]      # (2,2) is a union point with both sides free: final-and adds it


def test_tune_null_threshold():
    rows = [FWD, FWD, FWD]
    gold = [[0, -1, -1], [0, -1, -1], [0, -1, -1]]
    theta = dec.tune_null_threshold(rows, gold, rev_per_pair=[REV] * 3, n_source_per_pair=[3] * 3)
    assert dec.decode_default(FWD, theta=theta, rev_rows=REV) == [0, -1, -1]
    assert theta >= 0.55                        # the plateau's larger theta wins


def test_stack_equals_default():
    record = Record(id="d/1", level="gold", fold=4, source_work="", source_tokens=["a", "b", "c"],
                    reuse_work="", reuse_tokens=["x", "y", "z"], pair_label="cit", links=[Edge(0, 0, "COPY")])
    assert dec.decode_stack(FWD, record, w=0.0, weight=0.0, k=1.0, theta=0.3, rev_rows=REV) == \
        dec.decode_default(FWD, theta=0.3, rev_rows=REV)


def test_pinned_to_originals():
    from retexo.aligners.assignment import AssignmentPolicy, Reranker
    from retexo.formulations.change_detector import ChangeExample

    assert dec.hungarian(FWD) == AssignmentPolicy.links_hungarian([FWD])[0]
    assert dec.decode_raw(FWD) == AssignmentPolicy.links_argmax([FWD])[0]
    example = ChangeExample(source_tokens=["arma", "b", "c"], target_tokens=["arma", "y", "z"],
                            labels=[0, 1, 1], operations=["NOP", "INS", "INS"], n_operations=2)
    ours = dec.identity_bonus(FWD, example.source_tokens, example.target_tokens, 2.0)
    theirs = Reranker.rerank([FWD], [example], 2.0)[0]
    for a, b in zip(ours, theirs):
        assert {s: round(p, 9) for s, p in a} == {s: round(p, 9) for s, p in b}
    from retexo.aligners.agreement import AgreementDecoder
    ours = AgreementDecoder.null_scale(FWD, 3.0)
    theirs = Reranker.with_null_scale([FWD], 3.0)[0]
    for a, b in zip(ours, theirs):
        assert {s: round(p, 9) for s, p in a} == {s: round(p, 9) for s, p in b}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_decoder] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
