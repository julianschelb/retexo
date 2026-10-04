# tests/test_baseline_scorer.py
"""Every metric of the scorer on hand-made records (no resources, no model)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import scorer as sc  # noqa: E402
from retexo.baselines.adapters import write_dump  # noqa: E402
from retexo.baselines.base import Prediction  # noqa: E402
from retexo.baselines.record import Edge, Record, Span, links_of  # noqa: E402


def record(source, reuse, edges, spans=(), rid="t/1"):
    return Record(id=rid, level="gold", fold=4, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit",
                  links=[Edge(r, s, op, sure) for r, s, op, sure in edges],
                  spans=[Span(a, b, "FRAME") for a, b in spans])


def gold_pred(rec):
    links, tags, frame, _ = links_of(rec)
    return Prediction(links=links, tags=[t or "" for t in tags], frame=frame)


def test_gold_against_itself():
    rec = record(["arma", "uirum", "cano", "te"], ["ut", "ait", "arma", "uirumque", "cano"],
                 [(2, 0, "COPY", True), (3, 1, "MORPH", True), (4, 2, "COPY", True)], spans=[(0, 2)])
    pred = gold_pred(rec)
    assert sc.token_accuracy([rec], [pred]) == 1.0
    link = sc.link_prf([rec], [pred])
    assert (link["precision"], link["recall"], link["f1"]) == (1.0, 1.0, 1.0)
    for level in sc.LEVELS:
        assert sc.op_scores([rec], [pred], level)["macro_f1"] == 1.0, level
    assert sc.frame_span_prf([rec], [pred])["F1"] == 1.0
    assert sc.replay_rate([rec], [pred]) == 1.0


def test_sure_possible():
    rec = record(["a", "b", "c"], ["a", "b", "c"], [(0, 0, "COPY", True), (1, 1, "COPY", True), (2, 2, "COPY", False)])
    first = Prediction(links=[0, -1, 2], tags=["COPY", "", "COPY"], frame=[0, 0, 0])
    link = sc.link_prf([rec], [first])
    assert link["precision"] == 1.0 and link["recall"] == 0.5 and abs(link["f1"] - 2 / 3) < 1e-9
    assert abs(link["aer"] - 0.25) < 1e-9                       # 1 - (2 + 1) / (2 + 2)
    assert abs(sc.token_accuracy([rec], [first]) - 2 / 3) < 1e-9   # token 2 right (possible), token 1 wrong
    second = Prediction(links=[0, 1, 2], tags=["COPY"] * 3, frame=[0, 0, 0])
    link = sc.link_prf([rec], [second])
    assert link["precision"] == 1.0 and link["recall"] == 1.0
    assert sc.token_accuracy([rec], [second]) == 1.0


def test_fraser_marcu_example():
    # |A| = 100, |S| = 100, |P and A| = |S and A| = 50: F 0.5 and 1 - AER 0.5
    sure = [(t, t, "COPY", True) for t in range(100)]
    rec = record(["w"] * 200, ["w"] * 200, sure + [(t, t, "COPY", False) for t in range(100, 200)])
    links = [t if t < 50 else (t + 100 if t < 100 else -1) for t in range(200)]   # 50 sure hits, 50 misses onto non-links
    links = [-1] * 200
    for t in range(50):
        links[t] = t             # sure and possible hit
    for t in range(50, 100):
        links[t] = 150           # a wrong source: neither sure nor possible
    pred = Prediction(links=links, tags=["COPY" if s >= 0 else "" for s in links], frame=[0] * 200)
    link = sc.link_prf([rec], [pred])
    assert link["n_pred"] == 100 and abs(link["f1"] - 0.5) < 1e-9 and abs(1 - link["aer"] - 0.5) < 1e-9
    # |P and A| = 75, |S and A| = 25: F 0.375, 1 - AER still 0.5
    links = [-1] * 200
    for t in range(25):
        links[t] = t             # sure hits
    for t in range(100, 150):
        links[t] = t             # possible-only hits
    for t in range(25, 50):
        links[t] = 199           # wrong
    pred = Prediction(links=links, tags=["COPY" if s >= 0 else "" for s in links], frame=[0] * 200)
    link = sc.link_prf([rec], [pred])
    assert link["n_pred"] == 100 and abs(link["precision"] - 0.75) < 1e-9 and abs(link["recall"] - 0.25) < 1e-9
    assert abs(link["f1"] - 0.375) < 1e-9 and abs(1 - link["aer"] - 0.5) < 1e-9


def test_op_f1_levels():
    rec = record(["a", "b", "c", "d"], ["x", "y", "c", "ut"],
                 [(0, 0, "MORPH", True), (1, 1, "SUBST", True), (2, 2, "SPLIT", True)], spans=[(3, 4)])
    wrong_source = Prediction(links=[1, 1, 2, -1], tags=["MORPH", "SUBST", "SPLIT", ""], frame=[0, 0, 0, 1])
    v1 = sc.op_scores([rec], [wrong_source], "V1")["per_class"]["MORPH"]
    assert v1["P"] == 0.0 and v1["R"] == 0.0 and v1["support"] == 1     # FP and FN at once
    syn = Prediction(links=[0, 1, 2, -1], tags=["MORPH", "SYN", "SPLIT", ""], frame=[0, 0, 0, 1])
    assert sc.op_scores([rec], [syn], "V1")["per_class"]["SUBST"]["F1"] == 1.0
    assert sc.op_scores([rec], [syn], "V3")["per_class"]["SUBST"]["R"] == 0.0
    assert sc.op_scores([rec], [syn], "V3")["per_class"]["SYN"]["P"] == 0.0
    assert sc.op_scores([rec], [syn], "group")["per_class"]["cardinality"]["F1"] == 1.0
    mode = sc.op_scores([rec], [gold_pred(rec)], "mode")["per_class"]
    assert mode["FRAME"]["F1"] == 1.0
    assert sc.op_scores([rec], [gold_pred(rec)], "V1")["per_class"]["INS"]["support"] == 1   # the frame token is INS at V1


def test_ins_del_f1():
    rec = record(["a", "b", "c"], ["a", "x"], [(0, 0, "COPY", True)])
    pred = Prediction(links=[0, 2], tags=["COPY", "SUBST"], frame=[0, 0])
    out = sc.op_scores([rec], [pred], "V1")
    ins, dele = out["per_class"]["INS"], out["per_class"]["DEL"]
    # micro over INS and DEL: gold INS {1}, DEL {1, 2}; pred INS {}, DEL {1}
    tp, fp, fn = 1, 0, 2
    p, r = tp / (tp + fp), tp / (tp + fn)
    assert abs(out["ins_del_f1"] - 2 * p * r / (p + r)) < 1e-9
    assert ins["R"] == 0.0 and dele["R"] == 0.5


def test_structure():
    from retexo.aligners.decode import ScriptDecoder

    assert ScriptDecoder.reordered_targets([5, 3, 4]) == {0}
    assert ScriptDecoder.quote_spans([0, 1, -1], ["NOP", "NOP", "INS"], minimum=2) == [(0, 1)]
    assert ScriptDecoder.quote_spans([0, 1, -1], ["NOP", "NOP", "INS"], minimum=ScriptDecoder.QUOTE_MIN) == []
    rec = record(["a", "b", "c"], ["a", "b", "z"], [(0, 0, "COPY", True), (1, 1, "COPY", True)])
    out = sc.structure_scores([rec], [gold_pred(rec)])
    assert out["quote_span"]["F1"] == 1.0 and out["del"]["F1"] == 1.0


def test_invented_links():
    neg = [Record(id=f"n/{i}", level="gold", fold=4, source_work="", source_tokens=["a", "b", "c"], reuse_work="",
                  reuse_tokens=["x", "y", "z"], pair_label="no_match") for i in range(2)]
    preds = [Prediction(links=[-1, -1, -1], tags=["", "", ""], frame=[0, 0, 0]),
             Prediction(links=[0, 1, 2], tags=["SUBST"] * 3, frame=[0, 0, 0])]
    out = sc.invented_links(neg, preds)
    assert out["links"] == 3 and out["pairs_with_links"] == 0.5 and out["pairs"] == 2


def test_aggregate_and_split_half():
    per = [{"token_accuracy": v, "link": {"f1": v / 2}} for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    agg = sc.aggregate_folds(per)
    m = agg["metrics"]["token_accuracy"]
    assert abs(m["mean"] - 0.3) < 1e-9 and m["min"] == 0.1 and m["max"] == 0.5 and abs(m["std"] - 0.1414213562) < 1e-6
    recs = [record(["a", "b"], ["a", "b"], [(0, 0, "COPY", True), (1, 1, "COPY", True)], rid=f"p/{i}") for i in range(12)]
    good = [gold_pred(r) for r in recs]
    bad = [Prediction(links=[-1, -1], tags=["", ""], frame=[0, 0]) for _ in recs]
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
        write_dump(recs, good, a); write_dump(recs, bad, b)
        out = sc.split_half(a, b, "token_accuracy")
        assert out["agree"] == 20 and out["confirmed"]
        mixed = [good[i] if i % 2 else bad[i] for i in range(12)]
        c = Path(tmp) / "c.jsonl"
        write_dump(recs, mixed, c)
        write_dump(recs, [bad[i] if i % 2 else good[i] for i in range(12)], b)
        out = sc.split_half(c, b, "token_accuracy", seed=3)
        assert out["agree"] < 20


def test_agreement_and_flat():
    a = [(0, 0), (1, 1), (2, 2)]
    same = sc.agreement(a, a)
    assert same["iaa"] == 1.0 and same["kappa"] == 1.0
    assert sc.agreement(a, [(3, 3)])["iaa"] == 0.0
    rec = record(["a"], ["a"], [(0, 0, "COPY", True)])
    result = sc.score([rec], [gold_pred(rec)])
    flat = sc.flat(result)
    assert all(k.startswith("test_") for k in flat) and all(isinstance(v, float) for v in flat.values())
    assert "test_token_accuracy" in flat and "test_ops.V1.macro_f1" in flat


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_scorer] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
