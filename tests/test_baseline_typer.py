# tests/test_baseline_typer.py
"""The rule typer, the frame rule, the derivation and the gate with a stub featurizer (no resources)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import typer as typ  # noqa: E402
from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.edit_typing.link_features import FEATURE_NAMES  # noqa: E402


class Stub:
    """Pair features keyed by the two words; everything not given is 0 except the missing flags."""

    def __init__(self, table):
        self.table = table

    def __call__(self, source, target, s=0, t=0, n_s=1, n_t=1):
        f = {name: 0.0 for name in FEATURE_NAMES}
        f["cos_missing"] = 1.0
        f["wn_missing"] = 1.0
        f.update(self.table.get((source, target), {}))
        if "cos" in self.table.get((source, target), {}):
            f["cos_missing"] = 0.0
        return [f[name] for name in FEATURE_NAMES]


def record(source, reuse, links=()):
    return Record(id="t/1", level="gold", fold=4, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit", links=[Edge(r, s, op) for r, s, op in links])


def test_ordered_list():
    stub = Stub({("matrem", "matri"): {"same_lemma": 1}, ("gladius", "ensis"): {"wn_syn": 1},
                 ("gladius", "ferrum"): {"cos": 0.7, "same_pos": 1}, ("gladius", "uita"): {"cos": 0.6, "same_pos": 1},
                 ("ignotus", "ignota"): {"lemma_missing": 1}})
    rec = record(["arma", "matrem", "gladius", "gladius", "gladius", "ignotus"],
                 ["arma", "matri", "ensis", "ferrum", "uita", "ignota"])
    tags, outcomes, details = typ.rule_type(rec, [0, 1, 2, 3, 4, 5], stub)
    assert tags == ["COPY", "MORPH", "SYN", "SYN", "SUBST", "SUBST"], tags
    assert outcomes[4] == "no_rel_found" and outcomes[5] == "lemma_missing"
    # identical forms are COPY even when the stub also says same lemma
    rec2 = record(["arma"], ["arma"])
    assert typ.rule_type(rec2, [0], Stub({("arma", "arma"): {"same_lemma": 1}}))[0] == ["COPY"]


def test_spelling_fold():
    stub = Stub({("notus", "notum"): {"same_lemma": 1}})
    rec = record(["temptare", "arma", "haud", "notus"], ["tentare", "armaque", "haut", "notum"])
    tags, _, _ = typ.rule_type(rec, [0, 1, 2, 3], stub)
    assert tags == ["COPY", "COPY", "COPY", "MORPH"], tags
    # hostis / hostes is not folded: the -is / -es alternation is inflectional in canis / canes


def test_demotion_and_enclitics():
    stub = Stub({("caelum", "Trica"): {"wn_hyper": 1},
                 ("armaque", "arma"): {"enclitic_src": 1, "enclitic_stem_match": 1},
                 ("arma", "armaque"): {"enclitic_tgt": 1, "enclitic_stem_match": 1}})
    rec = record(["caelum", "armaque", "arma"], ["Trica", "arma", "armaque"])
    tags, _, details = typ.rule_type(rec, [0, 1, 2], stub, spelling_fold=False)
    assert tags == ["SUBST", "SPLIT", "MERGE"] and details[0] == "HYPER"


def test_frame_rule():
    reuse = ["ut", "ait", "Maro", ":", "arma", "uirumque", "cano"]
    rec = record(["arma", "uirumque", "cano"], reuse)
    links = [-1, -1, -1, -1, 0, 1, 2]
    frame = typ.frame_rule(rec, links, keywords={"ait"})
    assert frame == [1, 1, 1, 1, 0, 0, 0], frame
    far = record(["arma"], ["ut", "ait", "Maro", "a", "b", "c", "d", "e", "arma"])
    assert typ.frame_rule(far, [-1] * 8 + [0], keywords={"ait"}) == [0] * 9
    two = record(["arma", "cano"], ["ut", "ait", "Maro", "arma", "sic", "inquit", "ille", "cano"])
    frame = typ.frame_rule(two, [-1, -1, -1, 0, -1, -1, -1, 1], keywords={"ait", "inquit"})
    assert frame == [1, 1, 1, 0, 0, 0, 0, 0], frame          # only the span before the first link survives


def test_derive_fine_and_coverage():
    stub = Stub({("gladius", "ensis"): {"wn_syn": 1}, ("uita", "mors"): {}, ("matrem", "matri"): {"same_lemma": 1}})
    rec = record(["arma", "gladius", "uita", "matrem"], ["arma", "ensis", "mors", "matri"],
                 [(0, 0, "COPY"), (1, 1, "SUBST"), (2, 2, "SUBST"), (3, 3, "MORPH")])
    typ.derive_fine(rec, stub)
    ops = {e.r: e.op for e in rec.links}
    assert ops == {0: "COPY", 1: "SYN", 2: "SUBST", 3: "MORPH"}
    assert rec.provenance["no_rel_found"] == [2] and rec.provenance["fine_ops"] == "resource-lookup"
    cov = typ.coverage([rec, rec, rec])
    assert abs(cov["no_rel_found"] - 0.25) < 1e-9 and abs(cov["V2_named"] - 0.75) < 1e-9


def test_gate():
    assert typ.gate(["COPY", "SUBST"], [True, False], ["MORPH", "SYN"]) == ["COPY", "SYN"]


def test_gold_links_baseline():
    from retexo.baselines import get, load_all
    from retexo.baselines.adapters import to_script
    from retexo.metrics import ScriptScorer

    load_all()
    rec = record(["arma", "uirumque", "cano"], ["ut", "ait", "arma", "uirumque", "cano"],
                 [(2, 0, "COPY"), (3, 1, "COPY"), (4, 2, "COPY")])
    rec.spans = [Span(0, 2, "FRAME")]
    method = get("gold_links")(BaselineConfig(device="cpu"))
    pred = method.postprocess(rec, method.predict([rec])[0], {})
    assert pred.links == [-1, -1, 0, 1, 2] and pred.frame == [1, 1, 0, 0, 0] and pred.tags[2:] == ["COPY"] * 3
    gold = to_script(rec, pred)
    assert ScriptScorer.evaluate([to_script(rec, pred)], [gold], generative=False).alignment.f1 == 1.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_typer] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
