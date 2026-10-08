# tests/test_operations.py
"""
Behavioural tests for the operation inventory.

Each operation is exercised on its own — applied, replayed, inverted, and
replayed back — rather than only inside a composite script, so a failure names
the operation responsible. The structural invariants that the retired
prototype violated are asserted explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo import (  # noqa: E402
    CostModel,
    EditScript,
    OperationRegistry,
    Scriba,
    VariantBuilder,
    normalize,
)
from retexo.operations import EditOperation  # noqa: E402

REGISTRY = OperationRegistry.default()
SCRIBA = Scriba(REGISTRY)
SEED = "arma uirumque cano Troiae qui primus ab oris"


# =============================================================================
# Helpers
# =============================================================================


def roundtrip(builder: VariantBuilder, seed: str):
    """Build, replay, invert, replay back. Returns the variant."""
    seed_tokens = seed.split()
    variant, script = builder.build()

    assert SCRIBA.validate(script, len(seed_tokens)), SCRIBA.validate(script).problems
    assert SCRIBA.verify(script, seed_tokens, variant), "replay did not reproduce the variant"

    inverse = script.invert()
    restored = SCRIBA.execute(inverse, variant)
    assert [normalize(t) for t in restored] == [normalize(t) for t in seed_tokens], (
        f"inverse did not restore the seed: {restored} != {seed_tokens}"
    )
    return variant, script


# =============================================================================
# One operation at a time
# =============================================================================


def test_nop():
    b = VariantBuilder(SEED, REGISTRY)
    for i in range(len(SEED.split())):
        b.keep(i)
    variant, script = roundtrip(b, SEED)
    assert variant == SEED.split()
    assert script.op_counts() == {"NOP": 8}
    assert script.cost() == 0.0


def test_morph():
    b = VariantBuilder(SEED, REGISTRY)
    b.keep(0).keep(1).keep(2).inflect(3, "Troia").keep(4).keep(5).keep(6).keep(7)
    variant, script = roundtrip(b, SEED)
    assert variant[3] == "Troia"
    assert script.op_counts()["MORPH"] == 1


def test_substitution_family():
    for tag in ("SYN", "SYN-DIST", "HYPER", "HYPO", "ANT", "NE-SUB", "POS"):
        b = VariantBuilder(SEED, REGISTRY)
        for i in range(len(SEED.split())):
            if i == 4:
                b.substitute(i, "SUBST", tag=tag)
            else:
                b.keep(i)
        variant, script = roundtrip(b, SEED)
        assert variant[4] == "SUBST", tag
        assert script.op_counts()[tag] == 1, tag


def test_hyper_hypo_are_mutual_inverses():
    b = VariantBuilder(SEED, REGISTRY)
    for i in range(len(SEED.split())):
        b.substitute(i, "X", tag="HYPER") if i == 0 else b.keep(i)
    _, script = b.build()
    assert script.invert().operations[0].tag == "HYPO"
    assert script.invert().invert().operations[0].tag == "HYPER"


def test_insert_and_delete():
    b = VariantBuilder(SEED, REGISTRY)
    b.keep(0).delete(1).keep(2).insert("tandem")
    for i in range(3, 8):
        b.keep(i)
    variant, script = roundtrip(b, SEED)
    assert "uirumque" not in variant
    assert "tandem" in variant
    assert script.op_counts()["DEL"] == 1
    assert script.op_counts()["INS"] == 1


def test_ins_del_are_mutual_inverses():
    b = VariantBuilder(SEED, REGISTRY)
    b.delete(0)
    for i in range(1, 8):
        b.keep(i)
    _, script = b.build()
    tags = [op.tag for op in script.invert().operations]
    assert tags[0] == "INS"


def test_reorder_is_a_marker():
    b = VariantBuilder(SEED, REGISTRY)
    for i in range(len(SEED.split())):
        b.keep(i)
    b.mark_reordered(2)
    variant, script = roundtrip(b, SEED)
    assert variant == SEED.split(), "a marker must not change the text"
    assert script.op_counts()["REORDER"] == 1
    assert script.cost() == 0.50


def test_split_and_merge():
    b = VariantBuilder(SEED, REGISTRY)
    b.keep(0).split(1, ["uiros", "que"]).keep(2).merge([3, 4], "Troiaequi")
    for i in range(5, 8):
        b.keep(i)
    variant, script = roundtrip(b, SEED)
    assert "uiros" in variant and "que" in variant
    assert "Troiaequi" in variant
    assert script.op_counts()["SPLIT"] == 1
    assert script.op_counts()["MERGE"] == 1


def test_split_merge_are_mutual_inverses():
    b = VariantBuilder(SEED, REGISTRY)
    b.split(0, ["a", "b"])
    for i in range(1, 8):
        b.keep(i)
    _, script = b.build()
    assert script.invert().operations[0].tag == "MERGE"


def test_quote_covers_its_span():
    b = VariantBuilder(SEED, REGISTRY)
    b.quote([0, 1, 2, 3])
    for i in range(4, 8):
        b.keep(i)
    variant, script = roundtrip(b, SEED)
    assert variant[:4] == SEED.split()[:4]
    assert script.op_counts()["QUOTE"] == 1
    assert "NOP" not in script.op_counts() or script.op_counts().get("NOP", 0) == 4


def test_frame_inserts_and_inverts():
    b = VariantBuilder(SEED, REGISTRY)
    b.frame(["ut", "ait", "Cicero"])
    for i in range(len(SEED.split())):
        b.keep(i)
    variant, script = roundtrip(b, SEED)
    assert variant[:3] == ["ut", "ait", "Cicero"]
    assert script.op_counts()["FRAME"] == 1


def test_adapt_and_disperse_are_markers():
    for method, tag in (("mark_adapted", "ADAPT"), ("mark_dispersed", "DISPERSE")):
        b = VariantBuilder(SEED, REGISTRY)
        for i in range(len(SEED.split())):
            b.keep(i)
        getattr(b, method)([1, 2])
        variant, script = roundtrip(b, SEED)
        assert variant == SEED.split(), tag
        assert script.op_counts()[tag] == 1, tag


# =============================================================================
# Structural invariants
# =============================================================================


def test_cost_is_linear_in_op_counts():
    """The invariant the retired prototype violated."""
    b = VariantBuilder(SEED, REGISTRY)
    b.quote([0, 1, 2]).inflect(3, "Troia").substitute(4, "quisquis", tag="SYN")
    b.delete(5).insert("tandem").keep(6).keep(7).mark_reordered(0)
    _, script = b.build()
    costs = CostModel(registry=REGISTRY)
    reconstructed = sum(costs.costs[t] * n for t, n in script.op_counts().items())
    assert abs(reconstructed - script.cost()) < 1e-9


def test_double_inversion_is_identity():
    b = VariantBuilder(SEED, REGISTRY)
    b.quote([0, 1]).inflect(2, "canto").split(3, ["a", "b"]).delete(4)
    b.insert("x").keep(5).keep(6).keep(7)
    _, script = b.build()
    twice = script.invert().invert()
    assert [o.tag for o in twice.operations] == [o.tag for o in script.operations]
    assert [o.target_tokens for o in twice.operations] == [
        o.target_tokens for o in script.operations
    ]


def test_validity_rejects_double_write():
    ops = [
        EditOperation("NOP", (0,), (0,), ("a",), ("a",)),
        EditOperation("NOP", (1,), (0,), ("b",), ("b",)),
    ]
    validity = SCRIBA.validate(EditScript(["a", "b"], [], ops, REGISTRY), 2)
    assert not validity
    assert any("written twice" in p for p in validity.problems)


def test_validity_rejects_double_consume():
    ops = [
        EditOperation("NOP", (0,), (0,), ("a",), ("a",)),
        EditOperation("NOP", (0,), (1,), ("a",), ("a",)),
    ]
    validity = SCRIBA.validate(EditScript(["a", "b"], [], ops, REGISTRY), 2)
    assert not validity
    assert any("consumed twice" in p for p in validity.problems)


def test_validity_rejects_gap():
    ops = [
        EditOperation("NOP", (0,), (0,), ("a",), ("a",)),
        EditOperation("NOP", (1,), (2,), ("b",), ("b",)),
    ]
    validity = SCRIBA.validate(EditScript(["a", "b"], [], ops, REGISTRY), 2)
    assert not validity
    assert any("never written" in p for p in validity.problems)


def test_validity_rejects_out_of_range_source():
    ops = [EditOperation("NOP", (9,), (0,), ("a",), ("a",))]
    validity = SCRIBA.validate(EditScript(["a"], [], ops, REGISTRY), 1)
    assert not validity
    assert any("out of range" in p for p in validity.problems)


def test_execute_raises_on_invalid_script():
    ops = [
        EditOperation("NOP", (0,), (0,), ("a",), ("a",)),
        EditOperation("NOP", (1,), (0,), ("b",), ("b",)),
    ]
    try:
        SCRIBA.execute(EditScript(["a", "b"], [], ops, REGISTRY), ["a", "b"])
    except ValueError:
        return
    raise AssertionError("execute should refuse an invalid script")


def test_verify_returns_false_rather_than_raising():
    ops = [
        EditOperation("NOP", (0,), (0,), ("a",), ("a",)),
        EditOperation("NOP", (1,), (0,), ("b",), ("b",)),
    ]
    assert SCRIBA.verify(EditScript(["a", "b"], [], ops, REGISTRY), ["a", "b"], ["a"]) is False


# =============================================================================
# Edge cases
# =============================================================================


def test_delete_everything_then_insert_everything():
    b = VariantBuilder("a b c", REGISTRY)
    b.delete(0).delete(1).delete(2).insert("x").insert("y")
    variant, script = roundtrip(b, "a b c")
    assert variant == ["x", "y"]
    assert script.cost() == 5 * 1.00


def test_single_token():
    b = VariantBuilder("solus", REGISTRY)
    b.keep(0)
    variant, _ = roundtrip(b, "solus")
    assert variant == ["solus"]


def test_normalization_folds_orthography():
    assert normalize("Uox,") == normalize("uox")
    assert normalize("iam") == normalize("jam")


def test_every_operation_is_exercised():
    """No tag in the registry goes untested."""
    b = VariantBuilder(SEED, REGISTRY)
    b.frame(["ut"]).quote([0, 1, 2]).keep(3).inflect(4, "X")
    b.substitute(5, "Y", tag="SYN").substitute(6, "Z", tag="SYN-DIST")
    b.substitute(7, "W", tag="HYPER").insert("i").mark_reordered(0)
    b.mark_adapted([0]).mark_dispersed([1])
    _, s1 = b.build()
    c = VariantBuilder("a b c d e f", REGISTRY)
    c.substitute(0, "A", tag="HYPO").substitute(1, "B", tag="ANT")
    c.substitute(2, "C", tag="NE-SUB").substitute(3, "D", tag="POS")
    c.split(4, ["e1", "e2"]).delete(5)
    _, s2 = c.build()
    d = VariantBuilder("g h", REGISTRY)
    d.merge([0, 1], "gh")
    _, s3 = d.build()
    e = VariantBuilder("k l m n", REGISTRY)
    e.subst(0, "K").keep(1).form(2, "M").sense(3, "N")
    _, s4 = e.build()
    seen = set(s1.op_counts()) | set(s2.op_counts()) | set(s3.op_counts()) | set(s4.op_counts())
    missing = set(REGISTRY.tags()) - seen
    assert not missing, f"never exercised: {sorted(missing)}"


# =============================================================================
# The tabular view
# =============================================================================


def test_rows_cover_every_operation():
    rows = REGISTRY.rows()
    assert len(rows) == len(REGISTRY)
    assert [r["tag"] for r in rows] == list(REGISTRY.tags()), "rows must keep registration order"
    fields = {"tag", "level", "role", "cost", "detectable", "requires"}
    assert all(set(r) == fields for r in rows)


def test_rows_report_costs_and_requirements():
    by_tag = {r["tag"]: r for r in REGISTRY.rows()}
    assert by_tag["SYN"]["cost"] == REGISTRY["SYN"].default_cost
    assert by_tag["SYN"]["requires"] == "wordnet"
    # An operation needing nothing must not render as an empty string, which
    # would read as missing data rather than as "no requirement".
    assert by_tag["NOP"]["requires"] == "\u2014"


def test_to_frame_is_indexed_by_tag():
    try:
        import pandas  # noqa: F401
    except ImportError:
        return
    frame = REGISTRY.to_frame()
    assert frame.index.name == "tag"
    assert len(frame) == len(REGISTRY)
    assert list(frame.columns) == ["level", "role", "cost", "detectable", "requires"]


# =============================================================================
# The builder interface
# =============================================================================


def test_every_tag_has_a_builder_method():
    """One named method per operation, so no script needs a tag string."""
    mapping = VariantBuilder.METHOD_FOR_TAG
    assert set(mapping) == set(REGISTRY.tags()), (
        f"unmapped: {set(REGISTRY.tags()) - set(mapping)}; "
        f"stale: {set(mapping) - set(REGISTRY.tags())}"
    )
    builder = VariantBuilder(SEED, REGISTRY)
    for tag, name in mapping.items():
        assert callable(getattr(builder, name, None)), f"{tag}: no method {name!r}"


def test_substitution_helpers_match_substitute():
    """Each helper is exactly its tag's substitute call, not a variant of it."""
    for tag, name in [
        ("SYN", "syn"),
        ("SYN-DIST", "syn_dist"),
        ("HYPER", "hyper"),
        ("HYPO", "hypo"),
        ("ANT", "ant"),
        ("NE-SUB", "ne_sub"),
        ("POS", "pos"),
    ]:
        generic = VariantBuilder(SEED, REGISTRY)
        generic.substitute(0, "aliud", tag=tag, detail="d")
        _, expected = generic.build()

        named = VariantBuilder(SEED, REGISTRY)
        getattr(named, name)(0, "aliud", detail="d")
        _, actual = named.build()

        assert actual.serialize() == expected.serialize(), f"{name} diverges from substitute"


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
