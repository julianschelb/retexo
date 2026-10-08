"""Tests for what the generator is willing to sample."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.datasets.generation import (  # noqa: E402
    GenerationConfig,
    MockSubstitutionSource,
    SyntheticGenerator,
)
from retexo.datasets.substitution import CandidatePool, ContextualSubstitutionSource  # noqa: E402

PASSED = FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}")


def contextual():
    return SyntheticGenerator(
        ContextualSubstitutionSource(resources=None, pool=CandidatePool()),
        GenerationConfig(),
    )


def test_mock_realises_everything():
    generator = SyntheticGenerator(MockSubstitutionSource(), GenerationConfig())
    check("mock drops nothing", generator.dropped_tags() == [])
    check(
        "mock weights unchanged",
        generator._usable_weights()
        == {
            k: v / sum(generator.config.weights.values())
            for k, v in generator.config.weights.items()
        },
    )


def test_contextual_drops_wordnet_relations():
    dropped = contextual().dropped_tags()
    for tag in ("SYN", "HYPER", "HYPO", "ANT"):
        check(f"{tag} dropped", tag in dropped)
    for tag in ("MORPH", "SYN-DIST"):
        check(f"{tag} kept", tag not in dropped)


def test_dropping_does_not_inflate_insertion():
    generator = contextual()
    weights = generator._usable_weights()
    original = generator.config.weights
    before = (original["INS"] + original["DEL"]) / sum(original.values())
    after = weights["INS"] + weights["DEL"]
    check("insertion share preserved", abs(before - after) < 1e-9)


def test_weights_are_a_distribution():
    weights = contextual()._usable_weights()
    check("sums to one", abs(sum(weights.values()) - 1.0) < 1e-9)
    check("all non-negative", all(v >= 0 for v in weights.values()))


def test_sampled_tags_are_realisable():
    import random

    generator = contextual()
    rng = random.Random(0)
    structural = {"INS", "DEL", "REORDER"}
    sampled = {generator._sample_tag(rng) for _ in range(400)}
    unrealisable = {
        t for t in sampled if t not in structural and not generator.substitutions.realises(t)
    }
    check("never samples an unrealisable tag", not unrealisable)


for fn in [
    test_mock_realises_everything,
    test_contextual_drops_wordnet_relations,
    test_dropping_does_not_inflate_insertion,
    test_weights_are_a_distribution,
    test_sampled_tags_are_realisable,
]:
    fn()

print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
