"""Tests for span localisation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.datasets.localize import WINDOW_SLACK, Localized  # noqa: E402

PASSED = FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}")


def test_short_pair_untouched():
    source = "arma uirumque cano".split()
    target = "arma uirumque cecini".split()
    span = Localized.find(source, target)
    check("equal lengths untouched", span.source == source and span.target == target)
    check("nothing dropped", not span.trimmed)


def test_window_finds_the_reuse():
    source = "decem tulerunt fastidia menses".split()
    target = (
        "quae autem potest maior esse clementia quam ut filius "
        "decem mensum fastidia sustineret partus expectaret aduentum "
        "inuolueretur pannis subiceretur parentibus"
    ).split()
    span = Localized.find(source, target)
    check("source kept whole", span.source == source)
    check("target trimmed", len(span.target) < len(target))
    check("dropped is recorded", span.dropped == len(target) - len(span.target))
    check("window holds the reuse", "fastidia" in span.target and "decem" in span.target)


def test_window_respects_slack():
    source = ["a"] * 10
    target = ["a"] * (10 + WINDOW_SLACK)
    span = Localized.find(source, target)
    check("within slack is untouched", span.target == target)


def test_longer_source_is_trimmed():
    target = "decem tulerunt fastidia".split()
    source = (
        "incipe parue puer risu cognoscere matrem matri longa decem tulerunt fastidia menses"
    ).split()
    span = Localized.find(source, target)
    check("source trimmed", len(span.source) < len(source))
    check("target kept whole", span.target == target)
    check("window holds the reuse", "fastidia" in span.source)
    # The reuse sits at the end here, so the window cannot start at zero.
    check("offset recorded", span.source_offset > 0)


def test_disabled_is_identity():
    source, target = ["a"], ["b"] * 40
    span = Localized.find(source, target, enabled=False)
    check("disabled keeps both", span.source == source and span.target == target)


def test_empty_is_safe():
    span = Localized.find([], ["a", "b"])
    check("empty source safe", span.source == [] and span.target == ["a", "b"])


for fn in [
    test_short_pair_untouched,
    test_window_finds_the_reuse,
    test_window_respects_slack,
    test_longer_source_is_trimmed,
    test_disabled_is_identity,
    test_empty_is_safe,
]:
    fn()

print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
