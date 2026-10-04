# tests/test_resources.py
"""
Tests for the ported resources and the oracle that uses them.

Resources are exercised offline against the pinned cache, so the suite does not
depend on a remote server being up and does not spend a network round-trip per
assertion.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from retexo import EditPlanOracle, Resources, Scriba  # noqa: E402
from retexo.core.oracle import GreedyAligner, OptimalAligner  # noqa: E402

RESOURCES = Resources(offline=True)
ORACLE = EditPlanOracle(resources=RESOURCES)
SCRIBA = Scriba()


# =============================================================================
# Resources
# =============================================================================


def test_status_reports_every_backend():
    status = RESOURCES.status()
    for key in ("morphology", "wordnet", "wordnet_cached", "vectors"):
        assert key in status


def test_morphology_lemmatizes():
    assert RESOURCES.morphology.lemma("armis") == "arma"
    assert RESOURCES.morphology.lemma("canit") == "cano"


def test_same_lemma_excludes_identity():
    """An identical surface is a copy, not an inflection."""
    assert RESOURCES.morphology.same_lemma("arma", "armis")
    assert not RESOURCES.morphology.same_lemma("arma", "arma")


def test_pos_candidates_are_ordered_and_complete():
    candidates = RESOURCES.morphology.pos_candidates("cano")
    assert candidates[0] == "v", "a marked verb should be queried as a verb first"
    assert set(candidates) == {"n", "v", "a", "r"}, "every part of speech stays reachable"


def test_wordnet_reads_the_cache_offline():
    record = RESOURCES.wordnet.lookup("animus", "n")
    assert isinstance(record.get("synonyms"), list)
    assert RESOURCES.wordnet.cached_count() > 0


def test_missing_vectors_report_unavailable_rather_than_zero():
    """An absent model must not look like universal dissimilarity."""
    if not RESOURCES.vectors.available:
        assert RESOURCES.has("vectors") is False


def test_lemma_or_surface_falls_back():
    disabled = Resources(offline=True, enabled={"morphology": False})
    assert disabled.lemma_or_surface("Armis,") == "armis"


# =============================================================================
# Oracle
# =============================================================================


def test_detects_identity_and_reordering():
    script = ORACLE.plan("uox faucibus haesit", "haesit uox faucibus")
    counts = script.op_counts()
    assert counts.get("NOP") == 3
    assert counts.get("REORDER") == 1


def test_detects_inflection():
    script = ORACLE.plan("arma uirumque cano", "arma uirosque cano")
    assert script.op_counts().get("MORPH") == 1


def test_plans_replay_to_the_target():
    for source, target in [
        ("uox faucibus haesit", "haesit uox faucibus"),
        ("arma uirumque cano", "arma uirosque cano"),
        ("ingentes animos", "animos ingentes"),
    ]:
        script = ORACLE.plan(source, target)
        assert SCRIBA.verify(script, source.split(), target.split()), (source, target)


def test_unrelated_pairs_cost_more_than_related_ones():
    """The distance must separate reuse from non-reuse, or it is not a metric."""
    related = ORACLE.plan("arma uirumque cano", "arma uirosque cano")
    unrelated = ORACLE.plan("arma uirumque cano", "quis talia fando temperet")
    assert unrelated.normalized_cost() > related.normalized_cost()


def test_excluding_operations_changes_the_analysis():
    """The vocabulary ablation must actually remove an operation's reach."""
    full = ORACLE.plan("arma uirumque cano", "arma uirosque cano")
    ablated = EditPlanOracle(resources=RESOURCES, exclude={"MORPH"}).plan(
        "arma uirumque cano", "arma uirosque cano"
    )
    assert "MORPH" in full.op_counts()
    assert "MORPH" not in ablated.op_counts()
    assert ablated.cost() > full.cost(), "removing a relation must not make a pair cheaper"


def test_optimal_and_greedy_aligners_both_produce_valid_scripts():
    for aligner in (OptimalAligner(), GreedyAligner()):
        oracle = EditPlanOracle(resources=RESOURCES, aligner=aligner)
        script = oracle.plan("uox faucibus haesit", "haesit uox faucibus")
        assert SCRIBA.verify(script, "uox faucibus haesit".split(),
                             "haesit uox faucibus".split()), type(aligner).__name__


def test_relation_lookups_are_cached():
    oracle = EditPlanOracle(resources=RESOURCES)
    oracle.plan("arma uirumque cano", "arma uirosque cano")
    assert oracle._cache, "relation answers must be memoised for bulk scoring"


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
