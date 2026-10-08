# tests/test_cltk_relations.py
"""
Relations that CLTK made available: derivation, named entities, enclitics.

Each was previously recorded as impossible. These tests pin down what is now
detected, and — more importantly — what must still be refused, since the risk
with all three is firing too readily rather than too rarely.

Tests needing the network or a corpus skip rather than fail, so the suite stays
runnable offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.operations import OperationRegistry  # noqa: E402
from retexo.resources import Resources  # noqa: E402
from retexo.resources.entities import Entities  # noqa: E402
from retexo.resources.morphology import Morphology  # noqa: E402
from retexo.resources.wordnet import EMPTY  # noqa: E402

REGISTRY = OperationRegistry.default()


class StubVectors:
    """Vectors with a fixed similarity table."""

    available = True

    def __init__(self, table):
        self.table = table

    def similarity(self, a, b):
        return self.table.get(frozenset((a, b)))


class StubResources(Resources):
    """Real resources, with the vectors replaced by a fixed table."""

    def __init__(self, table):
        super().__init__()
        self._vectors = StubVectors(table)

    def has(self, name):
        return True if name == "vectors" else super().has(name)


# =============================================================================
# Inventory
# =============================================================================


def test_derivation_and_entities_are_declared_resources():
    resources = Resources()
    # "derivation" is answered by the wordnet, but the operation asks for the
    # capability rather than the backend.
    assert resources.has("derivation") == resources.has("wordnet")
    assert "derivatives" in EMPTY


def test_pos_and_ne_sub_are_now_detectable():
    assert REGISTRY["POS"].detectable
    assert REGISTRY["NE-SUB"].detectable
    assert REGISTRY["NE-SUB"].requires == ("entities", "vectors")


# =============================================================================
# Named entities
# =============================================================================


def test_entities_match_case_sensitively():
    """The capital is the whole guard; folding case makes the list useless."""
    entities = Entities()
    if not entities.available:
        return
    assert entities.is_name("Troiae")
    assert entities.is_name("Italiam")
    # Capitalized but not a name: the sentence-initial failure mode that the
    # earlier prototype was rejected for.
    assert not entities.is_name("Arma")
    assert not entities.is_name("arma")


def test_ne_sub_needs_both_names_and_similarity():
    entities = Entities()
    if not entities.available:
        return
    resources = StubResources(
        {
            frozenset(("italia", "hesperia")): 0.71,
            frozenset(("italia", "caesar")): 0.12,
        }
    )
    detect = REGISTRY["NE-SUB"].detect
    assert detect("Italiam", "Hesperiam", resources)  # both names, similar
    assert detect("Italiam", "Caesar", resources) is None  # names, unrelated
    assert detect("Arma", "Italiam", resources) is None  # not a name
    assert detect("Italiam", "Italia", resources) is None  # same name is NOP


def test_ne_sub_declines_without_vectors():
    """Similarity is required, not advisory — no vectors means no detection."""
    resources = Resources(vectors_path=None)
    if resources.has("vectors"):
        return
    assert REGISTRY["NE-SUB"].detect("Italiam", "Hesperiam", resources) is None


# =============================================================================
# Derivation
# =============================================================================


def test_pos_detects_derivational_pairs():
    resources = Resources()
    if not resources.has("derivation") or resources.offline:
        return
    detect = REGISTRY["POS"].detect
    # The pair the old docstring offered as proof this was impossible.
    assert detect("amo", "amor", resources)
    assert detect("cano", "carmen", resources)
    assert detect("fugio", "fuga", resources)


def test_pos_declines_same_lemma_and_unrelated():
    resources = Resources()
    if not resources.has("derivation") or resources.offline:
        return
    # Same lemma is MORPH, not a derivation.
    assert REGISTRY["POS"].detect("arma", "arma", resources) is None
    assert REGISTRY["POS"].detect("arma", "uirum", resources) is None


# =============================================================================
# Morphological features
# =============================================================================


def test_morpho_features_decode_collatinus_tags():
    morphology = Morphology()
    assert morphology.morpho_features("armis") == "pl.dat."
    assert morphology.morpho_features("regem") == "sg.acc."
    assert morphology.morpho_features("amabam") == "1.sg.impf.ind.act."
    # A lemma Collatinus cannot decline is a miss, not an error.
    assert morphology.morpho_features("Lauiniaque") is None


def test_morph_justification_names_the_features():
    resources = Resources()
    if not resources.has("morphology"):
        return
    justification = REGISTRY["MORPH"].detect("arma", "armis", resources)
    assert justification and "lemma=" in justification
    assert "->" in justification, f"features not reported: {justification!r}"


# =============================================================================
# Enclitics
# =============================================================================


def test_enclitics_split_where_they_should():
    morphology = Morphology()
    assert morphology.split_enclitic("uirumque") == ("uirum", "que")
    assert morphology.split_enclitic("Lauiniaque") == ("Lauinia", "que")
    assert morphology.split_enclitic("armaque") == ("arma", "que")
    assert morphology.split_enclitic("bonusne") == ("bonus", "ne")


def test_enclitics_refuse_words_that_merely_end_that_way():
    """The failure mode that matters — CLTK's exceptions list is the guard."""
    morphology = Morphology()
    for word in ["neque", "denique", "quoque", "itaque", "atque", "usque", "namque"]:
        assert morphology.split_enclitic(word) is None, f"{word} was split"
    assert morphology.split_enclitic("arma") is None


def test_enclitic_splitting_is_not_wired_into_detection():
    """SPLIT stays undetectable: the aligner is still one-to-one."""
    assert not REGISTRY["SPLIT"].detectable
    assert not REGISTRY["MERGE"].detectable


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
