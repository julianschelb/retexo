# tests/test_passage.py
"""
The standalone annotation layer.

Two things matter here. That a ``Passage`` still behaves enough like a passage
to be substitutable, and that unknown stays distinguishable from empty -- a
field that is ``None`` because a resource was missing must never be confused
with one that was asked and came back with nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.datasets.passage import Passage, TokenInfo  # noqa: E402
from retexo.resources import Resources  # noqa: E402

TEXT = "arma virumque cano Troiae qui primus ab oris"


# =============================================================================
# Sequence behaviour
# =============================================================================


def test_passage_behaves_like_its_text():
    passage = Passage(TEXT)
    assert str(passage) == TEXT
    assert len(passage) == len(TEXT.split())
    assert passage.surfaces == TEXT.split()


def test_passage_accepts_tokens_as_well_as_a_string():
    tokens = TEXT.split()
    assert Passage(tokens).surfaces == tokens
    assert str(Passage(tokens)) == TEXT


def test_indexing_and_iteration_give_annotated_tokens():
    passage = Passage(TEXT)
    assert isinstance(passage[0], TokenInfo)
    assert passage[0].surface == "arma"
    assert [t.index for t in passage] == list(range(len(passage)))


# =============================================================================
# Annotation
# =============================================================================


def test_construction_reads_nothing():
    """Annotation is lazy: building a passage must not touch a resource."""
    passage = Passage(TEXT)
    assert passage._tokens is None


def test_local_layer_populates_on_access():
    passage = Passage(TEXT)
    if not passage.resources.has("morphology"):
        return
    assert all(t.lemma for t in passage)
    assert passage[1].enclitic == ("virum", "que")


def test_unknown_is_not_the_same_as_empty():
    """None means the resource could not say; () means it said nothing."""
    passage = Passage("arma")
    token = passage[0]
    assert token.synonyms is None, "lexical layer must not run unasked"
    passage.annotate(wordnet=True)
    if passage.resources.has("wordnet"):
        assert passage[0].synonyms is not None, "asking must replace None"


def test_annotate_is_a_no_op_without_flags():
    passage = Passage("arma")
    before = passage[0]
    assert passage.annotate() is passage
    assert passage[0] is before


def test_readings_keep_ambiguity():
    passage = Passage("arma oris")
    if not passage.resources.has("morphology"):
        return
    readings = passage[0].readings
    if readings:  # Collatinus may not know every lemma
        assert len(readings) > 1, "arma is nom/voc/acc plural"
        assert passage[0].features in readings


# =============================================================================
# Views
# =============================================================================


def test_rows_have_one_record_per_token():
    passage = Passage(TEXT)
    rows = passage.rows()
    assert len(rows) == len(passage)
    assert rows[0]["surface"] == "arma"
    # Nothing unestablished may render as an empty string, which would read as
    # missing data rather than as "not established".
    assert all(v != "" or k in ("name", "func") for r in rows for k, v in r.items())


def test_render_names_its_fields():
    passage = Passage(TEXT)
    text = passage.render()
    assert text.startswith("PASSAGE:")
    assert f"TOKENS: {len(passage)}" in text
    assert len(text.splitlines()) >= len(passage)


def test_summary_counts_what_is_known():
    passage = Passage(TEXT)
    summary = passage.summary()
    assert summary["tokens"] == len(passage)
    assert 0.0 <= summary["feature_coverage"] <= 1.0
    assert summary["lemmatized"] <= summary["tokens"]


def test_works_with_no_resources_at_all():
    """A passage with everything switched off is still a passage."""
    bare = Resources(
        vectors_path=None,
        enabled={
            "morphology": False,
            "wordnet": False,
            "vectors": False,
            "entities": False,
        },
    )
    passage = Passage(TEXT, resources=bare)
    assert len(passage) == len(TEXT.split())
    assert passage[0].lemma is None
    assert passage[0].normalized == "arma"
    assert passage.render()


# =============================================================================
# Isolation
# =============================================================================


def test_passage_is_not_wired_into_the_pipeline():
    """It is deliberately standalone; importing it must stay optional."""
    import retexo

    assert not hasattr(retexo, "Passage"), (
        "Passage was exported from the package -- it is meant to stay out of "
        "the pipeline until an experiment says it helps"
    )
    # The word "passage" is ordinary prose in this codebase, so look for the
    # import and the constructor rather than the noun.
    root = Path(__file__).resolve().parents[1] / "retexo"
    for source_file in root.rglob("*.py"):
        if source_file.name == "passage.py":
            continue
        source = source_file.read_text()
        assert "retexo.datasets.passage" not in source, f"{source_file.name} imports it"
        assert "Passage(" not in source, f"{source_file.name} constructs it"


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
