"""Script-style test modules (a ``check()`` counter and ``sys.exit`` at module level) run as subprocesses, see
``test_script_style.py``; pytest must not import them, or their ``sys.exit`` ends the session."""

import re
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT_STYLE_MARKER = re.compile(r"^sys\.exit\(1 if FAILED else 0\)", re.MULTILINE)


def script_style_modules():
    return sorted(
        p
        for p in HERE.glob("test_*.py")
        if SCRIPT_STYLE_MARKER.search(p.read_text(encoding="utf-8"))
    )


collect_ignore = [p.name for p in script_style_modules()]


# ---------------------------------------------------------------------------
# What a test may need that a plain install does not bring: the lexical extra, a spaCy model, the annotation files.
# A test that needs one is skipped with the reason, not failed.
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402

import pytest  # noqa: E402

from retexo.paths import home  # noqa: E402


def _spacy_english() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


_REQUIREMENTS = {
    "cltk": (
        importlib.util.find_spec("cltk") is not None,
        "needs the lexical extra: pip install 'retexo[lexical]'",
    ),
    "spacy-en": (
        _spacy_english(),
        "needs a spaCy English model: python -m spacy download en_core_web_sm",
    ),
    "annotation": (
        (home() / "data" / "gold_full" / "gold_2026-09-24.records.jsonl").exists(),
        "needs the word-level annotation under $RETEXO_HOME/data/gold_full (not part of the repository)",
    ),
}

#: Test node ids (a path, or path::test) and what they need.
NEEDS = {
    "cltk": [
        "tests/test_cltk_relations.py::test_ne_sub_needs_both_names_and_similarity",
        "tests/test_cltk_relations.py::test_morpho_features_decode_collatinus_tags",
        "tests/test_cltk_relations.py::test_enclitics_split_where_they_should",
        "tests/test_cltk_relations.py::test_enclitics_refuse_words_that_merely_end_that_way",
        "tests/test_resources.py::test_morphology_lemmatizes",
        "tests/test_resources.py::test_same_lemma_excludes_identity",
        "tests/test_resources.py::test_pos_candidates_are_ordered_and_complete",
        "tests/test_resources.py::test_detects_inflection",
        "tests/test_resources.py::test_excluding_operations_changes_the_analysis",
        "tests/test_passage.py::test_unknown_is_not_the_same_as_empty",
    ],
    "spacy-en": [
        "tests/test_baseline_sultan.py::test_the_driver_never_touches_evidence",
        "tests/test_baseline_sultan.py::test_sultan_aligner_english_mode_end_to_end",
        "tests/test_baseline_sultan.py::test_english_featurizer_word_sim_tiers",
        "tests/test_baseline_sultan.py::test_english_dependency_parser_root_and_pos",
    ],
    "annotation": [
        "tests/test_baseline_synthetic.py::test_coverage_of_the_gold_against_itself_has_no_gaps",
        "tests/test_baseline_synthetic.py::test_a_tiny_pool_replays_comes_in_both_orientations_and_excludes_held_out_passages",
        "tests/test_baseline_harness.py::test_gold_to_records",
    ],
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for need, ids in NEEDS.items():
            available, reason = _REQUIREMENTS[need]
            if (
                not available
                and any(
                    item.nodeid == i or item.nodeid.startswith(i + "[") for i in ids if "::" in i
                )
                or (
                    not available
                    and any(item.nodeid.startswith(i + "::") for i in ids if "::" not in i)
                )
            ):
                item.add_marker(pytest.mark.skip(reason=reason))
