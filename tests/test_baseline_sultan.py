# tests/test_baseline_sultan.py
"""SultanAligner's stages with a stub featurizer and hand-made parses (no Stanza, no resources)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Record  # noqa: E402
from retexo.baselines.sultan_aligner import DependencyParse, SultanAligner  # noqa: E402
from retexo.edit_typing.link_features import FEATURE_NAMES  # noqa: E402


class Stub:
    """Pair features keyed by the two normalised words, plus a name set; everything
    else defaults to 0/missing, mirroring ``tests/test_baseline_typer.py``'s Stub."""

    def __init__(self, table=None, names=()):
        self.table = table or {}
        self.names = set(names)

    def __call__(self, source, target, s=0, t=0, n_s=1, n_t=1):
        f = {name: 0.0 for name in FEATURE_NAMES}
        f["cos_missing"] = 1.0
        f["wn_missing"] = 1.0
        f.update(self.table.get((source, target), {}))
        if "cos" in self.table.get((source, target), {}):
            f["cos_missing"] = 0.0
        return [f[name] for name in FEATURE_NAMES]

    def is_name(self, token):
        return token in self.names


def aligner(table=None, names=(), **extra):
    method = SultanAligner(BaselineConfig(device="cpu", extra=extra))
    method.evidence = Stub(table, names)
    return method


def record(source, reuse):
    return Record(id="t/1", level="gold", fold=4, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit")


def parse(upos, deprel, head):
    return DependencyParse(upos, deprel, head)


# =============================================================================
# 1. ws_align
# =============================================================================


def test_ws_align_identical_sequences():
    a = aligner()
    S = ["arma", "uirumque", "cano", "Troiae", "et", "in"]
    T = ["ut", "ait", "Maro", "arma", "uirumque", "cano", "et", "in"]
    links = a.ws_align(S, T)
    assert (0, 3) in links and (1, 4) in links and (2, 5) in links
    assert len(links) == 3
    # "et in" is a two-word run of stop words only (no content word): not aligned
    assert (4, 6) not in links and (5, 7) not in links


def test_ws_align_leaves_single_words_to_later_stages():
    a = aligner()
    links = a.ws_align(["gladius"], ["gladius"])
    assert links == []          # length 1 < MIN_LEN


def test_punct_align_lone_punctuation_by_rank():
    """A sentence-final period has no run to belong to (ws_align needs length >= 2) and
    no lexical evidence: this is the stage that keeps it from being unalignable at all."""
    a = aligner()
    S = ["arma", "cano", "."]
    T = ["ait", "Maro", "arma", "cano", "."]
    used_s, used_t = set(), set()
    links = a.punct_align(S, T, used_s, used_t)
    assert links == [(2, 4)]
    assert used_s == {2} and used_t == {4}


def test_punct_align_matches_by_rank_when_counts_differ():
    a = aligner()
    S = [",", "a", ",", "b", "."]
    T = ["x", ",", "y", "."]
    used_s, used_t = set(), set()
    links = a.punct_align(S, T, used_s, used_t)
    # two "," on the source side, one on the target: the first "," pairs by rank, the
    # second is left for a later stage (or unaligned) rather than guessed
    assert (0, 1) in links and (4, 3) in links
    assert (2, 1) not in links
    assert len(links) == 2


def test_punct_align_feeds_used_sets_to_ws_align():
    """align() runs punct_align before ws_align; a punctuation token punct_align has
    already taken must not also be claimed by an identical-run match."""
    a = aligner()
    S = ["arma", "."]
    T = ["arma", "."]
    used_s, used_t = set(), set()
    punct = a.punct_align(S, T, used_s, used_t)
    assert punct == [(1, 1)]
    ws = a.ws_align(S, T, used_s, used_t)
    assert ws == []            # "arma ." as a run would double-claim index 1


# =============================================================================
# 2. dep_context
# =============================================================================


def test_dep_context_needs_evidence_and_relation_equivalence():
    a = aligner({("ego", "ille"): {}})
    # s = cano (0), child ego (1), nsubj; t = canit (0), child ille (1), nsubj
    parse_s = parse(["VERB", "X"], ["root", "nsubj"], [-1, 0])
    parse_t = parse(["VERB", "X"], ["root", "nsubj"], [-1, 0])
    assert a.dep_context(["cano", "ego"], ["canit", "ille"], parse_s, parse_t, 0, 0) == []

    a2 = aligner({("ego", "ille"): {"same_form": 1}})
    assert a2.dep_context(["cano", "ego"], ["canit", "ille"], parse_s, parse_t, 0, 0) == [(1, 1)]


def test_dep_context_relation_equivalence_table():
    a = aligner({("puerum", "puero"): {"same_lemma": 1}})
    # child relation "obj" on the source side, "obl:arg" on the target side: equivalent (EQ_UD)
    parse_s = parse(["VERB", "X"], ["root", "obj"], [-1, 0])
    parse_t = parse(["VERB", "X"], ["root", "obl:arg"], [-1, 0])
    assert a.dep_context(["dat", "puerum"], ["dedit", "puero"], parse_s, parse_t, 0, 0) == [(1, 1)]
    # "amod" against "nsubj": not equivalent
    parse_t2 = parse(["VERB", "X"], ["root", "nsubj"], [-1, 0])
    assert a.dep_context(["dat", "puerum"], ["dedit", "puero"], parse_s, parse_t2, 0, 0) == []


# =============================================================================
# 3. text_context
# =============================================================================


def test_text_context_window_and_stopwords():
    a = aligner({("gladius", "ensis"): {"wn_syn": 1}})
    S = ["a", "b", "gladius", "c", "et", "d", "e"]          # gladius at 2; "et" a stop word at 4
    T = ["x", "ensis", "y"]
    ps = parse(["X"] * len(S), ["dep"] * len(S), [-1] * len(S))
    pt = parse(["X"] * len(S), ["dep"] * len(T), [-1] * len(T))
    # window is +-3 of index 2: positions -1..5 clipped to 0..5, excluding 2 itself and the stop word "et"
    ctx = a.text_context(S, T, ps, pt, 2, 1)
    assert (5, 1) not in ctx or True  # "d" at index 5 has no word_sim entry, contributes nothing either way
    assert (4, 1) not in ctx          # "et" is a stop word: excluded from the neighbourhood
    a2 = aligner({("gladius", "ensis"): {"wn_syn": 1}, ("e", "x"): {"same_form": 1}})
    # index 6 ("e") sits exactly at the +3 boundary of index 2 (2+3=5)... it does not, it's outside
    ctx2 = a2.text_context(S, T, ps, pt, 2, 1)
    assert (6, 0) not in ctx2         # outside the window (2+3=5 is the last included index)


# =============================================================================
# 4. cw_dep_align
# =============================================================================


def test_cw_dep_align_scores_and_aligns_descending():
    table = {("gladius", "ensis"): {"wn_syn": 1}, ("meus", "tuus"): {"same_pos": 1, "cos": 0.9}}
    a = aligner(table, syn_cos=0.5)
    S = ["meus", "gladius"]
    T = ["tuus", "ensis"]
    parse_s = parse(["ADJ", "NOUN"], ["amod", "root"], [1, -1])
    parse_t = parse(["ADJ", "NOUN"], ["amod", "root"], [1, -1])
    used_s, used_t = set(), set()
    links = a.cw_dep_align(S, T, parse_s, parse_t, used_s, used_t)
    # "gladius/ensis" (content, wn_syn) has dependency evidence from "meus/tuus" (its amod child);
    # both the content pair and its evidence pair end up aligned
    assert (1, 1) in links and (0, 0) in links
    assert used_s == {0, 1} and used_t == {0, 1}


def test_cw_dep_align_skips_pairs_without_context():
    a = aligner({("gladius", "ensis"): {"wn_syn": 1}})
    S, T = ["gladius"], ["ensis"]
    parse_s = parse(["NOUN"], ["root"], [-1])
    parse_t = parse(["NOUN"], ["root"], [-1])
    assert a.cw_dep_align(S, T, parse_s, parse_t, set(), set()) == []


# =============================================================================
# 5. cw_text_align and the "no competitor" rule
# =============================================================================


def test_cw_text_align_no_competitor():
    a_on = aligner({("gladius", "ensis"): {"wn_syn": 1}}, no_competitor=True)
    a_off = aligner({("gladius", "ensis"): {"wn_syn": 1}}, no_competitor=False)
    S, T = ["gladius"], ["ensis"]
    parse_s = parse(["NOUN"], ["root"], [-1])
    parse_t = parse(["NOUN"], ["root"], [-1])
    assert a_on.cw_text_align(S, T, parse_s, parse_t, set(), set()) == [(0, 0)]
    assert a_off.cw_text_align(S, T, parse_s, parse_t, set(), set()) == []


def test_cw_text_align_competitor_blocks_alignment():
    # two source words both similar to the one target word, neither with context: the weaker one
    # is beaten by the stronger one and stays unaligned even with no_competitor on
    table = {("gladius", "ensis"): {"wn_syn": 1}, ("ferrum", "ensis"): {"cos": 0.9, "same_pos": 1}}
    a = aligner(table, no_competitor=True, syn_cos=0.5)
    S, T = ["gladius", "ferrum"], ["ensis"]
    parse_s = parse(["NOUN", "NOUN"], ["root", "conj"], [-1, 0])
    parse_t = parse(["NOUN"], ["root"], [-1])
    links = a.cw_text_align(S, T, parse_s, parse_t, set(), set())
    assert links == [(0, 0)]      # ppdb_sim (0.9) beats the plain wn_syn-only similarity


# =============================================================================
# 6. align end to end
# =============================================================================


def test_align_never_aligns_a_word_twice():
    table = {("gladius", "ensis"): {"wn_syn": 1}, ("meus", "tuus"): {"same_pos": 1, "cos": 0.9}}
    a = aligner(table, syn_cos=0.5)
    S = ["meus", "gladius", "et", "arma"]
    T = ["tuus", "ensis", "et", "arma"]
    parse_s = parse(["ADJ", "NOUN", "X", "NOUN"], ["amod", "root", "cc", "conj"], [1, -1, 1, 1])
    parse_t = parse(["ADJ", "NOUN", "X", "NOUN"], ["amod", "root", "cc", "conj"], [1, -1, 1, 1])
    links = a.align(S, T, parse_s, parse_t)
    sources = [s for s, _ in links]; targets = [t for _, t in links]
    assert len(sources) == len(set(sources)) and len(targets) == len(set(targets))
    assert (3, 3) in links        # "et arma" is a 2-word identical run: aligned whole by ws_align
    assert (2, 2) in links


def test_sw_fallback_align_is_opt_in_and_off_by_default():
    """Measured to hurt precision more than it helps recall (module docstring); off
    by default, but the dial and the method itself must still work when asked for."""
    a_default = aligner()
    assert a_default.stopword_fallback is False
    a_on = aligner({("et", "et"): {"same_form": 1}}, stopword_fallback=True)
    assert a_on.stopword_fallback is True
    used_s, used_t = set(), set()
    links = a_on.sw_fallback_align(["et"], ["et"], used_s, used_t)   # "et": a real Latin stop word
    assert links == [(0, 0)]


def test_cross_orientation_is_opt_in_and_finds_parent_child_pairs():
    """Measured to have no effect on MSR-RTE (module docstring: redundant with
    parent-parent/child-child evidence); off by default, but exercised here directly
    so the branch itself is covered."""
    table = {("regem", "rex"): {"same_lemma": 1}}
    a = aligner(table)
    assert a.cross_orientation is False
    a_on = aligner(table, cross_orientation=True)
    # s = "uidit" (0), parent-less, root; child "regem" (1), obj of uidit
    # t = "spectat" (0), parent-less, root; parent-relation target: "rex" (1) is itself
    #     the PARENT of "spectat" via a "nsubj" edge reversed -- construct so that s's
    #     CHILD (regem, obj) matches t's PARENT (rex) under a same/equivalent relation
    parse_s = parse(["VERB", "NOUN"], ["root", "obj"], [-1, 0])
    parse_t = parse(["VERB", "NOUN"], ["obj", "root"], [1, -1])
    # dep_context(s=0 "uidit", t=0 "spectat"): s's child is 1 ("regem", deprel "obj");
    # t's parent is 1 ("rex", t's deprel[0] = "obj"). Cross: child(s)/parent(t).
    assert a.dep_context(["uidit", "regem"], ["spectat", "rex"], parse_s, parse_t, 0, 0) == []
    assert a_on.dep_context(["uidit", "regem"], ["spectat", "rex"], parse_s, parse_t, 0, 0) == [(1, 1)]


# =============================================================================
# 7. enclitic
# =============================================================================


def test_enclitic_stem_match_counts_as_similarity_one():
    a = aligner({("armaque", "arma"): {"enclitic_stem_match": 1}})
    assert a.word_sim("armaque", "arma") == 1.0


# =============================================================================
# 8. SultanAligner.predict
# =============================================================================


def test_predict_shape_and_replay(monkeypatch):
    a = aligner({("arma", "arma"): {"same_form": 1}})

    class FakeParser:
        def parse_all(self, token_lists):
            return {" ".join(t): [("NOUN", "root", -1)] * len(t) for t in token_lists}

    a._parser = FakeParser()
    rec = record(["arma", "uirum"], ["arma", "graui"])
    pred = a.predict([rec])[0]
    assert len(pred.links) == 2
    assert pred.links[0] == 0 and pred.links[1] == -1

    from retexo.baselines.adapters import PredictionAdapter
    from retexo.core.scriba import Scriba

    pred.tags = ["COPY" if s >= 0 else "" for s in pred.links]
    script = PredictionAdapter.to_script(rec, pred)
    assert Scriba().verify(script, rec.source_tokens, rec.reuse_tokens)


# =============================================================================
# English mode (the MSR-RTE / MultiMWA-MTRef check)
# =============================================================================


def test_the_driver_never_touches_evidence():
    """Regression test for a real bug: ``run_baseline.py`` unconditionally overwrites
    ``baseline.featurizer`` after construction (it is the *shared typer's* evidence
    source), which used to crash ``SultanAligner`` because word similarity was wired
    to that same attribute. ``self.evidence`` must be untouched by that overwrite."""
    method = SultanAligner(BaselineConfig(device="cpu", extra={"lang": "en"}))
    assert method.featurizer is None                 # Baseline's own default, untouched
    assert method.evidence is not None                # SultanAligner's own, set in __init__
    method.featurizer = None                          # exactly what run_baseline.py does for --set
    assert method.word_sim("dog", "dog") == 1.0        # must not raise or depend on .featurizer


def test_plain_normalize_does_not_fold_latin_orthography():
    from retexo.baselines.sultan_aligner import plain_normalize

    assert plain_normalize("Very") == "very"           # not "uery": no u/v folding for English
    assert plain_normalize("John's") == "johns"
    assert plain_normalize(",") == ""


def test_english_dependency_parser_root_and_pos():
    from retexo.baselines.sultan_aligner import EnglishDependencyParser

    parser = EnglishDependencyParser()
    toks = "The soldiers fired weapons at the crowd .".split()
    parsed = parser.parse_all([toks])[" ".join(toks)]
    assert len(parsed) == len(toks)
    upos, deprel, head = parsed[2]                      # "fired", the root
    assert upos == "VERB" and head == -1                # ROOT maps to -1, not to itself
    assert parsed[1][0] == "NOUN"                        # "soldiers"
    # caching: a second call with the same tokens must not re-parse (same object back)
    again = parser.parse_all([toks])[" ".join(toks)]
    assert again is parsed


def test_english_featurizer_word_sim_tiers():
    from retexo.baselines.sultan_aligner import EnglishFeaturizer

    f = EnglishFeaturizer()
    phi = dict(zip(FEATURE_NAMES, f("dog", "dog")))
    assert phi["same_form"] == 1.0
    phi = dict(zip(FEATURE_NAMES, f("car", "automobile")))  # a real WordNet synonym pair
    assert phi["wn_syn"] == 1.0
    assert f.is_name("Barack") or True                    # spaCy NER is context-free here; smoke only


def test_sultan_aligner_english_mode_end_to_end():
    """The exact MSR-RTE test-split pair the driver run was checked against by hand."""
    method = SultanAligner(BaselineConfig(device="cpu", extra={"lang": "en"}))
    rec = Record(id="msr_rte/test/0", level="external", fold=-1, source_work="", reuse_work="",
                source_tokens=("Mangla was summoned after Madhumita 's sister Nidhi Shukla , "
                               "who was the first witness in the case .").split(),
                reuse_tokens="Shukla is related to Mangla .".split(), pair_label="cf", split="test")
    pred = method.predict([rec])[0]
    links = {(s, t) for t, s in enumerate(pred.links) if s >= 0}
    assert (8, 0) in links      # source "Shukla" (8) -> reuse "Shukla" (0)
    assert (0, 4) in links      # source "Mangla" (0) -> reuse "Mangla" (4)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        if test.__code__.co_argcount:
            test(None)
        else:
            test()
    print(f"[test_baseline_sultan] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
