"""The attestation tiers, the combiner and the regimes, checked on hand-made evidence."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.edit_typing.attest import (  # noqa: E402
    FORM,
    LEMMA,
    NONE,
    RELATION,
    Attestation,
    Attester,
    FrameLexicon,
)
from retexo.edit_typing.link_features import FEATURE_NAMES  # noqa: E402

_F = {n: i for i, n in enumerate(FEATURE_NAMES)}
attester = Attester(min_tier=LEMMA)


def phi(**on):
    v = [0.0] * len(FEATURE_NAMES)
    for k, x in on.items():
        v[_F[k]] = float(x)
    return v


def test_tiers_follow_the_strongest_evidence():
    assert Attester.pair_tier(phi(same_form=1, wn_syn=1)) == FORM
    assert Attester.pair_tier(phi(same_lemma=1)) == LEMMA
    assert Attester.pair_tier(phi(wn_any=1)) == RELATION
    assert Attester.pair_tier(phi(both_names=1)) == RELATION
    assert Attester.pair_tier(phi(edit_ratio=0.9)) == NONE


def test_combine_keeps_the_model_null_by_default():
    att = [Attestation(tier=FORM, candidates=[3]), Attestation()]
    rows = [[(3, 0.2), (5, 0.5), (-1, 0.3)], [(1, 0.6), (-1, 0.4)]]
    out = Attester.combine_scores(att, rows)  # mode "whether"
    assert dict(out[0]) == {3: 0.4, -1: 0.6}  # 5 is closed; null kept
    assert out[1] == rows[1]  # open word untouched
    fixed = Attester.combine_scores(att, rows, mode="fix")
    assert fixed[0][0] == (3, 1.0)


def test_combine_restricts_among_several_candidates():
    att = [Attestation(tier=FORM, candidates=[2, 7], by_order=7)]
    rows = [[(2, 0.1), (7, 0.3), (9, 0.5), (-1, 0.1)]]
    out = Attester.combine_scores(att, rows)
    probs = dict(out[0])
    assert set(probs) == {2, 7, -1} and abs(sum(probs.values()) - 1) < 1e-9
    assert Attester.combine_scores(att, rows, mode="order")[0][0] == (7, 1.0)


def test_regimes_and_types():
    assert Attestation(tier=FORM, candidates=[1]).regime() == "form:1"
    assert Attestation(tier=LEMMA, candidates=[1, 2]).regime() == "lemma:n"
    assert Attestation(tier=RELATION, candidates=[1]).regime() == "weak"
    assert Attestation().regime() == "none"
    assert Attester.attest_type(phi(same_form=1)) == ("NOP", True)
    assert Attester.attest_type(phi(same_lemma=1)) == ("MORPH", True)
    assert Attester.attest_type(phi(cos=0.9)) == ("SYN-DIST", False)
    assert Attester.attest_type(phi()) == ("SUBST", False)


class _Ex:
    def __init__(self, source, target, align):
        self.source_tokens, self.target_tokens, self.alignments = source, target, align


def test_training_restriction_adds_the_gold_when_the_resources_missed_it():
    ex = _Ex(["arma", "uirumque", "cano"], ["arma", "uirum", "canit"], [0, 1, -1])
    att = [
        Attestation(tier=FORM, candidates=[0], phi={0: phi(same_form=1)}),
        Attestation(tier=LEMMA, candidates=[2], phi={2: phi(same_lemma=1)}),
        Attestation(),
    ]
    allowed, type_open = attester.training_restriction(ex, att)
    assert allowed == [[0], [2, 1], None]
    assert type_open == [False, True, True]  # NOP named; gold 1 not in phi -> open


def test_lexicon_frames_match_whole_templates():
    ex = _Ex([], ["quod", "et", "Vergilius", "ait", "arma", "uirumque"], None)
    flags = FrameLexicon([["quod", "et"], ["ait"], ["nihil"]]).frames(ex)
    assert flags == [1, 1, 0, 0, 0, 0]  # one-word templates are below min_len
