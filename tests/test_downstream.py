"""The paper's aggregator and threshold rule, and the script features, on hand-made input."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.edit_typing.downstream import (FORM, LEMMA, NONE, SCRIPT_FEATURES,  # noqa: E402
                            DownstreamScorer, ScriptFeaturizer)
from retexo.edit_typing.link_features import FEATURE_NAMES  # noqa: E402


def test_macro_is_per_query_over_queries_with_a_positive():
    gold = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]])
    pred = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]])
    macro, micro, rows = DownstreamScorer.macro_micro(["a", "b", "c"], gold, pred)
    # a: tp1 fp1 fn0 -> P .5 R 1 F1 .667 ; c: tp1 fp0 fn1 -> P 1 R .5 F1 .667 ; b has no positive
    assert abs(macro["f1"] - 2 / 3) < 1e-9 and abs(macro["precision"] - 0.75) < 1e-9
    assert macro["fp"] == 2 and macro["fn"] == 1            # b's false positive counts in the rates
    assert abs(macro["fpr"] - (1 / 3 + 1 / 3 + 0) / 3) < 1e-9
    assert abs(micro["precision"] - 2 / 3) < 1e-9 and abs(micro["recall"] - 2 / 3) < 1e-9


def test_plateau_high_takes_the_largest_threshold_near_the_best():
    labels = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.6, 0.55, 0.2, 0.1])
    assert abs(DownstreamScorer.find_threshold(labels, probs, method="max_f1") - 0.56) < 1e-9
    assert DownstreamScorer.find_threshold(labels, probs) >= 0.56          # plateau_high: the high end of the plateau
    assert DownstreamScorer.find_threshold(labels, probs) <= 0.60


def test_recall_at_k():
    gold = np.array([[1, 0, 0, 1], [0, 0, 0, 0]])
    score = np.array([[0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])
    r = DownstreamScorer.recall_at_k(["a", "b"], gold, score, ks=(1, 2, 4))
    assert r[1] == 0.5 and r[2] == 0.5 and r[4] == 1.0


def test_tier_grid_and_features():
    F = {n: i for i, n in enumerate(FEATURE_NAMES)}
    pf = np.zeros((3, 4, len(FEATURE_NAMES)), dtype=np.float16)
    pf[0, 1, F["same_form"]] = 1
    pf[1, 2, F["same_lemma"]] = 1
    tiers = ScriptFeaturizer.tier_grid(pf)
    assert tiers[0, 1] == FORM and tiers[1, 2] == LEMMA and tiers[2].max() == NONE

    class Op:
        def __init__(self, tag, t, s):
            self.tag, self.target_indices, self.source_indices = tag, t, s

    class Script:
        operations = [Op("QUOTE", [0, 1], [1, 2]), Op("DEL", [], [0])]
    view = {"tags": ["NOP", "MORPH", "INS"], "link": [1, 2, -1], "frame": [0, 0, 1],
            "reorder": [0, 0, 0], "quote": [1, 1, 0]}
    row = ScriptFeaturizer.features(view, Script, 4, [0.9, 0.8, 0.0], [0.1, 0.2, 0.95], tiers)
    d = dict(zip(SCRIPT_FEATURES, row))
    assert len(row) == len(SCRIPT_FEATURES)
    assert d["n_links"] == 2 and d["longest_run"] == 2 and d["n_runs"] == 1
    assert d["quote_longest"] == 2 and d["frame_n"] == 1 and d["del_share"] == 0.25
    assert d["form1_link_share"] == 0.5 and d["lemma_link_share"] == 0.5
    assert abs(d["mean_link_p"] - 0.85) < 1e-9 and d["mean_null_p_unlinked"] == 0.95
