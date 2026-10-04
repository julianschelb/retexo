# tests/test_baseline_downstream.py
"""Note 35 on hand-made dumps: the script-only columns, the mode shares, the
negative sampling and the plateau threshold, the aggregator, the cit./cf.
classifier on a separable set, the coefficients table, the duplicate rule."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import Prediction  # noqa: E402
from retexo.baselines.downstream import (EXCLUDED_FEATURES, MODE_FEATURES, SCRIPT_ONLY_FEATURES, Explanation,  # noqa: E402
                                             ReferenceClassifier, ScriptOnlyFeatures, TypeClassifier)
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.edit_typing.downstream import DownstreamScorer  # noqa: E402


def record(rid, source, reuse, edges=(), spans=(), label="cit", fold=0, query="q"):
    return Record(id=rid, level="gold", fold=fold, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label=label, links=list(edges), spans=list(spans), annotation={"query_id": query})


def three_pairs():
    verbatim = record("v", ["arma", "virum", "cano"], ["arma", "virum", "cano"],
                      [Edge(0, 0, "COPY"), Edge(1, 1, "COPY"), Edge(2, 2, "COPY")])
    allusion = record("a", ["rex", "regem", "amat", "pater"], ["regem", "amat", "ense", "hostem"],
                      [Edge(0, 0, "MORPH"), Edge(1, 2, "COPY")], label="cf")
    empty = record("e", ["nox", "erat"], ["dies", "est"], [], label="no_match")
    return verbatim, allusion, empty


# =============================================================================
# 1. Features
# =============================================================================


def test_script_only_columns_exclude_the_lexicon_and_carry_the_mode_shares():
    assert not set(EXCLUDED_FEATURES) & set(SCRIPT_ONLY_FEATURES)
    assert set(MODE_FEATURES) <= set(SCRIPT_ONLY_FEATURES) and len(MODE_FEATURES) == 4
    frame = ScriptOnlyFeatures.from_gold(three_pairs())
    assert list(frame.columns)[: len(SCRIPT_ONLY_FEATURES)] == list(SCRIPT_ONLY_FEATURES)
    assert set(frame.columns) - set(SCRIPT_ONLY_FEATURES) == {"id", "query_id", "fold", "pair_label", "is_reference", "is_cit"}


def test_mode_shares_and_the_gold_ceilings_confidence():
    frame = ScriptOnlyFeatures.from_gold(three_pairs()).set_index("id")
    assert frame.loc["v", "mode_verbatim"] == 1.0 and frame.loc["v", "mean_link_p"] == 1.0
    # MORPH and COPY are the form group, which the mode collapse calls VERBATIM; the two INS words are NOMATCH
    assert abs(frame.loc["a", "mode_verbatim"] - 0.5) < 1e-9 and abs(frame.loc["a", "mode_nomatch"] - 0.5) < 1e-9
    assert frame.loc["e", "mode_nomatch"] == 1.0 and frame.loc["e", "n_links"] == 0
    assert frame.loc["a", "morph_share"] == 0.5 and frame.loc["v", "link_rate"] == 1.0   # QUOTE needs QUOTE_MIN words; three is under it
    pred = Prediction(links=[0, 1, 2], tags=["COPY", "COPY", "COPY"], frame=[0, 0, 0], link_p=[0.9, 0.8, 0.7])
    row = ScriptOnlyFeatures.row(three_pairs()[0], pred)
    assert abs(row["mean_link_p"] - 0.8) < 1e-9 and row["is_reference"] == 1 and row["is_cit"] == 1


def test_dedupe_keeps_the_first_id_of_a_repeated_text_pair():
    a = record("p0002", ["x"], ["y"]); b = record("p0001", ["x"], ["y"], label="cf"); c = record("p0003", ["x"], ["z"])
    kept, dropped = ScriptOnlyFeatures.dedupe([(a, None), (b, None), (c, None)])
    assert [r.id for r, _ in kept] == ["p0001", "p0003"] and dropped == [("p0001", "p0002")]


# =============================================================================
# 2. The classifiers
# =============================================================================


def separable(n=40):
    import pandas as pd

    rows = []
    for i in range(n):
        ref = i % 2
        row = {f: 0.0 for f in SCRIPT_ONLY_FEATURES}
        row["link_rate"] = 0.9 if ref else 0.1
        row["mean_link_p"] = 0.8 if ref else 0.2
        row.update({"id": f"r{i}", "query_id": f"q{i % 5}", "fold": i % 5, "pair_label": "cit" if i % 4 == 1 else ("cf" if ref else "no_match"),
                    "is_reference": ref, "is_cit": int(i % 4 == 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_negative_sampling_and_the_plateau_threshold_are_deterministic():
    import numpy as np

    frame = separable(60)
    frame.loc[frame.index[:6], "is_reference"] = 1
    sample_a = ReferenceClassifier.sample_negatives(frame, neg_ratio=2, seed=42)
    sample_b = ReferenceClassifier.sample_negatives(frame, neg_ratio=2, seed=42)
    n_pos = int((frame["is_reference"] == 1).sum())
    assert list(sample_a["id"]) == list(sample_b["id"]) and len(sample_a) <= 3 * n_pos
    labels = np.array([1] * 5 + [0] * 5)
    probs = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.2, 0.1, 0.05, 0.0])
    assert abs(DownstreamScorer.find_threshold(labels, probs) - 0.69) < 1e-9        # the largest grid threshold still on the plateau


def test_fit_and_evaluate_reproduce_the_aggregator_and_recall_at_one():
    frame = separable(40)
    model, threshold = ReferenceClassifier.fit(frame, neg_ratio=10)
    result = ReferenceClassifier.evaluate(model, threshold, frame)
    assert result["macro"]["f1"] == 1.0 and result["micro"]["f1"] == 1.0 and result["n_queries"] == 5
    import numpy as np

    qids = ["q"]
    gold = [np.array([1, 0, 0])]
    assert DownstreamScorer.recall_at_k(qids, gold, [np.array([0.9, 0.1, 0.2])], ks=(1,))[1] == 1.0
    macro, micro, _ = DownstreamScorer.macro_micro(qids, gold, [np.array([1, 1, 0])])
    assert macro["precision"] == 0.5 and macro["recall"] == 1.0


def test_the_type_classifier_separates_a_separable_set_and_the_coefficients_table_has_signs():
    frame = separable(40)
    positives = frame[frame["is_reference"] == 1].copy()
    positives["is_cit"] = (positives["mean_link_p"] > 0.5).astype(int)
    positives.loc[positives.index[::2], "mean_link_p"] = 0.95
    positives.loc[positives.index[1::2], "mean_link_p"] = 0.6
    positives["is_cit"] = (positives["mean_link_p"] > 0.9).astype(int)
    result = TypeClassifier.fit_and_evaluate(positives, folds=(0, 1, 2, 3, 4))
    assert result["macro_f1"] == 1.0 and len(result["per_fold"]) >= 3
    table = Explanation.coefficients(result["models"])
    assert list(table.index)[: len(SCRIPT_ONLY_FEATURES)] and "sign agreement" in table.columns
    assert table.loc["mean_link_p", "sign agreement"] == len(result["models"])


def test_the_similarity_row_reads_the_cross_encoder_score_of_positives_and_pool():
    from retexo.baselines.downstream import SimilarityScore

    verbatim, allusion, _ = three_pairs()
    key = f"{' '.join(verbatim.reuse_tokens)}\t{' '.join(verbatim.source_tokens)}"
    positives = SimilarityScore.positives([verbatim, allusion], {0: {key: 0.9}})
    assert list(positives["similarity"]) == [0.9] and list(positives["is_reference"]) == [1]     # the unscored one left out
    pooled = Record(id="pool/f0/0", level="pool", fold=0, source_work="", source_tokens=["nox"], reuse_work="",
                    reuse_tokens=["arma", "virum", "cano"], pair_label="no_match", provenance={"prob_no_match": 0.75})
    pool = SimilarityScore.pool([pooled])
    assert list(pool["similarity"]) == [0.25] and list(pool["is_reference"]) == [0]
    assert pool["query_id"][0] == " ".join(pooled.reuse_tokens)
