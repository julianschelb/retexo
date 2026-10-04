# tests/test_baseline_span_aligner.py
"""Query marking, span_scores, rows, extra edges and queries_of on hand-built data;
SpanAligner.predict with a fake encoder and heads (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.decoder import BaselineDecoder  # noqa: E402
from retexo.baselines.record import Edge, Record  # noqa: E402
from retexo.baselines.span_aligner import (QEND_TOKEN, Q_TOKEN, QueryResult, SpanAligner,  # noqa: E402
                                               SpanEncoder, span_scores)


def record(source, reuse, edges=()):
    return Record(id="t/1", level="gold", fold=4, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit", links=list(edges))


# =============================================================================
# 1. Query marking
# =============================================================================


def test_mark_inserts_markers_around_only_the_asked_occurrence():
    words = ["quod", "sed", "quod", "iterum"]
    marked = SpanEncoder._mark(words, 2)
    assert marked == ["quod", "sed", Q_TOKEN, "quod", QEND_TOKEN, "iterum"]
    # the first "quod" (index 0) is untouched
    assert marked[0] == "quod" and marked[1] == "sed"


def test_mark_at_the_first_and_last_word():
    words = ["arma", "cano"]
    assert SpanEncoder._mark(words, 0) == [Q_TOKEN, "arma", QEND_TOKEN, "cano"]
    assert SpanEncoder._mark(words, 1) == ["arma", Q_TOKEN, "cano", QEND_TOKEN]


# =============================================================================
# 2. span_scores
# =============================================================================


def test_span_scores_best_span_is_the_argmax_within_max_words():
    import torch

    # context has 3 words at subword spans (1,2) (2,4) (4,5); position 0 is [CLS]/null
    spans_b = [(1, 2), (2, 4), (4, 5)]
    start_logits = torch.full((6,), -10.0)
    end_logits = torch.full((6,), -10.0)
    start_logits[2] = 10.0   # word 1 starts at subword 2
    end_logits[3] = 10.0     # word 1 ends at subword 3 (its span is (2,4), last position 3)
    start_logits[0] = -10.0  # null is not favoured
    end_logits[0] = -10.0
    result = span_scores(start_logits, end_logits, spans_b, max_words=2)
    assert result.best_words == (1,)
    assert result.omega_best > 0.9


def test_span_scores_s_null_is_the_cls_position():
    import torch

    spans_b = [(1, 2)]
    start_logits = torch.tensor([5.0, -5.0])
    end_logits = torch.tensor([5.0, -5.0])
    result = span_scores(start_logits, end_logits, spans_b, max_words=3)
    assert result.s_null > 0.9        # position 0 (CLS) dominates both softmaxes
    assert result.best_words == (0,)  # still the best (only) real candidate, decoder decides null vs. real


def test_span_scores_respects_max_words():
    import torch

    # three one-subword words; with max_words=1 the two-word span (0,1) cannot win
    spans_b = [(1, 2), (2, 3), (3, 4)]
    start_logits = torch.full((5,), -10.0)
    end_logits = torch.full((5,), -10.0)
    start_logits[1] = 10.0
    end_logits[2] = 10.0     # the true best span covers words 0 and 1 together
    result_wide = span_scores(start_logits, end_logits, spans_b, max_words=2)
    result_narrow = span_scores(start_logits, end_logits, spans_b, max_words=1)
    assert result_wide.best_words == (0, 1)
    assert result_narrow.best_words != (0, 1)   # max_words=1 forbids the two-word span


def test_span_scores_empty_context_returns_only_the_null():
    import torch

    result = span_scores(torch.tensor([1.0]), torch.tensor([1.0]), [], max_words=3)
    assert result.best_words == ()


# =============================================================================
# 3. Rows
# =============================================================================


def test_row_carries_omega_best_on_every_span_word_and_s_null_on_the_null():
    result = QueryResult(best_words=(1, 2), omega_best=0.7, s_null=0.2)
    row = SpanAligner._row(result)
    assert dict(row) == {1: 0.7, 2: 0.7, -1: 0.2}
    assert row[0][1] >= row[-1][1]   # sorted best first


# =============================================================================
# 4. symmetrise_average is Nagata's eq. 3 once rows carry ω_best on every span word
# =============================================================================


def test_symmetrise_average_matches_nagatas_figure_5_arithmetic():
    # reuse word 0 -> source word 0 only in the forward direction, omega 0.8;
    # reuse word 1 -> source word 1 only in the forward direction, omega 0.6
    rows = [[(0, 0.8), (-1, 0.1)], [(1, 0.6), (-1, 0.3)]]
    rev_rows = [[(-1, 1.0)], [(-1, 1.0)]]   # neither source word points back
    merged = BaselineDecoder.symmetrise_average(rows, rev_rows)
    assert abs(dict(merged[0])[0] - 0.4) < 1e-9
    assert abs(dict(merged[1])[1] - 0.3) < 1e-9
    links = BaselineDecoder.decode_threshold(merged, theta=0.4)
    assert links == [0, -1]   # 0.4 clears theta, 0.3 does not


# =============================================================================
# 5. Extra edges: multi-word spans
# =============================================================================


def test_extra_edges_forward_direction_is_a_split_sharing_r():
    results = [QueryResult(best_words=(2, 3), omega_best=0.9, s_null=0.1)]
    extra = SpanAligner._extra_edges(results, reverse=False)
    assert extra == [Edge(r=0, s=3, op="")]


def test_extra_edges_reverse_direction_is_a_merge_sharing_s():
    results = [QueryResult(best_words=(2, 3), omega_best=0.9, s_null=0.1)]
    extra = SpanAligner._extra_edges(results, reverse=True)
    assert extra == [Edge(r=3, s=0, op="")]


def test_extra_edges_empty_for_single_word_spans():
    results = [QueryResult(best_words=(2,), omega_best=0.9, s_null=0.1), QueryResult((), 0.0, 1.0)]
    assert SpanAligner._extra_edges(results, reverse=False) == []


# =============================================================================
# 6. queries_of
# =============================================================================


def test_queries_of_one_query_per_word_both_directions_with_null_targets():
    rec = record(["arma", "cano"], ["cano", "gladius"], edges=[Edge(r=0, s=1, op="COPY", sure=True)])
    method = SpanAligner(BaselineConfig(device="cpu"))
    queries = method.queries_of([rec])
    fwd = [(i, t) for q, c, i, t in queries if q is rec.reuse_tokens]
    rev = [(i, t) for q, c, i, t in queries if q is rec.source_tokens]
    assert len(fwd) == rec.n_reuse and len(rev) == rec.n_source
    assert dict(fwd)[0] == (1, 1)    # "cano" (reuse 0) links to source word 1
    assert dict(fwd)[1] is None      # "gladius" (reuse 1) is a null question
    assert dict(rev)[1] == (0, 0)    # source word 1 links back to reuse 0
    assert dict(rev)[0] is None


def test_queries_of_sure_only_drops_possible_links():
    rec = record(["arma"], ["arma"], edges=[Edge(r=0, s=0, op="COPY", sure=False)])
    method = SpanAligner(BaselineConfig(device="cpu", extra={"train_on": "sure"}))
    queries = method.queries_of([rec])
    fwd_target = next(t for q, c, i, t in queries if q is rec.reuse_tokens)
    assert fwd_target is None


def test_tune_seeds_a_fixed_theta_only_when_asked():
    assert SpanAligner(BaselineConfig(device="cpu")).tune([]) == {}
    fixed = SpanAligner(BaselineConfig(device="cpu", extra={"theta": 0.4}))
    assert fixed.tune([]) == {"theta": 0.4}


# =============================================================================
# 7. predict() with a fake encoder and heads (no model download)
# =============================================================================


class _FakeEncoder:
    """``encode_queries`` echoes the query index through so ``_FakeHeads`` can look up
    pre-registered logits per item; ``forward_hidden`` is the identity."""

    def __init__(self, spans_by_len, logits_by_index):
        self.spans_by_len = spans_by_len
        self.logits_by_index = logits_by_index

    def encode_queries(self, items):
        spans_b = [self.spans_by_len[len(context)] for _, context, _ in items]
        return items, spans_b

    def forward_hidden(self, batch):
        return batch


class _FakeHeads:
    def __init__(self, logits_by_index):
        self.logits_by_index = logits_by_index

    def __call__(self, items):
        import torch

        starts = torch.stack([self.logits_by_index[i][0] for _, _, i in items])
        ends = torch.stack([self.logits_by_index[i][1] for _, _, i in items])
        return starts, ends


def test_predict_returns_scores_and_extra_from_a_fake_encoder():
    import torch

    rec = record(["arma", "cano"], ["cano", "bellum"])
    # both directions have 2 queries (indices 0, 1); build logits that make index 0's
    # best span the null and index 1's best span word 0 of a 2-word context
    null_win = (torch.tensor([10.0, -10.0, -10.0]), torch.tensor([10.0, -10.0, -10.0]))
    word_win = (torch.tensor([-10.0, 10.0, -10.0]), torch.tensor([-10.0, 10.0, -10.0]))
    method = SpanAligner(BaselineConfig(device="cpu"))
    method.encoder = _FakeEncoder(spans_by_len={2: [(1, 2), (2, 3)]}, logits_by_index={0: null_win, 1: word_win})
    method.heads = _FakeHeads({0: null_win, 1: word_win})
    preds = method.predict([rec])
    assert len(preds) == 1
    pred = preds[0]
    assert len(pred.scores) == rec.n_reuse
    assert len(pred.rev_scores) == rec.n_source
    assert dict(pred.scores[0])[-1] > dict(pred.scores[0]).get(0, 0.0)   # query 0: null wins
    assert dict(pred.scores[1]).get(0, 0.0) > 0.0                        # query 1: word 0 wins


def test_predict_empty_passage_returns_empty_prediction_without_error():
    method = SpanAligner(BaselineConfig(device="cpu"))
    method.heads = _FakeHeads({})
    pred = method.predict([record(["arma"], [])])[0]
    assert pred.links == []
    assert pred.scores is None


def test_predict_before_fit_returns_empty_predictions():
    method = SpanAligner(BaselineConfig(device="cpu"))
    pred = method.predict([record(["arma", "cano"], ["cano", "arma"])])[0]
    assert pred.links == [-1, -1]
    assert pred.scores is None
