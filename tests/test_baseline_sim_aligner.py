# tests/test_baseline_sim_aligner.py
"""SimilarityMatrix's rows, Itermax and distortion on hand-built matrices; SimAligner.predict
with a fake embedder (no model download); the Latin BERT span bookkeeping with a fake model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.decoder import BaselineDecoder  # noqa: E402
from retexo.baselines.record import Record  # noqa: E402
from retexo.baselines.sim_aligner import Embedder, SimAligner, SimilarityMatrix  # noqa: E402


def record(source, reuse):
    return Record(
        id="t/1",
        level="gold",
        fold=4,
        source_work="",
        source_tokens=source,
        reuse_work="",
        reuse_tokens=reuse,
        pair_label="cit",
    )


# =============================================================================
# 1. rows, rev_rows
# =============================================================================


def test_rows_dot_sum_to_one_and_contain_every_source_index():
    S = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2]])
    matrix = SimilarityMatrix(S, sim="dot")
    for row in matrix.rows():
        assert {s for s, _ in row} == {0, 1, 2}
        assert all(s >= 0 for s, _ in row)
        assert abs(sum(p for _, p in row) - 1.0) < 1e-9


def test_rows_cos_values_lie_in_the_unit_interval():
    S = np.array([[0.9, -0.2], [0.1, 0.99]])
    matrix = SimilarityMatrix(S, sim="cos")
    for row in matrix.rows():
        assert all(0.0 <= p <= 1.0 for _, p in row)


def test_rev_rows_equals_rows_of_the_transposed_matrix():
    S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    assert matrix.rev_rows() == SimilarityMatrix(S.T, sim="dot").rows()


def test_rows_empty_matrix_returns_no_error():
    matrix = SimilarityMatrix(np.zeros((0, 3)), sim="dot")
    assert matrix.rows() == []
    assert matrix.rev_rows() == [[] for _ in range(3)]


# =============================================================================
# 2. Mutual argmax (SimAlign's Argmax) over rows built here
# =============================================================================


def test_mutual_argmax_recovers_an_unambiguous_diagonal():
    S = np.eye(3) * 5.0 + 0.1
    matrix = SimilarityMatrix(S, sim="dot")
    links = BaselineDecoder.decode_mutual(matrix.rows(), matrix.rev_rows())
    assert links == [0, 1, 2]


def test_mutual_argmax_drops_the_loser_of_a_shared_column():
    S = np.array([[5.0, 0.1], [4.0, 0.1], [0.1, 5.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    links = BaselineDecoder.decode_mutual(matrix.rows(), matrix.rev_rows())
    assert links[0] == 0  # reuse 0 wins source 0 (5.0 > 4.0)
    assert links[1] == -1  # reuse 1 loses it and has no other candidate
    assert links[2] == 1


# =============================================================================
# 3. Intersection (awesome-align) over rows built here
# =============================================================================


def test_decode_intersect_matches_hand_computed_awesome_align_with_a_tie():
    S = np.array([[1.0, 1.0], [1.0, 1.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    links, extra = BaselineDecoder.decode_intersect(matrix.rows(), matrix.rev_rows(), c=0.001)
    # every candidate is a 0.5 probability in both directions and clears c=0.001,
    # so both source words survive for both reuse words: primary the tie-break winner
    # (larger source index), the other landing in extra -- the many-to-one case.
    assert links == [1, 1]
    assert extra == [(0, 0), (1, 0)]


def test_decode_intersect_drops_a_candidate_below_c():
    S = np.array([[10.0, 0.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    links, extra = BaselineDecoder.decode_intersect(matrix.rows(), matrix.rev_rows(), c=0.001)
    assert links == [0]
    assert extra == []


# =============================================================================
# 4. Itermax (Algorithm 1)
# =============================================================================


def test_itermax_finds_the_best_remaining_partner_in_pass_two():
    # reuse 0 wins source 0 outright in pass one; reuse 1 (still free) and source 1
    # (still free) only become each other's best once source 0 is faded by alpha.
    S = np.array([[5.0, 0.0], [4.0, 4.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    assert matrix.itermax(n_max=1, alpha=0.9) == {(0, 0)}
    assert matrix.itermax(n_max=2, alpha=0.9) == {(0, 0), (1, 1)}


def test_itermax_is_idempotent_once_no_pair_beats_the_fade():
    S = np.array([[5.0, 0.0], [4.0, 4.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    assert matrix.itermax(n_max=3, alpha=0.9) == {(0, 0), (1, 1)}


def test_itermax_empty_matrix_returns_no_pairs():
    assert SimilarityMatrix(np.zeros((0, 0)), sim="dot").itermax() == set()


# =============================================================================
# 5. Distortion
# =============================================================================


def test_distorted_matches_the_formula_and_keeps_the_diagonal_at_one():
    n = 4
    S = np.ones((n, n))
    matrix = SimilarityMatrix(S, sim="dot").distorted(0.5)
    t = np.arange(n)[:, None] / n
    s = np.arange(n)[None, :] / n
    expected = 1.0 - 0.5 * (s - t) ** 2
    assert np.allclose(matrix.values, expected)
    assert np.allclose(np.diag(matrix.values), 1.0)
    assert matrix.values.min() >= 0.5 - 1e-9
    assert matrix.values.max() <= 1.0 + 1e-9


def test_distorted_kappa_zero_is_a_no_op():
    S = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix = SimilarityMatrix(S, sim="dot")
    assert matrix.distorted(0.0) is matrix


# =============================================================================
# 6. Latin BERT tokenizer path: span bookkeeping on a fake model
# =============================================================================


class _FakeLatinModel:
    """Returns ``hidden[i] = [4i, 4i + 1, 4i + 2, 4i + 3]`` so spans are checkable by hand."""

    class _Config:
        hidden_size = 4

    config = _Config()

    def __call__(self, input_ids, output_hidden_states=True):
        import torch

        n = input_ids.shape[1]
        hidden = torch.arange(n * 4, dtype=torch.float32).reshape(1, n, 4)

        class _Out:
            pass

        out = _Out()
        out.hidden_states = [hidden] * 13  # 12 transformer layers plus the embedding layer
        return out


class _FakeLatinEncoder:
    """Two subword pieces per word, ignoring the word itself."""

    def encode_word(self, word):
        return [10, 11]


def test_latin_bert_span_bookkeeping_on_a_fake_model():
    import torch

    embedder = Embedder("ashleygong03/bamman-burns-latin-bert")
    embedder._model = _FakeLatinModel()
    embedder._latin_encoder = _FakeLatinEncoder()
    embedder._specials = {"[CLS]": 2, "[SEP]": 3}
    embedder.torch = torch

    vectors = embedder._encode_latin_bert(["arma", "cano"])
    # ids: [CLS, 10, 11, 10, 11, SEP] -> arma spans positions 1:3, cano spans 3:5
    assert vectors.shape == (2, 4)
    assert torch.allclose(
        vectors[0], torch.tensor([6.0, 7.0, 8.0, 9.0])
    )  # mean(hidden[1], hidden[2])
    assert torch.allclose(
        vectors[1], torch.tensor([14.0, 15.0, 16.0, 17.0])
    )  # mean(hidden[3], hidden[4])


# =============================================================================
# 7. SimAligner.predict with a fake embedder
# =============================================================================


class _FakeEmbedder:
    VECTORS = {"arma": [1.0, 0.0], "cano": [0.0, 1.0], "puella": [0.9, 0.1], "bellum": [0.0, 0.9]}

    def encode(self, words):
        return np.array([self.VECTORS.get(w, [0.0, 0.0]) for w in words])


def test_predict_returns_one_prediction_per_record_with_full_rows():
    method = SimAligner(BaselineConfig(device="cpu"))
    method.embedder = _FakeEmbedder()
    rec = record(["arma", "cano"], ["cano", "arma", "bellum"])
    preds = method.predict([rec])
    assert len(preds) == 1
    pred = preds[0]
    assert len(pred.scores) == rec.n_reuse
    assert all(len(row) == rec.n_source for row in pred.scores)
    assert len(pred.rev_scores) == rec.n_source
    assert all(len(row) == rec.n_reuse for row in pred.rev_scores)


def test_predict_empty_reuse_passage_returns_empty_rows_without_error():
    method = SimAligner(BaselineConfig(device="cpu"))
    method.embedder = _FakeEmbedder()
    pred = method.predict([record(["arma", "cano"], [])])[0]
    assert pred.links == []
    assert pred.scores is None


def test_extract_itermax_bypasses_the_shared_decoder():
    method = SimAligner(BaselineConfig(device="cpu", extra={"extract": "itermax"}))
    method.embedder = _FakeEmbedder()
    rec = record(["arma", "bellum"], ["puella", "cano"])
    pred = method.predict([rec])[0]
    assert "itermax_links" in pred.meta
    typed = method.postprocess(rec, pred, {"theta": 0.45, "frame_rule": "none"})
    assert typed.links[0] == 0  # puella <-> arma: [1.0, 0.0] vs [0.9, 0.1], each other's best
    assert typed.links[1] == 1  # cano <-> bellum: [0.0, 1.0] vs [0.0, 0.9], each other's best
