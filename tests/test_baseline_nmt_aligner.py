# tests/test_baseline_nmt_aligner.py
"""Note 13 on fabricated attention tensors and the tiny random T5: the shifted
reading, the subword pooling, the layer agreement grid, the alignment layer and
its losses, one fine-tuning epoch, and the row end to end."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.decoder import BaselineDecoder  # noqa: E402
from retexo.baselines.nmt_aligner import (AlignmentLayer, AttentionReader, LayerSelector, NMTAligner,  # noqa: E402
                                              Seq2SeqWrapper)
from retexo.baselines.record import Edge, Record  # noqa: E402

TINY = "hf-internal-testing/tiny-random-t5"


def record(rid, source, reuse, edges=()):
    return Record(id=rid, level="external", fold=-1, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit", links=list(edges), split="test")


# =============================================================================
# 1. Reading attention
# =============================================================================


def test_shift_reads_row_i_plus_one_and_never_the_start_row():
    import numpy as np

    n = 4
    att = np.zeros((2, n + 1, n))                      # [heads, dec_len = start + n pieces, src]
    for i in range(n):
        att[:, i + 1, i] = 1.0                          # decoder row i + 1 (input y_i) peaks at source i
    att[:, 0, n - 1] = 1.0                              # the start row points somewhere else entirely
    word_ids = list(range(n))
    shifted = AttentionReader.read(att, word_ids, word_ids, n, n, shift=True)
    naive = AttentionReader.read(att, word_ids, word_ids, n, n, shift=False)
    assert [int(r.argmax()) for r in shifted] == [0, 1, 2, 3]
    assert [int(r.argmax()) for r in naive] == [3, 0, 1, 2]                 # one off, and the start row read for word 0


def test_subword_pooling_sums_source_pieces_and_averages_target_pieces():
    import numpy as np

    # source word 0 = pieces 0, 1; source word 1 = piece 2; target word 0 = pieces 0, 1 (rows 1, 2)
    att = np.zeros((1, 3, 3))
    att[0, 1] = [0.3, 0.3, 0.4]
    att[0, 2] = [0.1, 0.1, 0.8]
    rows = AttentionReader.read(att, [0, 0], [0, 0, 1], 1, 2, shift=True)
    assert abs(rows[0, 0] - 0.4) < 1e-6 and abs(rows[0, 1] - 0.6) < 1e-6 and abs(rows[0].sum() - 1.0) < 1e-6
    as_rows = AttentionReader.rows(rows)
    assert as_rows[0][0][0] == 1 and abs(as_rows[0][0][1] - 0.6) < 1e-6 and any(s == -1 for s, _ in as_rows[0])


def test_select_layer_picks_the_pair_that_agrees_best():
    fwd = [[[0, 1, 2]], [[2, 1, 0]]]                    # layer 0 monotone, layer 1 reversed
    bwd = [[[2, 1, 0]], [[0, 1, 2]]]                    # backward: source -> reuse
    assert LayerSelector.select(fwd, bwd) == (0, 1)
    assert LayerSelector.mutual_aer([[0, 1, 2]], [[0, 1, 2]]) == 0.0


# =============================================================================
# 2. The alignment layer
# =============================================================================


def test_alignment_layer_has_a_null_column_and_the_guided_loss_prefers_its_argmax():
    import torch

    torch.manual_seed(0)
    layer = AlignmentLayer(8, device="cpu")
    keys, queries = torch.randn(5, 8), torch.randn(3, 8)
    probs = layer(keys, queries)
    assert probs.shape == (3, 6) and torch.allclose(probs.sum(dim=1), torch.ones(3))
    forced = layer(keys, queries, null_logit=1e9)
    assert forced[:, -1].min() > 0.999
    links = [int(p[:-1].argmax()) for p in probs]
    shuffled = [links[1], links[2], links[0]]
    assert AlignmentLayer.guided_loss(probs, links) <= AlignmentLayer.guided_loss(probs, shuffled) + 1e-6
    assert AlignmentLayer.guided_loss(forced, [-1, -1, -1]) < 1e-3


def test_contiguity_loss_prefers_a_diagonal_to_a_scatter():
    import torch

    diagonal = torch.eye(4)
    diagonal = torch.cat([diagonal * 0.9 + 0.025, torch.full((4, 1), 0.0)], dim=1)
    scatter = torch.tensor([[0.9, 0.03, 0.03, 0.04], [0.03, 0.03, 0.9, 0.04], [0.9, 0.03, 0.03, 0.04], [0.03, 0.03, 0.9, 0.04]])
    scatter = torch.cat([scatter, torch.zeros(4, 1)], dim=1)
    assert AlignmentLayer.contiguity_loss(diagonal) < AlignmentLayer.contiguity_loss(scatter)


# =============================================================================
# 3. The model and the row
# =============================================================================


def pairs():
    return [(["the", "cat", "sat"], ["le", "chat", "assis"]), (["a", "dog", "runs"], ["un", "chien", "court"]),
            (["red", "house"], ["maison", "rouge"]), (["we", "go", "home"], ["nous", "rentrons"])]


def test_the_tiny_seq2seq_fits_one_epoch_and_reads_attentions():
    model = Seq2SeqWrapper(TINY, device="cpu")
    src, labels, src_words, tgt_words = model.encode(pairs()[:2])
    assert labels.shape[0] == 2 and len(src_words) == 2
    before = None
    losses = []
    model.fit(pairs(), epochs=1, lr=1e-3, batch_size=2, log=lambda line: losses.append(line))
    assert losses and "epoch 1/1" in losses[-1]
    mats = model.attentions(pairs(), layer=0, batch_size=2)
    assert len(mats) == 4 and mats[0].shape == (3, 3)
    assert before is None


def test_the_row_runs_end_to_end_with_both_readings(tmp_path):
    recs = [record("e/1", ["the", "cat", "sat"], ["le", "chat", "assis"], [Edge(0, 0, "COPY"), Edge(1, 1, "COPY")]),
            record("e/2", ["a", "dog", "runs"], ["un", "chien", "court"], [Edge(1, 1, "COPY")])]
    cfg = BaselineConfig(device="cpu", base_model=TINY, batch_size=2, seed=1, smoke=2,
                         extra={"backbone": TINY, "reading": "layer", "epochs": 1, "layer_updates": 5, "batch_size": 2,
                                "dev_pairs": 2})
    method = NMTAligner(cfg).fit(recs, recs, log=None)
    assert method.align_fwd is not None and method.align_bwd is not None
    preds = method.predict(recs)
    for rec, pred in zip(recs, preds):
        assert len(pred.scores) == rec.n_reuse and len(pred.rev_scores) == rec.n_source
        for row in pred.scores:
            assert abs(sum(p for _, p in row) - 1.0) < 1e-4 and any(s == -1 for s, _ in row)
        links = BaselineDecoder.decode_default(pred.scores, theta=0.3, rev_rows=pred.rev_scores, n_source=rec.n_source)
        assert len(links) == rec.n_reuse
    method.dials["reading"] = "shift_att"
    shifted = method.predict(recs)
    assert all(len(p.scores) == r.n_reuse for p, r in zip(shifted, recs))
    method.save(tmp_path / "m")
    again = NMTAligner.load(tmp_path / "m", cfg)
    assert again.layer_fwd == method.layer_fwd and again.align_fwd is not None
