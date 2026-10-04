# tests/test_baseline_span_pair.py
"""Note 21 on hand-made data: spans and pruning, the gold runs, the segmentation
dynamic programme, the expansion, the word rows; then the head learns one record
on fixed vectors and the row runs end to end on the tiny backbone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import labels  # noqa: E402
from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.baselines.span_pair import (ADAPT, FRAME, INS, QUOTE, Expander, Run, RunReader,  # noqa: E402
                                            Segmenter, SpanChoice, SpanEnumerator, SpanPairHead, SpanPairScorer)

TINY = "hf-internal-testing/tiny-random-bert"


def record(rid, source, reuse, edges=(), spans=()):
    return Record(id=rid, level="gold", fold=1, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit", links=list(edges), spans=list(spans))


# =============================================================================
# 1. Spans and pruning
# =============================================================================


def test_enumerate_spans_lists_every_span_up_to_the_length_in_order():
    spans = SpanEnumerator.enumerate_spans(5, 3)
    assert len(spans) == 12 and spans[:4] == [(0, 0), (0, 1), (0, 2), (1, 1)] and spans[-1] == (4, 4)


def test_candidate_pairs_keep_diagonals_through_cells_unequal_rectangles_and_always_the_gold():
    import numpy as np

    grid = np.zeros((3, 4))
    grid[0, 1] = 0.5                                   # the only cell above eps
    reuse, source = ["a", "b", "Ense"], ["x", "y", "z", "ense"]
    pairs = SpanEnumerator.candidate_pairs(grid, reuse, source, L_max=2, eps=0.01, K_pairs=1000)
    assert ((0, 0), (1, 1)) in pairs and ((0, 1), (1, 2)) in pairs      # the cell and the diagonal through it
    assert ((0, 1), (0, 1)) not in pairs                                 # an equal-length rectangle off its diagonal
    assert ((0, 0), (0, 1)) in pairs and ((0, 1), (1, 1)) in pairs      # unequal rectangles holding the cell
    assert ((2, 2), (3, 3)) in pairs                                     # equal normalised forms (u/v, case)
    assert ((1, 1), (2, 2)) not in pairs                                 # nothing attests it
    capped = SpanEnumerator.candidate_pairs(grid, reuse, source, L_max=2, eps=0.01, K_pairs=0, gold_pairs=[((1, 1), (2, 2))])
    assert ((1, 1), (2, 2)) in capped and ((0, 0), (0, 1)) not in capped   # the gold is appended past the cap


# =============================================================================
# 2. Gold runs
# =============================================================================


def test_gold_runs_split_at_gaps_and_crossings_and_read_frames():
    source = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    reuse = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
    edges = [Edge(0, 0, "COPY"), Edge(1, 1, "COPY"), Edge(2, 2, "MORPH"),
             Edge(5, 7, "COPY"), Edge(6, 5, "COPY"), Edge(7, 6, "COPY")]
    runs = RunReader.gold_runs(record("t/1", source, reuse, edges, spans=[Span(3, 5, "FRAME")]))
    assert runs == [Run(0, 2, ADAPT, (0, 2)), Run(3, 4, FRAME, None), Run(5, 5, QUOTE, (7, 7)), Run(6, 7, QUOTE, (5, 6))]


# =============================================================================
# 3. Segmentation and expansion
# =============================================================================


def opt(reuse, source, tag, p):
    import math

    return SpanChoice(reuse, source, tag, math.log(p), p)


def test_segment_picks_the_best_partition_declines_below_theta_and_repairs_source_overlap():
    options = [opt((0, 1), (0, 1), QUOTE, 0.8), opt((0, 1), None, INS, 0.1),
               opt((0, 0), (0, 0), QUOTE, 0.5), opt((0, 0), None, INS, 0.4),
               opt((1, 1), (1, 1), QUOTE, 0.5), opt((1, 1), None, INS, 0.4),
               opt((2, 2), (5, 5), ADAPT, 0.2), opt((2, 2), None, INS, 0.7), opt((2, 2), None, FRAME, 0.1)]
    chosen = Segmenter.segment(options, 3)
    assert [(c.reuse, c.source, c.tag) for c in chosen] == [((0, 1), (0, 1), QUOTE), ((2, 2), None, INS)]
    assert sum(c.reuse[1] - c.reuse[0] + 1 for c in chosen) == 3          # a partition of [0, 3)
    # theta above the pair's probability: the null wins the span
    chosen = Segmenter.segment(options, 3, theta=0.9)
    assert all(c.source is None for c in chosen)
    # two pairs on the same source span: the weaker becomes its null
    clash = [opt((0, 0), (0, 0), QUOTE, 0.9), opt((0, 0), None, INS, 0.05),
             opt((1, 1), (0, 0), QUOTE, 0.6), opt((1, 1), None, FRAME, 0.3)]
    chosen = Segmenter.segment(clash, 2)
    assert [(c.source, c.tag) for c in chosen] == [((0, 0), QUOTE), (None, FRAME)]
    # an uncovered position gets a length-one INS
    assert [c.reuse for c in Segmenter.segment([opt((0, 0), None, INS, 1.0)], 2)] == [(0, 0), (1, 1)]


def test_expand_links_position_wise_uses_the_hungarian_on_unequal_lengths_and_reads_enclitics():
    import numpy as np

    reuse = ["arma", "virum", "que", "cano", "hostem"]
    source = ["arma", "virumque", "canit"]
    grid = np.zeros((5, 3))
    grid[3, 2] = 0.9; grid[4, 2] = 0.1
    chosen = [opt((0, 0), (0, 0), QUOTE, 0.9), opt((1, 2), (1, 1), ADAPT, 0.7), opt((3, 4), (2, 2), ADAPT, 0.6)]
    links, tags, frame, extra, rows = Expander.expand(chosen, grid, reuse, source, lambda t, s: "MORPH")
    assert links[0] == 0 and tags[0] == "COPY"
    assert links[1] == 1 and links[2] == 1 and tags[1] == "SPLIT" and tags[2] == "SPLIT"    # virumque: an enclitic SPLIT
    assert links[3] == 2 and tags[3] == "MORPH" and links[4] == -1                          # Hungarian inside the 2x1 rectangle
    assert frame == [0] * 5 and extra == []
    for t, row in enumerate(rows):
        assert abs(sum(p for _, p in row) - 1.0) < 1e-6 and any(s == -1 for s, _ in row)
    merged = [opt((0, 0), (0, 1), ADAPT, 0.8)]
    links, tags, _, extra, _ = Expander.expand(merged, np.array([[0.9, 0.1]]), ["virumque"], ["virum", "que"], lambda t, s: "MORPH")
    assert links == [0] and tags == ["MERGE"] and extra and extra[0].s == 1 and extra[0].op == "MERGE"


# =============================================================================
# 4. The head learns, the row runs
# =============================================================================


def test_the_head_learns_one_record_on_fixed_vectors():
    import torch

    torch.manual_seed(0)
    rec = record("t/2", ["arma", "virumque", "cano"], ["arma", "virum", "canit", "poeta"],
                 edges=[Edge(0, 0, "COPY"), Edge(1, 1, "MORPH"), Edge(2, 2, "MORPH")])
    cfg = BaselineConfig(device="cpu", base_model=TINY, extra={"L_max": 3, "eps": 0.0})
    method = SpanPairScorer(cfg)
    method.head = SpanPairHead(8, L_max=3, hidden=32, device="cpu")
    h_r, h_s = torch.randn(4, 8), torch.randn(3, 8)
    optimizer = torch.optim.Adam(method.head.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        loss, n = method.loss(rec, h_r, h_s)
        first = float(loss.detach()) if first is None else first
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(loss.detach()) < first * 0.3
    _, per_span = method.options_for(rec, h_r, h_s)
    entries, logits = per_span[(0, 2)]
    assert entries[int(logits.argmax())] == ((0, 2), ADAPT)              # the gold run wins its span


def test_the_row_runs_end_to_end_on_the_tiny_backbone(tmp_path):
    pairs = [record("t/1", ["arma", "virumque", "cano"], ["arma", "virum", "canit", "poeta"],
                    edges=[Edge(0, 0, "COPY"), Edge(1, 1, "MORPH"), Edge(2, 2, "MORPH")]),
             record("t/2", ["rex", "regem", "amat"], ["ut", "ait", "rex", "amat"],
                    edges=[Edge(2, 0, "COPY"), Edge(3, 2, "COPY")], spans=[Span(0, 2, "FRAME")])]
    cfg = BaselineConfig(fold=4, dev_fold=0, device="cpu", base_model=TINY, batch_size=2, seed=1,
                         extra={"gold_passes": 1, "L_max": 3})
    method = SpanPairScorer(cfg).fit(pairs, [], log=None)
    preds = method.predict(pairs)
    for rec, pred in zip(pairs, preds):
        assert pred.scores is not None and len(pred.scores) == rec.n_reuse
        for row in pred.scores:
            assert abs(sum(p for _, p in row) - 1.0) < 1e-4
    done = [method.postprocess(r, p, {"theta": 0.0}) for r, p in zip(pairs, preds)]
    for rec, pred in zip(pairs, done):
        assert len(pred.links) == rec.n_reuse and len(pred.tags) == rec.n_reuse and len(pred.frame) == rec.n_reuse
        assert sum(b - a + 1 for (a, b), _, _, _ in pred.meta["runs"]) == rec.n_reuse   # the runs partition the reuse
        used = [s for s in pred.links if s >= 0]
        assert len(used) == len(set(used)) or any(t == "SPLIT" for t in pred.tags)
        for s, tag in zip(pred.links, pred.tags):
            assert (s < 0) == (tag == "")
            if s >= 0:
                assert tag in labels.EDGE_OPS
    assert method.validation_loss(pairs) > 0                            # the run loss without a gradient
    recall = method.candidate_recall(pairs)                            # the pruning's ceiling on records with links
    assert recall["gold_pairs"] > 0 and 0 <= recall["gold_pairs_in_candidates"] <= recall["gold_pairs"]
    assert recall["recall"] == round(recall["gold_pairs_in_candidates"] / recall["gold_pairs"], 5)
    method.save(tmp_path / "m")
    again = SpanPairScorer.load(tmp_path / "m", cfg)
    assert again.head is not None and again.predict(pairs)[0].scores is not None
