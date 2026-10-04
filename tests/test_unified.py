# tests/test_unified.py
"""Runs, the structured decoder and the gate on hand-built grids and logits; no model, no download."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.unified import (FRAME, INS, QUOTE, Augment, CellGrid, DecoderConfig, Run,  # noqa: E402
                                Segmentation, StructuredDecoder, gate_log_probs, gate_loss, gate_targets)

LOW = math.log(0.01)
HIGH = math.log(0.9)


def grid(cells, ins, frame):
    """A grid from dense lists: ``cells[t][s]`` is a probability (0 = not a candidate)."""
    m, n = len(cells), len(cells[0]) if cells else 0
    log_p = [[math.log(p) if p > 0 else float("-inf") for p in row] for row in cells]
    return CellGrid(log_p, [math.log(p) for p in ins], [math.log(p) for p in frame], m, n)


# =============================================================================
# 1. Runs
# =============================================================================


def test_from_links_reads_the_notebook_pair_into_three_runs():
    # quod et illustris poeta testatur dicens: | sed fugit interea fugit irreparabile tempus | et iterum
    links = [-1] * 6 + [0, 1, 2, 3, 4, 5] + [-1, -1]
    frame = [1] * 6 + [0] * 8
    seg = Segmentation.from_links(links, frame)
    assert seg.runs == [Run(0, 6, FRAME), Run(6, 12, QUOTE, source_start=0), Run(12, 14, INS)]
    assert seg.links() == links
    assert seg.frame() == frame


def test_from_links_breaks_a_run_at_a_gap_and_at_a_crossing():
    links = [0, 1, 3, 4, 2]              # 0-1 contiguous, 3-4 contiguous, then back to 2
    seg = Segmentation.from_links(links)
    assert [r.source_start for r in seg.runs] == [0, 3, 2]
    assert [r.length for r in seg.runs] == [2, 2, 1]


def test_segmentation_must_tile_the_reuse():
    import pytest

    with pytest.raises(ValueError):
        Segmentation([Run(0, 2, INS), Run(3, 4, INS)], 4)
    with pytest.raises(ValueError):
        Segmentation([Run(0, 2, QUOTE)], 2)     # a quote needs a source_start


# =============================================================================
# 2. The structured decoder
# =============================================================================


def test_decoder_keeps_a_quotation_together_against_a_stray_cell():
    # reuse words 0..3 come from source 0..3; word 2 also has a slightly *higher* cell at source 7
    cells = [[0.6, 0, 0, 0, 0, 0, 0, 0],
             [0, 0.6, 0, 0, 0, 0, 0, 0],
             [0, 0, 0.5, 0, 0, 0, 0, 0.55],
             [0, 0, 0, 0.6, 0, 0, 0, 0]]
    g = grid(cells, ins=[0.1] * 4, frame=[0.01] * 4)
    seg = StructuredDecoder(DecoderConfig(contiguity=0.5, crossing=1.0)).decode(g)
    assert seg.links() == [0, 1, 2, 3]          # the per-word argmax would send word 2 to source 7
    assert len(seg.runs) == 1 and seg.runs[0].kind == QUOTE


def test_decoder_prefers_the_null_run_where_the_nulls_win():
    cells = [[0.05, 0.05], [0.05, 0.05], [0.6, 0], [0, 0.6]]
    g = grid(cells, ins=[0.8, 0.8, 0.1, 0.1], frame=[0.01] * 4)
    seg = StructuredDecoder(DecoderConfig(contiguity=0.0)).decode(g)
    assert seg.links() == [-1, -1, 0, 1]
    assert seg.runs[0] == Run(0, 2, INS)


def test_decoder_marks_a_frame_run_where_frame_beats_ins():
    cells = [[0.02, 0.02], [0.02, 0.02], [0.7, 0], [0, 0.7]]
    g = grid(cells, ins=[0.2, 0.2, 0.05, 0.05], frame=[0.7, 0.7, 0.01, 0.01])
    seg = StructuredDecoder().decode(g)
    assert seg.frame() == [1, 1, 0, 0]
    assert seg.links() == [-1, -1, 0, 1]


def test_decoder_charges_a_crossing_and_forbids_overlap_with_the_previous_run():
    # words 0-1 from source 2-3, words 2-3 from source 0-1: a REORDER, allowed but charged
    cells = [[0, 0, 0.7, 0], [0, 0, 0, 0.7], [0.7, 0, 0, 0], [0, 0.7, 0, 0]]
    g = grid(cells, ins=[0.01] * 4, frame=[0.001] * 4)
    seg = StructuredDecoder(DecoderConfig(crossing=0.5)).decode(g)
    assert seg.links() == [2, 3, 0, 1]
    # overlap: words 0-1 from 0-1 and words 2-3 *also* from 0-1 -- the weaker run loses to INS
    cells = [[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0]]
    g = grid(cells, ins=[0.3, 0.3, 0.3, 0.3], frame=[0.001] * 4)
    seg = StructuredDecoder(DecoderConfig(crossing=0.0)).decode(g)
    assert seg.links() == [0, 1, -1, -1]


def test_decoder_respects_max_run():
    cells = [[0.9 if s == t else 0 for s in range(4)] for t in range(4)]
    g = grid(cells, ins=[0.01] * 4, frame=[0.001] * 4)
    seg = StructuredDecoder(DecoderConfig(max_run=2, contiguity=0.0)).decode(g)
    assert seg.links() == [0, 1, 2, 3]
    assert all(r.length <= 2 for r in seg.runs)


def test_empty_reuse_decodes_to_nothing():
    g = CellGrid([], [], [], 0, 3)
    assert StructuredDecoder().decode(g).runs == []


# =============================================================================
# 3. The structured hinge
# =============================================================================


def test_loss_augmented_finds_a_wrong_segmentation_and_the_margin_is_positive_when_close():
    cells = [[0.5, 0.45], [0.45, 0.5]]
    g = grid(cells, ins=[0.05, 0.05], frame=[0.001, 0.001])
    decoder = StructuredDecoder(DecoderConfig(contiguity=0.0, crossing=0.0, cost=1.0))
    gold = Segmentation.from_links([0, 1])
    wrong = decoder.loss_augmented(g, gold)
    assert wrong != gold
    assert wrong.links() != gold.links()
    # gold: 2 log 0.5 = -1.386; wrong [1, 0]: 2 log 0.45 = -1.597, two disagreements -> delta 2
    loss = decoder.margin_loss(g, gold, wrong)
    assert abs(loss - (-1.597 + 2.0 + 1.386)) < 0.01


def test_margin_is_zero_when_the_gold_wins_by_more_than_the_margin():
    cells = [[0.98, 0.0001], [0.0001, 0.98]]
    g = grid(cells, ins=[0.0001, 0.0001], frame=[0.0001, 0.0001])
    decoder = StructuredDecoder(DecoderConfig(contiguity=0.0, cost=0.5))
    gold = Segmentation.from_links([0, 1])
    wrong = decoder.loss_augmented(g, gold)
    assert wrong == gold                         # nothing wrong comes within the cost of the gold
    assert decoder.margin_loss(g, gold, wrong) == 0.0


def test_augment_charges_exactly_the_disagreements():
    gold = Segmentation.from_links([0, -1, -1], frame=[0, 1, 0])
    a = Augment.against(gold)
    assert a.cell(0, 0) == 0.0 and a.cell(0, 1) == 1.0
    assert a.null(0, INS) == 1.0                 # word 0 is linked in the gold
    assert a.null(1, FRAME) == 0.0 and a.null(1, INS) == 1.0
    assert a.null(2, INS) == 0.0 and a.null(2, FRAME) == 1.0


def test_margin_loss_backpropagates_through_a_tensor_grid():
    import torch

    logits = torch.tensor([[0.0, 0.0, 2.0, 1.5], [0.0, 0.0, 1.5, 2.0]], requires_grad=True)
    log_soft = torch.log_softmax(logits, dim=-1)
    live = CellGrid(log_soft[:, 2:], log_soft[:, 0], log_soft[:, 1], 2, 2)
    frozen = CellGrid(log_soft[:, 2:].detach().tolist(), log_soft[:, 0].detach().tolist(),
                      log_soft[:, 1].detach().tolist(), 2, 2)
    decoder = StructuredDecoder(DecoderConfig(contiguity=0.0, cost=2.0))
    gold = Segmentation.from_links([0, 1])
    wrong = decoder.loss_augmented(frozen, gold)
    loss = decoder.margin_loss(live, gold, wrong)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# =============================================================================
# 4. The gate
# =============================================================================


def test_gate_log_probs_factorises_the_location_row():
    import torch

    loc_flat = torch.tensor([[0.0, 0.0, 3.0, 1.0, 1.0]])           # nulls, then 3 source columns
    gate = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))               # reused / not / frame
    out = gate_log_probs(loc_flat, gate)
    p = torch.softmax(out, dim=-1)[0]
    assert abs(p[0].item() - 0.3) < 1e-5 and abs(p[1].item() - 0.2) < 1e-5
    assert abs(p[2:].sum().item() - 0.5) < 1e-5
    assert p[2] > p[3] and abs(p[3].item() - p[4].item()) < 1e-6   # column ranking kept


def test_gate_log_probs_with_no_source_columns_keeps_mass_on_the_nulls():
    import torch

    loc_flat = torch.tensor([[1.0, 0.0, float("-inf"), float("-inf")]])
    gate = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    p = torch.softmax(gate_log_probs(loc_flat, gate), dim=-1)[0]
    assert torch.isfinite(p[:2]).all() and abs(p[:2].sum().item() - 1.0) < 1e-5
    assert p[2:].sum().item() == 0.0


def test_gate_targets_follow_the_gold():
    allowed = gate_targets([3, -1, -1, -1, -100], [0, 1, 0, -100, 0])
    assert allowed == [[True, False, False], [False, False, True], [False, True, False],
                       [False, True, True], [False, False, False]]


def test_gate_loss_prefers_logits_that_put_mass_on_the_allowed_class():
    import torch

    allowed = [[True, False, False], [False, True, False]]
    good = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    bad = torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    assert gate_loss(good, allowed) < gate_loss(bad, allowed)
    assert gate_loss(good, [[False, False, False]] * 2) == 0.0


def test_gate_log_probs_backward_is_finite_with_a_candidate_free_row():
    import torch

    loc_flat = torch.tensor([[1.0, 0.0, float("-inf"), float("-inf")],
                             [0.0, 0.0, 2.0, 1.0]], requires_grad=False)
    gate = torch.zeros((2, 3), requires_grad=True)
    out = gate_log_probs(loc_flat, gate)
    loss = -(out[0, 0] + out[1, 2])          # a null on the first row, a source on the second
    loss.backward()
    assert torch.isfinite(gate.grad).all()


def test_resolve_overlaps_catches_a_clash_with_a_non_adjacent_earlier_run():
    # run A: words 0-1 <- source 0-1; a null run; run B: words 3-4 <- source 4-5;
    # run C: words 5-6 <- source 0-1 again (clashes with A, two runs back)
    cells = [[0.9, 0, 0, 0, 0, 0], [0, 0.9, 0, 0, 0, 0],
             [0.01] * 6,
             [0, 0, 0, 0, 0.9, 0], [0, 0, 0, 0, 0, 0.9],
             [0.6, 0, 0, 0, 0, 0], [0, 0.6, 0, 0, 0, 0]]
    g = grid(cells, ins=[0.05, 0.05, 0.9, 0.05, 0.05, 0.3, 0.3], frame=[0.001] * 7)
    seg = StructuredDecoder(DecoderConfig(crossing=0.0)).decode(g)
    assert seg.links() == [0, 1, -1, 4, 5, -1, -1]      # C lost to A, the stronger run


def test_decoder_does_not_buy_a_crossing_overlap_with_a_junk_link():
    # the p0017 case of the fold-4 error analysis: words 0-3 <- source 0-3 (a
    # quotation), word 4 has no source (its best cell is 0.001), words 5-6
    # repeat source 0-1 with high cells. A decoder that checks overlap only
    # against the previous run would link word 4 to the junk cell (a new
    # state), then land words 5-6 on source 0-1 as a crossing run, and the
    # overlap repair would strip the run but leave the junk link behind.
    cells = [[0.9, 0, 0, 0, 0, 0], [0, 0.9, 0, 0, 0, 0], [0, 0, 0.9, 0, 0, 0], [0, 0, 0, 0.9, 0, 0],
             [0.0005, 0, 0, 0, 0, 0.001],
             [0.9, 0, 0, 0, 0, 0], [0, 0.9, 0, 0, 0, 0]]
    g = grid(cells, ins=[0.05, 0.05, 0.05, 0.05, 0.96, 0.005, 0.005], frame=[0.001] * 7)
    seg = StructuredDecoder(DecoderConfig(crossing=1.0)).decode(g)
    assert seg.links() == [0, 1, 2, 3, -1, -1, -1]
    assert all(r.kind != QUOTE or r.source_start is not None for r in seg.runs)
    # and the same repetition is linked when it does not clash: source 6-7 free
    cells = [row + [0, 0] for row in cells]
    cells[5], cells[6] = [0] * 6 + [0.9, 0], [0] * 6 + [0, 0.9]
    g = grid(cells, ins=[0.05, 0.05, 0.05, 0.05, 0.96, 0.005, 0.005], frame=[0.001] * 7)
    assert StructuredDecoder(DecoderConfig(crossing=1.0)).decode(g).links() == [0, 1, 2, 3, -1, 6, 7]
