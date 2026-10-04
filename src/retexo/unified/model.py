# retexo/unified/model.py
"""The unified pointer: the typed pointer with a mode gate on the null and a
structured decoder behind the grid.

Everything that already works is inherited unchanged from ``TypedPointer`` --
the joint encoding, the factorised cell grid (locate + name, the evidence term),
both orientations, the pair head. Two things are added, each switchable so the
feasibility arms of the brainstorm note can isolate it:

- the **mode gate** (``use_gate``): a three-way head on the reuse word's own
  vector that factorises the location softmax, ``gate.py``;
- the **structured decoder** (``predict_structured``, and ``structured_margin_weight``
  to train against it): a semi-Markov programme over runs in place of the
  per-word assignment and the three repairs, ``decoder.py``.

A better-initialised encoder (stage 0, ``retexo.pretraining``) enters
through ``config.base_model`` and needs nothing here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from retexo.formulations.change_detector import ChangeExample
from retexo.formulations.typed_pointer import TypedPointer
from retexo.unified.decoder import CellGrid, DecoderConfig, StructuredDecoder
from retexo.unified.gate import gate_log_probs, gate_loss, gate_targets
from retexo.unified.runs import Segmentation


class UnifiedPointer(TypedPointer):
    """``TypedPointer`` plus the gate and the structured decoder.

    Example:
        ```python
        model = UnifiedPointer(config, use_gate=True, structured_margin_weight=0.5)
        model.fit(examples)
        links, frame = model.predict_structured(examples)[0]     # one pair
        rows = model.predict_alignment_scores(examples)          # interface (I), gated
        ```

    ``gate_weight`` scales the gate's set cross-entropy; ``gate_class_weights``
    are ``(reused, not reused, frame)`` -- frame defaults to the parent's
    ``frame_null_weight`` for the same reason it exists there (3% of words).
    """

    def __init__(self, config=None, *, use_gate: bool = True, gate_weight: float = 1.0,
                 decoder: Optional[DecoderConfig] = None, structured_margin_weight: float = 0.0) -> None:
        import torch

        super().__init__(config)
        hidden = self._pointer_null.shape[0]
        self._gate = torch.nn.Linear(hidden, 3).to(self.config.device)
        self.use_gate = use_gate
        self.gate_weight = gate_weight
        #: ``(reused, not reused, frame)``; ``None`` follows ``frame_null_weight`` at loss time
        self.gate_class_weights: Optional[Tuple[float, float, float]] = None
        self.decoder = StructuredDecoder(decoder)
        self.structured_margin_weight = structured_margin_weight
        self._last_gate = None
        self._last_rows = None
        self._last_t_words = None

    def _extra_fresh_parameters(self):
        return list(self._gate.parameters())

    # ---------- the grid, gated ----------

    def _cells_from_vectors(self, vectors, s_vectors, rows, t_words, s_rows, s_words, examples):
        got = super()._cells_from_vectors(vectors, s_vectors, rows, t_words, s_rows, s_words, examples)
        if got is None:
            return None
        loc_flat, name, frame_logits = self._last_factorized
        self._last_gate = self._gate(vectors.to(self.config.device))
        self._last_rows, self._last_t_words = rows, t_words
        if self.use_gate:
            self._last_factorized = (gate_log_probs(loc_flat, self._last_gate), name, frame_logits)
        return got

    # ---------- training ----------

    def _loss_from_cells(self, flat, word_at, align, fine_t, frame_t, restrict=None):
        loss = super()._loss_from_cells(flat, word_at, align, fine_t, frame_t, restrict=restrict)
        parts = {"base": float(loss.detach()), "gate": 0.0, "margin": 0.0}
        if self.use_gate and self._last_gate is not None:
            allowed = gate_targets(align.tolist(), frame_t.tolist())
            weights = self.gate_class_weights or (1.0, 1.0, float(self.frame_null_weight))
            gate = self.gate_weight * gate_loss(self._last_gate, allowed, weights)
            parts["gate"] = float(gate.detach())
            loss = loss + gate
        if self.structured_margin_weight > 0:
            margin = self._structured_margin(word_at, align, frame_t)
            if margin is not None:
                margin = self.structured_margin_weight * margin
                parts["margin"] = float(margin.detach())
                loss = loss + margin
        #: the components of the last batch's loss, for a trainer that logs them
        self.last_loss_parts = parts
        return loss

    def _structured_margin(self, word_at, align, frame_t):
        """The hinge of ``StructuredDecoder`` averaged over the batch's examples:
        the gold read as runs, the loss-augmented decode on a detached copy of
        the grid, the margin on the live one."""
        import torch

        loc_flat, _, frame_logits = self._last_factorized
        rows = self._last_rows.tolist()
        t_words = self._last_t_words.tolist()
        word_l = word_at.tolist()
        align_l, frame_l = align.tolist(), frame_t.tolist()
        log_soft = torch.log_softmax(loc_flat, dim=-1)
        # without the gate the location row's INS/FRAME split is untrained: take the
        # null mass from the row and split it by the frame head, as predict_structured does
        if not self.use_gate and frame_logits is not None:
            log_null = torch.logsumexp(log_soft[:, :2], dim=-1, keepdim=True)
            log_head = torch.log_softmax(frame_logits, dim=-1)
            log_soft = torch.cat([log_null + log_head, log_soft[:, 2:]], dim=-1)
        per_example: Dict[int, List[int]] = {}
        for i, r in enumerate(rows):
            per_example.setdefault(r, []).append(i)
        losses = []
        for r, idx in per_example.items():
            # an unknown gold link, or a gold source word truncated out of the grid,
            # leaves no gold segmentation to score against; a source word linked twice
            # (a SPLIT, the enclitic case) is a segmentation the decoder cannot produce,
            # so the hinge would push against an unreachable target: skip those examples
            linked = [align_l[i] for i in idx if align_l[i] >= 0]
            if any(align_l[i] == -100 or (align_l[i] >= 0 and align_l[i] not in word_l[i]) for i in idx) \
                    or len(linked) != len(set(linked)):
                continue
            m = max(t_words[i] for i in idx) + 1
            n = max([s for i in idx for s in word_l[i] if s >= 0] + [-1]) + 1
            if n == 0:
                continue
            log_p = loc_flat.new_full((m, n), float("-inf"))
            log_ins = loc_flat.new_full((m,), 0.0)
            log_frame = loc_flat.new_full((m,), float("-inf"))
            gold_links, gold_frame = [-1] * m, [0] * m
            for i in idx:
                t = t_words[i]
                log_ins[t] = log_soft[i, 0]
                log_frame[t] = log_soft[i, 1]
                for c, s in enumerate(word_l[i]):
                    if s >= 0:
                        log_p[t, s] = log_soft[i, 2 + c]
                gold_links[t] = align_l[i] if align_l[i] >= 0 else -1
                gold_frame[t] = 1 if frame_l[i] == 1 else 0
            gold = Segmentation.from_links(gold_links, gold_frame)
            live = CellGrid(log_p, log_ins, log_frame, m, n)
            frozen = CellGrid(log_p.detach().tolist(), log_ins.detach().tolist(), log_frame.detach().tolist(), m, n)
            wrong = self.decoder.loss_augmented(frozen, gold)
            if wrong == gold:
                continue
            # per reuse word, so the hinge sits on the scale of the per-word cross-entropy
            losses.append(self.decoder.margin_loss(live, gold, wrong) / m)
        if not losses:
            return None
        return torch.stack(losses).mean()

    # ---------- inference ----------

    def predict_structured(self, examples: Sequence[ChangeExample]) -> List[Tuple[List[int], List[int]]]:
        """Per example ``(links, frame)`` from the structured decoder over the
        (gated) grid -- the replacement for the per-word assignment and the three
        repairs. FRAME is the decoder's null run of that kind: split by the gate
        when it is on, by the parent's frame head when it is off. Words lost to
        truncation come back as INS."""
        out = []
        for ex, words in zip(examples, self.predict_cells(examples)):
            grid = CellGrid.from_word_rows(words, len(ex.source_tokens), frame_from_head=not self.use_gate)
            seg = self.decoder.decode(grid)
            out.append((seg.links(), seg.frame()))
        return out

    def gate_probabilities(self, examples: Sequence[ChangeExample]) -> List[List[Tuple[float, float, float]]]:
        """Per example, per reuse word ``(p_reused, p_not, p_frame)`` -- the gate
        read on its own, for diagnostics."""
        import torch

        self._encoder.eval(); self._typer.eval(); self._loc_mlp.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start:start + self.config.batch_size])
                if not chunk:
                    continue
                batch, rows, starts, ends, _, source, _, _, t_words = self._encode(chunk)
                per = [[(0.0, 1.0, 0.0)] * len(e.target_tokens) for e in chunk]
                if source is not None and source[0].numel() and rows.numel():
                    batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                    hidden = self._encoder(**batch_on).last_hidden_state
                    if self._cells(hidden, rows, starts, ends, t_words, source, chunk) is not None:
                        probs = torch.softmax(self._last_gate, dim=-1).cpu().tolist()
                        for i, (r, t) in enumerate(zip(rows.tolist(), t_words.tolist())):
                            per[r][t] = tuple(probs[i])
                out.extend(per)
        self._encoder.train(); self._typer.train(); self._loc_mlp.train()
        return out
