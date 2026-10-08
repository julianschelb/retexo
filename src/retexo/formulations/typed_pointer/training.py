# formulations/typed_pointer/training.py
"""The training loops: ``fit`` over batches and the cached-vector joint refinement."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from retexo.formulations.change_detector import ChangeExample

if TYPE_CHECKING:
    from retexo.formulations.typed_pointer.model import TypedPointer


# =============================================================================
# Training
# =============================================================================
class TrainingMixin:
    """The optimizer's parameter groups and the two training loops."""

    # ---------- the optimizer's parameter groups ----------

    def _extra_fresh_parameters(self):
        """Parameters a subclass adds to the heads' learning-rate group: the span view's, the stretch head's, the slot convolution's."""
        out = []
        if self._slot_conv is not None:
            out += list(self._slot_conv.parameters())
        if self._span_q is not None:
            out += (
                list(self._span_q.parameters())
                + list(self._span_start.parameters())
                + list(self._span_end.parameters())
                + [self._span_null]
            )
        if self._stretch_head is not None:
            out += list(self._stretch_head.parameters())
            if self._null_fuse is not None:
                out.append(self._null_fuse)
        return out

    # ---------- training ----------

    def fit(
        self,
        examples: Sequence[ChangeExample],
        *,
        log=None,
        on_batch: Optional[Callable[[int, int], None]] = None,
    ) -> TypedPointer:
        """Train on ``examples``; ``on_batch(done, total)`` is called after every optimizer step."""
        import torch

        encoder_params = list(self._encoder.parameters())
        encoder_params += list(self._pointer_source.parameters())
        encoder_params += list(self._pointer_target.parameters())
        encoder_params += [self._pointer_null, self._frame_null]
        if self._tower is not None:
            encoder_params += list(self._tower.parameters())
        fresh = list(self._typer.parameters())
        if self._typer_evidence is not None:
            fresh += list(self._typer_evidence.parameters())
        if self._loc_evidence is not None:
            fresh += list(self._loc_evidence.parameters())
        fresh += list(self._loc_mlp.parameters())
        fresh += list(self._pair_head.parameters())
        fresh += list(self._sk_bin.parameters())
        fresh += list(self._sk_src.parameters())
        fresh += list(self._match_pair.parameters()) + list(self._match_label.parameters())
        fresh += (
            list(self._state_reuse.parameters())
            + list(self._state_source.parameters())
            + list(self._state_link.parameters())
        )
        if self._frame_head is not None:
            fresh += list(self._frame_head.parameters())
        fresh += list(self._extra_fresh_parameters())
        groups = [
            {"params": encoder_params, "lr": self.config.learning_rate},
            {"params": fresh, "lr": self.config.typer_lr},
        ]
        if self._channels is not None:
            if self.config.channel_lr > 0:
                groups.append(
                    {"params": list(self._channels.parameters()), "lr": self.config.channel_lr}
                )
            else:
                fresh += list(self._channels.parameters())
        optimizer = torch.optim.AdamW(groups)
        order = list(examples)
        rng = random.Random(self.config.seed)
        self._encoder.train()
        self._typer.train()
        self._loc_mlp.train()
        for epoch in range(self.config.epochs):
            rng.shuffle(order)
            total, n = 0.0, 0
            for start in range(0, len(order), self.config.batch_size):
                chunk = order[start : start + self.config.batch_size]
                if not chunk:
                    continue
                loss = self._typed_loss(chunk)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                n += 1
                if on_batch is not None:
                    on_batch(start + len(chunk), len(order))
            self.losses.append(total / max(n, 1))
            if log:
                log(f"    epoch {epoch + 1}/{self.config.epochs}  loss {total / max(n, 1):.4f}")
        return self

    def refine_joint(
        self,
        examples: Sequence[ChangeExample],
        *,
        epochs: int = 4,
        lr: float = 5e-4,
        log=None,
        freeze_locate: bool = True,
    ) -> None:
        """More epochs of the *same* cell loss, with the encoder frozen.

        The parent's refinement retrains the name and evidence terms as a
        standalone classifier. That is right when they feed a separate head
        and wrong here, where they sit inside one softmax with the locate term
        and the nulls: rescaling them freely destroyed the alignment (v2:
        0.943 -> 0.884). So the extra epochs use the joint loss over cached
        word vectors -- every non-encoder parameter trains, calibrated against
        the others, at a fraction of a full epoch's cost.
        """
        import torch

        cache = []
        self._encoder.eval()
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, rows, starts, ends, _, source, align, extra, t_words = self._encode(chunk)
                if source is None or not source[0].numel() or not rows.numel():
                    continue
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch).last_hidden_state
                vectors = self._word_vectors(hidden, rows, starts, ends).half().cpu()
                s_rows, s_starts, s_ends, _, s_words = source
                s_vectors = self._word_vectors(hidden, s_rows, s_starts, s_ends).half().cpu()
                fine_t, frame_t, _ = extra
                cache.append(
                    (
                        vectors,
                        s_vectors,
                        rows,
                        t_words,
                        s_rows,
                        s_words,
                        chunk,
                        align,
                        fine_t,
                        frame_t,
                    )
                )
        self._encoder.train()
        # v3c refined everything and lost the alignment again (0.947 -> 0.890):
        # the refinement set is synthetic-heavy, and re-exposing the *locate*
        # projections to it is the extra synthetic epoch E7 showed costs gold
        # score. The locate term is frozen by default; the name terms and the
        # nulls train against it, with the joint loss, and stay calibrated.
        params = [self._pointer_null, self._frame_null] + list(self._typer.parameters())
        if self._typer_evidence is not None:
            params += list(self._typer_evidence.parameters())
        if not freeze_locate:
            params += list(self._pointer_source.parameters()) + list(
                self._pointer_target.parameters()
            )
        opt = torch.optim.Adam(params, lr=lr)
        rng = random.Random(self.config.seed + 1)
        for epoch in range(epochs):
            rng.shuffle(cache)
            total, n = 0.0, 0
            for (
                vectors,
                s_vectors,
                rows,
                t_words,
                s_rows,
                s_words,
                chunk,
                align,
                fine_t,
                frame_t,
            ) in cache:
                got = self._cells_from_vectors(
                    vectors.float(), s_vectors.float(), rows, t_words, s_rows, s_words, chunk
                )
                if got is None:
                    continue
                flat, _, word_at, _ = got
                loss = self._loss_from_cells(flat, word_at, align, fine_t, frame_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
            if log:
                log(
                    f"    joint refine {epoch + 1}/{epochs}  loss {total / max(n, 1):.4f}"
                    f"  ({len(cache)} batches)"
                )
