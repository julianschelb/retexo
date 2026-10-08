# formulations/change_detector/training.py
"""The training loops of the change detector: the main fit and the head refinement."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from retexo.formulations.change_detector.example import ChangeExample

if TYPE_CHECKING:
    from retexo.formulations.change_detector.detector import ChangeDetector


# =============================================================================
# _TrainingMixin
# =============================================================================


class _TrainingMixin:
    """The training loops of the change detector."""

    # ---------- Training ----------

    def fit(
        self,
        examples: Sequence[ChangeExample],
        *,
        log=None,
        on_batch: Optional[Callable[[int, int], None]] = None,
    ) -> ChangeDetector:
        """Train on ``examples``; ``on_batch(done, total)`` is called after every optimizer step (a training
        monitor's evaluations inside the epoch)."""
        import torch

        parameters = list(self._encoder.parameters()) + list(self._head.parameters())
        if self._source_head is not None:
            parameters += list(self._source_head.parameters())
        if self._channels is not None:
            parameters += list(self._channels.parameters())
        if self.config.pointer:
            parameters += list(self._pointer_source.parameters())
            parameters += list(self._pointer_target.parameters())
            parameters += [self._pointer_null]
        groups = [{"params": parameters, "lr": self.config.learning_rate}]
        fresh = []
        if self._typer is not None:
            fresh += list(self._typer.parameters())
        if self._typer_evidence is not None:
            fresh += list(self._typer_evidence.parameters())
        if self._frame_head is not None:
            fresh += list(self._frame_head.parameters())
        if fresh:
            groups.append({"params": fresh, "lr": self.config.typer_lr})
        if self.config.pointer and self._pointer_temperature is not None:
            groups.append({"params": [self._pointer_temperature], "lr": self.config.temperature_lr})
        optimizer = torch.optim.AdamW(groups, lr=self.config.learning_rate)
        fns = self._loss_functions()
        order = list(examples)
        rng = random.Random(self.config.seed)

        self._encoder.train()
        self._head.train()
        if self._source_head is not None:
            self._source_head.train()
        if self._typer is not None:
            self._typer.train()
        if self._frame_head is not None:
            self._frame_head.train()
        if self.config.pointer:
            self._pointer_source.train()
            self._pointer_target.train()
        for epoch in range(self.config.epochs):
            rng.shuffle(order)
            epoch_losses = []
            for start in range(0, len(order), self.config.batch_size):
                chunk = order[start : start + self.config.batch_size]
                if not chunk:
                    continue
                loss = self._chunk_loss(chunk, fns)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                if on_batch is not None:
                    on_batch(start + len(chunk), len(order))
            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            self.losses.append(mean_loss)
            if log:
                log(f"    epoch {epoch + 1}/{self.config.epochs}  loss {mean_loss:.4f}")
        return self

    # ---------- Head refinement ----------

    def refine_heads(
        self,
        examples: Sequence[ChangeExample],
        *,
        epochs: int = 6,
        lr: float = 1e-3,
        batch_size: int = 512,
        log=None,
        frame_examples: Optional[Sequence[ChangeExample]] = None,
    ) -> None:
        """E24: train the typer and frame head alone, on cached encodings.

        The two heads ride along with the encoder for a few hundred steps
        during the main fit, which is not enough for a head that starts from
        random weights to converge. Encoding every example once with the
        encoder frozen and then fitting the heads for several epochs over the
        cached vectors costs seconds per epoch and lets them actually fit --
        without touching the pointer, which is what the shared encoder would
        otherwise pay for.
        """
        import torch

        if self._typer is None and self._frame_head is None:
            return
        device = self.config.device
        link_rows, frame_rows = [], []
        # the frame head may be refined on a different (gold-heavier) set than
        # the typer: constructed frames repeat 117 templates, and a head that
        # memorises them misses the formulas the held-out authors actually use
        frame_set = {id(e) for e in (frame_examples if frame_examples is not None else examples)}
        todo = list(examples)
        if frame_examples is not None:
            seen = {id(e) for e in todo}
            todo += [e for e in frame_examples if id(e) not in seen]
        typer_set = {id(e) for e in examples}
        self._encoder.eval()
        with torch.no_grad():
            for start in range(0, len(todo), self.config.batch_size):
                chunk = list(todo[start : start + self.config.batch_size])
                if not chunk:
                    continue
                # a subclass may return more (the typed pointer adds the word index)
                batch, rows, starts, ends, _, source, align, extra = self._encode(chunk)[:8]
                batch = {k: v.to(device) for k, v in batch.items()}
                hidden = self._run_encoder(batch).last_hidden_state
                vectors = self._word_vectors(hidden, rows, starts, ends)
                fine_t, frame_t, phi = extra
                in_frame_set = torch.tensor([id(chunk[r]) in frame_set for r in rows.tolist()])
                in_typer_set = torch.tensor([id(chunk[r]) in typer_set for r in rows.tolist()])
                if self._frame_head is not None:
                    keep = (frame_t != -100) & in_frame_set
                    if bool(keep.any()):
                        frame_rows.append((vectors[keep.to(device)].half().cpu(), frame_t[keep]))
                fine_t = torch.where(in_typer_set, fine_t, torch.full_like(fine_t, -100))
                if self._typer is not None and source is not None and source[0].numel():
                    s_rows, s_starts, s_ends, _, s_words = source
                    s_vectors = self._word_vectors(hidden, s_rows, s_starts, s_ends)
                    flat = {
                        (r, w): k for k, (r, w) in enumerate(zip(s_rows.tolist(), s_words.tolist()))
                    }
                    sel_t, sel_s = [], []
                    for k, (r, a, f) in enumerate(
                        zip(rows.tolist(), align.tolist(), fine_t.tolist())
                    ):
                        if a >= 0 and f != -100 and (r, a) in flat:
                            sel_t.append(k)
                            sel_s.append(flat[(r, a)])
                    if sel_t:
                        it = torch.tensor(sel_t, device=device)
                        link_rows.append(
                            (
                                vectors[it].half().cpu(),
                                s_vectors[torch.tensor(sel_s, device=device)].half().cpu(),
                                phi[torch.tensor(sel_t)],
                                fine_t[torch.tensor(sel_t)],
                            )
                        )
        self._encoder.train()

        fine_weights = None
        if self._fine and self.config.fine_class_weights:
            table = dict(self.config.fine_class_weights)
            fine_weights = torch.tensor(
                [float(table.get(c, 1.0)) for c in self._fine], device=device
            )
        frame_loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(self.config.frame_positive_weight)], device=device)
        )

        if link_rows and self._typer is not None:
            h_t = torch.cat([r[0] for r in link_rows])
            h_s = torch.cat([r[1] for r in link_rows])
            phi = torch.cat([r[2] for r in link_rows])
            y = torch.cat([r[3] for r in link_rows])
            params = list(self._typer.parameters()) + (
                list(self._typer_evidence.parameters()) if self._typer_evidence is not None else []
            )
            opt = torch.optim.Adam(params, lr=lr)
            n = len(y)
            for epoch in range(epochs):
                order = torch.randperm(n)
                total = 0.0
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    logits = self._typer_forward(
                        h_t[idx].float().to(device), h_s[idx].float().to(device), phi[idx]
                    )
                    loss = self._typer_loss(logits, y[idx].to(device), fine_weights)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * len(idx)
                if log:
                    log(
                        f"    typer refine {epoch + 1}/{epochs}  loss {total / n:.4f}  ({n:,} links)"
                    )
        if frame_rows and self._frame_head is not None:
            h = torch.cat([r[0] for r in frame_rows])
            y = torch.cat([r[1] for r in frame_rows])
            opt = torch.optim.Adam(self._frame_head.parameters(), lr=lr)
            n = len(y)
            for epoch in range(epochs):
                order = torch.randperm(n)
                total = 0.0
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    loss = frame_loss_fn(
                        self._frame_head(h[idx].float().to(device)), y[idx].to(device)
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * len(idx)
                if log:
                    log(
                        f"    frame refine {epoch + 1}/{epochs}  loss {total / n:.4f}  ({n:,} words)"
                    )
