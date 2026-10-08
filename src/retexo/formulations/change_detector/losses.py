# formulations/change_detector/losses.py
"""The losses of the change detector and the loss evaluations used while training."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from retexo.formulations.change_detector.constants import GROUP_TARGET
from retexo.formulations.change_detector.example import ChangeExample

# =============================================================================
# _LossMixin
# =============================================================================


class _LossMixin:
    """Loss terms and held-out loss evaluation of the change detector."""

    # ---------- Loss terms ----------

    def _loss_functions(self) -> Dict[str, object]:
        """The training losses and their class weights (one place for ``fit`` and ``evaluation_loss``)."""
        import torch

        weights = None
        if self.config.class_weights and self._classes:
            table = dict(self.config.class_weights)
            weights = torch.tensor(
                [float(table.get(c, 1.0)) for c in self._classes], device=self.config.device
            )
        fine_weights = None
        if self._fine and self.config.fine_class_weights:
            table = dict(self.config.fine_class_weights)
            fine_weights = torch.tensor(
                [float(table.get(c, 1.0)) for c in self._fine], device=self.config.device
            )
        return {
            "weights": weights,
            "fine_weights": fine_weights,
            "loss": torch.nn.CrossEntropyLoss(ignore_index=-100, weight=weights),
            # The source head is binary (deleted / not), so it must not inherit a
            # weight vector sized for the operation classes.
            "source": torch.nn.CrossEntropyLoss(ignore_index=-100),
            "frame": torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                weight=torch.tensor(
                    [1.0, float(self.config.frame_positive_weight)], device=self.config.device
                ),
            ),
        }

    def _chunk_loss(self, chunk: Sequence[ChangeExample], fns: Dict[str, object]):
        """The training loss of one batch, or ``None`` when it has no target."""
        logits, targets, source_out, pointer_out, extra = self._logits(chunk)
        if targets.numel() == 0:
            return None
        loss = (
            self._focal(logits, targets, fns["weights"])
            if self.config.focal_gamma
            else fns["loss"](logits, targets)
        )
        if source_out is not None:
            # Both sides weigh equally: a missed deletion is as wrong
            # as a missed substitution.
            loss = loss + fns["source"](*source_out)
        if pointer_out is not None:
            loss = loss + self._pointer_loss(*pointer_out)
        typer_out, frame_out = extra
        if typer_out is not None:
            loss = loss + self._typer_loss(*typer_out, fns["fine_weights"])
        if frame_out is not None and bool((frame_out[1] != -100).any()):
            loss = loss + fns["frame"](*frame_out)
        return loss

    def _typer_loss(self, logits, targets, weights):
        """Cross-entropy on links whose kind is known; a group term on the rest.

        A hand-labelled SUBST says only "changed, lexically". Its loss is the
        negative log of the probability mass on the lexical classes together,
        so real data teaches the boundary between NOP / MORPH and the lexical
        group while the constructed data, where every link is typed, decides
        the relation inside the group.
        """
        import torch
        import torch.nn.functional as F

        exact = targets >= 0
        group = targets == GROUP_TARGET
        loss = logits.sum() * 0.0
        if bool(exact.any()):
            loss = loss + F.cross_entropy(logits[exact], targets[exact], weight=weights)
        if bool(group.any()) and self._lexical_index:
            log_p = F.log_softmax(logits[group], dim=-1)
            idx = torch.tensor(self._lexical_index, device=logits.device)
            in_group = torch.logsumexp(log_p[:, idx], dim=-1)
            loss = loss + self.config.group_loss_weight * (-in_group).mean()
        return loss

    def _pointer_loss(self, scores, gold):
        """Cross-entropy over the candidates, with the null down-weighted.

        Only 14.5% of reuse words align, so a pointer trained on a plain
        objective learns to point nowhere -- the same corner E5's unweighted
        SUBST fell into. The weight applies to the *gold* being null rather than
        to the prediction, so it re-prices the examples the model would
        otherwise be right to ignore.
        """
        import torch
        import torch.nn.functional as F

        losses = F.cross_entropy(scores, gold, ignore_index=-100, reduction="none")
        labelled = gold != -100
        if not bool(labelled.any()):
            return scores.sum() * 0.0
        weights = torch.where(
            gold == 0,
            torch.full_like(losses, self.config.null_pointer_weight),
            torch.ones_like(losses),
        )
        weights = weights * labelled.float()
        return (losses * weights).sum() / weights.sum().clamp(min=1e-6)

    def _focal(self, logits, targets, weights):
        """Focal loss: down-weight what the model already gets right.

        With 86% of tokens INS, plain cross-entropy is dominated by examples
        that are already correct. Focal loss scales each by ``(1 - p)^gamma``,
        so the rare classes keep influencing the gradient.
        """
        import torch.nn.functional as F

        log_probability = F.log_softmax(logits, dim=-1)
        valid = targets != -100
        if not valid.any():
            return logits.sum() * 0.0
        logits, targets = log_probability[valid], targets[valid]
        chosen = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - chosen.exp()) ** self.config.focal_gamma) * chosen
        if weights is not None:
            loss = loss * weights[targets]
        return loss.mean()

    # ---------- Evaluation ----------

    def evaluation_loss(self, examples: Sequence[ChangeExample]) -> Optional[float]:
        """The training loss on ``examples`` without a gradient (the validation loss), the mean over batches."""
        import torch

        fns = self._loss_functions()
        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                loss = self._chunk_loss(list(examples[start : start + self.config.batch_size]), fns)
                if loss is not None:
                    total += float(loss.item())
                    n += 1
        return total / n if n else None

    def evaluate_loss(self, examples: Sequence[ChangeExample]) -> float:
        """Mean loss on held-out examples, for the early-stopping criterion."""
        import torch

        self._encoder.eval()
        self._head.eval()
        if self._source_head is not None:
            self._source_head.eval()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        losses = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                logits, targets, source_out, _, _ = self._logits(chunk)
                if targets.numel() == 0:
                    continue
                loss = (
                    self._focal(
                        logits, targets, None
                    )  # unweighted, like the plain branch's loss_fn
                    if self.config.focal_gamma
                    else loss_fn(logits, targets)
                )
                if source_out is not None:
                    loss = loss + loss_fn(*source_out)
                losses.append(float(loss.item()))
        self._encoder.train()
        self._head.train()
        if self._source_head is not None:
            self._source_head.train()
        return sum(losses) / max(len(losses), 1)
