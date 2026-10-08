# retexo/formulations/operation_typer.py
"""
E2: given the position that changed, say which operation changed it.

The typing question on its own. Position is not asked -- E0 covers that -- so
the loss and the prediction are taken at one known position per example, and a
poor score cannot be blamed on failing to find the change.

Shares the pair encoder and the pooling choice with the change detector, so a
result here is comparable with E0's rather than confounded by a different input
path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from retexo.formulations.pair_encoding import PairEncoder


@dataclass(frozen=True)
class TyperConfig:
    """Knobs for the operation classifier."""

    base_model: str = "FacebookAI/xlm-roberta-base"
    pooling: str = "mean"
    max_length: int = 256
    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 2e-5
    device: str = "cuda"
    seed: int = 42


@dataclass(frozen=True)
class TypedExample:
    """A pair with exactly one operation, at a known position."""

    source_tokens: List[str]
    target_tokens: List[str]
    position: int
    tag: str
    source_word: str
    target_word: str


class OperationTyper:
    """Classifies the operation at one marked reuse position."""

    def __init__(self, tags: Sequence[str], config: Optional[TyperConfig] = None) -> None:
        import torch
        from transformers import AutoModel

        self.config = config or TyperConfig()
        self.tags = list(tags)
        self.index = {tag: i for i, tag in enumerate(self.tags)}

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

        self._pair_encoder = PairEncoder.build(self.config.base_model)
        self._encoder = AutoModel.from_pretrained(self.config.base_model)
        hidden = self._encoder.config.hidden_size
        self._head = torch.nn.Linear(hidden, len(self.tags))
        for module in (self._encoder, self._head):
            module.to(self.config.device)
        self.losses: List[float] = []
        #: Examples whose marked position fell outside the encoded sequence.
        self.unreachable = 0

    def _vectors(self, examples: Sequence[TypedExample]):
        import torch

        pairs = [(list(e.source_tokens), list(e.target_tokens)) for e in examples]
        batch, spans = self._pair_encoder.encode(pairs, self.config.max_length)
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        hidden = self._encoder(**batch).last_hidden_state

        rows, starts, ends, labels, kept = [], [], [], [], []
        for row, example in enumerate(examples):
            if example.position >= len(spans[row]):
                self.unreachable += 1
                continue
            start, end = spans[row][example.position]
            rows.append(row)
            starts.append(start)
            ends.append(max(end, start + 1))
            labels.append(self.index[example.tag])
            kept.append(row)
        if not rows:
            return None, None, []

        rows_t = torch.tensor(rows)
        starts_t = torch.tensor(starts)
        ends_t = torch.tensor(ends)
        if self.config.pooling == "first":
            vectors = hidden[rows_t, starts_t]
        else:
            widths = ends_t - starts_t
            longest = int(widths.max().item())
            offsets = torch.arange(longest, device=hidden.device).unsqueeze(0)
            index = starts_t.unsqueeze(1).to(hidden.device) + offsets
            mask = offsets < widths.unsqueeze(1).to(hidden.device)
            index = index.clamp(max=hidden.shape[1] - 1)
            gathered = hidden[rows_t.unsqueeze(1).to(hidden.device), index]
            gathered = gathered * mask.unsqueeze(-1)
            vectors = gathered.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (self._head(vectors), torch.tensor(labels).to(self.config.device), kept)

    def fit(self, examples: Sequence[TypedExample], *, log=None) -> OperationTyper:
        import torch

        optimizer = torch.optim.AdamW(
            list(self._encoder.parameters()) + list(self._head.parameters()),
            lr=self.config.learning_rate,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        order = list(examples)
        rng = random.Random(self.config.seed)

        self._encoder.train()
        self._head.train()
        for epoch in range(self.config.epochs):
            rng.shuffle(order)
            epoch_losses = []
            for start in range(0, len(order), self.config.batch_size):
                chunk = order[start : start + self.config.batch_size]
                logits, labels, _ = self._vectors(chunk)
                if logits is None:
                    continue
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            self.losses.append(mean_loss)
            if log:
                log(f"    epoch {epoch + 1}/{self.config.epochs}  loss {mean_loss:.4f}")
        return self

    def predict(self, examples: Sequence[TypedExample]) -> List[Optional[str]]:
        """Predicted tag per example; ``None`` where the position was unreachable."""
        import torch

        self._encoder.eval()
        self._head.eval()
        out: List[Optional[str]] = [None] * len(examples)
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                logits, _, kept = self._vectors(chunk)
                if logits is None:
                    continue
                chosen = logits.argmax(dim=-1).tolist()
                for local, guess in zip(kept, chosen):
                    out[start + local] = self.tags[guess]
        return out


__all__ = ["OperationTyper", "TyperConfig", "TypedExample"]
