# baselines/span_pair/substrate.py
"""The joint encoder substrate: one vector per word of each side and the pruning grid."""

from __future__ import annotations

from typing import Sequence, Tuple

from retexo.baselines.span_pair.constants import Span

# =============================================================================
# Substrate
# =============================================================================


class Substrate:
    """The joint encoder: one vector per word of both sides, and a word grid
    (row-softmaxed cosine at ``temperature``) that prunes the candidate pairs.

    Example:
        ```python
        sub = Substrate("bert-base-cased", device="cpu")
        (h_r, h_s), grid = sub.encode_one(reuse, source)
        ```
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        max_length: int = 256,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self._model = None
        self._pair_encoder = None

    # ---------- model state ----------

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        from retexo.formulations.pair_encoding import PairEncoder

        kwargs = {"reference_compile": False} if "modernbert" in self.model_name.lower() else {}
        self._model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device)
        self._pair_encoder = PairEncoder.build(self.model_name)

    @property
    def hidden_size(self) -> int:
        self._ensure()
        return int(self._model.config.hidden_size)

    def parameters(self):
        self._ensure()
        return self._model.parameters()

    def train(self) -> None:
        self._ensure()
        self._model.train()

    def eval(self) -> None:
        self._ensure()
        self._model.eval()

    def state_dict(self):
        self._ensure()
        return self._model.state_dict()

    def load_state_dict(self, state) -> None:
        self._ensure()
        self._model.load_state_dict(state)

    # ---------- encoding ----------

    @staticmethod
    def _pool(hidden_row, spans: Sequence[Span]):
        import torch

        return (
            torch.stack([hidden_row[a : max(b, a + 1)].mean(dim=0) for a, b in spans])
            if spans
            else hidden_row[:0]
        )

    def encode(self, pairs: Sequence[Tuple[Sequence[str], Sequence[str]]]):
        """Per pair ``(reuse_words [n_t, H], source_words [n_s, H])``; words the
        encoder truncated away are absent (callers cut to the shorter length)."""
        self._ensure()
        batch, reuse_spans = self._pair_encoder.encode(
            [(list(s), list(r)) for s, r in pairs], self.max_length
        )
        source_spans = list(self._pair_encoder.last_source_spans)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self._model(**batch).last_hidden_state
        out = []
        for i in range(len(pairs)):
            out.append(
                (self._pool(hidden[i], reuse_spans[i]), self._pool(hidden[i], source_spans[i]))
            )
        return out

    def grid(self, h_r, h_s):
        """Row-softmaxed cosine similarity, a probability per (reuse, source) cell."""
        import torch

        if h_r.shape[0] == 0 or h_s.shape[0] == 0:
            return torch.zeros((h_r.shape[0], h_s.shape[0]))
        r = torch.nn.functional.normalize(h_r.detach(), dim=-1)
        s = torch.nn.functional.normalize(h_s.detach(), dim=-1)
        return torch.softmax((r @ s.T) / self.temperature, dim=-1).cpu()
