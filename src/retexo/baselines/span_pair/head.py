# baselines/span_pair/head.py
"""The span-pair head: span vectors, the pair and null scorers, and the word-type scorer."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from retexo.baselines.span_pair.constants import NULL_TAGS, PAIR_TAGS, WORD_TYPES, Span

# =============================================================================
# SpanPairHead
# =============================================================================


class SpanPairHead:
    """The span representation, the pair scorer and the null scorer.

    A span is ``[h_start ; h_end ; mean(h) ; phi_len]``; a pair
    ``[r ; s ; |r - s| ; r * s ; phi_dlen ; phi_pos]`` scored for QUOTE and ADAPT;
    a reuse span alone scored for INS and FRAME.

    Example:
        ```python
        head = SpanPairHead(hidden_size=768, L_max=6, device="cpu")
        pair_logits = head.score_pairs(vec_r, vec_s, dlen, pos)     # [n, 2]
        ```
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        L_max: int,
        hidden: int = 256,
        device: str = "cpu",
        embed: int = 16,
    ):
        import torch

        self.L_max = L_max
        self.embed = embed
        span_dim = 3 * hidden_size + embed
        self.len_embedding = torch.nn.Embedding(L_max + 4, embed).to(device)
        self.dlen_embedding = torch.nn.Embedding(2 * (L_max + 3) + 1, embed).to(device)
        self.pair_mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * span_dim + embed + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(PAIR_TAGS)),
        ).to(device)
        self.null_mlp = torch.nn.Sequential(
            torch.nn.Linear(span_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(NULL_TAGS)),
        ).to(device)
        self.type_mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(WORD_TYPES)),
        ).to(device)
        self.device = device

    # ---------- state ----------

    def modules(self):
        return [
            self.len_embedding,
            self.dlen_embedding,
            self.pair_mlp,
            self.null_mlp,
            self.type_mlp,
        ]

    def parameters(self):
        return [p for m in self.modules() for p in m.parameters()]

    def train(self) -> None:
        for m in self.modules():
            m.train()

    def eval(self) -> None:
        for m in self.modules():
            m.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {f"{i}": m.state_dict() for i, m in enumerate(self.modules())}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for i, m in enumerate(self.modules()):
            m.load_state_dict(state[f"{i}"])

    # ---------- scoring ----------

    def span_vectors(self, words, spans: Sequence[Span]):
        """``[h_start ; h_end ; mean ; phi_len]`` for each span over ``words`` [n, H]."""
        import torch

        if not spans:
            return words.new_zeros((0, 3 * words.shape[1] + self.embed))
        starts = torch.tensor([a for a, _ in spans], device=words.device)
        ends = torch.tensor([b for _, b in spans], device=words.device)
        means = torch.stack([words[a : b + 1].mean(dim=0) for a, b in spans])
        lengths = torch.tensor(
            [min(b - a + 1, self.L_max + 3) for a, b in spans], device=words.device
        )
        return torch.cat([words[starts], words[ends], means, self.len_embedding(lengths)], dim=-1)

    def score_pairs(self, vec_r, vec_s, dlen, pos):
        """``[n, 2]`` logits (QUOTE, ADAPT) for ``n`` pairs of span vectors."""
        import torch

        dlen_idx = dlen.clamp(-(self.L_max + 3), self.L_max + 3) + self.L_max + 3
        feats = torch.cat(
            [
                vec_r,
                vec_s,
                (vec_r - vec_s).abs(),
                vec_r * vec_s,
                self.dlen_embedding(dlen_idx),
                pos.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.pair_mlp(feats)

    def score_nulls(self, vec_r):
        return self.null_mlp(vec_r)

    def word_types(self, h_t, h_s):
        """``[n, 3]`` logits over COPY, MORPH, SUBST for ``n`` word pairs."""
        import torch

        return self.type_mlp(torch.cat([h_t, h_s, (h_t - h_s).abs(), h_t * h_s], dim=-1))
