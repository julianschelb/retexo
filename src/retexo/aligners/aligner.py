# retexo/aligners/aligner.py
"""
Contextual word alignment, for the teacher's neural half.

Wraps a (base or fine-tuned) Latin BERT as a per-word embedder and offers
mutual-best matching over two passages. Stage 1b trains the checkpoint; the
teacher consumes it to convert resource-silent DEL+INS residue into arrows.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from retexo.formulations.pair_encoding import latin_bert_pieces

BASE_MODEL = "ashleygong03/bamman-burns-latin-bert"

#: Layer the Stage-1 diagnostic measures and the Stage-1b objective trains.
LAYER = 8


class ContextualAligner:
    """Per-word vectors and mutual-best matches from Latin BERT.

    Example:
        ```python
        aligner = ContextualAligner("runs/aligner_v1", device="cuda")
        for i, j, sim in aligner.mutual_best(source_words, target_words):
            ...
        ```
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        import huggingface_hub
        import torch
        from locisimiles.tokenization.latin_bert import SubwordTextEncoder
        from transformers import AutoModel

        vocab = huggingface_hub.hf_hub_download(BASE_MODEL, "vocab.txt")
        self.encoder = SubwordTextEncoder.from_file(vocab)
        self.specials = {
            s: i for i, s in enumerate(self.encoder._subtokens) if s in ("[CLS]", "[SEP]")
        }
        self.model = AutoModel.from_pretrained(model_path or BASE_MODEL)
        self.model.to(device).eval()
        self.device = device
        self.torch = torch

    def embed(self, words: Sequence[str]):
        """One L2-normalised vector per word."""
        torch = self.torch
        ids = [self.specials["[CLS]"]]
        spans = []
        for word in words:
            pieces = latin_bert_pieces(self.encoder, word)
            spans.append((len(ids), len(ids) + len(pieces)))
            ids.extend(pieces)
        ids.append(self.specials["[SEP]"])
        with torch.no_grad():
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device), output_hidden_states=True
            )
        hidden = out.hidden_states[LAYER][0]
        vectors = torch.stack([hidden[a:b].mean(0) for a, b in spans])
        return torch.nn.functional.normalize(vectors, dim=-1)

    def mutual_best(
        self,
        source_words: Sequence[str],
        target_words: Sequence[str],
        *,
        threshold: float = 0.0,
    ) -> List[Tuple[int, int, float]]:
        """Pairs that are each other's best match, at or above ``threshold``."""
        if not source_words or not target_words:
            return []
        sim = self.embed(source_words) @ self.embed(target_words).T
        row = sim.argmax(1)
        col = sim.argmax(0)
        out = []
        for i in range(sim.shape[0]):
            j = int(row[i])
            score = float(sim[i, j])
            if int(col[j]) == i and score >= threshold:
                out.append((i, j, score))
        return out
