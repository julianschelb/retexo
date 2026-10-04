# retexo/formulations/seq2seq_full.py
"""
Formulation 2: generate the whole script in one pass.

The pair goes in, the serialized script comes out, and the model decides
content and position at the same time. This is the natural formulation and the
baseline the other two are measured against, so it is built to be genuinely
competitive rather than as a straw man: it gets the same budget, and its two
encodings — operation names as words, or collapsed to single vocabulary tokens
— are variants of this arm rather than separate models.

Its known weakness is that indices are written as ordinary output, so nothing
constrains them to stay consistent with the length of the reuse. Whether that
is the dominant failure is what the comparison exists to decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from retexo.formulations.encoding import ScriptEncoder
from retexo.formulations.base import FormulationConfig, ScriptExample, ScriptModel
from retexo.operations import OperationRegistry
from retexo.datasets.parsing import ScriptParser
from retexo.core.script import EditScript

# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class Seq2SeqFullConfig(FormulationConfig):
    """Configuration for one-pass script generation."""

    base_model: str = "google/flan-t5-base"

    #: Collapse each operation name to a single added vocabulary token.
    atomic_operation_tokens: bool = True

    num_beams: int = 1

    #: Generate several candidates and keep the first Scriba accepts.
    best_of_n: int = 1


# =============================================================================
# Model
# =============================================================================


class Seq2SeqFullModel(ScriptModel):
    """Sequence-to-sequence model emitting a serialized script."""

    name = "seq2seq_full"

    def __init__(self, config: Optional[Seq2SeqFullConfig] = None, registry=None):
        super().__init__(config or Seq2SeqFullConfig())
        self.registry = registry or OperationRegistry.default()
        self._tokenizer = None
        self._model = None

    # ---------- Setup ----------

    def _build(self):
        """Load the backbone, adding operation tokens if configured."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.manual_seed(self.config.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model)
        if self.config.atomic_operation_tokens:
            self._tokenizer.add_tokens(ScriptEncoder.operation_tokens(self.registry.tags()))
            self._model.resize_token_embeddings(len(self._tokenizer))
        self._model.to(self.config.device)

    # ---------- Training ----------

    def fit(self, train: Sequence[ScriptExample]) -> "Seq2SeqFullModel":
        """Train on whole serialized scripts, one target string per pair."""
        import torch

        self._build()
        pairs = [
            ScriptEncoder.to_seq2seq(e.script, atomic_tags=self.config.atomic_operation_tokens)
            for e in train
        ]
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.config.learning_rate)
        self._model.train()

        for _ in range(self.config.epochs):
            total, batches = 0.0, 0
            for start in range(0, len(pairs), self.config.batch_size):
                chunk = pairs[start : start + self.config.batch_size]
                loss = self._step(chunk)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total += loss.item()
                batches += 1
            self.log.losses.append(total / max(batches, 1))
        return self

    def _step(self, chunk):
        """Loss for one batch of (input, target) string pairs."""
        inputs = self._tokenizer(
            [c[0] for c in chunk], padding=True, truncation=True,
            max_length=self.config.max_length, return_tensors="pt",
        ).to(self.config.device)
        labels = self._tokenizer(
            [c[1] for c in chunk], padding=True, truncation=True,
            max_length=self.config.max_length, return_tensors="pt",
        ).input_ids.to(self.config.device)
        labels[labels == self._tokenizer.pad_token_id] = -100
        return self._model(**inputs, labels=labels).loss

    # ---------- Inference ----------

    def predict(
        self, source_tokens: Sequence[str], target_tokens: Sequence[str]
    ) -> Optional[EditScript]:
        """Generate a script in one pass, keeping the first that parses."""
        import torch

        self._build()
        self._model.eval()
        stub = EditScript(list(source_tokens), list(target_tokens), [], self.registry)
        model_input, _ = ScriptEncoder.to_seq2seq(stub, atomic_tags=self.config.atomic_operation_tokens)
        encoded = self._tokenizer(
            model_input, return_tensors="pt", truncation=True,
            max_length=self.config.max_length,
        ).to(self.config.device)

        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.config.max_length,
                num_beams=self.config.num_beams,
                num_return_sequences=self.config.best_of_n,
            )
        for row in generated:
            text = self._tokenizer.decode(row, skip_special_tokens=False)
            script = ScriptParser.parse(text, source_tokens, target_tokens, self.registry)
            if script is not None:
                return script
        return None

    # ---------- Persistence ----------

    def save(self, path: Path) -> None:
        """Write the backbone and the tokenizer, including added operation tokens."""
        self._build()
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
