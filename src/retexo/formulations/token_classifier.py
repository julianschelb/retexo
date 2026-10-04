# retexo/formulations/token_classifier.py
"""
Formulation 1: label every reuse position.

The pair is encoded once and the model emits, for each token of the reuse,
which operation produced it and which source token it came from. Position is
never a value the model writes down: the operation for reuse token *j* is read
off the representation at position *j*, so the output cannot drift out of
alignment with the text the way a generated sequence can.

What remains is the source index, predicted as a distribution over source
positions — a pointer in all but name. That makes this arm the natural place
to see whether pointing is sufficient on its own.

Deletions have no reuse position to attach to, so they are recovered after
decoding: any source token that no reuse token points at was deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from retexo.formulations.encoding import DERIVED_TAGS, NULL_SOURCE, ScriptEncoder
from retexo.formulations.base import FormulationConfig, ScriptExample, ScriptModel
from retexo.operations import EditOperation, OperationRegistry
from retexo.formulations.pair_encoding import PairEncoder
from retexo.core.script import EditScript

# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class TokenClassifierConfig(FormulationConfig):
    """Configuration for the tagging formulation."""

    base_model: str = "FacebookAI/xlm-roberta-base"

    #: Weight of the source-pointer loss against the operation loss.
    pointer_loss_weight: float = 1.0

    #: Longest source passage the pointer head can address.
    max_source_positions: int = 128

    #: Reweight the operation head by inverse class frequency. The typed
    #: relations are rare -- in the benchmark's real pairs, SYN is 0.19% of
    #: operations and HYPER is six instances -- so an unweighted cross-entropy
    #: is minimised by ignoring them. Weighting trades precision on the common
    #: operations for any signal at all on the rare ones.
    balance_operation_loss: bool = False

    #: Weight of the null class in the pointer head, relative to the real
    #: source positions. A reuse token pointing nowhere is an insertion, and
    #: insertions are 40% of gold, so the cheapest way to lower the pointer
    #: loss is to point nowhere. Below 1.0 this discourages that without
    #: forbidding it.
    null_pointer_weight: float = 1.0

    #: Cap on any single class weight, so a tag with two instances cannot
    #: dominate the gradient.
    max_class_weight: float = 20.0

    #: Strength of the cost regulariser. The cost model in Table 2 makes
    #: insertion and deletion the most expensive operations, but that cost
    #: binds the oracle, which minimises it, and never reaches the model, which
    #: is fit with plain cross-entropy. "Insertion is expensive" is therefore a
    #: statement about the teacher that the student never hears. This adds the
    #: expected cost of the predicted script to the loss, so the
    #: minimum-description-length argument applies during training too: a model
    #: that can explain a position by a cheap typed relation is pushed to
    #: prefer it over an insertion that explains nothing. Zero disables it.
    cost_weight: float = 0.0


# =============================================================================
# Model
# =============================================================================


class TokenClassifierModel(ScriptModel):
    """Encoder with an operation head and a source-pointer head."""

    name = "token_classifier"

    def __init__(self, config: Optional[TokenClassifierConfig] = None, registry=None):
        super().__init__(config or TokenClassifierConfig())
        self.registry = registry or OperationRegistry.default()
        #: Insertion and deletion are deliberately absent from the operation
        #: head. Both are recovered at decode time from the pointer -- a reuse
        #: position pointing nowhere is an insertion, a source position nothing
        #: points at is a deletion -- so predicting them here would give the
        #: model a second, unsupervised way to say the same thing. Since most
        #: gold positions in this data are insertions, that second channel is
        #: also the cheapest way to lower the loss without learning anything,
        #: which is exactly the degenerate solution the cost model exists to
        #: prevent. Leaving them out makes it unavailable rather than merely
        #: expensive: the head answers only "given that this reuse token came
        #: from that source token, what is the relation?"
        self.labels: Dict[str, int] = ScriptEncoder.label_vocabulary(
            [tag for tag in self.registry.tags() if tag not in DERIVED_TAGS]
        )
        self.inverse_labels = {i: tag for tag, i in self.labels.items()}
        self._pair_encoder = None
        self._encoder = None
        self._op_weights = None
        self._op_head = None
        self._pointer_head = None

    # ---------- Setup ----------

    def _build(self):
        """Load the encoder and attach the two prediction heads."""
        if self._encoder is not None:
            return
        import torch
        from transformers import AutoModel

        torch.manual_seed(self.config.seed)
        # Latin BERT ships a tensor2tensor vocabulary that AutoTokenizer reads
        # as WordPiece, mapping most Latin words to [UNK] without error; the
        # pair encoder selects the right one for the backbone.
        self._pair_encoder = PairEncoder.build(self.config.base_model)
        self._encoder = AutoModel.from_pretrained(self.config.base_model)
        hidden = self._encoder.config.hidden_size
        self._op_head = torch.nn.Linear(hidden, len(self.labels))
        self._pointer_head = torch.nn.Linear(hidden, self.config.max_source_positions + 1)
        for module in (self._encoder, self._op_head, self._pointer_head):
            module.to(self.config.device)

    def _parameters(self):
        """Encoder and both heads, for one optimizer over all of them."""
        return (
            list(self._encoder.parameters())
            + list(self._op_head.parameters())
            + list(self._pointer_head.parameters())
        )

    # ---------- Batching ----------

    def _encode(self, examples: Sequence[ScriptExample]):
        """Encode pairs and align labels to the first sub-token of each reuse word."""
        import torch

        pairs = [(list(e.source_tokens), list(e.target_tokens)) for e in examples]
        batch, spans = self._pair_encoder.encode(pairs, self.config.max_length)
        shape = batch["input_ids"].shape
        op_labels = torch.full(shape, -100, dtype=torch.long)
        pointers = torch.full(shape, -100, dtype=torch.long)

        for row, example in enumerate(examples):
            encoded = ScriptEncoder.to_token_labels(example.script)
            for word, (start, _) in enumerate(spans[row]):
                if word >= len(encoded["op_labels"]):
                    break
                tag = encoded["op_labels"][word]
                #: Insertions carry no operation label: the pointer head alone
                #: supervises them, through NULL_SOURCE below. Left at -100 the
                #: cross-entropy ignores the position entirely.
                if tag in self.labels:
                    op_labels[row, start] = self.labels[tag]
                source = encoded["source_indices"][word]
                pointers[row, start] = (
                    self.config.max_source_positions
                    if source == NULL_SOURCE or source >= self.config.max_source_positions
                    else source
                )
        return batch, op_labels, pointers, spans

    def _class_weights(self, train: Sequence[ScriptExample]):
        """Inverse-frequency weights for the operation head, from the data.

        Computed once from the training split rather than per batch, so the
        weighting is a property of the run and appears in its record.
        """
        import torch

        counts = torch.zeros(len(self.labels))
        for example in train:
            for tag, n in example.script.op_counts().items():
                if tag in self.labels:
                    counts[self.labels[tag]] += n
        seen = counts > 0
        if not seen.any():
            return None
        weights = torch.ones(len(self.labels))
        # Only classes present in the data are reweighted; an absent class
        # keeps weight one, since inverse frequency is undefined for zero.
        weights[seen] = (counts[seen].sum() / (seen.sum() * counts[seen]))
        return weights.clamp(max=self.config.max_class_weight).to(self.config.device)

    def _cost_vector(self):
        """Cost of each operation-head class, ordered by label index."""
        import torch

        costs = self.registry.costs()
        vector = torch.zeros(len(self.labels))
        for tag, index in self.labels.items():
            vector[index] = costs.get(tag, 0.0)
        return vector.to(self.config.device)

    def _expected_cost(self, op_logits, pointer_logits, valid):
        """Mean cost the model expects to pay, under its own distributions.

        A reuse position costs the insertion price weighted by how likely the
        model thinks it has no source, plus the cost of the relation it would
        assert if it does. Differentiable in both heads, so the pressure lands
        on the pointer as well as on the tag.
        """
        import torch

        if not valid.any():
            return op_logits.sum() * 0.0

        insert_cost = self.registry.costs().get("INS", 1.0)
        op_probs = torch.softmax(op_logits[valid], dim=-1)
        pointer_probs = torch.softmax(pointer_logits[valid], dim=-1)
        null_probability = pointer_probs[:, self.config.max_source_positions]

        typed_cost = (op_probs * self._cost_vector()).sum(dim=-1)
        return (null_probability * insert_cost
                + (1.0 - null_probability) * typed_cost).mean()

    def _pointer_weights(self):
        """Pointer-head weights, down-weighting the null (insertion) class."""
        import torch

        if self.config.null_pointer_weight == 1.0:
            return None
        size = self.config.max_source_positions + 1
        weights = torch.ones(size)
        weights[self.config.max_source_positions] = self.config.null_pointer_weight
        return weights.to(self.config.device)

    def _losses(self, batch, op_labels, pointers):
        """Operation loss plus weighted pointer loss."""
        import torch

        moved = {k: v.to(self.config.device) for k, v in batch.items()}
        hidden = self._encoder(**moved).last_hidden_state
        op_loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100, weight=self._op_weights
        )
        pointer_loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100, weight=self._pointer_weights()
        )
        op_loss = op_loss_fn(
            self._op_head(hidden).view(-1, len(self.labels)),
            op_labels.to(self.config.device).view(-1),
        )
        pointer_logits = self._pointer_head(hidden).view(
            -1, self.config.max_source_positions + 1
        )
        pointer_loss = pointer_loss_fn(
            pointer_logits, pointers.to(self.config.device).view(-1)
        )
        total = op_loss + self.config.pointer_loss_weight * pointer_loss

        if self.config.cost_weight:
            flat_ops = self._op_head(hidden).view(-1, len(self.labels))
            # Only positions carrying supervision contribute, so padding does
            # not dilute the term.
            valid = pointers.to(self.config.device).view(-1) != -100
            total = total + self.config.cost_weight * self._expected_cost(
                flat_ops, pointer_logits, valid
            )
        return total

    # ---------- Training ----------

    def fit(self, train: Sequence[ScriptExample]) -> "TokenClassifierModel":
        """Train both heads jointly on per-reuse-token operations and pointers."""
        import torch

        self._build()
        if self.config.balance_operation_loss and self._op_weights is None:
            self._op_weights = self._class_weights(train)
        optimizer = torch.optim.AdamW(self._parameters(), lr=self.config.learning_rate)
        self._encoder.train()

        for _ in range(self.config.epochs):
            total, batches = 0.0, 0
            for start in range(0, len(train), self.config.batch_size):
                chunk = list(train[start : start + self.config.batch_size])
                batch, op_labels, pointers, _ = self._encode(chunk)
                loss = self._losses(batch, op_labels, pointers)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total += loss.item()
                batches += 1
            self.log.losses.append(total / max(batches, 1))
        return self

    # ---------- Inference ----------

    def predict(
        self, source_tokens: Sequence[str], target_tokens: Sequence[str]
    ) -> Optional[EditScript]:
        """Tag every reuse position, then assemble a script from the tags."""
        import torch

        self._build()
        self._encoder.eval()
        stub = ScriptExample(
            source_tokens, target_tokens,
            EditScript(list(source_tokens), list(target_tokens), [], self.registry),
        )
        batch, _, _, spans = self._encode([stub])
        moved = {k: v.to(self.config.device) for k, v in batch.items()}
        with torch.no_grad():
            hidden = self._encoder(**moved).last_hidden_state
            op_ids = self._op_head(hidden).argmax(-1)[0]
            pointer_ids = self._pointer_head(hidden).argmax(-1)[0]

        tags: List[str] = []
        sources: List[int] = []
        for start, _ in spans[0]:
            tags.append(self.inverse_labels[int(op_ids[start])])
            pointer = int(pointer_ids[start])
            sources.append(NULL_SOURCE if pointer >= len(source_tokens) else pointer)
        return self._decode(source_tokens, target_tokens, tags, sources)

    def _decode(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        tags: Sequence[str],
        sources: Sequence[int],
    ) -> EditScript:
        """Assemble a script from per-token predictions.

        Predictions can conflict — two reuse tokens may point at one source
        token — so the first claim wins and later ones become insertions.
        Source tokens nothing points at are emitted as deletions, which is how
        deletion is recovered without ever being predicted directly.
        """
        operations: List[EditOperation] = []
        claimed: set = set()
        #: Unaligned positions the model called FRAME; contiguous runs merge
        #: into one operation below, because framing is a single act and the
        #: whole point of the tag is to describe a span rather than charge it
        #: token by token.
        frame_run: List[int] = []

        def flush_frames() -> None:
            if frame_run:
                operations.append(EditOperation(
                    "FRAME", (), tuple(frame_run), (),
                    tuple(target_tokens[i] for i in frame_run),
                ))
                frame_run.clear()

        for index, token in enumerate(target_tokens):
            tag = tags[index] if index < len(tags) else "INS"
            source = sources[index] if index < len(sources) else NULL_SOURCE
            unaligned = (source == NULL_SOURCE or source in claimed
                         or source >= len(source_tokens))
            if unaligned:
                if tag == "FRAME":
                    frame_run.append(index)
                else:
                    flush_frames()
                    operations.append(
                        EditOperation("INS", (), (index,), (), (token,))
                    )
                continue
            flush_frames()
            claimed.add(source)
            operations.append(
                EditOperation(
                    tag if tag not in ("INS", "DEL", "FRAME") else "NOP",
                    (source,), (index,), (source_tokens[source],), (token,),
                )
            )
        flush_frames()
        for index, token in enumerate(source_tokens):
            if index not in claimed:
                operations.append(EditOperation("DEL", (index,), (), (token,), ()))
        return EditScript(list(source_tokens), list(target_tokens), operations, self.registry)

    # ---------- Persistence ----------

    @classmethod
    def load(cls, path: "Path") -> "TokenClassifierModel":
        """Reconstruct a saved tagger for inference.

        Reads the fine-tuned encoder in place of the base checkpoint, restores
        both heads and the exact label order, so ``predict`` behaves as it did
        at save time.
        """
        import torch
        from transformers import AutoModel

        from pathlib import Path as _Path

        path = _Path(path)
        heads = torch.load(path / "heads.pt", map_location="cpu")
        model = cls(TokenClassifierConfig(base_model=str(path), device="cpu"))
        model.labels = heads["labels"]
        model.inverse_labels = {i: t for t, i in model.labels.items()}
        model._pair_encoder = PairEncoder.build(heads["pair_encoder"])
        model._encoder = AutoModel.from_pretrained(str(path))
        hidden = model._encoder.config.hidden_size
        model._op_head = torch.nn.Linear(hidden, len(model.labels))
        model._pointer_head = torch.nn.Linear(
            hidden, model.config.max_source_positions + 1
        )
        model._op_head.load_state_dict(heads["op_head"])
        model._pointer_head.load_state_dict(heads["pointer_head"])
        for module in (model._encoder, model._op_head, model._pointer_head):
            module.to(model.config.device)
        model._encoder.eval()
        return model

    def save(self, path: Path) -> None:
        """Write the encoder, the tokenizer, and both heads with their labels."""
        import torch

        self._build()
        path.mkdir(parents=True, exist_ok=True)
        self._encoder.save_pretrained(path)
        torch.save(
            {"op_head": self._op_head.state_dict(),
             "pointer_head": self._pointer_head.state_dict(),
             "labels": self.labels,
             "pair_encoder": self._pair_encoder.name},
            path / "heads.pt",
        )
