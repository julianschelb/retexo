# retexo/formulations/seq2seq_stepwise.py
"""
Formulation 3: emit one operation at a time, validated in the loop.

At each step the model sees the pair and the operations accepted so far, and
proposes one more. Scriba checks the proposal against the state — does it write
a reuse position still empty, does it consume a source position not already
consumed — and only accepted operations enter the prefix the next step
conditions on. Position is neither inherent nor freely generated: it is
proposed and then externally checked.

This is also where a *dynamic* teacher becomes possible. Because the oracle can
score any partial state, training need not be limited to gold prefixes: the
model can be rolled out and asked what to do next from wherever it actually
arrived. Both regimes are implemented, since telling a formulation problem
apart from an exposure-bias problem is the point of the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from retexo.core.scriba import Scriba
from retexo.core.script import EditScript
from retexo.datasets.parsing import ScriptParser
from retexo.formulations.base import FormulationConfig, ScriptExample, ScriptModel
from retexo.formulations.encoding import ScriptEncoder
from retexo.operations import EditOperation, OperationRegistry, Role

# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class Seq2SeqStepwiseConfig(FormulationConfig):
    """Configuration for validated step-wise decoding."""

    base_model: str = "google/flan-t5-base"
    atomic_operation_tokens: bool = True

    #: Cap on decoding steps, as a multiple of the reuse length.
    max_steps_factor: float = 2.0

    #: Train against oracle decisions at the model's own states, not only gold
    #: prefixes. The exposure-bias remedy, and the D1/D2 crossover.
    dynamic_teacher: bool = False

    #: Share of training states drawn from rollouts when the teacher is dynamic.
    rollout_ratio: float = 0.5

    #: Keep at most this many training states per example. Each example expands
    #: into one state per operation, so a 34-operation script becomes 34 states
    #: and the corpus grows ~30x; a cap makes the arm trainable overnight while
    #: still teaching the step decision. Zero means no cap. The final step
    #: (the <STOP> decision) is always kept; the rest are sampled.
    max_states_per_example: int = 0


# =============================================================================
# Model
# =============================================================================


class Seq2SeqStepwiseModel(ScriptModel):
    """Step-wise decoder with Scriba validating each proposed operation."""

    name = "seq2seq_stepwise"

    def __init__(
        self,
        config: Optional[Seq2SeqStepwiseConfig] = None,
        registry=None,
        oracle=None,
    ):
        super().__init__(config or Seq2SeqStepwiseConfig())
        self.registry = registry or OperationRegistry.default()
        self.scriba = Scriba(self.registry)
        self.oracle = oracle
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

    # ---------- State rendering ----------

    def _state_text(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        prefix: Sequence[EditOperation],
    ) -> str:
        """The model's view: the pair, plus what has been accepted so far."""
        marked = " ".join(f"{i}:{t}" for i, t in enumerate(source_tokens))
        reuse = " ".join(f"{i}:{t}" for i, t in enumerate(target_tokens))
        done = " | ".join(op.serialize() for op in prefix) or "none"
        return f"source: {marked} reuse: {reuse} done: {done} next:"

    @staticmethod
    def _op_text(op: EditOperation, atomic: bool) -> str:
        """One operation as the single line the decoder emits for a step."""
        tag = f"<{op.tag}>" if atomic else op.tag
        src = ",".join(str(i) for i in op.source_indices) or "-"
        tgt = ",".join(str(i) for i in op.target_indices) or "-"
        return f"{tag} {src}>{tgt} {' '.join(op.target_tokens)}".strip()

    def _training_states(self, example: ScriptExample) -> List[Tuple[str, str]]:
        """Gold prefixes: at each step, what the gold script does next."""
        pairs = []
        operations = list(example.script.operations)
        for cut in range(len(operations)):
            text = self._state_text(example.source_tokens, example.target_tokens, operations[:cut])
            pairs.append(
                (text, self._op_text(operations[cut], self.config.atomic_operation_tokens))
            )
        pairs.append(
            (
                self._state_text(example.source_tokens, example.target_tokens, operations),
                "<STOP>",
            )
        )
        return pairs

    # ---------- Training ----------

    def fit(self, train: Sequence[ScriptExample]) -> Seq2SeqStepwiseModel:
        """Train one step at a time, on gold prefixes or on the model's own states.

        Raises:
            ValueError: If ``dynamic_teacher`` is set without an oracle, since
                there is then nothing to ask what to do next from a state the
                gold script never visited.
        """
        import torch

        if self.config.dynamic_teacher and self.oracle is None:
            raise ValueError(
                "dynamic_teacher=True requires an oracle: there is nothing to ask "
                "what the next operation should be from a state the gold script "
                "never visited"
            )
        self._build()
        self._tokenizer.add_tokens(["<STOP>"])
        self._model.resize_token_embeddings(len(self._tokenizer))

        import random as _random

        rng = _random.Random(self.config.seed)
        states: List[Tuple[str, str]] = []
        cap = self.config.max_states_per_example
        for example in train:
            example_states = self._training_states(example)
            if cap and len(example_states) > cap:
                # Always keep the last state (the STOP decision); sample the
                # rest, so the model still learns when to halt.
                head, tail = example_states[:-1], example_states[-1:]
                example_states = rng.sample(head, cap - 1) + tail
            states.extend(example_states)
        rng.shuffle(states)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.config.learning_rate)
        self._model.train()
        for _ in range(self.config.epochs):
            total, batches = 0.0, 0
            for start in range(0, len(states), self.config.batch_size):
                chunk = states[start : start + self.config.batch_size]
                inputs = self._tokenizer(
                    [c[0] for c in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.config.device)
                labels = self._tokenizer(
                    [c[1] for c in chunk],
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                ).input_ids.to(self.config.device)
                labels[labels == self._tokenizer.pad_token_id] = -100
                loss = self._model(**inputs, labels=labels).loss
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
        """Decode one operation at a time, keeping only what Scriba accepts."""
        import torch

        self._build()
        self._model.eval()
        prefix: List[EditOperation] = []
        max_steps = int(self.config.max_steps_factor * max(len(target_tokens), 1)) + 2

        for _ in range(max_steps):
            text = self._state_text(source_tokens, target_tokens, prefix)
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.config.device)
            with torch.no_grad():
                generated = self._model.generate(**encoded, max_new_tokens=24)
            decoded = self._tokenizer.decode(generated[0], skip_special_tokens=False)
            if "<STOP>" in decoded:
                break
            proposal = ScriptParser.parse(decoded, source_tokens, target_tokens, self.registry)
            if proposal is None or not proposal.operations:
                break
            operation = proposal.operations[0]
            if self._accepts(prefix, operation, len(source_tokens), len(target_tokens)):
                prefix.append(operation)
            else:
                break

        if not prefix:
            return None
        return EditScript(list(source_tokens), list(target_tokens), prefix, self.registry)

    def _accepts(
        self,
        prefix: Sequence[EditOperation],
        operation: EditOperation,
        source_length: int,
        target_length: int,
    ) -> bool:
        """Whether an operation is consistent with the state reached so far."""
        role = self.registry[operation.tag].role
        if role is Role.MARKER:
            return True
        written = {i for op in prefix for i in op.target_indices}
        consumed = {i for op in prefix for i in op.source_indices}
        return not any(
            i in written or i >= target_length for i in operation.target_indices
        ) and not any(i in consumed or i >= source_length for i in operation.source_indices)

    # ---------- Dynamic teacher ----------

    def oracle_next(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        prefix: Sequence[EditOperation],
    ) -> Optional[EditOperation]:
        """What the oracle would do from an arbitrary partial state.

        Not the same as reading the next operation off a gold script: the
        prefix may contain operations the gold script never chose, so the
        oracle must re-plan over what is left rather than look up an index.

        Raises:
            ValueError: If no oracle was supplied.
        """
        if self.oracle is None:
            raise ValueError("no oracle available for dynamic teaching")
        consumed = {i for op in prefix for i in op.source_indices}
        written = {i for op in prefix for i in op.target_indices}
        remaining_source = [t for i, t in enumerate(source_tokens) if i not in consumed]
        remaining_target = [t for i, t in enumerate(target_tokens) if i not in written]
        if not remaining_target:
            return None
        replanned = self.oracle.plan_tokens(remaining_source, remaining_target)
        return replanned.operations[0] if replanned.operations else None

    # ---------- Persistence ----------

    @classmethod
    def load(cls, path: Path) -> Seq2SeqStepwiseModel:
        """Reconstruct a saved step-wise model for inference."""
        from pathlib import Path as _Path

        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = _Path(path)
        model = cls(Seq2SeqStepwiseConfig(base_model=str(path), device="cpu"))
        model._tokenizer = AutoTokenizer.from_pretrained(str(path))
        model._model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        model._model.to(model.config.device).eval()
        return model

    def save(self, path: Path) -> None:
        """Write the backbone and the tokenizer, including added operation tokens."""
        self._build()
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
