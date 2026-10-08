# retexo/formulations/base.py
"""
The contract every output formulation implements.

The three formulations differ only in how a script is represented as a
prediction target, and therefore in how positional bookkeeping is handled.
They take the same supervision, are scored by the same metrics, and return the
same :class:`EditScript`, so the formulation is the only variable in the
comparison.

Training is a plain loop rather than the ``Trainer`` API: the arms differ in
how a batch is built and in what a decoding step means, and a shared explicit
loop keeps those differences visible instead of hidden behind subclassed
callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.core.script import EditScript

# =============================================================================
# Training data
# =============================================================================


@dataclass(frozen=True)
class ScriptExample:
    """One supervised example: a pair and the script that relates them."""

    source_tokens: Sequence[str]
    target_tokens: Sequence[str]
    script: EditScript
    origin: str = "oracle"
    difficulty: Optional[Any] = None

    #: Whether the target embeds the fragment in framing context, and where the
    #: fragment sits inside it. Carried so later stages can supervise span
    #: localisation without re-deriving it.
    framed: bool = False
    fragment_span: Optional[Any] = None

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> ScriptExample:
        """Build from a generator record."""
        return cls(
            source_tokens=record["source_tokens"],
            target_tokens=record["target_tokens"],
            script=record["script"],
            origin=record.get("origin", "synthetic"),
            difficulty=record.get("difficulty"),
            framed=record.get("framed", False),
            fragment_span=record.get("fragment_span"),
        )


@dataclass(frozen=True)
class FormulationConfig:
    """Knobs shared by every formulation.

    Anything specific to one arm belongs in its own subclass, so that a shared
    change cannot silently alter one side of the comparison.
    """

    base_model: str
    output_dir: Path = Path("runs")
    epochs: int = 3
    learning_rate: float = 3e-4
    batch_size: int = 8
    max_length: int = 256
    seed: int = 42
    device: str = "cpu"

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe view, for the experiment record."""
        out = asdict(self)
        out["output_dir"] = str(self.output_dir)
        return out


@dataclass
class TrainingLog:
    """Per-epoch losses, for the experiment record."""

    losses: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[int, Dict[str, float]]:
        """Per-epoch losses keyed by epoch number, as the run record expects."""
        return {i + 1: {"train_loss": loss} for i, loss in enumerate(self.losses)}


# =============================================================================
# Base model
# =============================================================================


class ScriptModel(ABC):
    """Predicts an :class:`EditScript` for a (source, reuse) pair."""

    #: Variant name used in experiment records.
    name: str = ""

    def __init__(self, config: FormulationConfig):
        self.config = config
        self.log = TrainingLog()

    # ---------- Training ----------

    @abstractmethod
    def fit(self, train: Sequence[ScriptExample]) -> ScriptModel:
        """Train on gold scripts. Returns self so calls can be chained."""

    # ---------- Inference ----------

    @abstractmethod
    def predict(
        self, source_tokens: Sequence[str], target_tokens: Sequence[str]
    ) -> Optional[EditScript]:
        """Predict the script relating one pair, or ``None`` if the output
        could not be read as a script at all.

        Returning ``None`` rather than raising keeps the invalid-output rate a
        measurable quantity instead of an exception to catch.
        """

    def predict_batch(
        self, pairs: Sequence[Tuple[Sequence[str], Sequence[str]]]
    ) -> List[Optional[EditScript]]:
        """Predict for many pairs, ``None`` where the output was unusable."""
        return [self.predict(source, target) for source, target in pairs]

    # ---------- Persistence ----------

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write weights and whatever is needed to decode scripts again."""

    # ---------- Reporting ----------

    def training_parameters(self) -> Dict[str, Any]:
        """Hyperparameters, for the experiment record."""
        return {"formulation": self.name, **self.config.as_dict()}
