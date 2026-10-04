# retexo/baselines/base.py
"""The one contract every method row implements.

A ``Baseline`` is built from a ``BaselineConfig``, fitted on the training and
dev records, asked to ``predict`` a ``Prediction`` per record, and then handed
each prediction once to ``postprocess``, the single hook between the model
and the scorer. The default hook decodes score rows into links (the shared
decoder) and names the links (the shared typer); a method whose order differs
overrides the hook and nothing else. The driver never branches on what a
method emits.

Method-specific dials travel in ``cfg.extra``, one dictionary filled from the
command line and from ``tune``; there is no config subclass per method. The
package already had one abstract contract (``formulations/base.py``,
``ScriptModel``) that its two working models bypassed; this one is the only
entry point of the driver, wraps the working models instead of asking them to
inherit, and reads the record every level of data is stored in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines.record import Edge, Record

#: Interface (I): per reuse word, ``(source_index, p)`` best first, ``-1`` = null.
Rows = List[List[Tuple[int, float]]]

# =============================================================================
# Config and prediction
# =============================================================================


@dataclass
class BaselineConfig:
    """The one configuration, shared by every method.

    Attributes:
        fold: The test fold (``-1`` on an external set).
        dev_fold: The dev fold, ``(fold + 1) % 5`` on the gold.
        device: ``cuda`` or ``cpu``.
        seed: One seed per fold (protocol 5.2).
        base_model: The encoder every trainable row starts from.
        max_length, batch_size, epochs, learning_rate: Training defaults.
        train_on: ``all`` or ``sure`` (adjudicated links only).
        smoke: ``> 0`` truncates every split and the epochs for a smoke run.
        workers: CPU workers for data-side loops.
        out: The run directory.
        extra: Method-specific dials, read with a default at the point of use.

    Example:
        ```python
        cfg = BaselineConfig(fold=4, dev_fold=0, device="cpu", out=Path("runs/demo"),
                             extra={"model": "hmm", "p0": 0.2})
        ```
    """

    fold: int = -1
    dev_fold: int = -1
    device: str = "cuda"
    seed: int = 1
    base_model: str = "ashleygong03/bamman-burns-latin-bert"
    max_length: int = 256
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    train_on: str = "all"
    smoke: int = 0
    workers: int = 20
    out: Path = Path("runs/baseline")
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["out"] = str(self.out)
        return out


@dataclass
class Prediction:
    """What a method says about one pair.

    Attributes:
        links: Per reuse word the source index or ``-1``.
        tags: Per reuse word the edge operation, ``""`` until typed.
        frame: Per reuse word 0/1.
        scores, rev_scores: Interface (I) rows, reuse to source and source to reuse.
        link_p: Per reuse word the probability of its chosen link.
        frame_p: Per reuse word the frame probability.
        dels: Per source word 1 where the model says the word was dropped.
        extra: Edges the per-token lists cannot hold (a MERGE's second source).
        meta: Anything the method wants in the dump (pass counts, chosen spans).
        raw: A language model's reply when it could not be parsed; with empty
            ``links`` the prediction is invalid and is scored as such.

    Example:
        ```python
        pred = Prediction.empty(3)
        pred.links[0] = 0; pred.tags[0] = "COPY"
        ```
    """

    links: List[int]
    tags: List[str]
    frame: List[int]
    scores: Optional[Rows] = None
    rev_scores: Optional[Rows] = None
    link_p: Optional[List[float]] = None
    frame_p: Optional[List[float]] = None
    dels: Optional[List[int]] = None
    extra: List[Edge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[str] = None

    @staticmethod
    def empty(n_reuse: int) -> "Prediction":
        """The prediction that says nothing: no links, no tags, no frame."""
        return Prediction(links=[-1] * n_reuse, tags=[""] * n_reuse, frame=[0] * n_reuse)

    @property
    def invalid(self) -> bool:
        return self.raw is not None and not self.links


# =============================================================================
# Baseline
# =============================================================================


class Baseline:
    """One method row: ``fit``, ``tune``, ``predict``, ``postprocess``, ``save``, ``load``.

    Class attributes name the row: ``name`` is the registry key and the
    ``--method`` argument; ``emits`` (``scores`` | ``edges`` | ``types``) is
    documentation the driver copies into the run record so the scorer can
    blank the alignment columns of a types-only row; ``trainable`` says whether
    ``fit`` does anything; ``typer`` and ``decoder`` are the parameters of the
    default ``postprocess``.

    Available implementations: ``DoNothing`` and ``CopyInput`` (the two floors),
    ``GoldLinks`` (the typing ceiling), and one class per method note.
    Subclasses must implement ``predict``; ``fit`` and ``tune`` default to
    no-ops.

    Example:
        ```python
        from retexo.baselines import BaselineRegistry
        from retexo.baselines.base import BaselineConfig
        BaselineRegistry.load_all()
        cfg = BaselineConfig(device="cpu")
        method = BaselineRegistry.get("copy_input")(cfg)
        preds = method.predict(records)
        preds = [method.postprocess(r, p, {}) for r, p in zip(records, preds)]
        # python run_baseline.py --method copy_input --fold 4 --smoke 20 --device cpu
        ```
    """

    name: str = ""
    emits: str = "edges"
    trainable: bool = False
    typer: str = "rule"
    decoder: str = "default"
    #: Whether ``fit`` stops early on a validation sample (``retexo.baselines.early_stopping``);
    #: the driver then holds out 20 % of the training records as ``self.validation``.
    early_stopping_capable: bool = False
    #: The level and matching the validation score uses (typing-only rows: their own level, no source word).
    validation_level: str = "V1"
    validation_require_source: bool = True

    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.featurizer = None
        self.validation: List[Record] = []
        self.early_stopping = None
        self.monitor = None                    # the note's metrics over training time (early_stopping.TrainingMonitor)
        #: the shared training set's synthetic pairs and negatives (``--shared-data``), empty otherwise
        self.shared_synthetic: List[Record] = []
        self.shared_negatives: List[Record] = []

    # ---------- Training ----------

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()) -> "Baseline":
        """Train on the training records; a no-op for untrained rows.

        ``unlabeled`` is the text of the test records, labels never read: the
        unsupervised aligners add it to their EM corpus as the alignment
        literature aligns the whole bitext; every other row ignores it.
        """
        return self

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """Method-specific dials chosen on the dev fold; the driver adds theta."""
        return {}

    # ---------- Inference ----------

    def predict(self, records: List[Record]) -> List[Prediction]:
        raise NotImplementedError(f"{type(self).__name__}.predict")

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """Decode the rows, then name the links; the only hook the driver calls."""
        from retexo.baselines.adapters import PredictionAdapter

        pred = PredictionAdapter.decode_prediction(pred, self.decoder, dials, record)
        return PredictionAdapter.type_prediction(pred, record, self.typer, self.featurizer,
                               frame_rule=str(dials.get("frame_rule", "keyword")),
                               head=getattr(self, "typer_head", None))

    # ---------- Persistence ----------

    def save(self, path: Path) -> None:
        """Persist what ``load`` needs; untrained rows write nothing."""
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> "Baseline":
        return cls(cfg)
