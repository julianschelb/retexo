# datasets/synthetic/generator.py
"""The reusable generator object: ``GeneratorConfig`` and ``TypedGenerator``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.datasets.synthetic.inventory import ENCLITIC_RATE, FRAME_RATE
from retexo.datasets.synthetic.one_pair import TypedReport
from retexo.datasets.synthetic.parallel import generate_typed
from retexo.datasets.synthetic.pools import frame_pool, select_pool
from retexo.formulations.change_detector import ChangeExample

# =============================================================================
# A reusable generator object
# =============================================================================


@dataclass
class GeneratorConfig:
    """Every knob ``generate_typed`` takes, gathered so a caller can build one
    configuration once and reuse it across folds and pools."""

    vectors_path: Optional[str] = None
    attested: Optional[set] = None
    accept: Optional[float] = None
    frames: Sequence[Sequence[str]] = ()
    frame_rate: float = FRAME_RATE
    enclitic_rate: float = ENCLITIC_RATE
    reorder_inter: float = 0.0
    reorder_intra: float = 0.0
    mlm_subst: bool = False
    subst_weight: Optional[float] = None
    mlm_model: Optional[str] = None
    cohypo: bool = False
    dense_rate: float = 0.0
    workers: int = 24
    per_seed: int = 4
    seed: int = 1
    chunk_size: int = 200


class TypedGenerator:
    """The typed synthetic generator, as an object instead of free keywords.

    A thin wrapper over ``generate_typed``/``select_pool``/``frame_pool``: the
    functions do the multiprocess work (and stay directly importable, and
    directly unit-testable, for that reason); this class exists so a caller
    building several pools from one configuration -- one per fold, say --
    states the configuration once.

    Example:
        >>> generator = TypedGenerator(GeneratorConfig(
        ...     frames=frame_pool(gold_dir, folds_by_id, exclude_fold=4)))
        >>> pool, report = generator.generate(seeds, context_pool)
    """

    def __init__(self, config: Optional[GeneratorConfig] = None, **overrides):
        self.config = replace(config or GeneratorConfig(), **overrides)

    def generate(self, seeds, context_pool, *, log=None) -> Tuple[List[ChangeExample], TypedReport]:
        return generate_typed(seeds, context_pool, log=log, **asdict(self.config))

    @staticmethod
    def select_pool(pool, size: int, rng, rare_min: int = 1200):
        return select_pool(pool, size, rng, rare_min=rare_min)

    @staticmethod
    def frame_pool(gold_dir, folds_by_id: Dict[str, int], exclude_fold: int):
        return frame_pool(gold_dir, folds_by_id, exclude_fold)
