# retexo/core/__init__.py
"""The core object model: the replayable script, the oracle that derives one, the builder
that constructs one, the checker that replays one, and the orthographic normalizer every
comparison in the package goes through."""

from __future__ import annotations

from retexo.core.builder import VariantBuilder
from retexo.core.normalize import normalize
from retexo.core.oracle import (
    Aligner,
    EditPlanOracle,
    GreedyAligner,
    OptimalAligner,
    OracleConfig,
)
from retexo.core.scriba import Scriba, Validity
from retexo.core.script import CostModel, EditScript

__all__ = [
    "Aligner",
    "CostModel",
    "EditPlanOracle",
    "EditScript",
    "GreedyAligner",
    "OptimalAligner",
    "OracleConfig",
    "Scriba",
    "Validity",
    "VariantBuilder",
    "normalize",
]
