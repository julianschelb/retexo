# retexo/datasets/__init__.py
"""Building and loading the data the pipeline trains and evaluates on: the hand-labelled gold,
the typed synthetic generator, the negative-example builder, the older shape-realistic generator,
the benchmark data loader, the real-pair relabelling teacher, and the window localizer, span
parser and change-detection window measure that sit underneath them."""

from __future__ import annotations

from retexo.datasets.dataset import BenchmarkData, LabelledPair
from retexo.datasets.detection import DetectionScore
from retexo.datasets.generation import (
    Difficulty,
    GenerationConfig,
    GenerationReport,
    LexicalSubstitutionSource,
    MockSubstitutionSource,
    SubstitutionSource,
    SyntheticGenerator,
)
from retexo.datasets.gold import GoldPair, TypedScorer
from retexo.datasets.localize import Localized
from retexo.datasets.negatives import NegativeBuilder
from retexo.datasets.parsing import ScriptParser
from retexo.datasets.synthetic import GeneratorConfig, TypedGenerator, generate_typed
from retexo.datasets.teacher import RelabellingTeacher

__all__ = [
    "BenchmarkData",
    "DetectionScore",
    "Difficulty",
    "GenerationConfig",
    "GenerationReport",
    "GeneratorConfig",
    "GoldPair",
    "LabelledPair",
    "LexicalSubstitutionSource",
    "Localized",
    "MockSubstitutionSource",
    "NegativeBuilder",
    "RelabellingTeacher",
    "ScriptParser",
    "SubstitutionSource",
    "SyntheticGenerator",
    "TypedGenerator",
    "TypedScorer",
    "generate_typed",
]
