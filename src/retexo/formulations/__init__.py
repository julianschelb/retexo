# retexo/formulations/__init__.py
"""Output formulations for edit-script prediction."""

from __future__ import annotations

from retexo.formulations.base import (
    FormulationConfig,
    ScriptExample,
    ScriptModel,
    TrainingLog,
)
from retexo.formulations.seq2seq_full import Seq2SeqFullConfig, Seq2SeqFullModel
from retexo.formulations.seq2seq_stepwise import (
    Seq2SeqStepwiseConfig,
    Seq2SeqStepwiseModel,
)
from retexo.formulations.token_classifier import (
    TokenClassifierConfig,
    TokenClassifierModel,
)

#: Every formulation, keyed by the variant name used in experiment records.
FORMULATIONS = {
    TokenClassifierModel.name: (TokenClassifierModel, TokenClassifierConfig),
    Seq2SeqFullModel.name: (Seq2SeqFullModel, Seq2SeqFullConfig),
    Seq2SeqStepwiseModel.name: (Seq2SeqStepwiseModel, Seq2SeqStepwiseConfig),
}

__all__ = [
    "FORMULATIONS",
    "FormulationConfig",
    "ScriptExample",
    "ScriptModel",
    "Seq2SeqFullConfig",
    "Seq2SeqFullModel",
    "Seq2SeqStepwiseConfig",
    "Seq2SeqStepwiseModel",
    "TokenClassifierConfig",
    "TokenClassifierModel",
    "TrainingLog",
]
