# retexo/edit_typing/__init__.py
"""Naming which operation changed a token: the symbolic evidence features, the resource-first
attestation logic that separates what the static resources already know from the model's
residual territory, the dependency-parse features, the operation-label glosses, the post-decoding
repair rules, and the edit-script-as-classifier features for the downstream reuse/non-reuse
decision."""
from __future__ import annotations

from retexo.edit_typing.attest import Attestation, Attester, FrameLexicon
from retexo.edit_typing.dep_features import DependencyParser
from retexo.edit_typing.downstream import DownstreamScorer, EvaluationPool, ScriptFeaturizer
from retexo.edit_typing.glosses import GlossTable
from retexo.edit_typing.link_features import LinkFeaturizer, SymbolicTyper
from retexo.edit_typing.repair import Repairer

__all__ = [
    "Attestation",
    "Attester",
    "DependencyParser",
    "DownstreamScorer",
    "EvaluationPool",
    "FrameLexicon",
    "GlossTable",
    "LinkFeaturizer",
    "Repairer",
    "ScriptFeaturizer",
    "SymbolicTyper",
]
