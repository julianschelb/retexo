# retexo/aligners/__init__.py
"""Alignment models and the corpus-level and pair-level policies that turn their scores into
one source per reuse word: the classical pointer-score assignment policies, the bidirectional
agreement rules, the orthographic sameness predicates, the decoder that derives a full script
from a typed alignment, the Sinkhorn and whole-script DP variants, and the contextual embedding
aligner."""

from __future__ import annotations

from retexo.aligners.agreement import AgreementDecoder, PairSwap
from retexo.aligners.aligner import ContextualAligner
from retexo.aligners.assignment import AssignmentPolicy, Reranker
from retexo.aligners.decode import ScriptDecoder
from retexo.aligners.global_decode import GlobalDecoder
from retexo.aligners.sameness import SamenessPolicy
from retexo.aligners.sinkhorn import SinkhornBalancer

__all__ = [
    "AgreementDecoder",
    "AssignmentPolicy",
    "ContextualAligner",
    "GlobalDecoder",
    "PairSwap",
    "Reranker",
    "SamenessPolicy",
    "ScriptDecoder",
    "SinkhornBalancer",
]
