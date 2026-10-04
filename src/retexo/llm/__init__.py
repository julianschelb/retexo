# retexo/llm/__init__.py
"""LLM prompting for the type and propose jobs, and scoring their replies against adjudicated
hand labels, WordNet, and mechanical filters."""
from __future__ import annotations

from retexo.llm.llm_benchmark import EvidenceAnnotator, ProposeJob, TagParser, TypeJob
from retexo.llm.llm_prompts import GLOSS, PROPOSE, TYPE, PromptRegistry, PromptVariant

__all__ = [
    "GLOSS",
    "PROPOSE",
    "TYPE",
    "EvidenceAnnotator",
    "ProposeJob",
    "PromptRegistry",
    "PromptVariant",
    "TagParser",
    "TypeJob",
]
