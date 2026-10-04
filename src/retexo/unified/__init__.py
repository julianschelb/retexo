# retexo/unified/__init__.py
"""The unified model: the typed pointer with a mode gate on the null and a
structured decoder behind the grid; a stage-0 encoder enters through
``config.base_model``.

Design and the feasibility arms: the vault note ``Brainstorming/Unified
Architecture - Brainstorm`` (sections 4.1 to 4.3 for the data flow, the gate
and the decoder; section 5 for the arms). The parts that already work --
joint encoding, the factorised cell grid, both orientations, the pair head --
are inherited from ``formulations.typed_pointer`` unchanged.
"""
from __future__ import annotations

from retexo.unified.decoder import Augment, CellGrid, DecoderConfig, StructuredDecoder
from retexo.unified.gate import gate_log_probs, gate_loss, gate_targets
from retexo.unified.model import UnifiedPointer
from retexo.unified.runs import FRAME, INS, QUOTE, Run, Segmentation

__all__ = [
    "Augment",
    "CellGrid",
    "DecoderConfig",
    "FRAME",
    "INS",
    "QUOTE",
    "Run",
    "Segmentation",
    "StructuredDecoder",
    "UnifiedPointer",
    "gate_log_probs",
    "gate_loss",
    "gate_targets",
]
