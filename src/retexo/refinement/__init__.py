# retexo/refinement/__init__.py
"""Iterative refinement of a first-pass alignment: the per-token state fed back before the
heads (E36), and the second-pass 2-D convolutional grid refiner over the link/state grid plus
evidence features (E36b)."""
from __future__ import annotations

from retexo.refinement.grid_refiner import GridRefiner, PairGrid
from retexo.refinement.refine import State

__all__ = [
    "GridRefiner",
    "PairGrid",
    "State",
]
