# retexo/pretraining/__init__.py
"""Stage 0: continued pretraining of the encoder before the synthetic-then-gold schedule.

The design lives in the vault note ``Experiments/Pretraining Experiments -
Definition``: one encoder, three objectives (masked LM over the joint pair
with the mask biased toward reused words, contrastive lemma pairs in context,
pair identification), one frozen probe that decides whether to spend more,
and one checkpoint that replaces the default backbone through
``--base-model``. ``pool`` builds the real-pairs level; ``stage0`` holds the
objectives, the trainer and the probe (objective 1 implemented 2026-09-16;
objectives 2 and 3 keep their interfaces).
"""
from __future__ import annotations

from retexo.pretraining.pool import PairPool, PoolBuilder, PoolPair
from retexo.pretraining.stage0 import (
    ContrastiveLemmaPairs,
    Counterparts,
    GeometryProbe,
    MaskedPairLM,
    PairIdentification,
    ProbeResult,
    Stage0Config,
    Stage0Trainer,
)

__all__ = [
    "ContrastiveLemmaPairs", "Counterparts", "GeometryProbe", "MaskedPairLM", "PairIdentification", "PairPool",
    "PoolBuilder", "PoolPair", "ProbeResult", "Stage0Config", "Stage0Trainer",
]
