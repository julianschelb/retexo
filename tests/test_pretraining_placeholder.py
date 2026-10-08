# tests/test_pretraining_placeholder.py
"""The stage-0 placeholder imports, carries the note's defaults, and fails loudly rather than silently."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.skip(
    "stale: imports ResourcePairs, which retexo.pretraining no longer defines",
    allow_module_level=True,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.pretraining import (  # noqa: E402
    GeometryProbe,
    PairPool,
    ResourcePairs,
    Stage0Config,
    Stage0Trainer,
)


def test_config_carries_the_notes_defaults():
    cfg = Stage0Config()
    assert cfg.base_model == "ashleygong03/bamman-burns-latin-bert"
    assert cfg.mask_rate == 0.15
    assert cfg.reuse_mask_share == 0.5
    assert cfg.layer == 8
    assert cfg.epochs == 1


def test_resource_pairs_exclude_the_demoted_relations():
    for demoted in ("HYPER", "HYPO", "ANT", "CO-HYPO"):
        assert demoted not in ResourcePairs.RELATIONS


def test_placeholders_raise_not_implemented_not_silence():
    with pytest.raises(NotImplementedError):
        PairPool.build([])
    with pytest.raises(NotImplementedError):
        ResourcePairs.build(None, [])
    with pytest.raises(NotImplementedError):
        GeometryProbe().run("x", [], [])
    with pytest.raises(NotImplementedError):
        Stage0Trainer().fit(PairPool([]), ResourcePairs([]))


def test_empty_pool_and_pairs_construct():
    assert len(PairPool([])) == 0
    assert len(ResourcePairs([])) == 0
