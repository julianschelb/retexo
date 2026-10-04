"""The fine-tuned similarity aligner: awesome-align's supervised loss falls on the links it trains on."""

import pytest

from retexo.baselines.base import BaselineConfig
from retexo.baselines.record import Edge, Record
from retexo.baselines.sim_aligner import FineTunedSimAligner

TINY = "hf-internal-testing/tiny-random-bert"


def _record(i: int) -> Record:
    return Record(id=f"p{i}", level="gold", fold=1, source_work="", source_tokens=["arma", "virum", "cano", "troiae"],
                  reuse_work="", reuse_tokens=["cano", "arma", "que", "virum"], pair_label="cit",
                  links=[Edge(0, 2, "COPY"), Edge(1, 0, "COPY"), Edge(3, 1, "COPY")])


def test_the_supervised_loss_falls_and_a_negative_carries_none():
    pytest.importorskip("transformers")
    cfg = BaselineConfig(device="cpu", learning_rate=1e-3, extra={"model": TINY, "layer": 2, "sim": "cos", "norm": 0,
                                                                  "epochs": 1, "pairs_per_step": 4})
    method = FineTunedSimAligner(cfg)
    try:
        method.embedder._ensure_backend()
    except OSError:
        pytest.skip("tiny test model not cached")
    records = [_record(i) for i in range(8)]
    before = float(method.pair_loss(records[0]))
    for _ in range(5):
        method.fit(records, [], log=None)
    after = float(method.pair_loss(records[0]))
    assert after < before
    negative = Record(id="n", level="negative", fold=1, source_work="", source_tokens=["a", "b"], reuse_work="",
                      reuse_tokens=["c", "d"], pair_label="no_match")
    assert method.pair_loss(negative) is None
    pred = method.predict(records[:1])[0]
    assert pred.scores and len(pred.scores) == 4
