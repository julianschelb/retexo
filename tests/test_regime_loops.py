"""Real Unannotated Pairs: the real pairs as records, and a teacher's dump as distillation records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.adapters import write_dump  # noqa: E402
from retexo.baselines.base import Prediction  # noqa: E402
from retexo.baselines.record import Record, RecordCodec  # noqa: E402
from retexo.baselines.regimes import Distillation  # noqa: E402
from retexo.pretraining.pool import PairPool, PoolPair  # noqa: E402




def test_a_teacher_dump_becomes_distillation_records(tmp_path):
    record = Record(id="real/m_0", level="real_pairs", fold=-1, source_work="", source_tokens=["arma", "uirumque", "cano"],
                    reuse_work="", reuse_tokens=["ut", "ait", "arma", "canit"], pair_label="cf")
    pred = Prediction(links=[-1, -1, 0, 2], tags=["", "", "COPY", "MORPH"], frame=[1, 1, 0, 0])
    pred.scores = [[(-1, 1.0)], [(-1, 1.0)], [(0, 0.9)], [(2, 0.8)]]
    write_dump([record], [pred], tmp_path / "real.jsonl")
    [out] = Distillation.from_dump(tmp_path / "real.jsonl")
    assert out.id == "distill/real/m_0" and [(e.r, e.s, e.op) for e in out.links] == [(2, 0, "COPY"), (3, 2, "MORPH")]
    assert len(out.spans) == 1 and out.provenance["links"] == "merged_view"
    RecordCodec.save([out], tmp_path / "labels.jsonl")
    [back] = RecordCodec.load(tmp_path / "labels.jsonl")
    assert [(e.r, e.s, e.op) for e in back.links] == [(2, 0, "COPY"), (3, 2, "MORPH")]


def test_the_pairs_only_aligners_read_a_pool_as_extra_bitext(tmp_path):
    from retexo.baselines.em_aligner import extra_bitext
    from retexo.pretraining.pool import PoolBuilder

    pool = [PoolPair(f"m_{i}", ("arma", "uirum"), ("arma", "cano"), "ranked", 0.9) for i in range(4)]
    path = PoolBuilder.save(pool, tmp_path / "pairs.jsonl")
    records = extra_bitext(str(path))
    assert len(records) == 4 and all(r.level == "real_pairs" and not r.links for r in records)
    assert extra_bitext(None) == [] and extra_bitext("") == []
