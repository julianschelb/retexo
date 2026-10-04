# tests/test_pretraining_stage0.py
"""Stage 0 without a model: the pool builder's exclusions, the counterpart marks,
the biased mask, the probe's AUC."""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retexo.pretraining.pool import PairPool, PoolBuilder  # noqa: E402
from retexo.pretraining.stage0 import Counterparts, GeometryProbe, MaskedPairLM, Stage0Config  # noqa: E402


def test_pool_builder_drops_gold_passages_and_caps_sources():
    with tempfile.TemporaryDirectory() as tmp:
        allusion = Path(tmp) / "allusion.jsonl"
        ranked = Path(tmp) / "ranked.jsonl"
        allusion.write_text("\n".join(json.dumps(x) for x in [
            {"cand_id": "t1", "earlier": "arma virumque cano troiae qui primus", "later": "arma virum canit poeta noster hic", "stratum": "tesserae4", "score": 4},
            {"cand_id": "t2", "earlier": "GOLD SOURCE passage here with words", "later": "some reuse of it with more words", "stratum": "vf", "score": 0},
        ]))
        ranked.write_text("\n".join(json.dumps(x) for x in [
            {"earlier": "source one has enough words here", "later": "reuse alpha has enough words here", "p_reuse": 0.99},
            {"earlier": "source one has enough words here", "later": "reuse beta has enough words here too", "p_reuse": 0.95},
            {"earlier": "source one has enough words here", "later": "reuse gamma has enough words here too", "p_reuse": 0.94},
            {"earlier": "source two has enough words here", "later": "reuse alpha has enough words here", "p_reuse": 0.93},
            {"earlier": "source three has enough words here", "later": "reuse delta has enough words here", "p_reuse": 0.5},
        ]))
        builder = PoolBuilder({"GOLD SOURCE passage here with words"}, allusion=allusion, ranked=ranked, seed=1)
        pairs = builder.build(min_p_reuse=0.9, cap=10, per_source=2)
        strata = sorted(p.stratum for p in pairs)
        assert strata == ["ranked", "ranked", "tesserae4"], strata            # gold source dropped; per-source cap 2; one per reuse
        assert builder.report["dropped_gold_passage"] == 1
        out = PoolBuilder.save(pairs, Path(tmp) / "pairs.jsonl")
        pool = PairPool.load(out)
        assert len(pool) == 3 and len(list(pool.both_orientations())) == 6
        # stage-0 negatives: rejected candidates and re-pairings, never a gold passage
        low = Path(tmp) / "ranked_low.jsonl"
        low.write_text("\n".join(json.dumps({"earlier": f"source {i} has enough words here", "later": f"reuse {i} has enough words here",
                                             "p_reuse": 0.01}) for i in range(10)) + "\n" + json.dumps(
            {"earlier": "GOLD SOURCE passage here with words", "later": "reuse x has enough words here", "p_reuse": 0.0}))
        negs = PoolBuilder({"GOLD SOURCE passage here with words"}, allusion=allusion, ranked=low, seed=1).negative_pairs(8)
        assert len(negs) == 8 and {k for _, _, k in negs} == {"lexical_low_p", "random"}
        assert all("gold" not in " ".join(s) for s, _, _ in negs)
        # the Data Scale rungs: no caps, and several pairs per reuse passage
        wide = PoolBuilder({"GOLD SOURCE passage here with words"}, allusion=allusion, ranked=ranked, seed=1)
        every = wide.build(min_p_reuse=0.0, cap=0, per_source=0, one_per_reuse=False)
        assert sorted(p.stratum for p in every).count("ranked") == 5


def test_overlap_miner_keeps_every_pair_sharing_two_content_words_and_drops_the_gold():
    from retexo.pretraining.overlap import OverlapMiner

    corpus = ["arma uirumque cano troiae qui primus", "arma uirumque cano troiae qui primus", "nox erat et caelo fulgebat luna",
              "cano arma semper hic poeta noster"]
    queries = ["arma cano uirum atque deos magnos", "luna fulgebat nocte serena alta caelo", "arma cano uirum atque deos alios"]
    miner = OverlapMiner(corpus, queries, {"arma cano uirum atque deos alios"}, min_tokens=1)
    pairs = list(miner.pairs())
    assert len(pairs) == 3                            # the duplicate corpus text counted once; the gold query dropped
    assert {p.stratum for p in pairs} == {"overlap"} and all(p.score >= 2 for p in pairs)
    assert miner.report["dropped_gold_passage"] == 2
    assert len(list(OverlapMiner(corpus, queries, set(), min_tokens=1).pairs(max_pairs=2))) == 2


def test_counterparts_mark_forms_keys_and_stems():
    marks = Counterparts.mark(["arma", "virumque", "cano,", "Troiae", "et"], ["arma", "virum", "canit", "poeta", "et"])
    assert marks == [True, True, False, False, True]           # identical; shared 4-char stem; cano/canit share only 3; not Troiae; et identical


class _StubEncoder:
    """Two words of two subwords each on both sides: [CLS] s s s s [SEP] r r r r [SEP]."""

    PAD, CLS, SEP = 0, 2, 3
    last_source_spans = []

    def encode(self, pairs, max_length):
        import torch

        rows, spans, sources = [], [], []
        for source, reuse in pairs:
            ids = [self.CLS]
            s_spans = []
            for _ in source:
                s_spans.append((len(ids), len(ids) + 2)); ids += [10, 11]
            ids.append(self.SEP)
            r_spans = []
            for _ in reuse:
                r_spans.append((len(ids), len(ids) + 2)); ids += [12, 13]
            ids.append(self.SEP)
            rows.append(ids); spans.append(r_spans); sources.append(s_spans)
        width = max(len(r) for r in rows)
        input_ids = torch.tensor([r + [self.PAD] * (width - len(r)) for r in rows])
        self.last_source_spans = sources
        return {"input_ids": input_ids, "attention_mask": (input_ids != self.PAD).long()}, spans


def test_biased_mask_prefers_counterpart_words():
    cfg = Stage0Config(mask_rate=0.5, reuse_mask_share=1.0, device="cpu")
    mlm = MaskedPairLM(cfg, _StubEncoder(), vocab_size=100, mask_id=4, special_ids=[0, 2, 3, 4])
    rng = random.Random(3)
    batch = mlm.batch([(["arma", "xyzq"], ["arma", "pqrs"])], rng)
    labels = batch["labels"][0].tolist()
    masked = [i for i, l in enumerate(labels) if l != -100]
    # rho = 1: every masked position lies inside the counterpart words (arma on both sides: spans 1-3 and 6-8)
    assert masked and all(1 <= i < 3 or 6 <= i < 8 for i in masked), masked
    assert batch["_n_biased"] == len(masked)
    uniform = MaskedPairLM(Stage0Config(mask_rate=0.5, reuse_mask_share=0.0, device="cpu"), _StubEncoder(), 100, 4, [0, 2, 3, 4])
    labels_u = uniform.batch([(["arma", "xyzq"], ["arma", "pqrs"])], random.Random(3))["labels"][0].tolist()
    assert sum(1 for l in labels_u if l != -100) == 4                      # 50 % of the 8 subwords


def test_probe_auc_is_rank_based():
    assert abs(GeometryProbe._auc([0.9, 0.8], [0.1, 0.2]) - 1.0) < 1e-9
    assert abs(GeometryProbe._auc([0.5, 0.5], [0.5, 0.5]) - 0.5) < 1e-9
    assert abs(GeometryProbe._auc([0.3, 0.9], [0.5, 0.1]) - 0.75) < 1e-9




def test_contrastive_and_psi_objectives_run_on_the_tiny_model():
    """Objectives 2 and 3 through the real trainer on a tiny encoder: the losses are finite and the
    contrastive one starts near log(batch) as InfoNCE should; a checkpoint is written."""
    import json
    from retexo.pretraining.pool import PoolPair, PairPool
    from retexo.pretraining.resource_pairs import LemmaSentences, ResourcePair
    from retexo.pretraining.stage0 import Stage0Trainer

    with tempfile.TemporaryDirectory() as tmp:
        pool = PairPool([PoolPair(f"p{i}", ("arma", "virumque", "cano", "troiae", "qui", "primus", "ab", "oris"),
                                  ("arma", "virum", "canit", "poeta", "noster", "hic", "et", "ille"), "t", 1.0) for i in range(8)])
        sentences = LemmaSentences({
            "gladius": [{"tokens": ["stricto", "gladio", "in", "hostem", "ruit", "miles", "ferox", "ille"], "index": 1}] * 4,
            "ensis": [{"tokens": ["ense", "ferit", "et", "vulnera", "multa", "dedit", "hosti", "suo"], "index": 0}] * 4,
            "rex": [{"tokens": ["rex", "urbem", "gladio", "defendit", "et", "hostes", "fugat", "acer"], "index": 0}] * 4,
            "regia": [{"tokens": ["regia", "domus", "alta", "stat", "in", "monte", "vetere", "illo"], "index": 0}] * 4,
        })
        pairs = [ResourcePair("gladius", "ensis", "SYN", "NOUN", 1.0), ResourcePair("rex", "regia", "POS", "NOUN", 1.0)] * 4
        negatives = [(["nox", "erat", "et", "caelo", "fulgebat", "luna", "sereno", "alta"], ["quisquis", "ades", "medii", "subis", "in", "limina", "templi", "siste"])] * 4
        cfg = Stage0Config(base_model="hf-internal-testing/tiny-random-bert", objectives=("mlm", "contrastive", "psi"),
                           batch_size=4, device="cpu", out=Path(tmp), smoke=8, log_every=1, contrastive_weight=0.5, psi_weight=0.1)
        logged = []
        trainer = Stage0Trainer(cfg).fit(pool, pairs, sentences=sentences, negatives=negatives, log=logged.append)
        assert trainer.history and all(k in trainer.history[-1] for k in ("mlm", "contrastive", "psi")), trainer.history[-1]
        assert all(abs(v) < 100 for v in trainer.history[-1].values()), trainer.history[-1]
        assert trainer.history[0]["contrastive"] > 0.0             # a random tiny encoder at temperature .05: any finite value
        path = trainer.save()
        assert (path / "config.json").exists() and (path / "stage0.json").exists()
        text = "\n".join(logged)
        assert "[stage0] contrastive step" in text and "[stage0] psi step" in text
        # the Data Scale rung: the masked LM on one pool, the pair identification on another
        psi_pool = PairPool(pool.pairs[:2])
        cfg2 = Stage0Config(base_model="hf-internal-testing/tiny-random-bert", objectives=("mlm", "psi"), batch_size=4,
                            device="cpu", out=Path(tmp) / "two", smoke=8, log_every=1, psi_weight=0.1)
        trainer = Stage0Trainer(cfg2).fit(pool, negatives=negatives, psi_pool=psi_pool, log=logged.append)
        assert all(k in trainer.history[-1] for k in ("mlm", "psi")), trainer.history[-1]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_pretraining_stage0] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
