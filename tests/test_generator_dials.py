# tests/test_generator_dials.py
"""The dials of 2026-09-27: copy variants, related-substitution share, frameless fillers, the frame channel, and the
pair head's columns in the Downstream features."""

import random

import numpy as np

from retexo.datasets import synthetic as syn
from retexo.edit_typing.link_features import FEATURE_NAMES, N_FEATURES
from retexo.edit_typing.repair import Repairer


class _Featurizer:
    """No enclitics, no evidence: the generator's cardinality branch never fires."""

    def enclitic(self, token):
        return None

    def __call__(self, *args):
        return [0.0] * N_FEATURES


def _one(substitute, source="arma uirumque cano troiae qui primus ab oris", **kw):
    source = source.split()
    rng = random.Random(0)
    return syn._make_one_typed(source, [source], substitute, _Featurizer(), rng, enclitic_rate=0.0, **kw)


def test_frame_channel_counts_in_order_anchors():
    arr = np.zeros((5, 6, N_FEATURES), dtype=np.float16)
    for t, s in [(0, 0), (1, 1), (3, 3), (4, 4)]:
        arr[t, s, FEATURE_NAMES.index("same_form")] = 1
    ch = syn.frame_support_channel(arr)
    assert ch[2, 2] == 1.0          # four in-order anchors around the middle cell
    assert ch[0, 5] == 0.0          # nothing in order around a far cell
    assert 0 < ch[2, 5] < 1


def test_spelling_variant_keeps_the_key_and_needs_attestation():
    rng = random.Random(1)
    assert syn.spelling_variant("temptare", rng, attested=None) is None
    v = syn.spelling_variant("temptare", rng, attested={"tentare"})
    assert v == "tentare" and Repairer.spelling_key(v) == Repairer.spelling_key("temptare")
    assert syn.spelling_variant("Grai,", random.Random(2), attested={"graii"}) == "Graii,"


def test_copy_variants_write_spelling_variants_as_copy():
    # every word has an attested variant under the same spelling key
    attested = {"temptare", "tentare", "haud", "haut", "grai", "graii", "caelum", "celum"}
    ex, stats = _one(lambda *a, **k: None, source="temptare haud grai caelum temptare haud grai caelum",
                     copy_variants=1.0, attested=attested)
    assert ex is not None and stats["spelling"] > 0
    for t, s in enumerate(ex.alignments):
        if s >= 0:
            assert ex.operations[t] == "COPY"
            assert Repairer.spelling_key(ex.target_tokens[t]) == Repairer.spelling_key(ex.source_tokens[s])


def test_frameless_filler_becomes_an_insertion():
    # every word replaced by an unrelated filler: no kept word anywhere, so no filler keeps its link
    ex, stats = _one(lambda token, rng, context=None, force=False: ("zz" + token, "SUBST"), frameless_fill="nolink")
    assert ex is not None and stats["frameless"] > 0
    assert all(s < 0 for s in ex.alignments)
    ex2, _ = _one(lambda token, rng, context=None, force=False: ("zz" + token, "SUBST"))
    assert any(s >= 0 for s in ex2.alignments)


def test_pair_head_columns_reach_the_downstream_features():
    from retexo.baselines.base import Prediction
    from retexo.baselines.downstream import PAIR_HEAD_FEATURES, ScriptOnlyFeatures
    from retexo.baselines.record import Edge, Record

    record = Record(id="p1", source_tokens=["arma", "cano"], reuse_tokens=["arma", "cano"], pair_label="cit",
                    links=[Edge(0, 0, "COPY"), Edge(1, 1, "COPY")], level="gold", fold=4, source_work="", reuse_work="")
    pred = Prediction(links=[0, 1], tags=["COPY", "COPY"], frame=[0, 0])
    pred.link_p = [0.9, 0.9]
    pred.meta["pair_head"] = [0.1, 0.7, 0.2]
    frame = ScriptOnlyFeatures.from_pairs([(record, pred)])
    assert all(c in frame.columns for c in PAIR_HEAD_FEATURES)
    assert frame["head_cit"].iloc[0] == 0.7
    plain = ScriptOnlyFeatures.from_pairs([(record, Prediction(links=[0, 1], tags=["COPY", "COPY"], frame=[0, 0]))])
    assert not any(c in plain.columns for c in PAIR_HEAD_FEATURES)


def test_changed_form_edges_need_a_kept_frame():
    from retexo.baselines.record import Edge, Record
    from retexo.baselines.regimes import SelfTraining

    record = Record(id="r", source_tokens="arma uirum cano troiae".split(), reuse_tokens="arma uirum canit troiae".split(),
                    pair_label="cit", links=[], level="gold", fold=4, source_work="", reuse_work="")
    framed = [Edge(0, 0, "LINK"), Edge(1, 1, "LINK"), Edge(2, 2, "LINK"), Edge(3, 3, "LINK")]
    assert len(SelfTraining.framed(record, framed)) == 4                 # canit <- cano sits in the frame
    alone = [Edge(2, 2, "LINK")]
    assert SelfTraining.framed(record, alone) == []                      # fewer than three links: dropped
    far = [Edge(0, 0, "LINK"), Edge(1, 1, "LINK"), Edge(2, 0, "LINK")]
    assert Edge(2, 0, "LINK") not in SelfTraining.framed(record, far)    # out of order: no frame
