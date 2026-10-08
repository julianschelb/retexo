# tests/test_word_channels.py
"""Chain 2 row 6: the word-level channels, on a stub morphology and the tiny random backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Edge, Record  # noqa: E402
from retexo.formulations.word_channels import SIZES, WordChannels  # noqa: E402

TINY = "hf-internal-testing/tiny-random-bert"


class StubMorphology:
    LEMMAS = {"armis": "arma", "arma": "arma", "uirum": "uir", "cano": "cano"}
    LABELS = {"armis": "--p---nb-", "arma": "--p---na-", "uirum": "--s---ma-", "cano": "v1spia---"}

    def lemma(self, word):
        return self.LEMMAS.get(word.lower().strip(",."), "")

    def morpho_label(self, word):
        return self.LABELS.get(word.lower().strip(",."))


def channels(kinds=("lemma", "pos", "morph"), hidden=8):
    ch = WordChannels(kinds, hidden)
    ch._morphology = StubMorphology()
    return ch


def test_word_ids_share_the_lemma_and_differ_on_the_form():
    ch = channels()
    armis, arma, uirum, cano, punct = (
        ch.word_ids(w) for w in ("armis", "arma,", "uirum", "cano", "--")
    )
    assert armis[0] == arma[0] != uirum[0]  # lemma channel: arma = arma, not uir
    assert armis[2] != arma[2]  # morphology channel: ablative vs accusative
    assert armis[1] == arma[1] == uirum[1] != cano[1]  # POS: nominal ('-') vs verb ('v')
    assert punct == (0, 0, 0)  # no letters: the padding row everywhere
    assert 1 <= armis[0] <= SIZES["lemma"] and 1 <= cano[2] <= SIZES["morph"]


def test_ids_fill_every_piece_of_a_word_and_nothing_else():
    ch = channels(("lemma",))
    pairs = [(["armis", "cano"], ["arma"])]
    reuse_spans = [[(5, 7)]]
    source_spans = [[(1, 3), (3, 4)]]
    ids = ch.ids((1, 9), reuse_spans, source_spans, pairs)
    assert ids.shape == (1, 9, 1)
    lemma_arma = ch.word_ids("arma")[0]
    assert (
        ids[0, 1, 0].item()
        == ids[0, 2, 0].item()
        == lemma_arma
        == ids[0, 5, 0].item()
        == ids[0, 6, 0].item()
    )
    assert ids[0, 3, 0].item() == ch.word_ids("cano")[0]
    assert (
        ids[0, 0, 0].item()
        == ids[0, 4, 0].item()
        == ids[0, 7, 0].item()
        == ids[0, 8, 0].item()
        == 0
    )
    assert ch(ids).shape == (1, 9, 8)
    assert torch.all(ch(ids)[0, 0] == 0)  # the padding row is zero


def test_pointer_trains_and_predicts_with_channels_on_the_tiny_backbone(monkeypatch):
    from retexo.baselines.typed_pointer import TypedPointerBaseline
    from retexo.formulations import word_channels

    monkeypatch.setattr(
        word_channels.WordChannels, "morphology", property(lambda self: StubMorphology())
    )
    cfg = BaselineConfig(
        fold=4,
        dev_fold=0,
        device="cpu",
        base_model=TINY,
        batch_size=4,
        seed=1,
        extra={"size": 0, "gold_passes": 1, "negatives": "none", "channels": "lemma+pos"},
    )
    recs = [
        Record(
            id=f"p{i}",
            level="gold",
            fold=1,
            source_work="",
            source_tokens=["arma", "uirum", "cano"],
            reuse_work="",
            reuse_tokens=["armis", "cano", "uirum"],
            pair_label="cit",
            links=[Edge(0, 0, "MORPH"), Edge(1, 2, "COPY"), Edge(2, 1, "COPY")],
        )
        for i in range(3)
    ]
    method = TypedPointerBaseline(cfg)
    assert method.recipe.channels == "lemma,pos"
    method.fit(recs, [], log=None)
    assert method.model._channels is not None and tuple(method.model._channels.kinds) == (
        "lemma",
        "pos",
    )
    preds = method.predict(recs)
    assert len(preds) == 3 and len(preds[0].scores) == 3
    done = method.postprocess(recs[0], preds[0], {"theta": 0.45})
    assert len(done.links) == 3


def test_parsed_channels_come_from_the_passage_parse(monkeypatch):
    from retexo.datasets import synthetic

    class StubParser:
        def parse_all(self, token_lists):
            # arma(obj of cano) uirum(obj) cano(root)
            full = [("NOUN", "obj", 2), ("NOUN", "obj", 2), ("VERB", "root", -1)]
            return {
                " ".join(t): (full if len(t) == 3 else [("NOUN", "root", -1)]) for t in token_lists
            }

    monkeypatch.setattr(synthetic, "_dep_parser", lambda: StubParser())
    ch = channels(("lemma", "dep", "head", "upos"))
    ids = ch.passage_ids(["arma", "uirum", "cano"])
    assert ids[0] == ids[1] and ids[0][0] != ids[2][0]  # obj = obj != root
    assert (
        ids[0][1] == ch._bucket("cano", SIZES["head"]) and ids[2][1] == 0
    )  # head lemma; the root has none
    assert ids[0][2] == _upos_index("NOUN") and ids[2][2] == _upos_index("VERB")
    grid = ch.ids(
        (1, 8), [[(5, 6)]], [[(1, 2), (2, 3), (3, 4)]], [(["arma", "uirum", "cano"], ["arma"])]
    )
    assert grid[0, 1].tolist() == [ch.word_ids("arma")[0], *ids[0]] and grid[0, 3].tolist() == [
        ch.word_ids("cano")[0],
        *ids[2],
    ]
    assert grid[0, 5].tolist() == [
        ch.word_ids("arma")[0],
        ch._bucket("root", SIZES["dep"]),
        0,
        _upos_index("NOUN"),
    ]
    assert grid[0, 0].tolist() == [0, 0, 0, 0]


def _upos_index(tag):
    from retexo.formulations.word_channels import _UPOS

    return _UPOS.index(tag) + 1


def test_cross_channels_say_whether_the_lemma_or_form_is_on_the_other_side():
    ch = channels(("shared", "sharedform", "pos"))
    ids = ch.cross_ids(["armis", "cano", "--"], ["arma", "uirum"])
    assert ids == [(2, 1), (1, 1), (0, 0)]  # armis: lemma arma shared, form not; cano: neither
    grid = ch.ids((1, 6), [[(4, 5)]], [[(1, 2), (2, 3)]], [(["armis", "cano"], ["arma"])])
    assert ch.kinds == ("pos", "shared", "sharedform")  # stored in the canonical order
    assert grid[0, 1].tolist() == [ch.word_ids("armis")[0], 2, 1] and grid[0, 2].tolist() == [
        ch.word_ids("cano")[0],
        1,
        1,
    ]
    assert grid[0, 4].tolist() == [
        ch.word_ids("arma")[0],
        2,
        1,
    ]  # arma on the reuse side: lemma shared with armis
    assert grid[0, 0].tolist() == [0, 0, 0]
