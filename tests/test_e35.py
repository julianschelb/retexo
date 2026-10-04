"""Label matching: the gloss tables, the label prefix in the Latin pair encoder
(positions and shifted word spans), and the matching logits' shape."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.datasets import synthetic  # noqa: E402
from retexo.edit_typing.glosses import GlossTable  # noqa: E402


def test_gloss_tables_cover_the_inventory_in_order():
    ops = tuple(synthetic.FINE_OPERATIONS)
    for table in (GlossTable.LATIN, GlossTable.ENGLISH):
        assert set(table) >= set(ops)
    words = GlossTable("latin").words(ops)
    assert len(words) == len(ops) and all(1 <= len(w) <= 6 for w in words)
    assert GlossTable("none").words(ops) == [[] for _ in ops]
    with pytest.raises(KeyError):
        GlossTable("latin").words(("NOT-AN-OP",))


def test_label_prefix_positions_and_shifted_spans():
    pytest.importorskip("locisimiles.tokenization.latin_bert")
    from retexo.formulations.pair_encoding import LatinBertPairEncoder
    try:
        enc = LatinBertPairEncoder("ashleygong03/bamman-burns-latin-bert")
    except Exception as exc:  # offline
        pytest.skip(f"encoder unavailable: {exc}")
    pairs = [(["arma", "virumque", "cano"], ["arma", "cano"])]
    batch0, spans0 = enc.encode(pairs, 64)
    enc.label_prefix = [["idem", "verbum"], [], ["aliud", "verbum"]]
    batch1, spans1 = enc.encode(pairs, 64)
    pos = enc.last_label_positions[0]
    assert len(pos) == 3 and pos[0] == 1                       # first [LBL] right after [CLS]
    ids = batch1["input_ids"][0].tolist()
    assert all(ids[p] == enc.LBL for p in pos)
    shift = spans1[0][0][0] - spans0[0][0][0]
    assert shift > 0 and all(b - a == d - c for (a, b), (c, d) in zip(spans0[0], spans1[0]))
    assert enc.last_source_spans[0][0][0] == 1 + shift                  # the source starts after the prefix
    assert batch1["input_ids"].shape[1] == batch0["input_ids"].shape[1] + shift
    enc.label_prefix = None
    batch2, _ = enc.encode(pairs, 64)
    assert batch2["input_ids"].shape == batch0["input_ids"].shape and enc.last_label_positions is None


def test_matching_logits_have_the_typer_shape():
    torch = pytest.importorskip("torch")
    c, width, H, K, d = 3, 5, 8, 4, 6
    pair = torch.randn(c, width, 4 * H)
    match_pair = torch.nn.Linear(4 * H, d); match_label = torch.nn.Linear(H, d)
    label_states = torch.randn(2, K, H)           # two examples in the batch
    rows = torch.tensor([0, 1, 1])
    q = match_pair(pair); lab = match_label(label_states)[rows]
    name = torch.einsum("cwd,ckd->cwk", q, lab) / d ** 0.5
    assert name.shape == (c, width, K)
    assert not torch.allclose(name[1], name[2])                 # rows 1 and 2 share labels, differ in pairs
