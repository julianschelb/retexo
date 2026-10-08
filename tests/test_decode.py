"""The structure layer is a set of definitions; each is checked on a small case."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.aligners.decode import ScriptDecoder  # noqa: E402
from retexo.core.scriba import Scriba  # noqa: E402
from retexo.edit_typing.link_features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    LinkFeaturizer,
    SymbolicTyper,
)


def test_reorder_is_the_lis_rule():
    # reuse order 0,1,2 -> sources 5,3,4: the first link crosses the other two
    assert ScriptDecoder.reordered_targets([5, 3, 4]) == {0}
    assert ScriptDecoder.reordered_targets([0, 1, 2]) == set()
    assert ScriptDecoder.reordered_targets([-1, 2, -1, 1]) == {
        1
    } or ScriptDecoder.reordered_targets([-1, 2, -1, 1]) == {3}
    assert ScriptDecoder.reordered_targets([]) == set()


def test_quote_needs_four_in_order_copies():
    tags = ["NOP"] * 6
    assert ScriptDecoder.quote_spans([0, 1, 2, 3, 4, 5], tags) == [(0, 5)]
    assert ScriptDecoder.quote_spans([0, 1, 2, 4, 5, 6], tags) == []  # gap breaks it
    assert ScriptDecoder.quote_spans([0, 1, 2, 3, -1, 5], tags) == [(0, 3)]
    tags[2] = "MORPH"
    assert ScriptDecoder.quote_spans([0, 1, 2, 3, 4, 5], tags) == []


def test_adapt_and_disperse():
    align = [0, 1, 2, -1, -1, -1, 7]
    tags = ["NOP", "SYN", "MORPH", "INS", "INS", "INS", "NOP"]
    assert ScriptDecoder.adapt_spans(align, tags) == [(0, 2)]
    reuse = ["a", "b", "c.", "d", "e", "f", "g"]
    assert ScriptDecoder.dispersed(reuse, align) is True
    assert ScriptDecoder.dispersed(["a", "b", "c", "g"], [0, 1, 2, 7]) is False


def test_decode_replays_and_prices():
    source = "uox faucibus haesit".split()
    reuse = "ut ait poeta haesit uox et faucibus".split()
    alignment = [-1, -1, -1, 2, 0, -1, 1]
    fine = ["INS", "INS", "INS", "NOP", "NOP", "INS", "NOP"]
    frame = [1, 1, 1, 0, 0, 0, 0]
    script = ScriptDecoder.decode_script(source, reuse, alignment, fine, frame)
    tags = [op.tag for op in script.operations]
    assert tags.count("FRAME") == 1 and tags.count("INS") == 1
    assert "REORDER" in tags  # haesit came out first
    assert script.cost() > 0
    assert Scriba().execute(script, source) == reuse
    view = ScriptDecoder.per_token_view(script)
    assert view["frame"] == frame
    assert view["link"] == alignment


def test_features_and_symbolic_typing_without_resources():
    f = LinkFeaturizer(None)
    typer = SymbolicTyper()
    vec = f("arma", "Arma", 0, 0, 3, 3)
    assert len(vec) == N_FEATURES == len(FEATURE_NAMES)
    assert typer(vec) == "NOP"
    assert typer(f("arma", "nauis", 0, 0, 3, 3)) == "SUBST"


def test_quote_stands_in_for_its_copies_and_replays():
    source = "arma uirumque cano Troiae qui primus ab oris".split()
    reuse = "ut ait arma uirumque cano Troiae qui primus".split()
    alignment = [-1, -1, 0, 1, 2, 3, 4, 5]
    fine = ["INS", "INS"] + ["NOP"] * 6
    script = ScriptDecoder.decode_script(source, reuse, alignment, fine, [1, 1, 0, 0, 0, 0, 0, 0])
    tags = [op.tag for op in script.operations]
    assert tags.count("QUOTE") == 1 and "NOP" not in tags  # one act, not six
    assert Scriba().verify(script, source, reuse)
    view = ScriptDecoder.per_token_view(script)
    assert view["quote"] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert view["link"] == alignment and view["tags"][2:] == ["NOP"] * 6


def test_frame_mask_cleaning():
    assert ScriptDecoder.clean_frame_mask([1, 1, 0, 1, 1, 0, 0, 1]) == [1, 1, 1, 1, 1, 0, 0, 0]
    assert ScriptDecoder.clean_frame_mask([0, 1, 0, 0, 0]) == [0, 0, 0, 0, 0]
    assert ScriptDecoder.clean_frame_mask([1, 1, 1]) == [1, 1, 1]


def test_evidence_veto_settles_definitions():
    import torch

    from retexo.formulations.change_detector import ChangeDetector

    index = {"NOP": 0, "MORPH": 1, "SYN": 2, "NE-SUB": 3, "SPLIT": 4, "MERGE": 5}
    logits = torch.zeros((3, 6))
    phi = torch.zeros((3, 23))
    phi[0, 0] = 1  # identical forms -> must be NOP
    phi[1, 12] = 1  # two names -> NE-SUB allowed
    phi[2, 14] = 1
    phi[2, 16] = 1  # enclitic on the source, stem matches -> SPLIT/MERGE allowed
    out = ChangeDetector.evidence_veto(logits, phi, index)
    assert out[0].argmax().item() == 0 and torch.isinf(out[0, 1:]).all()
    assert torch.isinf(out[1, 0]) and not torch.isinf(out[1, 3]) and torch.isinf(out[1, 4])
    assert torch.isinf(out[2, 3]) and not torch.isinf(out[2, 4]) and not torch.isinf(out[2, 5])
