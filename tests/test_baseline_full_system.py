# tests/test_baseline_full_system.py
"""Note 16 on hand-made rows: the ensemble arithmetic, the rater's bonus, the tag
rule, the gate, the four switches; then two tiny aligners end to end, no rater."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import labels  # noqa: E402
from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.decoder import BaselineDecoder  # noqa: E402
from retexo.baselines.full_system import SYSTEM_DEFAULTS, Ensemble, FullSystem, GatedTyper  # noqa: E402
from retexo.baselines.record import Edge, Record, Span  # noqa: E402
from retexo.edit_typing.repair import Repairer  # noqa: E402

TINY = "hf-internal-testing/tiny-random-bert"

FWD = [[(0, 0.9), (-1, 0.1)], [(1, 0.4), (-1, 0.6)], [(2, 0.7), (1, 0.2), (-1, 0.1)]]


def record(rid, source, reuse, edges=(), spans=()):
    return Record(id=rid, level="gold", fold=1, source_work="", source_tokens=source, reuse_work="",
                  reuse_tokens=reuse, pair_label="cit", links=list(edges), spans=list(spans))


def pairs():
    return [
        record("t/1", ["arma", "virumque", "cano"], ["arma", "virum", "canit", "poeta"],
               edges=[Edge(0, 0, "COPY"), Edge(1, 1, "MORPH"), Edge(2, 2, "MORPH")]),
        record("t/2", ["rex", "regem", "amat"], ["ut", "ait", "rex", "amat"],
               edges=[Edge(2, 0, "COPY"), Edge(3, 2, "COPY")], spans=[Span(0, 2, "FRAME")]),
        record("t/3", ["gladio", "ferit"], ["ense", "ferit", "hostem"],
               edges=[Edge(0, 0, "SUBST"), Edge(1, 1, "COPY")]),
    ]


# =============================================================================
# 1. The ensemble rows
# =============================================================================


def test_ensemble_rows_without_b_and_without_beta_are_a_unchanged():
    for row, orig in zip(Ensemble.rows(FWD, None), FWD):
        assert dict(row) == {s: round(p, 6) for s, p in orig}
    same = Ensemble.rows(FWD, FWD, weight_a=0.5)
    for row, orig in zip(same, FWD):
        assert dict(row) == {s: round(p, 6) for s, p in orig}
    assert dict(Ensemble.rows(FWD, [[(-1, 1.0)]] * 3, weight_a=1.0)[0]) == {0: 0.9, -1: 0.1}


def test_the_rater_bonus_lands_on_its_cell_only_and_creates_a_missing_cell():
    rows = Ensemble.rows(FWD, None, llm_links=[0, 2, -1], beta=0.3)
    assert dict(rows[0])[0] == 1.2 and dict(rows[0])[-1] == 0.1
    assert dict(rows[1])[2] == 0.3 and dict(rows[1])[1] == 0.4            # word 1: the rater's cell 2 had no mass
    assert dict(rows[2]) == {2: 0.7, 1: 0.2, -1: 0.1}                     # no bonus on an unlinked word


def test_the_greedy_rule_and_the_shared_decoder_on_the_ensemble_rows():
    rows = Ensemble.rows(FWD, None)
    assert Ensemble.greedy_links(rows, 0.5) == [0, -1, 2]
    assert Ensemble.greedy_links(rows, 0.3) == [0, 1, 2]                   # the champion's rule ignores the null mass
    clash = [[(1, 0.9), (-1, 0.1)], [(1, 0.6), (0, 0.35), (-1, 0.05)]]
    # the loser of a clash gets nothing: ensemble_dump scores each word's best cell only (no fallback
    # to its next candidate, whatever the note's prose says; the code is the reference)
    assert Ensemble.greedy_links(Ensemble.rows(clash, None), 0.3) == [1, -1]
    # the shared decoder lets the null compete: word 1's 0.4 loses to its null 0.6
    assert BaselineDecoder.decode_default(rows, theta=0.5, n_source=3) == [0, -1, 2]
    assert BaselineDecoder.decode_default(rows, theta=0.3, n_source=3) == [0, -1, 2]


# =============================================================================
# 2. The tag rule and the gate
# =============================================================================


def test_the_tag_rule_prefers_a_then_b_then_spelling_then_lemma_then_subst():
    source, reuse = ["rex", "amat", "gladio", "haesit"], ["regem", "amat", "ense", "haesit.", "hostem"]
    links_a, tags_a = [0, 1, -1, -1, -1], ["MORPH", "NOP", "INS", "INS", "INS"]
    links_b, tags_b = [0, -1, 2, -1, -1], ["POS", "INS", "SYN", "INS", "INS"]
    tag = lambda t, s, lemma=False: Ensemble.tag(t, s, source, reuse, links_a, tags_a, links_b, tags_b, lemma)  # noqa: E731
    assert tag(0, 0) == "MORPH"                       # A had the link
    assert tag(2, 2) == "SYN"                         # only B had it
    assert tag(3, 3) == "NOP"                         # haesit. / haesit: same spelling key
    assert tag(4, 2, lemma=True) == "MORPH"           # the lemmas agree
    assert tag(4, 2) == "SUBST"                       # nothing attests


def test_the_gate_passes_model_tags_through_without_evidence_and_overrides_where_attested():
    from retexo.edit_typing.link_features import FEATURE_NAMES

    rec = record("t/9", ["rex", "gladio"], ["rex", "ense"])
    assert GatedTyper(None).gate(rec, [0, 1], ["MORPH", "SUBST"]) == ["MORPH", "SUBST"]

    class Phi:
        def __call__(self, s, t, si, ti, ns, nt):
            v = [0.0] * len(FEATURE_NAMES)
            if s == t:
                v[FEATURE_NAMES.index("same_form")] = 1.0
            return v

    typer = GatedTyper(Phi())
    assert typer.gate(rec, [0, 1], ["MORPH", "SUBST"]) == ["NOP", "SUBST"]     # identical forms attest NOP; ense stays open
    assert not typer.same_lemma(rec, 1, 1)


def test_repairs_apply_the_three_rules_in_order():
    source = ["haesit", "rex", "gladio", "ferit", "arma"]
    reuse = ["ut", "ait", "haesit.", "rex", "arma", "ferit"]
    links, tags, frame = [-1, -1, 0, 1, 4, 3], ["INS", "INS", "MORPH", "NOP", "SUBST", "NOP"], [1, 1, 0, 0, 0, 0]
    out_links, out_tags, out_frame = Repairer.apply(source, reuse, links, tags, frame)
    assert out_tags[2] == "NOP"                          # spelling: haesit. / haesit
    assert out_links[4] == -1 and out_tags[4] == "INS"   # geometry: source 4 lies outside the gap (1, 3)
    assert out_frame == [1, 1, 0, 0, 0, 0]               # the frame span sits next to the first link


# =============================================================================
# 3. The switches and the pieces
# =============================================================================


def config(**extra):
    return BaselineConfig(fold=4, dev_fold=0, device="cpu", base_model=TINY, batch_size=4, seed=1,
                          out=Path("/tmp/fs_test"),
                          extra={"size": 0, "negatives": "none", "gold_passes": 1, "evidence": 0, "gate": 0,
                                 "rater": "none", **extra})


def test_switches_build_the_pieces_they_name():
    full = FullSystem(config())
    assert full.aligner_b is not None and full.rater is None and full.typer == "rule"
    assert full.aligner_a.recipe.evidence is False and full.aligner_b.recipe.dense_rate == SYSTEM_DEFAULTS["b_dense_rate"]
    assert full.aligner_b.cfg.seed == full.cfg.seed + 1
    alone = FullSystem(config(ensemble=0, beta=0))
    assert alone.aligner_b is None and alone.beta == 0.0
    with_evidence = FullSystem(config(evidence=1, gate=1))
    assert with_evidence.aligner_a.recipe.evidence and with_evidence.typer == "own"


def test_two_tiny_aligners_end_to_end_with_and_without_repairs(tmp_path):
    cfg = config()
    cfg.out = tmp_path
    method = FullSystem(cfg).fit(pairs(), [])
    dials = {"theta": 0.45, **method.tune([], log=None)}
    preds = method.predict(pairs())
    assert all(p.scores is not None and len(p.scores) == r.n_reuse for p, r in zip(preds, pairs()))
    done = [method.postprocess(r, p, dials) for r, p in zip(pairs(), preds)]
    for rec, pred in zip(pairs(), done):
        assert len(pred.links) == rec.n_reuse and len(pred.tags) == rec.n_reuse and len(pred.dels) == rec.n_source
        for s, tag in zip(pred.links, pred.tags):
            assert (s < 0) == (tag == "")
            if s >= 0:
                assert tag in labels.EDGE_OPS
        assert "tags_a" in pred.meta
    off = FullSystem(config(repairs=0))
    off.cfg.out = tmp_path / "off"
    off.aligner_a, off.aligner_b = method.aligner_a, method.aligner_b        # the same trained pieces
    again = [off.postprocess(r, p, dials) for r, p in zip(pairs(), off.predict(pairs()))]
    assert all(len(a.links) == len(b.links) for a, b in zip(done, again))
    method.save(tmp_path / "saved")
    assert (tmp_path / "saved" / "aligner_a" / "modules.pt").exists() and (tmp_path / "saved" / "aligner_b" / "modules.pt").exists()
