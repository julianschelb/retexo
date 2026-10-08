# tests/test_baseline_em_aligner.py
"""IBM1, the HMM and the diagonal prior on a toy corpus generated from a known 20-word dictionary."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.baselines import em_aligner as em  # noqa: E402
from retexo.baselines.base import BaselineConfig  # noqa: E402
from retexo.baselines.record import Record  # noqa: E402

WORDS = [f"s{i}" for i in range(20)]
TRANS = {w: f"t{i}" for i, w in enumerate(WORDS)}


def toy_corpus(n=200, monotone=False, seed=0):
    rng = random.Random(seed)
    src, tgt = [], []
    for _ in range(n):
        k = rng.randint(3, 8)
        s = rng.sample(WORDS, k)
        t = [TRANS[w] for w in s]
        if not monotone:
            rng.shuffle(t)
        for i in range(len(t)):
            if rng.random() < 0.1:
                t.insert(i, "ins")
        src.append(s)
        tgt.append(t)
    return src, tgt


def test_ibm1_recovers_dictionary():
    src, tgt = toy_corpus()
    corpus = em.Corpus(src, tgt)
    table = em.LexicalTable(corpus)
    model = em.IBM1(5).fit(corpus, table)
    ll = model.log_likelihood
    assert all(b >= a - 1e-6 for a, b in zip(ll, ll[1:])), ll
    counts = {w: sum(s.count(w) for s in src) for w in WORDS}
    for w in WORDS:
        if counts[w] < 3:
            continue
        sid = corpus.vs.get(w)
        mask = corpus.pair_s == sid
        best = corpus.pair_t[mask][np.argmax(table.theta[mask])]
        assert corpus.vt.words[best] == TRANS[w], (w, corpus.vt.words[best])


def test_hmm_jump_and_posteriors():
    src, tgt = toy_corpus(monotone=True)
    corpus = em.Corpus(src, tgt)
    table = em.LexicalTable(corpus)
    em.IBM1(5).fit(corpus, table)
    hmm = em.HMMAligner(5).fit(corpus, table)
    jump = hmm.jump / hmm.jump.sum()
    assert jump[hmm.max_jump + 1] > 0.5, jump
    for shape, pair_ids in list(corpus.group_ids.items())[:3]:
        post, _ = hmm.posteriors(table, pair_ids)
        assert np.allclose(post.sum(axis=1), 1.0, atol=1e-6)
    direction = em.Direction("hmm", {})
    direction.fit(src, tgt)
    post = direction.predict(["s1", "s2", "s3"], ["t1", "t2", "t3"])
    assert post.shape == (4, 3) and all(post[i + 1, i] >= 0.9 for i in range(3)), post
    post = direction.predict(["s1", "s2", "s3"], ["t1", "ins", "t3"])
    assert post[0, 1] >= 0.5, post[:, 1]


def test_empty_state_bookkeeping():
    src, tgt = toy_corpus(monotone=True)
    corpus = em.Corpus(src, tgt)
    table = em.LexicalTable(corpus)
    em.IBM1(3).fit(corpus, table)
    for p0, check in (
        (0.0, lambda null: null.max() < 1e-9),
        (1.0, lambda null: null.min() > 1 - 1e-9),
    ):
        hmm = em.HMMAligner(1, p0=p0)
        shape, pair_ids = next(iter(corpus.group_ids.items()))
        post, _ = hmm.posteriors(table, pair_ids)
        assert check(post[:, 0, :]), (p0, post[:, 0, :].max(), post[:, 0, :].min())


def test_diagonal_prior():
    src, tgt = toy_corpus(monotone=True)
    corpus = em.Corpus(src, tgt)
    table = em.LexicalTable(corpus)
    em.IBM1(3).fit(corpus, table)
    shape, pair_ids = next(iter(corpus.group_ids.items()))
    n_src = shape[0]
    # lambda = 0 with p0 = 1 / (I + 1) is exactly IBM1's uniform prior over the I + 1 positions
    flat = em.DiagonalIBM2(1, p0=1.0 / (n_src + 1), tension=0.0, optimise_tension=False)
    ibm1 = em.IBM1(1)
    a, _ = flat.posteriors(table, pair_ids)
    b, _ = ibm1.posteriors(table, pair_ids)
    assert np.allclose(a, b, atol=1e-9)
    sharp = em.DiagonalIBM2(1, p0=0.0, tension=8.0, optimise_tension=False)
    prior = sharp.prior(10, 10)[1:]
    j = 4  # target position 5 of 10, relative position 0.5
    near = prior[3:6, j].sum()
    # lambda = 8 on a 10 x 10 pair: 1 + 2 exp(-0.8) over the sum of exp(-0.8 |d|), 0.73 within one position
    # (the note's 0.9 needs lambda about 16); the check is that the prior is sharp, not its exact value
    assert near >= 0.7 and near > 2 * 0.3, near
    for m in range(1, 11):
        for n in range(1, 11):
            weights = np.exp(4.0 * em.DiagonalIBM2.h(m, n))
            brute = np.array(
                [
                    sum(np.exp(4.0 * -abs((j + 1) / n - (i + 1) / m)) for i in range(m))
                    for j in range(n)
                ]
            )
            assert np.allclose(weights.sum(axis=0), brute, atol=1e-9)


def test_rows_and_hungarian():
    from retexo.baselines.decoder import hungarian

    post = np.array([[0.1, 0.2, 0.05], [0.8, 0.1, 0.05], [0.05, 0.6, 0.1], [0.05, 0.1, 0.8]])
    rows = em.rows_from_posteriors(post)
    assert rows[0][0] == (0, 0.8) and rows[0][-1] == (-1, 0.1) or rows[0][0][0] == 0
    for row in rows:
        assert abs(sum(p for _, p in row) - 1.0) < 1e-6
        assert [p for _, p in row] == sorted([p for _, p in row], reverse=True)
    assert hungarian(rows) == [0, 1, 2]


def test_lemma_stream():
    rec = Record(
        id="t/1",
        level="gold",
        fold=4,
        source_work="",
        source_tokens=["Arma", "uirumque"],
        reuse_work="",
        reuse_tokens=["arma", "virum"],
        pair_label="cit",
        annotation={"lemma": ["arma", "uir"]},
    )
    assert em.lemma_stream(rec, "source") == [
        "arma",
        "uirumque",
    ]  # no cached lemmas: normalised forms
    assert em.lemma_stream(rec, "reuse") == ["arma", "uir"]
    assert em.lemma_stream(rec, "reuse", unit="form") == ["arma", "uirum"]
    # without cached lemmas the lemmatiser is asked; a silent lemmatiser falls back to the form; unit=form never asks
    lemmas = {"Arma": "arma", "uirumque": ""}.get
    assert em.lemma_stream(rec, "source", lemmatise=lemmas) == ["arma", "uirumque"]
    assert em.lemma_stream(rec, "source", lemmatise=lambda w: "uir" if w == "uirumque" else "") == [
        "arma",
        "uir",
    ]
    assert em.lemma_stream(rec, "source", unit="form", lemmatise=lambda w: "x") == [
        "arma",
        "uirumque",
    ]


class _Lemmas:
    """A featurizer stand-in: the driver sets ``baseline.featurizer``, whose ``lemma`` the aligner reads."""

    def lemma(self, token: str) -> str:
        return {"virum": "uir", "viro": "uir", "arma": "arma", "armis": "arma"}.get(token, "")


def test_baseline_reads_featurizer_lemmas():
    rec = Record(
        id="t/2",
        level="gold",
        fold=4,
        source_work="",
        source_tokens=["armis", "viro"],
        reuse_work="",
        reuse_tokens=["arma", "virum"],
        pair_label="cit",
    )
    method = em.EMAligner(BaselineConfig(device="cpu", extra={"model": "ibm1"}))
    assert method.corpus_from([rec]) == (
        [["armis", "uiro"]],
        [["arma", "uirum"]],
    )  # no featurizer: forms
    method.featurizer = _Lemmas()
    assert method.corpus_from([rec]) == ([["arma", "uir"]], [["arma", "uir"]])


def test_identity_dictionary():
    src, tgt = [["a", "b"], ["c"]], [["b", "d"], ["a"]]
    dict_s, dict_t = em.EMAligner.identity_dictionary(src, tgt, 2)
    assert dict_s == [["a"], ["a"], ["b"], ["b"]] and dict_t == dict_s
    assert em.EMAligner.identity_dictionary(src, tgt, 0) == ([], [])
    # a hapax pair that a garbage-collecting rare word would otherwise share: the dictionary puts it on the diagonal
    corpus_s = [["rare1", "rare2", "et"], ["et", "x"], ["et", "y"]]
    corpus_t = [["rare2", "z", "et"], ["et", "x"], ["et", "y"]]
    plain = em.Direction("ibm1", {}).fit(corpus_s, corpus_t).predict(corpus_s[0], corpus_t[0])
    assert abs(plain[1, 0] - plain[2, 0]) < 1e-9, plain[:, 0]  # without it: rare1 and rare2 tie
    dict_s, dict_t = em.EMAligner.identity_dictionary(corpus_s, corpus_t, 1)
    post = (
        em.Direction("ibm1", {})
        .fit(corpus_s + dict_s, corpus_t + dict_t)
        .predict(corpus_s[0], corpus_t[0])
    )
    assert post[2, 0] > 0.6 and post[2, 0] > 4 * post[1, 0], post[:, 0]  # with it: rare2 -> rare2


def test_baseline_both_directions_and_determinism():
    src, tgt = toy_corpus(n=120, monotone=True)
    records = [
        Record(
            id=f"t/{i}",
            level="external",
            fold=-1,
            source_work="",
            source_tokens=s,
            reuse_work="",
            reuse_tokens=t,
            pair_label="unknown",
        )
        for i, (s, t) in enumerate(zip(src, tgt))
    ]
    cfg = BaselineConfig(device="cpu", extra={"model": "hmm", "unit": "form"})
    method = em.EMAligner(cfg).fit(records[:100], records[100:110], unlabeled=records[110:])
    pred = method.predict(records[110:111])[0]
    fwd = [row[0][0] for row in pred.scores]
    rev = [row[0][0] for row in pred.rev_scores]
    for t, s in enumerate(fwd):
        if s >= 0 and records[110].reuse_tokens[t] != "ins":
            assert rev[s] == t, (t, s, rev)
    again = em.EMAligner(BaselineConfig(device="cpu", extra={"model": "hmm", "unit": "form"})).fit(
        records[:100], records[100:110], unlabeled=records[110:]
    )
    assert np.allclose(method.forward.theta, again.forward.theta)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"[test_baseline_em_aligner] {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
