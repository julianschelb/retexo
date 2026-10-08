# retexo/baselines/em_aligner.py
"""The classical baseline: IBM Model 1 and the HMM aligner, trained by EM on lemmas.

Table 1's "pairs only" row asks what an aligner learns from passage pairs
without word labels. The literature's answer is a translation table: count
which words co-occur across many pairs and link the ones that co-occur
suspiciously often (Brown et al. 1993), with an empty word for words that
come from nowhere and, in the HMM (Vogel et al. 1996; Och and Ney 2003), a
preference for staying near the previous link. On reuse the row is expected
to rediscover the lemmatiser's links, since substitutions do not repeat
across pairs; reviewers ask for it anyway. The paper reports it as
"IBM1 + HMM (EM)" with eflomal as the reference, or as eflomal-equivalent
if the check on the parallel gold lands within tolerance of eflomal.

Reimplemented rather than run as a binary: the posterior rows the shared
decoder needs (interface I) are a by-product of forward-backward, the
corpus is small, and one package with one record format is the point.
Three estimators share one corpus and one table: ``IBM1`` (uniform
alignment prior, five iterations, convex), ``HMMAligner`` (jump model with
``2I`` states, the second ``I`` being "null after position i", Och and Ney
eq. 12 to 16, ``p0`` = 0.2, initialised from Model 1) and ``DiagonalIBM2``
(fast_align's diagonal prior, Dyer et al. 2013, ``p0`` = 0.08, tension
re-estimated by a grid). The E-step is vectorised over sentence pairs of
equal shape so that a few hundred thousand pairs train in minutes; the
lexical table is sparse over co-occurring pairs only, with an optional
Dirichlet prior in the mean-field form (Riley and Gildea; eflomal's 0.001).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Record
from retexo.core.normalize import normalize

DEFAULT_ITERATIONS = 5  # Och and Ney 2003, section 6 (GIZA++ 1^5 H^5); fast_align ITERATIONS = 5
DEFAULT_P0_HMM = 0.2  # Och and Ney 2003, p. 24, the held-out value of the empty-word transition
DEFAULT_P0_DIAG = 0.08  # Dyer et al. 2013, p. 647, prob_align_null
DEFAULT_TENSION = 4.0  # Dyer et al. 2013, p. 647, diagonal_tension
DEFAULT_SMOOTHING = 0.05  # Och and Ney 2003, eq. 37, alpha (tuned on held-out data)
#: Plain EM by default: the Dirichlet smoothing adds ``alpha * n_target_types`` to every M-step denominator, which on a
#: corpus of a few thousand passages (ten or more per source type) flattens every table row towards the null word and
#: makes the likelihood fall after the first iteration (found on the Latin dry run, 2026-09-16); on parallel bitext of
#: 100k+ pairs it was harmless. ``prior_alpha`` stays a dial.
DEFAULT_PRIOR_ALPHA = 0.0
DEFAULT_MAX_JUMP = 10
NULL = 0

# =============================================================================
# Vocabulary and corpus
# =============================================================================


class Vocab:
    """Word types to ids; id 0 is the empty word."""

    def __init__(self):
        self.index: Dict[str, int] = {"<null>": NULL}
        self.words: List[str] = ["<null>"]

    def add(self, word: str) -> int:
        if word not in self.index:
            self.index[word] = len(self.words)
            self.words.append(word)
        return self.index[word]

    def get(self, word: str) -> int:
        return self.index.get(word, -1)

    def __len__(self) -> int:
        return len(self.words)


class Corpus:
    """Sentence pairs as id arrays, grouped by shape, with the sparse pair index.

    ``pairs`` holds every distinct (source id including null, target id)
    that co-occurs; ``pair_s`` its source id; each group of sentences with
    the same ``(I, J)`` holds an int32 array ``[B, I + 1, J]`` of pair ids.
    """

    def __init__(
        self,
        source: Sequence[Sequence[str]],
        target: Sequence[Sequence[str]],
        vs: Optional[Vocab] = None,
        vt: Optional[Vocab] = None,
        *,
        grow: bool = True,
    ):
        self.vs = vs or Vocab()
        self.vt = vt or Vocab()
        self.sentences: List[Tuple[np.ndarray, np.ndarray]] = []
        for s, t in zip(source, target):
            if not s or not t:
                continue
            s_ids = np.array(
                [NULL] + [self.vs.add(w) if grow else max(self.vs.get(w), 0) for w in s],
                dtype=np.int64,
            )
            t_ids = np.array(
                [self.vt.add(w) if grow else max(self.vt.get(w), 0) for w in t], dtype=np.int64
            )
            self.sentences.append((s_ids, t_ids))
        self.groups: Dict[Tuple[int, int], List[int]] = {}
        for n, (s_ids, t_ids) in enumerate(self.sentences):
            self.groups.setdefault((len(s_ids) - 1, len(t_ids)), []).append(n)
        self.n_target_types = max(len(self.vt), 1)
        self._index_pairs()

    def _index_pairs(self) -> None:
        keys = []
        for s_ids, t_ids in self.sentences:
            keys.append((s_ids[:, None] * self.n_target_types + t_ids[None, :]).ravel())
        flat = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
        self.pair_keys, inverse = np.unique(flat, return_inverse=True)
        self.pair_s = (self.pair_keys // self.n_target_types).astype(np.int64)
        self.pair_t = (self.pair_keys % self.n_target_types).astype(np.int64)
        self.n_pairs = len(self.pair_keys)
        self.group_ids: Dict[Tuple[int, int], np.ndarray] = {}
        offsets = np.cumsum([0] + [len(k) for k in keys])
        for shape, members in self.groups.items():
            n_src, J = shape
            block = np.stack(
                [inverse[offsets[n] : offsets[n + 1]].reshape(n_src + 1, J) for n in members]
            )
            self.group_ids[shape] = block.astype(np.int32)

    def __len__(self) -> int:
        return len(self.sentences)


# =============================================================================
# The lexical table shared by every estimator
# =============================================================================


class LexicalTable:
    """Sparse ``theta[s, t]`` over the corpus's co-occurring pairs."""

    def __init__(self, corpus: Corpus, *, prior_alpha: float = 0.0, variational: bool = False):
        self.corpus = corpus
        self.prior_alpha = prior_alpha
        self.variational = variational
        self.n_source_types = int(corpus.pair_s.max()) + 1 if corpus.n_pairs else 1
        self.theta = np.zeros(corpus.n_pairs, dtype=np.float64)
        self.uniform()

    def uniform(self) -> None:
        counts = np.bincount(self.corpus.pair_s, minlength=self.n_source_types).astype(np.float64)
        self.theta = 1.0 / np.maximum(counts[self.corpus.pair_s], 1.0)

    def m_step(self, counts: np.ndarray) -> None:
        totals = np.bincount(self.corpus.pair_s, weights=counts, minlength=self.n_source_types)
        if self.variational and self.prior_alpha > 0:
            from scipy.special import digamma

            alpha = self.prior_alpha
            self.theta = np.exp(
                digamma(counts + alpha)
                - digamma(totals[self.corpus.pair_s] + alpha * self.corpus.n_target_types)
            )
        else:
            self.theta = (counts + self.prior_alpha) / np.maximum(
                totals[self.corpus.pair_s] + self.prior_alpha * self.corpus.n_target_types, 1e-12
            )
        self.theta = np.maximum(self.theta, 1e-12)

    def sub(self, pair_ids: np.ndarray) -> np.ndarray:
        """``theta`` gathered into ``[B, I + 1, J]``."""
        return self.theta[pair_ids]


# =============================================================================
# Estimators
# =============================================================================


class IBM1:
    """Uniform alignment prior; the posterior is the normalised lexical column."""

    name = "ibm1"

    def __init__(self, iterations: int = DEFAULT_ITERATIONS):
        self.iterations = iterations
        self.log_likelihood: List[float] = []

    def posteriors(
        self, table: LexicalTable, pair_ids: np.ndarray, **_
    ) -> Tuple[np.ndarray, float]:
        """``[B, I + 1, J]`` posteriors and the batch log-likelihood."""
        sub = table.sub(pair_ids)
        total = sub.sum(axis=1, keepdims=True)
        post = sub / np.maximum(total, 1e-300)
        n_src = pair_ids.shape[1] - 1
        ll = float(
            np.log(np.maximum(total, 1e-300)).sum()
            - pair_ids.shape[0] * pair_ids.shape[2] * math.log(n_src + 1)
        )
        return post, ll

    def fit(self, corpus: Corpus, table: LexicalTable, *, log=None) -> IBM1:
        for it in range(self.iterations):
            counts = np.zeros(corpus.n_pairs, dtype=np.float64)
            ll = 0.0
            for _shape, pair_ids in corpus.group_ids.items():
                post, batch_ll = self.posteriors(table, pair_ids)
                counts += np.bincount(
                    pair_ids.ravel(), weights=post.ravel(), minlength=corpus.n_pairs
                )
                ll += batch_ll
            self.log_likelihood.append(ll)
            table.m_step(counts)
            if log:
                log(
                    f"[em_aligner] ibm1 iteration {it + 1}/{self.iterations} log-likelihood {ll:.1f}"
                )
        return self


class DiagonalIBM2(IBM1):
    """fast_align: IBM1 with a prior ``exp(lambda * h)`` that pins links to the diagonal (Dyer et al. 2013)."""

    name = "diag"

    def __init__(
        self,
        iterations: int = DEFAULT_ITERATIONS,
        p0: float = DEFAULT_P0_DIAG,
        tension: float = DEFAULT_TENSION,
        optimise_tension: bool = True,
    ):
        super().__init__(iterations)
        self.p0 = p0
        self.tension = tension
        self.optimise_tension = optimise_tension

    # ---------- Diagonal prior ----------

    @staticmethod
    def h(n_src: int, J: int) -> np.ndarray:
        """``-|j/J - i/I|`` for source position ``i`` (1..I) and target position ``j`` (1..J): shape ``[I, J]``."""
        i = np.arange(1, n_src + 1, dtype=np.float64)[:, None] / n_src
        j = np.arange(1, J + 1, dtype=np.float64)[None, :] / J
        return -np.abs(j - i)

    def prior(self, n_src: int, J: int, tension: Optional[float] = None) -> np.ndarray:
        """``[I + 1, J]``: ``p0`` for the null row, the normalised diagonal prior below it."""
        lam = self.tension if tension is None else tension
        weights = np.exp(lam * self.h(n_src, J))
        z = weights.sum(axis=0, keepdims=True)
        out = np.empty((n_src + 1, J))
        out[0] = self.p0
        out[1:] = (1.0 - self.p0) * weights / z
        return out

    # ---------- EM ----------

    def posteriors(
        self, table: LexicalTable, pair_ids: np.ndarray, **_
    ) -> Tuple[np.ndarray, float]:
        n_src, J = pair_ids.shape[1] - 1, pair_ids.shape[2]
        joint = table.sub(pair_ids) * self.prior(n_src, J)[None]
        total = joint.sum(axis=1, keepdims=True)
        return joint / np.maximum(total, 1e-300), float(np.log(np.maximum(total, 1e-300)).sum())

    def fit(self, corpus: Corpus, table: LexicalTable, *, log=None) -> DiagonalIBM2:
        for it in range(self.iterations):
            counts = np.zeros(corpus.n_pairs, dtype=np.float64)
            ll = 0.0
            expected: Dict[Tuple[int, int], np.ndarray] = {}
            for shape, pair_ids in corpus.group_ids.items():
                post, batch_ll = self.posteriors(table, pair_ids)
                counts += np.bincount(
                    pair_ids.ravel(), weights=post.ravel(), minlength=corpus.n_pairs
                )
                expected[shape] = post[:, 1:, :].sum(axis=0)
                ll += batch_ll
            self.log_likelihood.append(ll)
            table.m_step(counts)
            if self.optimise_tension:
                self.tension = self._best_tension(expected)
            if log:
                log(
                    f"[em_aligner] diag iteration {it + 1}/{self.iterations} log-likelihood {ll:.1f} tension {self.tension:.2f}"
                )
        return self

    def _best_tension(self, expected: Dict[Tuple[int, int], np.ndarray]) -> float:
        """The grid value of lambda that maximises the expected log prior (Dyer eq. 2, solved by search)."""
        best, best_value = self.tension, -math.inf
        for lam in [x / 2 for x in range(2, 17)]:
            value = 0.0
            for (n_src, J), post in expected.items():
                weights = np.exp(lam * self.h(n_src, J))
                log_prior = lam * self.h(n_src, J) - np.log(weights.sum(axis=0, keepdims=True))
                value += float((post * log_prior).sum())
            if value > best_value:
                best, best_value = lam, value
        return best


class HMMAligner:
    """The homogeneous HMM with ``I`` empty states (Och and Ney 2003, eq. 12 to 16), by Baum-Welch."""

    name = "hmm"

    def __init__(
        self,
        iterations: int = DEFAULT_ITERATIONS,
        p0: float = DEFAULT_P0_HMM,
        smoothing_alpha: float = DEFAULT_SMOOTHING,
        max_jump: int = DEFAULT_MAX_JUMP,
    ):
        self.iterations = iterations
        self.p0 = p0
        self.smoothing_alpha = smoothing_alpha
        self.max_jump = max_jump
        self.jump = np.ones(2 * max_jump + 1, dtype=np.float64)  # c(d), d in [-max_jump, max_jump]
        self.log_likelihood: List[float] = []
        self._transitions: Dict[int, np.ndarray] = {}

    # ---------- Jump model ----------

    def _bucket(self, d: np.ndarray) -> np.ndarray:
        return np.clip(d, -self.max_jump, self.max_jump) + self.max_jump

    def transitions(self, n_src: int) -> np.ndarray:
        """``[2I, 2I]`` row-stochastic matrix over real states ``0..I-1`` and null states ``I..2I-1``."""
        if n_src in self._transitions:
            return self._transitions[n_src]
        pos = np.arange(n_src)
        c = self.jump[self._bucket(pos[None, :] - pos[:, None])]  # c(i - i') for real -> real
        p = c / c.sum(axis=1, keepdims=True)
        p = (1.0 - self.smoothing_alpha) * p + self.smoothing_alpha / n_src
        real = (1.0 - self.p0) * p
        T = np.zeros((2 * n_src, 2 * n_src))
        T[:n_src, :n_src] = real
        T[:n_src, n_src:] = self.p0 * np.eye(n_src)
        T[n_src:, :n_src] = real
        T[n_src:, n_src:] = self.p0 * np.eye(n_src)
        self._transitions[n_src] = T
        return T

    def initial(self, n_src: int) -> np.ndarray:
        start = np.full(2 * n_src, 0.0)
        start[:n_src] = (1.0 - self.p0) / n_src
        start[n_src:] = self.p0 / n_src
        return start

    # ---------- Emissions and forward-backward ----------

    def _emissions(self, table: LexicalTable, pair_ids: np.ndarray) -> np.ndarray:
        """``[B, 2I, J]``: real states emit ``theta[s_i, t_j]``, null states ``theta[null, t_j]``."""
        sub = table.sub(pair_ids)
        n_src = pair_ids.shape[1] - 1
        null = np.repeat(sub[:, :1, :], n_src, axis=1)
        return np.concatenate([sub[:, 1:, :], null], axis=1)

    def forward_backward(
        self, table: LexicalTable, pair_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Posteriors ``[B, 2I, J]``, expected transitions ``[2I, 2I]`` summed over the batch, log-likelihood."""
        B, I1, J = pair_ids.shape
        n_src = I1 - 1
        E = self._emissions(table, pair_ids)
        T = self.transitions(n_src)
        alpha = np.zeros((B, 2 * n_src, J))
        beta = np.zeros((B, 2 * n_src, J))
        scale = np.zeros((B, J))
        alpha[:, :, 0] = self.initial(n_src)[None, :] * E[:, :, 0]
        scale[:, 0] = alpha[:, :, 0].sum(axis=1)
        alpha[:, :, 0] /= np.maximum(scale[:, :1], 1e-300)
        for j in range(1, J):
            alpha[:, :, j] = (alpha[:, :, j - 1] @ T) * E[:, :, j]
            scale[:, j] = alpha[:, :, j].sum(axis=1)
            alpha[:, :, j] /= np.maximum(scale[:, j : j + 1], 1e-300)
        beta[:, :, J - 1] = 1.0
        for j in range(J - 2, -1, -1):
            beta[:, :, j] = ((beta[:, :, j + 1] * E[:, :, j + 1]) @ T.T) / np.maximum(
                scale[:, j + 1 : j + 2], 1e-300
            )
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi = np.zeros((2 * n_src, 2 * n_src))
        for j in range(1, J):
            right = (E[:, :, j] * beta[:, :, j]) / np.maximum(
                scale[:, j : j + 1], 1e-300
            )  # [B, 2I]
            xi += np.einsum("bi,bk->ik", alpha[:, :, j - 1], right) * T
        ll = float(np.log(np.maximum(scale, 1e-300)).sum())
        return gamma, xi, ll

    # ---------- EM ----------

    def posteriors(
        self, table: LexicalTable, pair_ids: np.ndarray, **_
    ) -> Tuple[np.ndarray, float]:
        """``[B, I + 1, J]`` with the null mass in row 0 (the ``I`` empty states summed)."""
        gamma, _, ll = self.forward_backward(table, pair_ids)
        n_src = pair_ids.shape[1] - 1
        out = np.concatenate(
            [gamma[:, n_src:, :].sum(axis=1, keepdims=True), gamma[:, :n_src, :]], axis=1
        )
        return out, ll

    def fit(self, corpus: Corpus, table: LexicalTable, *, log=None) -> HMMAligner:
        for it in range(self.iterations):
            counts = np.zeros(corpus.n_pairs, dtype=np.float64)
            jump_counts = np.zeros_like(self.jump)
            ll = 0.0
            for shape, pair_ids in corpus.group_ids.items():
                n_src = shape[0]
                gamma, xi, batch_ll = self.forward_backward(table, pair_ids)
                post = np.concatenate(
                    [gamma[:, n_src:, :].sum(axis=1, keepdims=True), gamma[:, :n_src, :]], axis=1
                )
                counts += np.bincount(
                    pair_ids.ravel(), weights=post.ravel(), minlength=corpus.n_pairs
                )
                pos = np.arange(n_src)
                origin = np.concatenate([pos, pos])  # the position a null state remembers
                d = self._bucket(
                    pos[None, :] - origin[:, None]
                )  # [2I, I] jump widths into real states
                jump_counts += np.bincount(
                    d.ravel(), weights=xi[:, :n_src].ravel(), minlength=len(self.jump)
                )
                ll += batch_ll
            self.log_likelihood.append(ll)
            table.m_step(counts)
            self.jump = jump_counts + 1e-3
            self._transitions = {}
            if log:
                log(
                    f"[em_aligner] hmm iteration {it + 1}/{self.iterations} log-likelihood {ll:.1f} "
                    f"p(+1) {self.jump[self.max_jump + 1] / self.jump.sum():.3f}"
                )
        return self


# =============================================================================
# From posteriors to interface (I), and the baseline
# =============================================================================


ESTIMATORS = {"ibm1": IBM1, "hmm": HMMAligner, "diag": DiagonalIBM2}


class Direction:
    """One direction's corpus, table and estimator; ``predict`` gives ``[I + 1, J]`` posteriors per pair."""

    @staticmethod
    def rows_from_posteriors(post: np.ndarray) -> Rows:
        """``[I + 1, J]`` posteriors of one pair to rows per target word, null as ``-1``, best first."""
        rows = []
        for j in range(post.shape[1]):
            column = post[:, j]
            entries = [(i - 1, float(column[i])) for i in range(1, post.shape[0])] + [
                (-1, float(column[0]))
            ]
            rows.append(sorted(entries, key=lambda x: -x[1]))
        return rows

    def __init__(self, model: str, extra: Dict):
        self.model = model
        self.extra = extra
        self.vs, self.vt = Vocab(), Vocab()
        self.table: Optional[LexicalTable] = None
        self.ibm1: Optional[IBM1] = None
        self.estimator = None
        self.pair_keys = np.zeros(0, dtype=np.int64)
        self.theta = np.zeros(0)
        self.n_target_types = 1

    def fit(self, source: List[List[str]], target: List[List[str]], *, log=None) -> Direction:
        corpus = Corpus(source, target, self.vs, self.vt)
        table = LexicalTable(
            corpus,
            prior_alpha=float(self.extra.get("prior_alpha", DEFAULT_PRIOR_ALPHA)),
            variational=bool(self.extra.get("variational", 0)),
        )
        it_ibm1 = int(self.extra.get("iterations_ibm1", DEFAULT_ITERATIONS))
        it_hmm = int(self.extra.get("iterations_hmm", DEFAULT_ITERATIONS))
        if self.model == "diag":
            self.estimator = DiagonalIBM2(
                it_ibm1,
                p0=float(self.extra.get("p0", DEFAULT_P0_DIAG)),
                tension=float(self.extra.get("tension", DEFAULT_TENSION)),
                optimise_tension=bool(self.extra.get("optimise_tension", 1)),
            )
            self.estimator.fit(corpus, table, log=log)
        else:
            self.ibm1 = IBM1(it_ibm1).fit(corpus, table, log=log)
            if self.model == "hmm":
                self.estimator = HMMAligner(
                    it_hmm,
                    p0=float(self.extra.get("p0", DEFAULT_P0_HMM)),
                    smoothing_alpha=float(self.extra.get("smoothing_alpha", DEFAULT_SMOOTHING)),
                    max_jump=int(self.extra.get("max_jump", DEFAULT_MAX_JUMP)),
                )
                self.estimator.fit(corpus, table, log=log)
            else:
                self.estimator = self.ibm1
        # keep only what prediction needs: the sorted pair keys and theta, not the corpus blocks
        self.pair_keys = corpus.pair_keys
        self.theta = table.theta
        self.n_target_types = corpus.n_target_types
        self.table = None
        return self

    def _theta_matrix(self, s_ids: np.ndarray, t_ids: np.ndarray) -> np.ndarray:
        """``[I + 1, J]`` of ``theta`` for one pair; pairs never seen in training get a floor."""
        floor = 1e-6
        keys = s_ids[:, None] * self.n_target_types + t_ids[None, :]
        idx = np.searchsorted(self.pair_keys, keys)
        idx = np.clip(idx, 0, max(len(self.pair_keys) - 1, 0))
        found = (self.pair_keys[idx] == keys) & (s_ids[:, None] >= 0) & (t_ids[None, :] >= 0)
        return np.where(found, self.theta[idx], floor)

    def predict(self, source: List[str], target: List[str]) -> np.ndarray:
        s_ids = np.array([NULL] + [self.vs.get(w) for w in source])
        t_ids = np.array([self.vt.get(w) for w in target])
        theta = self._theta_matrix(s_ids, t_ids)
        n_src, J = len(source), len(target)
        if self.model == "diag":
            joint = theta * self.estimator.prior(n_src, J)
            return joint / np.maximum(joint.sum(axis=0, keepdims=True), 1e-300)
        if self.model == "ibm1":
            return theta / np.maximum(theta.sum(axis=0, keepdims=True), 1e-300)
        table = _FixedTable(theta)
        pair_ids = np.arange((n_src + 1) * J, dtype=np.int32).reshape(1, n_src + 1, J)
        post, _ = self.estimator.posteriors(table, pair_ids)
        return post[0]


class _FixedTable:
    """A dense ``theta`` for one pair, quacking like ``LexicalTable`` for the estimators."""

    def __init__(self, theta: np.ndarray):
        self.flat = theta.ravel()

    def sub(self, pair_ids: np.ndarray) -> np.ndarray:
        return self.flat[pair_ids]


def extra_bitext(path, *, log=None, name: str = "aligner") -> List[Record]:
    """The real reuse pairs of a stage-0 pool as unlabelled bitext for a pairs-only aligner (no gold passage in
    it, any fold); ``[]`` without a path."""
    if not path:
        return []
    from pathlib import Path as _Path

    from retexo.pretraining.pool import PairPool

    records = PairPool.load(_Path(str(path))).to_records()
    if log:
        log(f"[{name}] extra bitext: {len(records)} real pairs from {path}")
    return records


@BaselineRegistry.register
class EMAligner(Baseline):
    """IBM1 / HMM / diagonal-prior aligner on lemmas, both directions, posteriors as score rows.

    ``cfg.extra`` dials (defaults in brackets): ``model`` (hmm | ibm1 | diag),
    ``unit`` (lemma | form), ``iterations_ibm1`` (5), ``iterations_hmm`` (5),
    ``p0`` (0.2 for hmm, 0.08 for diag), ``smoothing_alpha`` (0.05),
    ``prior_alpha`` (0.001), ``variational`` (0), ``tension`` (4.0),
    ``optimise_tension`` (1), ``max_jump`` (10), ``both_directions`` (1),
    ``with_synthetic`` (0), ``include_unlabeled`` (1: the test sentences'
    text joins the EM corpus, as fast_align and eflomal align the whole
    bitext), ``extra_bitext`` (a stage-0 pool of real reuse pairs without
    labels, e.g. ``data/pretrain/pairs.jsonl``: the family's natural data),
    ``identity_pairs`` (0: copies of the one-word pair ``(w, w)`` per shared
    type appended to the corpus, the identity dictionary). With
    ``unit=lemma`` the lemmas come from the driver's featurizer.

    Example:
        ```python
        method = EMAligner(BaselineConfig(device="cpu", extra={"model": "hmm"}))
        method.fit(train, dev, unlabeled=test)
        pred = method.postprocess(test[0], method.predict(test[:1])[0], {"theta": 0.45})
        # python run_baseline.py --method em_aligner --set wpt_enfr --decoder gdf --extra unit=form,model=diag --max-train 200000
        ```
    """

    name = "em_aligner"
    emits = "scores"
    trainable = True
    typer = "rule"
    decoder = "default"

    # ---------- Setup ----------

    @staticmethod
    def lemma_stream(
        record: Record,
        side: str,
        unit: str = "lemma",
        lemmatise: Optional[Callable[[str], str]] = None,
    ) -> List[str]:
        """The tokens of one side as lemmas or normalised forms.

        With ``unit == "lemma"`` a token's lemma is the record's cached one (``annotation.lemma`` /
        ``lemma_source``), else ``lemmatise(token)`` (the driver's Latin lemmatiser), else its normalised
        form. No gold or pool record carries cached lemmas, so without ``lemmatise`` the "lemma" row ran on
        forms (found on the five-fold run, 2026-09-28: 0 to 4 of 134 to 164 lemma-only gold links per fold).
        """
        tokens = record.source_tokens if side == "source" else record.reuse_tokens
        key = "lemma_source" if side == "source" else "lemma"
        lemmas = record.annotation.get(key) if unit == "lemma" else None
        out = []
        for i, token in enumerate(tokens):
            lemma = lemmas[i] if lemmas and i < len(lemmas) and lemmas[i] else ""
            if not lemma and unit == "lemma" and lemmatise is not None:
                lemma = lemmatise(token) or ""
            out.append(lemma.lower() if lemma else (normalize(token) or token.lower()))
        return out

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.model = str(cfg.extra.get("model", "hmm"))
        if self.model not in ESTIMATORS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {sorted(ESTIMATORS)}")
        self.unit = str(cfg.extra.get("unit", "lemma"))
        self.forward: Optional[Direction] = None
        self.reverse: Optional[Direction] = None

    # ---------- Training ----------

    def stream(self, record: Record, side: str) -> List[str]:
        """``lemma_stream`` with the driver's lemmatiser (the rule typer's featurizer) when one is set."""
        lemmatise = self.featurizer.lemma if self.featurizer is not None else None
        return self.lemma_stream(record, side, self.unit, lemmatise)

    def corpus_from(self, records: Sequence[Record]) -> Tuple[List[List[str]], List[List[str]]]:
        source = [self.stream(r, "source") for r in records]
        target = [self.stream(r, "reuse") for r in records]
        return source, target

    @staticmethod
    def identity_dictionary(
        source: List[List[str]], target: List[List[str]], copies: int
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """``copies`` one-word pairs ``(w, w)`` for every type on both sides of the corpus: the identity of
        lemmas as a bilingual dictionary appended to the training corpus (the note's open dictionary prior)."""
        if copies <= 0:
            return [], []
        shared = sorted({w for s in source for w in s} & {w for t in target for w in t})
        pairs = [[w] for w in shared for _ in range(copies)]
        return pairs, [list(p) for p in pairs]

    def fit(
        self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
    ) -> EMAligner:
        # the trained-on pairs and the shared synthetic pairs, as text (no label is read); the dev and test
        # passages join only with ``include_unlabeled`` (the transductive practice of the alignment literature)
        records = list(train) + list(self.shared_synthetic)
        if int(self.cfg.extra.get("include_unlabeled", 1)):
            records += list(dev) + list(unlabeled)
        records += extra_bitext(self.cfg.extra.get("extra_bitext"), log=log, name="em_aligner")
        if self.cfg.smoke:
            for key in ("iterations_ibm1", "iterations_hmm"):
                self.cfg.extra[key] = min(int(self.cfg.extra.get(key, DEFAULT_ITERATIONS)), 2)
        source, target = self.corpus_from(records)
        dict_s, dict_t = self.identity_dictionary(
            source, target, int(self.cfg.extra.get("identity_pairs", 0))
        )
        source, target = source + dict_s, target + dict_t
        if log:
            lemmatised = (
                "the driver's lemmatiser"
                if self.unit == "lemma" and self.featurizer is not None
                else "no lemmatiser"
            )
            log(
                f"[em_aligner] corpus {len(source)} pairs ({len(dict_s)} identity pairs), model {self.model}, "
                f"unit {self.unit} ({lemmatised})"
            )
        self.forward = Direction(self.model, self.cfg.extra).fit(source, target, log=log)
        if int(self.cfg.extra.get("both_directions", 1)):
            self.reverse = Direction(self.model, self.cfg.extra).fit(target, source, log=log)
        return self

    # ---------- Inference ----------

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            source, target = self.stream(record, "source"), self.stream(record, "reuse")
            pred = Prediction.empty(record.n_reuse)
            if self.forward is None or not source or not target:
                out.append(pred)
                continue
            pred.scores = Direction.rows_from_posteriors(self.forward.predict(source, target))
            if self.reverse is not None:
                pred.rev_scores = Direction.rows_from_posteriors(
                    self.reverse.predict(target, source)
                )
            out.append(pred)
        return out

    # ---------- Persistence ----------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "em_aligner.pkl", "wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "unit": self.unit,
                    "forward": self.forward,
                    "reverse": self.reverse,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> EMAligner:
        with open(Path(path) / "em_aligner.pkl", "rb") as handle:
            state = pickle.load(handle)
        cfg.extra.setdefault("model", state["model"])
        cfg.extra.setdefault("unit", state["unit"])
        method = cls(cfg)
        method.forward, method.reverse = state["forward"], state["reverse"]
        return method


#: Backward-compatible module-level aliases.
rows_from_posteriors = Direction.rows_from_posteriors
lemma_stream = EMAligner.lemma_stream
