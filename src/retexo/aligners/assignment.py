# retexo/aligners/assignment.py
"""How a table of pointer scores becomes one source per reuse word.

Ported verbatim from the preliminary runners ``run_e8.py`` (E8, the assignment
policies), ``run_e9.py`` (E9, the identity bonus), ``run_e18.py`` (E18, the null
scale) and ``run_e23.py`` (E23, the lemma re-ranking of the shortlist), which now
live under ``attic/scripts/``. Every method here is corpus-level: ``scores``
is a list over pairs, each a list over reuse words, each a list of
``(source_index, probability)`` candidates sorted best first with the null as
source ``-1``. The pair-level restatements used by the baselines live in
``retexo.baselines.decoder``; the unit tests pin those to these originals.

The three call sites inside the package (``Attester.symbolic_links``,
``grid_refiner`` and ``formulations.typed_pointer``) import
``AssignmentPolicy.links_hungarian`` from here, so the package no longer
depends on any top-level script.
"""

from __future__ import annotations

import math
from typing import Callable, ClassVar, Dict, List

from retexo.core.normalize import normalize


class AssignmentPolicy:
    """Assignment policies -- how a table of scores becomes one source per
    reuse word.

    Example:
        ```python
        links = AssignmentPolicy.links_hungarian(scores)
        links = AssignmentPolicy.POLICIES["greedy-2"](scores)
        ```
    """

    @staticmethod
    def _null_probability(candidates) -> float:
        for source, probability in candidates:
            if source < 0:
                return probability
        return 0.0

    @staticmethod
    def links_argmax(scores) -> List[List[int]]:
        """What the model does today: each reuse word takes its own best candidate.

        The candidate lists arrive sorted best first, so this is the head of each.
        Nothing coordinates the choices, which is the behaviour under test.
        """
        return [
            [(candidates[0][0] if candidates else -1) for candidates in example]
            for example in scores
        ]

    @classmethod
    def links_greedy(cls, scores, cap: int = 1) -> List[List[int]]:
        """Highest-confidence links first, refusing a source that is already spent.

        Null is never excluded: a reuse word whose best candidate loses to its own
        null probability is left unaligned rather than pushed onto a source it did
        not want. That matters because 84% of gold tokens align to nothing, so a
        policy that forces every word onto some source would be far worse than the
        baseline it is trying to improve.

        ``cap`` is how many reuse words may share one source. One is the strict
        constraint; two leaves room for the genuine splits the hand labels contain
        (*necessest* -> *necesse est*), which a strict rule makes unrepresentable.
        """
        out = []
        for example in scores:
            nulls = [cls._null_probability(c) for c in example]
            claims = [
                (probability, word, source)
                for word, candidates in enumerate(example)
                for source, probability in candidates
                if source >= 0 and probability > nulls[word]
            ]
            claims.sort(key=lambda claim: -claim[0])
            assigned = [-1] * len(example)
            spent: Dict[int, int] = {}
            for _, word, source in claims:
                if assigned[word] >= 0 or spent.get(source, 0) >= cap:
                    continue
                assigned[word] = source
                spent[source] = spent.get(source, 0) + 1
            out.append(assigned)
        return out

    @classmethod
    def links_hungarian(cls, scores) -> List[List[int]]:
        """The globally optimal one-to-one assignment of reuse words to source words.

        Each reuse word gets a private null column carrying its own null
        probability, so declining to align is always available and costs what the
        model says it costs. Without that the solver would be forced to align every
        reuse word to some source, which the data does not support.

        Optimal for the sum of probabilities -- which is not obviously the right
        objective here, since each reuse word's probabilities come from its own
        softmax over its own candidate set and so are not calibrated against another
        word's. Whether that theoretical objection costs anything measurable is what
        this arm is for.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        out = []
        for example in scores:
            words = len(example)
            sources = sorted({s for c in example for s, _ in c if s >= 0})
            if not words or not sources:
                out.append([-1] * words)
                continue
            column = {source: j for j, source in enumerate(sources)}
            blocked = 1e6
            cost = np.full((words, len(sources) + words), blocked)
            for word, candidates in enumerate(example):
                for source, probability in candidates:
                    if source >= 0:
                        cost[word, column[source]] = -probability
                cost[word, len(sources) + word] = -cls._null_probability(candidates)
            rows, columns = linear_sum_assignment(cost)
            assigned = [-1] * words
            for word, j in zip(rows, columns):
                if j < len(sources) and cost[word, j] < blocked:
                    assigned[word] = sources[j]
            out.append(assigned)
        return out

    POLICIES: ClassVar[Dict[str, Callable]] = {}


AssignmentPolicy.POLICIES = {
    "argmax": AssignmentPolicy.links_argmax,
    "greedy": lambda s: AssignmentPolicy.links_greedy(s, cap=1),
    "greedy-2": lambda s: AssignmentPolicy.links_greedy(s, cap=2),
    "hungarian": AssignmentPolicy.links_hungarian,
}
#: Kept for callers that want the mapping without naming the class.
ASSIGNMENT = AssignmentPolicy.POLICIES


class Reranker:
    """Re-ranking a score table after it is produced: the identity bonus, the
    null scale, the lemma re-rank of the shortlist.

    Example:
        ```python
        scores = Reranker.rerank(scores, examples, weight=2.0)
        scores = Reranker.with_null_scale(scores, factor=1.5)
        ```
    """

    @staticmethod
    def rerank(scores, examples, weight: float):
        """Add ``weight`` to every candidate spelled the same as the reuse word.

        Works in logit space -- ``log p`` differs from the pre-softmax score only by
        a constant, and softmax is shift-invariant -- then renormalises, so the
        result is what the model would have produced had the bonus been part of the
        score all along.

        The bonus fires on identity alone, so a genuine substitution is untouched:
        where the words differ the bracket is zero and the candidate keeps its
        score. It also cannot separate one *et* from another, since every copy
        earns the same bonus; that is left to the context term and the assignment.
        """
        if weight == 0.0:
            return scores
        out = []
        for per_word, example in zip(scores, examples):
            source = [normalize(w) for w in example.source_tokens]
            rescored = []
            for word, candidates in enumerate(per_word):
                # A reuse word can arrive with no candidates at all -- the pair was
                # truncated past it, or the source side is empty. Leave it alone
                # rather than renormalising an empty distribution.
                if not candidates:
                    rescored.append([])
                    continue
                target = (
                    normalize(example.target_tokens[word])
                    if word < len(example.target_tokens)
                    else None
                )
                bumped = [
                    (
                        s,
                        math.log(max(p, 1e-12))
                        + (weight if s >= 0 and source[s] == target else 0.0),
                    )
                    for s, p in candidates
                ]
                top = max(logp for _, logp in bumped)
                weights = [(s, math.exp(logp - top)) for s, logp in bumped]
                total = sum(v for _, v in weights) or 1.0
                rescored.append(sorted(((s, v / total) for s, v in weights), key=lambda c: -c[1]))
            out.append(rescored)
        return out

    @staticmethod
    def with_null_scale(scores, factor):
        """Re-weight the null against every candidate.

        ``factor`` < 1 makes declining cheaper, > 1 makes it dearer. Applied to the
        probability then renormalised, so the ordering among the real candidates is
        untouched and only the align-or-decline decision moves.
        """
        out = []
        for per_word in scores:
            rescored = []
            for candidates in per_word:
                if not candidates:
                    rescored.append([])
                    continue
                bumped = [(s, p * (factor if s < 0 else 1.0)) for s, p in candidates]
                total = sum(p for _, p in bumped) or 1.0
                rescored.append(sorted(((s, p / total) for s, p in bumped), key=lambda c: -c[1]))
            out.append(rescored)
        return out

    @classmethod
    def similarity_rerank(cls, scores, examples, morph, vectors, weight, top_k, cache):
        """Nudge the top-k real candidates by lemma similarity, and only those.

        Restricting to the shortlist is the whole point: it is where the correct
        answer already lives, and it is what keeps the vectors' poor global
        discrimination from mattering. A candidate the context term has excluded is
        never reconsidered, however similar it looks.
        """
        if weight == 0.0:
            return scores
        out = []
        for per_word, example in zip(scores, examples):
            rescored = []
            for word, candidates in enumerate(per_word):
                if not candidates:
                    rescored.append([])
                    continue
                target = example.target_tokens[word] if word < len(example.target_tokens) else None
                real = [c for c in candidates if c[0] >= 0][:top_k]
                eligible = {s for s, _ in real}
                bumped = []
                for s, p in candidates:
                    bonus = 0.0
                    if s in eligible and target is not None:
                        sim = cls._similarity(
                            example.source_tokens[s], target, morph, vectors, cache
                        )
                        if sim is not None:
                            bonus = weight * sim
                    bumped.append((s, math.log(max(p, 1e-12)) + bonus))
                top = max(v for _, v in bumped)
                exp = [(s, math.exp(v - top)) for s, v in bumped]
                total = sum(p for _, p in exp) or 1.0
                rescored.append(sorted(((s, p / total) for s, p in exp), key=lambda c: -c[1]))
            out.append(rescored)
        return out

    @staticmethod
    def _similarity(source, target, morph, vectors, cache):
        """Cosine between lemmas, or ``None`` when either is out of vocabulary."""
        key = (source, target)
        if key in cache:
            return cache[key]
        value = None
        try:
            a, b = morph.lemma(source), morph.lemma(target)
            if a and b:
                # Same lemma is maximal similarity, not missing similarity. Handing
                # it None gives every *worse* candidate a bonus the right one never
                # receives, which is a penalty on MORPH wearing a bonus's clothes.
                value = 1.0 if a == b else vectors.similarity(a, b)
        except Exception:
            value = None
        cache[key] = value
        return value
