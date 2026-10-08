# retexo/edit_typing/attest.py
"""
Resource-first decisions: what the static resources *attest*, and where they are silent.

E26 asks a question the earlier experiments only answered in pieces: for how much
of an edit script do we already know the answer without a model? If a reuse word
has exactly one source word of the same form in the window, the link is not a
prediction. If two linked words share a lemma and differ in form, MORPH is not a
prediction. The typed pointer currently spends its capacity re-deriving both,
and then the identity bonus and the lemma re-rank push it toward what a lookup
would have said in the first place.

This module makes the split explicit, gathered on :class:`Attester`:

    Attester().attest_links(example, featurizer)   per reuse word: the tier of
                                                    evidence, the candidates at
                                                    that tier, and the unique
                                                    choice if there is one
    Attester().symbolic_links(examples, ...)       links from the resources alone
    Attester.combine_links(attested, scores)       the model's rows where the
                                                    resources are silent; the
                                                    resources' answer where not
    Attester.attest_type(phi)                      (type, attested) from the
                                                    evidence alone
    Attestation.regime()                           the bucket a word is scored in

The tiers are ordered by how far a Latinist would trust them without looking:

    FORM      identical after normalization           certain
    LEMMA     same lemma, different surface            a lemmatizer's word
    ENCLITIC  -que / -ue / -ne stem match              a rule's word
    RELATION  a wordnet relation, or two names         a hypothesis
    NONE      nothing attests a source                 the model's territory

A tier is *unique* when exactly one source word sits in it. With several (three
*et* in the source), the resources know the word is linked but not to which;
"order" picks the one nearest to where a monotone alignment would put it, the
model may pick among them, and both are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from retexo.edit_typing.link_features import FEATURE_NAMES, SymbolicTyper

# =============================================================================
# Tiers
# =============================================================================

NONE, RELATION, ENCLITIC, LEMMA, FORM = 0, 1, 2, 3, 4
TIER_NAMES = {
    NONE: "none",
    RELATION: "relation",
    ENCLITIC: "enclitic",
    LEMMA: "lemma",
    FORM: "form",
}

#: the score a candidate at each tier gets in the resource-only aligner; the
#: null sits between RELATION and ENCLITIC, so a relation alone never links
TIER_SCORE = {FORM: 1.0, LEMMA: 0.8, ENCLITIC: 0.7, RELATION: 0.4}
NULL_SCORE = 0.5

REGIMES = ("form:1", "form:n", "lemma:1", "lemma:n", "weak", "none")

_F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def pair_tier(phi: Sequence[float]) -> int:
    """The highest tier the evidence vector supports for one (source, reuse) pair."""
    if phi[_F["same_form"]]:
        return FORM
    if phi[_F["same_lemma"]]:
        return LEMMA
    if phi[_F["enclitic_stem_match"]] and phi[_F["enclitic_src"]] != phi[_F["enclitic_tgt"]]:
        return ENCLITIC
    if phi[_F["wn_any"]] or phi[_F["both_names"]]:
        return RELATION
    return NONE


@dataclass
class Attestation:
    """What the resources say about one reuse word."""

    tier: int = NONE
    candidates: List[int] = field(default_factory=list)  # source indices at ``tier``
    by_order: int = -1  # nearest to the monotone slot
    phi: Dict[int, List[float]] = field(default_factory=dict)  # evidence per candidate

    @property
    def unique(self) -> bool:
        return len(self.candidates) == 1

    @property
    def chosen(self) -> int:
        """The one candidate, or -1 when the resources leave a choice open."""
        return self.candidates[0] if self.unique else -1

    def regime(self) -> str:
        """The bucket this reuse word is scored in: how much the resources knew."""
        if self.tier == FORM:
            return "form:1" if self.unique else "form:n"
        if self.tier == LEMMA:
            return "lemma:1" if self.unique else "lemma:n"
        if self.tier in (ENCLITIC, RELATION):
            return "weak"
        return "none"


def phi_source(example, featurizer=None):
    """``phi(s, t)`` for one example: the precomputed pair grid when the example
    carries one (``featurize_pairs``), the featurizer otherwise."""
    pf = getattr(example, "pair_features", None)
    if pf is not None:
        return lambda s, t: pf[t, s].tolist()
    if featurizer is None:
        raise ValueError("no pair_features on the example and no featurizer given")
    source, target = example.source_tokens, example.target_tokens
    n_s, n_t = len(source), len(target)
    return lambda s, t: featurizer(source[s], target[t], s, t, n_s, n_t)


#: what attest_type returns when nothing attests a relation
OPEN_TYPES = ("SUBST", "SYN-DIST")


class Attester:
    """Resource-first attestation: what the static resources already know about
    a pair, so a model only has to answer where they are silent.

    Args:
        min_tier: The weakest tier that counts as attested; below it a reuse
            word is the model's.
        order: Whether :meth:`symbolic_scores` breaks ties among same-tier
            candidates by nearness to the monotone slot (E17's "+order"
            baseline).

    Example:
        ```python
        attester = Attester(min_tier=LEMMA)
        attested = attester.attest_links(example, featurizer)
        links = attester.symbolic_links(examples, featurizer)
        ```
    """

    def __init__(self, *, min_tier: int = LEMMA, order: bool = True):
        self.min_tier = min_tier
        self.order = order

    @staticmethod
    def pair_tier(phi: Sequence[float]) -> int:
        return pair_tier(phi)

    def attest_links(self, example, featurizer=None) -> List[Attestation]:
        """Per reuse word, the best-attested source candidates.

        Below :attr:`min_tier` the word is the model's. Candidates at weaker
        tiers are still recorded in ``phi`` so the type decision can use them
        later.
        """
        source, target = example.source_tokens, example.target_tokens
        n_s, n_t = len(source), len(target)
        phi_of = phi_source(example, featurizer)
        out = []
        for t, _token in enumerate(target):
            best, cands, phis = NONE, [], {}
            for s, _src in enumerate(source):
                phi = phi_of(s, t)
                tier = pair_tier(phi)
                if tier == NONE:
                    continue
                phis[s] = phi
                if tier > best:
                    best, cands = tier, [s]
                elif tier == best:
                    cands.append(s)
            att = Attestation(
                tier=best, candidates=cands if best >= self.min_tier else [], phi=phis
            )
            if att.candidates:
                expected = t * n_s / max(n_t, 1)
                att.by_order = min(att.candidates, key=lambda s: abs(s - expected))
            out.append(att)
        return out

    # ---------- links from the resources alone ----------

    def symbolic_scores(self, example, featurizer) -> List[List[Tuple[int, float]]]:
        """Per reuse word, [(source, score)...] best first, no model anywhere.

        E17's "+order" baseline generalised to the tiers: a candidate scores its
        tier, minus a hair per position it sits from the monotone slot, and the
        null sits at a fixed floor. The Hungarian assignment then makes it
        one-to-one, exactly as it does for the model.
        """
        source, target = example.source_tokens, example.target_tokens
        n_s, n_t = len(source), len(target)
        phi_of = phi_source(example, featurizer)
        rows = []
        for t, _token in enumerate(target):
            row = [(-1, NULL_SCORE)]
            expected = t * n_s / max(n_t, 1)
            for s, _src in enumerate(source):
                tier = pair_tier(phi_of(s, t))
                if tier == NONE or tier < self.min_tier:
                    continue
                score = TIER_SCORE[tier]
                if self.order:
                    score -= 0.001 * abs(s - expected)
                row.append((s, score))
            rows.append(sorted(row, key=lambda c: -c[1]))
        return rows

    def symbolic_links(self, examples, featurizer=None) -> List[List[int]]:
        from retexo.aligners.assignment import AssignmentPolicy

        return AssignmentPolicy.links_hungarian(
            [self.symbolic_scores(ex, featurizer) for ex in examples]
        )

    # ---------- combining: the resources where they speak, the model where they do not ----------

    @staticmethod
    def combine_scores(
        attested: List[Attestation],
        model_row: List[List[Tuple[int, float]]],
        *,
        mode: str = "whether",
    ) -> List[List[Tuple[int, float]]]:
        """One example's model rows, restricted by the attestations.

        The measurement behind the modes: the resources know *where* (1,026 of
        1,028 unique attestations are the gold source) but not *whether* (413
        attested words are unlinked in the gold: frames and coincidental function
        words). So the default keeps the model's null.

        mode "whether"   the model's probabilities over {attested candidates, null},
                         renormalised -- resources decide where, the model whether
                         (and which, when several candidates tie)
        mode "fix"       a unique attestation is linked outright; several are left
                         to the model entirely; the null is gone for unique words
        mode "order"     like "fix", with several settled by order (E17's baseline)
        """
        out = []
        for att, row in zip(attested, model_row):
            if not att.candidates:
                out.append(row)
                continue
            if mode in ("fix", "order"):
                if att.unique or mode == "order":
                    s = att.chosen if att.unique else att.by_order
                    out.append([(s, 1.0), (-1, 0.0)])
                else:
                    out.append(row)
                continue
            probs = dict(row)
            keep = {s: probs.get(s, 0.0) for s in att.candidates}
            keep[-1] = probs.get(-1, 0.0)
            total = sum(keep.values()) or 1.0
            out.append(sorted([(s, p / total) for s, p in keep.items()], key=lambda c: -c[1]))
        return out

    @staticmethod
    def combine_links(attested_all, model_scores, **kw) -> List[List[int]]:
        from retexo.aligners.assignment import AssignmentPolicy

        return AssignmentPolicy.links_hungarian(
            [Attester.combine_scores(a, r, **kw) for a, r in zip(attested_all, model_scores)]
        )

    @staticmethod
    def ceiling_links(links: Sequence[int], gold_align) -> List[int]:
        """Resources decide where, the gold decides whether: the resources' own
        links, kept only where the gold links the word. The ceiling of any system
        whose links on attested words come from the resources."""
        return [
            s if (gold_align is not None and t < len(gold_align) and gold_align[t] >= 0) else -1
            for t, s in enumerate(links)
        ]

    def training_restriction(self, example, attested: List[Attestation]):
        """What route B hands the loss: per reuse word the allowed sources (the
        attested candidates, plus the gold source if the resources missed it -- the
        labels are known at training time), or None for an open word; and whether
        the type at the gold link is open (the lookup did not name it)."""
        gold = example.alignments
        allowed, type_open = [], []
        for t, att in enumerate(attested):
            s = gold[t] if gold is not None and t < len(gold) else -1
            if att.candidates:
                ok = list(att.candidates)
                if s >= 0 and s not in ok:
                    ok.append(s)
                allowed.append(ok)
            else:
                allowed.append(None)
            if s >= 0 and s in att.phi:
                type_open.append(self.attest_type(att.phi[s])[1] is False)
            else:
                type_open.append(True)
        return allowed, type_open

    # ---------- types ----------

    @staticmethod
    def attest_type(phi: Sequence[float]) -> Tuple[str, bool]:
        """The evidence's own name for a link, and whether it is attested at all.

        SYN-DIST (a vector cosine over a threshold) is a guess, not a lookup; it
        counts as open, like the residual SUBST.
        """
        said = SymbolicTyper()(phi)
        return said, said not in OPEN_TYPES

    @staticmethod
    def combine_types(said: str, attested: bool, model_type: str) -> str:
        return said if attested else model_type


class FrameLexicon:
    """Frame flags from attribution formulas seen in the training folds.

    A reuse span is a frame if it matches a template word for word after
    normalization. The weakest honest resource for frames: formulas repeat in
    shape (*ut ait X*, *quod et X testatur dicens*), far less in words.

    Example:
        ```python
        lexicon = FrameLexicon(templates, min_len=2)
        flags = lexicon.frames(example)
        ```
    """

    def __init__(self, templates: Sequence[Sequence[str]], *, min_len: int = 2):
        self.templates = templates
        self.min_len = min_len

    def frames(self, example) -> List[int]:
        from retexo.core.normalize import normalize

        words = [normalize(w) for w in example.target_tokens]
        flags = [0] * len(words)
        keyed: Dict[Tuple[str, ...], None] = {}
        for tpl in self.templates:
            norm = tuple(normalize(w) for w in tpl)
            if len(norm) >= self.min_len:
                keyed[norm] = None
        longest = max((len(k) for k in keyed), default=0)
        for a in range(len(words)):
            for n in range(min(longest, len(words) - a), self.min_len - 1, -1):
                if tuple(words[a : a + n]) in keyed:
                    for i in range(a, a + n):
                        flags[i] = 1
                    break
        return flags
