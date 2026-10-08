# datasets/synthetic/substitution.py
"""The substitution sources: attestation, acceptance calibration, the untyped probe, the typed source."""

from __future__ import annotations

from typing import Dict, List, Optional

from retexo.core.normalize import normalize
from retexo.datasets.synthetic.inventory import RELATED_TAGS, RESIDUAL_MAX_COS, TYPED_WEIGHTS
from retexo.datasets.synthetic.shape import TARGET

# ---------- Attestation and acceptance ----------


def attested_forms(passages, *, min_count: int = 3) -> set:
    """Normalized forms the corpus contains at least ``min_count`` times.

    The floor matters as much as the filter: a hapax is technically attested and
    still reads as odd, which leaves the same tell in weaker form.
    """
    counts = {}
    for passage in passages:
        for word in passage:
            key = normalize(word)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return {form for form, n in counts.items() if n >= min_count}


def _make_substitute(vectors_path, attested=None, accept=1.0):
    """The untyped substitution function ``calibrate_acceptance`` measures.

    ``accept`` thins the successes: see ``calibrate_acceptance``. Thinning here
    rather than in the caller keeps the fragment loop a plain "try every token",
    which is the only shape that reaches the target rate on the one-token
    fragments that are 62% of the corpus.
    """
    from retexo.datasets.generation import LexicalSubstitutionSource
    from retexo.resources import Resources

    resources = Resources(offline=True, vectors_path=vectors_path)
    source = LexicalSubstitutionSource(resources, fallback=None)
    tags = ("SYN", "SYN-DIST", "HYPER", "HYPO", "ANT", "MORPH")

    def substitute(token, rng):
        if accept < 1.0 and rng.random() >= accept:
            return None
        for tag in rng.sample(tags, len(tags)):
            replacement = source.replacement(token, tag, rng)
            if not replacement:
                continue
            if attested is not None and normalize(replacement) not in attested:
                continue  # a form no author wrote is not a substitution
            return replacement
        return None

    return substitute


def calibrate_acceptance(
    passages, vectors_path=None, attested=None, *, target=None, sample=3000, seed=11
):
    """Acceptance rate that makes the realized SUBST share meet the target.

    Only some tokens have a substitute the filter will pass, so proposing at the
    target rate realizes ``target x coverage``. Measuring coverage and dividing
    it out means the filter can be tightened without also thinning the class --
    the two were previously confounded, and tightening the filter looked like it
    had cost 55% of the SUBST examples when it had only moved this number.
    """
    import random as _random

    target = TARGET["subst_share"] if target is None else target
    probe = _make_substitute(vectors_path, attested, accept=1.0)
    rng = _random.Random(seed)
    tokens = [word for passage in passages for word in passage]
    rng.shuffle(tokens)
    tokens = tokens[:sample]
    hits = 0
    for token in tokens:
        replacement = probe(token, rng)
        if replacement and normalize(replacement) != normalize(token):
            hits += 1
    coverage = hits / max(1, len(tokens))
    if coverage <= 0:
        return 1.0, 0.0
    return min(1.0, target / coverage), coverage


# =============================================================================
# Typed substitution
# =============================================================================


def _make_typed_substitute(
    vectors_path,
    attested=None,
    accept=1.0,
    mlm=None,
    subst_weight=None,
    cohypo=False,
    norel_share=0.0,
    stem_subst=0.0,
    lexical_share=None,
    rel_share=None,
):
    """Like ``_make_substitute``, but returns the tag beside the replacement.

    ``mlm`` (E38): a ContextualSubstituter; the SUBST tag then draws a word the
    masked LM finds likely in the slot instead of a random same-POS word. ``subst_weight``
    re-weights the SUBST tag (the others scaled to keep the sum)."""
    from retexo.datasets.generation import LexicalSubstitutionSource
    from retexo.resources import Resources

    resources = Resources(offline=True, vectors_path=vectors_path)
    source = LexicalSubstitutionSource(resources, fallback=None, cohypo=cohypo)
    tags = list(TYPED_WEIGHTS)
    weights = [TYPED_WEIGHTS[t] for t in tags]
    if subst_weight is not None:
        rest = sum(w for t, w in zip(tags, weights) if t != "SUBST")
        weights = [
            subst_weight if t == "SUBST" else w * (1 - subst_weight) / rest
            for t, w in zip(tags, weights)
        ]
    if lexical_share is not None:
        # the error analysis of 2026-09-26: the share of every lexical tag together against MORPH; 0 = inflections
        # only, the synthetic data then teaches no substitution at all
        rest = sum(w for t, w in zip(tags, weights) if t != "MORPH")
        weights = [
            (1 - lexical_share) if t == "MORPH" else w * lexical_share / rest
            for t, w in zip(tags, weights)
        ]
    names: Optional[List[str]] = None
    # ``rel_share`` (2026-09-27, Synthetic Stage): the realised share of relation-bearing substitutions among the
    # lexical ones. A lexical draw tries the related tags while the running share is below the target and, if none
    # is found, substitutes nothing (the word stays a copy) instead of falling back to a slot filler; above the
    # target it takes the slot filler. So the share is met exactly, at the cost of fewer substitutions.
    counts = {"related": 0, "lexical": 0}
    related_tags = [t for t in tags if t in RELATED_TAGS]
    related_weights = [w for t, w in zip(tags, weights) if t in RELATED_TAGS]

    def name_pool():
        nonlocal names
        if names is None:
            names = []
            if resources.has("entities") and attested is not None:
                names = sorted(n for n in resources.entities.names if normalize(n) in attested)
        return names

    def ne_sub(token, rng):
        if not resources.has("entities") or not resources.entities.is_name(token):
            return None
        pool = name_pool()
        if len(pool) < 2:
            return None
        for _ in range(8):
            other = rng.choice(pool)
            if normalize(other) != normalize(token):
                return other
        return None

    def pos_shift(token, rng):
        if not resources.has("derivation"):
            return None
        lemma = resources.lemma_or_surface(token)
        for pos in resources.pos_candidates(token):
            options = resources.wordnet.lookup(lemma, pos).get("derivatives", [])
            options = [o for o in options if normalize(o) != normalize(token)]
            if options:
                return rng.choice(options)
        return None

    attested_list: Optional[List[str]] = None

    def residual(token, rng):
        """A real word in the same slot with no relation the resources attest.

        Same part-of-speech guess, at least four letters, not a name, not
        distributionally close, not the same lemma. What a parallel
        replacement, a textual variant or an allegorical substitution looks
        like from the typer's side: changed, and nothing named fits.
        """
        nonlocal attested_list
        if attested is None:
            return None
        if attested_list is None:
            attested_list = sorted(a for a in attested if len(a) >= 4)
        if len(attested_list) < 10:
            return None
        pos = resources.pos_candidates(token)
        pos0 = pos[0] if pos else None
        lemma = resources.lemma_or_surface(token)
        for _ in range(4):
            cand = rng.choice(attested_list)
            if normalize(cand) == normalize(token):
                continue
            if resources.has("entities") and resources.entities.is_name(cand):
                continue
            if resources.lemma_or_surface(cand) == lemma:
                continue
            cpos = resources.pos_candidates(cand)
            if pos0 and cpos and cpos[0] != pos0:
                continue
            if resources.has("vectors"):
                sim = resources.vectors.similarity(lemma, resources.lemma_or_surface(cand))
                if sim is not None and sim >= RESIDUAL_MAX_COS:
                    continue
            return cand
        return None

    stem_index: Optional[Dict[str, List[str]]] = None

    def stem_family(token, rng):
        """Failure-mode row 15 (``stem_subst``): a derivational relative by surface stem, no
        dictionary -- an attested word that carries the token's first five letters at offset 0
        to 3 (*luxuriem / luxuriant*, *lapidea / lapidem*, *trahebat / extrahit*), another lemma,
        not a name. What the POS misses of fold 4 look like, and what ``wn_deriv`` (3,629 pairs)
        never covers; the grid's ``prefix_ratio`` / ``stem_match`` cells are the features it teaches."""
        nonlocal stem_index
        if attested is None:
            return None
        if stem_index is None:
            stem_index = {}
            for word in attested:
                if len(word) < 5:
                    continue
                for offset in range(0, min(4, len(word) - 4)):
                    stem_index.setdefault(word[offset : offset + 5], []).append(word)
        norm = normalize(token)
        if len(norm) < 5:
            return None
        cands = stem_index.get(norm[:5])
        if not cands or len(cands) < 2:
            return None
        lemma = resources.lemma_or_surface(token)
        for _ in range(6):
            cand = rng.choice(cands)
            if normalize(cand) == norm or resources.lemma_or_surface(cand) == lemma:
                continue
            if resources.has("entities") and resources.entities.is_name(cand):
                continue
            return cand
        return None

    def substitute(token, rng, context=None, force=False):
        """``force`` (E38c dense rewrites): no acceptance gate, the contextual SUBST
        source first, then the others."""
        if not force and accept < 1.0 and rng.random() >= accept:
            return None
        if stem_subst > 0 and rng.random() < stem_subst:
            relative = stem_family(token, rng)
            if relative and (attested is None or normalize(relative) in attested):
                return relative, "POS"
        order = rng.choices(tags, weights=weights, k=len(tags))
        if rel_share is not None and not force:
            # the lexical part of the order follows the target, also after a failed MORPH draw (the fallback would
            # otherwise take the generator's own weights and leak its related share, 0.46 in the smoke)
            want_related = counts["related"] < rel_share * (counts["lexical"] + 1)
            lexical = (
                rng.choices(related_tags, weights=related_weights, k=len(related_tags))
                if want_related
                else ["SUBST"]
            )
            order = (["MORPH"] if order[0] == "MORPH" else []) + lexical
        if force:
            order = ["SUBST"] + [t for t in order if t != "SUBST"]
        seen = set()
        for tag in order:
            if tag in seen:
                continue
            seen.add(tag)
            if tag == "NE-SUB":
                replacement = ne_sub(token, rng)
            elif tag == "POS":
                replacement = pos_shift(token, rng)
            elif tag == "SUBST":
                if mlm is not None and not (norel_share > 0 and rng.random() < norel_share):
                    # E38: the slot-fitting word or nothing -- no random fallback
                    replacement = (
                        mlm.propose(
                            context[0], context[1], rng, attested=attested, resources=resources
                        )
                        if context is not None
                        else None
                    )
                elif mlm is not None:
                    # failure-mode row 15 (``norel_share``): the relation-free residual beside the
                    # slot filler -- a real same-POS word with no attested relation and no vector
                    # closeness, so "changed, and nothing named fits" is a linked case the model has seen
                    replacement = residual(token, rng)
                else:
                    replacement = residual(token, rng)
            else:
                replacement = source.replacement(token, tag, rng)
            if not replacement or normalize(replacement) == normalize(token):
                continue
            if attested is not None and normalize(replacement) not in attested:
                continue
            if rel_share is not None and tag != "MORPH":
                counts["lexical"] += 1
                counts["related"] += tag in RELATED_TAGS
            return replacement, tag
        return None

    return substitute, resources
