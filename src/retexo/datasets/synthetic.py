# retexo/datasets/synthetic.py
"""
The typed synthetic generator: shaped like real reuse, typed like real reuse.

Two preliminary experiments, merged. **E4** (``attic/retexo/e4.py``)
established the shape: four operations decidable without a lexicon (``COPY``,
``SUBST``, ``INS``, ``DEL``), generated to the measured statistics of real
reuse -- short scattered fragments inside real prose, not label balance alone
-- because a generator matching only the label mix reached synthetic macro
0.98 while gold macro sat at 0.075. **E24** kept that shape and added
everything it left out: typed links (the substitute's tag -- ``MORPH``,
``SYN``, ``HYPER``, ``HYPO``, ``ANT``, ``SYN-DIST``, ``NE-SUB``, ``POS`` --
carried through with its evidence vector), ``SPLIT``/``MERGE`` (an enclitic
detached or attached), ``FRAME`` (an attribution formula drawn from the
*training folds'* hand-labelled frames), and switchable ``REORDER``.
``QUOTE``, ``ADAPT`` and ``DISPERSE`` need no construction: they fall out of
the alignment at decode time, on synthetic and real pairs alike.

The two are one system in practice -- E24 always calls through to E4's
distributions and its acceptance calibration -- so they live in one module
under one name. What did not survive the merge: E4's own single-typed
generator ``generate`` (superseded by ``generate_typed`` below), its report
dataclass, and its realism-comparison helpers' now-dead call sites; those stay
in ``attic/retexo/e4.py`` for the record. Everything importable from here
today was load-bearing before the merge.

    pool, report = generate_typed(seeds, context, frames=frame_pool(...), ...)

or, through the thin config object:

    generator = TypedGenerator(GeneratorConfig(frames=frame_pool(...)))
    pool, report = generator.generate(seeds, context)
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.formulations.change_detector import ChangeExample
from retexo.edit_typing.link_features import LinkFeaturizer
from retexo.core.normalize import normalize
from retexo.edit_typing.dep_features import DEP_FEATURES, DependencyParser

# =============================================================================
# The untyped shape (E4): operations, and the measured targets
# =============================================================================

OPERATIONS = ("COPY", "SUBST", "INS")   # reuse side; DEL has its own head

#: Measured from data/gold/labels.json. Generation aims at these, and the run record
#: reports what it actually produced so drift stays visible.
TARGET = {
    # Share of *aligned* tokens reworked rather than copied. The hand labels put
    # SUBST at 3.9% of reuse tokens against COPY's 13.2%, so 0.228 of the tokens
    # that align. With the deficit carry above this is realized, not merely
    # asked for, which is why it is the measured number and not a higher guess.
    "subst_share": 0.228,
}

#: How many contiguous aligned fragments a pair has, measured over the 100 hand
#: labels: 62 pairs have one, 31 have two, 3 have three, 2 have four, 2 have none.
FRAGMENTS = ((1, 0.62), (2, 0.31), (3, 0.03), (4, 0.02), (0, 0.02))

#: Fragment length, same source. **89 of 134 fragments are a single token** --
#: a lone shared word, not a phrase. A generator that only makes contiguous
#: runs of several tokens produces the wrong phenomenon, especially for `cf.`
#: pairs, whose fragments average 1.1 tokens.
#: The tail matters: cit32 in the hand labels is a twenty-token verbatim
#: quotation. Capping at eight starves the model of the long runs that make a
#: citation a citation.
FRAGMENT_LENGTHS = ((1, 0.62), (2, 0.18), (3, 0.03), (4, 0.05), (5, 0.02),
                    (6, 0.02), (7, 0.02), (8, 0.01), (11, 0.02), (14, 0.02),
                    (20, 0.01))

#: Share of COPY tokens whose surface differs from the source's, measured over
#: the hand labels: **75 of 255 aligned copies, 29.4%**. Punctuation and
#: capitalisation dominate -- *fuit;* against *fuit:*, *Ille* against *ille* --
#: with orthographic variation (*tura*/*thura*, *vultus*/*uultus*) behind them.
#: Generating COPY only as an exact string match teaches the model that a copy
#: is a character-identical token, which it is not: two of the five errors in
#: the first inspected epoch were exactly this.
COPY_VARIANT_SHARE = 0.40

#: Applied in the direction `normalize` folds *away*, so the variant and the
#: original still normalize to the same string and the COPY label stays true.
#: Aspiration is word-initial only: it marks a Greek loan (*thura*, *Chalybes*),
#: so turning *fuit* into *fuith* would round-trip correctly while teaching the
#: model a string no editor has ever printed.
_INITIAL = (("t", "th"), ("p", "ph"), ("c", "ch"), ("r", "rh"))
_ANYWHERE = (("i", "y"), ("mp", "np"), ("mb", "nb"))
_TRAILING = (",", ".", ";", ":", "?", "!", "")


def _variant(token: str, rng) -> str:
    """A differently-spelled token that still normalizes to the same string.

    Returns the token unchanged when no safe variant applies, so the caller can
    label it COPY either way.
    """
    body = token.rstrip(",.;:?!")
    if not body:
        return token
    kind = rng.random()
    if kind < 0.45:                                   # punctuation
        candidate = body + rng.choice(_TRAILING)
    elif kind < 0.75:                                 # capitalisation
        candidate = (body.lower() if body[:1].isupper() else body.capitalize()) \
                    + token[len(body):]
    elif rng.random() < 0.5:                          # aspiration, word-initial
        source, target = rng.choice(_INITIAL)
        if not body.lower().startswith(source):
            return token
        candidate = target + body[len(source):] + token[len(body):]
        if body[:1].isupper():
            candidate = candidate.capitalize()
    else:                                             # medial variation
        source, target = rng.choice(_ANYWHERE)
        if source not in body.lower():
            return token
        at = body.lower().index(source)
        candidate = body[:at] + target + body[at + len(source):] + token[len(body):]
    # Only accept a variant the comparison actually folds; otherwise the label
    # would be wrong, which is worse than a missing example.
    return candidate if normalize(candidate) == normalize(token) else token


def _draw(distribution, rng):
    r, acc = rng.random(), 0.0
    for value, weight in distribution:
        acc += weight
        if r <= acc:
            return value
    return distribution[-1][0]


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
                continue        # a form no author wrote is not a substitution
            return replacement
        return None

    return substitute


def calibrate_acceptance(passages, vectors_path=None, attested=None, *,
                         target=None, sample=3000, seed=11):
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


def coarse(op: str) -> str:
    """The E4-era three-way collapse: ``COPY`` / ``SUBST`` / ``INS``.

    Not the same thing as ``retexo.baselines.labels``' canonical levels --
    this folds ``MORPH`` into ``SUBST`` (labels.py's V1 keeps them apart) -- and
    it is kept only because ``fine_from_gold`` uses it to give the untyped
    pointer head exactly the supervision E4-era training gave it. Do not reach
    for it as a general-purpose collapse; use ``retexo.baselines.labels``
    for that.
    """
    return {"NOP": "COPY", "REORDER": "COPY", "QUOTE": "COPY",
            "MORPH": "SUBST", "FRAME": "INS"}.get(op, op if op in OPERATIONS else "SUBST")


def _fragments(operations):
    """Contiguous runs of aligned (non-INS) tokens."""
    runs, current = [], 0
    for op in operations:
        if op != "INS":
            current += 1
        elif current:
            runs.append(current); current = 0
    if current:
        runs.append(current)
    return runs


def realism_report(examples, gold, vocabulary) -> dict:
    """Compare generated data against the hand labels, statistic by statistic.

    Three runs have now failed because the synthetic data was easier than
    reality in a way nobody had measured first: random filler let the model find
    aligned tokens by spotting incoherence; a single contiguous fragment taught
    the wrong shape when two thirds of real fragments are one token; and
    substitutes Collatinus invented (*simpliior*, *mobiliium*) let the model
    find SUBST by spotting a non-word.

    Each of those was discovered *after* a training run. This computes every
    statistic we have had to learn the hard way and prints it beside the gold
    value, so drift is visible in the first minute instead of the fourth hour.
    """
    import statistics

    def token_share(ops_lists):
        counts = {}
        for ops in ops_lists:
            for op in ops:
                counts[op] = counts.get(op, 0) + 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    gen_ops = [e.operations for e in examples]
    gold_ops = [g.target_ops for g in gold]
    gen, ref = token_share(gen_ops), token_share(gold_ops)

    gen_frag = [_fragments(o) for o in gen_ops]
    ref_frag = [_fragments(o) for o in gold_ops]
    gen_lengths = [l for f in gen_frag for l in f] or [0]
    ref_lengths = [l for f in ref_frag for l in f] or [0]

    gen_del = sum(sum(e.source_labels or []) for e in examples)
    gen_src = sum(len(e.source_labels or []) for e in examples) or 1
    ref_del = sum(sum(g.source_del) for g in gold)
    ref_src = sum(len(g.source_del) for g in gold) or 1

    # COPY tokens whose surface differs from the source's
    def copy_variant_share(items, ops_lists, source_of):
        exact = variant = 0
        for item, ops in zip(items, ops_lists):
            surfaces = set(source_of(item))
            normalized = {normalize(w) for w in source_of(item)}
            for token, op in zip(item.target_tokens, ops):
                if op != "COPY":
                    continue
                if token in surfaces:
                    exact += 1
                elif normalize(token) in normalized:
                    variant += 1
        return variant / max(exact + variant, 1)

    substitutes = [t for e in examples for t, o in zip(e.target_tokens, e.operations)
                   if o == "SUBST"]
    attested = sum(1 for w in substitutes if normalize(w) in vocabulary)

    return {
        "COPY share": (gen.get("COPY", 0), ref.get("COPY", 0)),
        "SUBST share": (gen.get("SUBST", 0), ref.get("SUBST", 0)),
        "INS share": (gen.get("INS", 0), ref.get("INS", 0)),
        "DEL share": (gen_del / gen_src, ref_del / ref_src),
        "fragments/pair": (statistics.mean([len(f) for f in gen_frag] or [0]),
                           statistics.mean([len(f) for f in ref_frag] or [0])),
        "fragment length": (statistics.mean(gen_lengths), statistics.mean(ref_lengths)),
        "source length": (statistics.mean([len(e.source_tokens) for e in examples]),
                          statistics.mean([len(g.source_tokens) for g in gold])),
        "target length": (statistics.mean([len(e.target_tokens) for e in examples]),
                          statistics.mean([len(g.target_tokens) for g in gold])),
        "COPY differing surface": (
            copy_variant_share(examples, gen_ops, lambda e: e.source_tokens),
            copy_variant_share(gold, gold_ops, lambda g: g.source_tokens)),
        "substitutes attested": (attested / max(len(substitutes), 1), 1.0),
    }


def log_realism(report, log, tolerance: float = 0.25) -> None:
    """Print the comparison, flagging anything that has drifted."""
    log("  realism check (generated vs hand labels)")
    log(f"    {'statistic':<24}{'generated':>11}{'gold':>10}   ")
    for name, (generated, reference) in report.items():
        scale = max(abs(reference), 1e-9)
        drift = abs(generated - reference) / scale
        mark = "  <-- DRIFT" if drift > tolerance else ""
        fmt = ".1%" if "share" in name or "attested" in name or "surface" in name else ".1f"
        log(f"    {name:<24}{format(generated, fmt):>11}{format(reference, fmt):>10}{mark}")


def all_insert_baseline(examples):
    """Predict INS for every reuse token, DEL for every source token.

    The degenerate script, and a strong one: it is 83% correct on the reuse side
    of real pairs.
    """
    return ([["INS"] * len(e.target_tokens) for e in examples],
            [[1] * len(e.source_tokens) for e in examples])


# =============================================================================
# The typed inventory (E24)
# =============================================================================

#: Fine tags a link can carry. NOP is a copy; SUBST is a lexical change that no
#: named relation fits -- a class the typer must be able to *say*, because on
#: real links the reader says it for two in five (adjudication, fold 4) and a
#: head that lacks it spends those on the nearest relation it knows.
FINE_OPERATIONS = ("NOP", "MORPH", "SYN", "HYPER", "HYPO", "ANT", "SYN-DIST",
                   "NE-SUB", "POS", "SPLIT", "MERGE", "SUBST")

#: Tags a typed substitution source is asked for, in the mix the oracle finds on
#: real pairs (generation.DEFAULT_WEIGHTS), renormalised over what it can
#: realise. MORPH dominates because inflection dominates real substitution: the
#: hand labels carry 901 MORPH against 415 lexical SUBST. SUBST here is the
#: residual: a real word in the same slot with no attested relation and no
#: distributional closeness.
TYPED_WEIGHTS = {"MORPH": 0.50, "SYN": 0.14, "SYN-DIST": 0.07, "HYPER": 0.05,
                 "HYPO": 0.04, "ANT": 0.03, "NE-SUB": 0.04, "POS": 0.04,
                 "SUBST": 0.09}

#: Cosine above which a candidate is too close to count as unrelated.
RESIDUAL_MAX_COS = 0.35

#: Share of pairs that carry an attribution formula: 153 of 1,490 hand-labelled
#: pairs (10.3%), 150 of them citations. Applied per generated pair.
FRAME_RATE = 0.10

#: Enclitic events per linked token. The featurizer's criterion finds 41 such
#: links among fold 4's ~1,270 (3.2%); at 0.04 the generator realised 0.7%,
#: because it can only detach an enclitic from a word that carries one.
ENCLITIC_RATE = 0.15

ENCLITICS = ("que", "ue", "ne")

#: Fine tags that a random draw of 20k pairs starves: at the measured rates a
#: 20k pool holds ~160 HYPER links, ~60 HYPO, ~50 ANT, ~10 SPLIT.
RARE_TAGS = ("HYPER", "HYPO", "ANT", "NE-SUB", "SPLIT", "MERGE", "POS", "SYN", "SUBST")


def select_pool(pool, size: int, rng, rare_min: int = 1200):
    """The training pairs: a random sample, with rare relations guaranteed.

    For each rare tag, up to ``rare_min`` pairs that contain it are kept
    first; the rest of the budget is a plain random sample. This moves the
    pool's substitution share above the calibrated 0.228 -- the realism report
    prints where it lands -- and it is the price of a typer that has seen a
    few hundred antonyms rather than fifty.
    """
    pool = list(pool)
    rng.shuffle(pool)
    chosen, taken = [], set()
    for tag in RARE_TAGS:
        n = 0
        for k, ex in enumerate(pool):
            if n >= rare_min or len(chosen) >= size:
                break
            if k in taken or tag not in ex.fine_operations:
                continue
            chosen.append(ex); taken.add(k); n += 1
    for k, ex in enumerate(pool):
        if len(chosen) >= size:
            break
        if k not in taken:
            chosen.append(ex); taken.add(k)
    rng.shuffle(chosen)
    return chosen


def balanced_subset(held, size: int, rng, per_rare: int = 60):
    """A held-out set where every fine class has enough links to score."""
    return select_pool(held, size, rng, rare_min=per_rare)


def fine_mix(examples) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        for op in ex.fine_operations or ():
            if op != "INS":
                counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# =============================================================================
# Frame templates
# =============================================================================

def frame_pool(gold_dir, folds_by_id: Dict[str, int], exclude_fold: int) -> List[List[str]]:
    """Attribution formulas from the hand labels, minus the held-out fold.

    The frames are the citing author's own words -- *quod et illustris poeta
    testatur dicens:* -- and the only honest source of their shape. Taking them
    from the folds the model trains on anyway keeps the held-out frames unseen.
    """
    import json
    from pathlib import Path

    gold_dir = Path(gold_dir)
    pairs = {p["id"]: p for p in json.loads((gold_dir / "sample_pairs.json").read_text())}
    labels = json.loads((gold_dir / "labels.json").read_text())
    out = []
    for key, label in labels.items():
        if folds_by_id.get(key) == exclude_fold:
            continue
        target = pairs[key]["target"].split()
        for a, b in label.get("frame", []):
            span = target[a:b + 1]
            if 1 <= len(span) <= 14:
                out.append(span)
    return out


# =============================================================================
# Typed substitution
# =============================================================================

def _make_typed_substitute(vectors_path, attested=None, accept=1.0, mlm=None, subst_weight=None, cohypo=False,
                           norel_share=0.0, stem_subst=0.0, lexical_share=None, rel_share=None):
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
        weights = [subst_weight if t == "SUBST" else w * (1 - subst_weight) / rest for t, w in zip(tags, weights)]
    if lexical_share is not None:
        # the error analysis of 2026-09-26: the share of every lexical tag together against MORPH; 0 = inflections
        # only, the synthetic data then teaches no substitution at all
        rest = sum(w for t, w in zip(tags, weights) if t != "MORPH")
        weights = [(1 - lexical_share) if t == "MORPH" else w * lexical_share / rest for t, w in zip(tags, weights)]
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
                names = sorted(n for n in resources.entities.names
                               if normalize(n) in attested)
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
                    stem_index.setdefault(word[offset:offset + 5], []).append(word)
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
            lexical = rng.choices(related_tags, weights=related_weights, k=len(related_tags)) if want_related else ["SUBST"]
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
                    replacement = mlm.propose(context[0], context[1], rng, attested=attested, resources=resources) \
                        if context is not None else None
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


# =============================================================================
# One example
# =============================================================================

@dataclass
class TypedReport:
    attempted: int = 0
    kept: int = 0
    no_substitute: int = 0
    op_counts: Dict[str, int] = field(default_factory=dict)
    fine_counts: Dict[str, int] = field(default_factory=dict)
    frames: int = 0
    reorders: int = 0
    enclitics: int = 0
    spelling: int = 0
    frameless: int = 0


#: Relation-bearing substitution tags (the generator's lexical tags other than the unrelated slot filler).
RELATED_TAGS = ("SYN", "SYN-DIST", "HYPER", "HYPO", "ANT", "NE-SUB", "POS")

#: Orthographic alternations the COPY gate folds (``Repairer.spelling_key``), in the direction that writes a
#: variant: (pattern, replacement) on the lowercased body; the result must keep the spelling key and change the form.
_SPELLING_RULES = (("ae", "e"), ("e", "ae"), ("oe", "e"), ("i", "y"), ("nt", "mpt"), ("mpt", "nt"), ("f", "ph"),
                   ("ph", "f"), (r"^([aeiou])", r"h\1"), (r"^h", ""), (r"([bcdfglmnprst])\1", r"\1"),
                   (r"([aeiou])([lmnrst])([aeiou])", r"\1\2\2\3"), (r"ii$", "i"), (r"i$", "ii"), (r"d$", "t"),
                   (r"t$", "d"), (r"ies$", "iens"), (r"iens$", "ies"), (r"m$", "n"))


def spelling_variant(token: str, rng, attested=None) -> Optional[str]:
    """A spelling variant of ``token`` the gold counts as COPY (*temptare* / *tentare*, *Grai* / *Graii*,
    *haud* / *haut*): the same ``Repairer.spelling_key``, a different ``normalize`` form, and attested in the
    corpus vocabulary -- the rules alone write non-words (*temptarre*), so without ``attested`` there is no variant.
    None when no rule applies."""
    from retexo.edit_typing.repair import Repairer

    if not attested:
        return None
    bare = re.sub(r"[^A-Za-z]+$", "", token)
    trail = token[len(bare):]
    if len(bare) < 3:
        return None
    lower = bare.lower()
    key = Repairer.spelling_key(lower)
    for pattern, repl in rng.sample(_SPELLING_RULES, len(_SPELLING_RULES)):
        cand = re.sub(pattern, repl, lower, count=1)
        if cand == lower or Repairer.spelling_key(cand) != key or normalize(cand) == normalize(lower):
            continue
        if normalize(cand) not in attested:
            continue
        if bare[:1].isupper():
            cand = cand[:1].upper() + cand[1:]
        return cand + trail
    return None


def _attach_enclitic(token: str, rng, attested) -> Optional[str]:
    """``uirum`` -> ``uirumque``, if that is a form somebody wrote."""
    bare = re.sub(r"[^A-Za-z]+$", "", token)
    trail = token[len(bare):]
    for enclitic in rng.sample(ENCLITICS, len(ENCLITICS)):
        merged = bare + enclitic
        if attested is None or normalize(merged) in attested:
            return merged + trail
    return None


def _make_one_typed(seed_tokens, context_pool, substitute, featurizer, rng, *,
                    frames: Sequence[Sequence[str]] = (), frame_rate=FRAME_RATE,
                    enclitic_rate=ENCLITIC_RATE, reorder_inter=0.0,
                    reorder_intra=0.0, attested=None, dense_rate=0.0, ins_slot=0.0,
                    slot_words: Optional[List[str]] = None, copy_variants=0.0, frameless_fill=""):
    """One pair with typed links, optional frame, enclitic events, reorderings.

    ``copy_variants`` (2026-09-27, the convention wall of the error analysis): enclitic events on copied words are
    COPY, not SPLIT/MERGE (the gold counts *aristisque* <- *aristis* as a copy), and this share of the copied words
    is written in a spelling variant the gold also counts as COPY (``spelling_variant``). ``frameless_fill``
    (``nolink``, the Frame Support dry run): an unrelated slot filler (tag SUBST) with no copied or inflected word
    within two positions of its fragment is emitted as an insertion, its source word deleted -- a filler without a
    kept frame is not a substitution."""
    source = list(seed_tokens)
    if len(source) < 6:
        return None, None
    target = list(rng.choice(context_pool))
    if len(target) < 6:
        return None, None

    stats = {"frame": 0, "reorder": 0, "enclitic": 0, "spelling": 0, "frameless": 0}
    n_fragments = _draw(FRAGMENTS, rng)
    source_del = [1] * len(source)
    pieces = []
    taken = set()

    for _ in range(n_fragments):
        length = min(_draw(FRAGMENT_LENGTHS, rng), len(source))
        starts = [i for i in range(len(source) - length + 1)
                  if not any(j in taken for j in range(i, i + length))]
        if not starts:
            break
        start = rng.choice(starts)
        core = source[start:start + length]
        for offset in range(len(core)):
            taken.add(start + offset)
            source_del[start + offset] = 0
        # E38c: an allusion-shaped fragment -- every content word replaced in context,
        # only the skeleton (short words) kept, so the substitutions are the stretch
        dense = dense_rate > 0 and rng.random() < dense_rate

        reworked, ops, fine, links = [], [], [], []
        for offset, token in enumerate(core):
            links.append(start + offset)
            drawn = substitute(token, rng, (source, start + offset), force=(dense and len(normalize(token)) >= 4))
            if drawn:
                replacement, tag = drawn
                reworked.append(replacement); ops.append("SUBST"); fine.append(tag)
                continue
            # cardinality: detach or attach an enclitic on a copied word
            if rng.random() < enclitic_rate:
                split = featurizer.enclitic(token)
                if split and normalize(split[0]) != normalize(token):
                    if copy_variants > 0:
                        reworked.append(split[0]); ops.append("COPY"); fine.append("NOP")
                    else:
                        reworked.append(split[0]); ops.append("SUBST"); fine.append("SPLIT")
                    stats["enclitic"] += 1
                    continue
                merged = _attach_enclitic(token, rng, attested)
                if merged and featurizer.enclitic(merged) and not featurizer.enclitic(token):
                    if copy_variants > 0:
                        reworked.append(merged); ops.append("COPY"); fine.append("NOP")
                    else:
                        reworked.append(merged); ops.append("SUBST"); fine.append("MERGE")
                    stats["enclitic"] += 1
                    continue
            if copy_variants > 0 and rng.random() < copy_variants:
                variant = spelling_variant(token, rng, attested)
                if variant:
                    reworked.append(variant); ops.append("COPY"); fine.append("NOP")
                    stats["spelling"] += 1
                    continue
            if rng.random() < COPY_VARIANT_SHARE:
                token = _variant(token, rng)
            reworked.append(token); ops.append("COPY"); fine.append("NOP")
        if frameless_fill == "nolink":
            kept = [o == "COPY" or f == "MORPH" for o, f in zip(ops, fine)]
            for i, f in enumerate(fine):
                if f == "SUBST" and not any(kept[j] for j in range(max(0, i - 2), min(len(fine), i + 3)) if j != i):
                    source_del[links[i]] = 1
                    ops[i], fine[i], links[i] = "INS", "INS", -1
                    stats["frameless"] += 1
        # failure-mode row 15 (``ins_slot``): the later author's own word inside the reused
        # stretch -- an attested content word absent from the source, spliced between two
        # reused words as INS, so that "inside a stretch" is not itself the reason to link
        if ins_slot > 0 and slot_words and len(reworked) >= 3 and rng.random() < ins_slot:
            source_norm = {normalize(w) for w in source}
            for _ in range(4):
                word = rng.choice(slot_words)
                if normalize(word) not in source_norm:
                    at = rng.randrange(1, len(reworked))
                    reworked.insert(at, word); ops.insert(at, "INS"); fine.insert(at, "INS"); links.insert(at, -1)
                    break
        # intra-fragment inversion: a two-word fragment turned around, which is
        # what *iter durum* -> *durum iter* is. Only ever the whole fragment.
        if len(reworked) == 2 and rng.random() < reorder_intra:
            for row in (reworked, ops, fine, links):
                row.reverse()
            stats["reorder"] += 1
        pieces.append((reworked, ops, fine, links))

    # inter-fragment inversion: the second fragment comes out first
    if len(pieces) >= 2 and rng.random() < reorder_inter:
        pieces.reverse()
        stats["reorder"] += 1

    operations = ["INS"] * len(target)
    fine_ops = ["INS"] * len(target)
    alignments = [-1] * len(target)
    frame_mask = [0] * len(target)
    cuts = sorted(rng.sample(range(len(target) + 1), min(len(pieces), len(target))))
    grown = 0
    for offset, (position, (reworked, ops, fine, links)) in enumerate(zip(cuts, pieces)):
        at = position + grown
        # an attribution formula immediately before the first fragment, in the
        # citing author's own words, drawn from the training-fold hand labels
        if offset == 0 and frames and rng.random() < frame_rate:
            formula = list(rng.choice(frames))
            target[at:at] = formula
            operations[at:at] = ["INS"] * len(formula)
            fine_ops[at:at] = ["INS"] * len(formula)
            alignments[at:at] = [-1] * len(formula)
            frame_mask[at:at] = [1] * len(formula)
            grown += len(formula); at += len(formula)
            stats["frame"] = 1
        target[at:at] = reworked
        operations[at:at] = ops
        fine_ops[at:at] = fine
        alignments[at:at] = links
        frame_mask[at:at] = [0] * len(reworked)
        grown += len(reworked)

    # evidence for every link, computed where the resources live
    features = [None] * len(target)
    for t, s in enumerate(alignments):
        if s >= 0:
            features[t] = featurizer(source[s], target[t], s, t, len(source), len(target))

    example = ChangeExample(
        source_tokens=source, target_tokens=target,
        labels=[0 if o == "COPY" else 1 for o in operations],
        operations=operations, n_operations=len(taken),
        source_labels=source_del,
        source_operations=["DEL" if d else "COPY" for d in source_del],
        alignments=alignments,
        fine_operations=fine_ops, frame_labels=frame_mask, link_features=features)
    return example, stats


# =============================================================================
# Parallel generation
# =============================================================================

_WORKER: dict = {}
_MLM_CACHE: dict = {}


def _init_worker(vectors_path, attested, accept, frames, frame_rate, enclitic_rate,
                 reorder_inter, reorder_intra, mlm_subst=False, subst_weight=None, mlm_model=None, cohypo=False, dense_rate=0.0,
                 norel_share=0.0, stem_subst=0.0, ins_slot=0.0, lexical_share=None, rel_share=None, copy_variants=0.0,
                 frameless_fill=""):
    mlm = None
    if mlm_subst:
        from retexo.datasets.mlm_subst import ContextualSubstituter
        mlm = ContextualSubstituter(mlm_model) if mlm_model else ContextualSubstituter()
        mlm.cache = _MLM_CACHE.get("cache")            # inherited from the parent through fork
    substitute, resources = _make_typed_substitute(vectors_path, attested, accept, mlm=mlm, subst_weight=subst_weight, cohypo=cohypo, lexical_share=lexical_share,
                                                   norel_share=norel_share, stem_subst=stem_subst, rel_share=rel_share)
    slot_words = sorted(w for w in attested if len(w) >= 5) if (ins_slot > 0 and attested) else None
    _WORKER.update(substitute=substitute, featurizer=LinkFeaturizer(resources),
                   frames=frames, frame_rate=frame_rate, enclitic_rate=enclitic_rate,
                   reorder_inter=reorder_inter, reorder_intra=reorder_intra,
                   attested=attested, dense_rate=dense_rate, ins_slot=ins_slot, slot_words=slot_words,
                   copy_variants=copy_variants, frameless_fill=frameless_fill)


def _chunk(args):
    seeds, context_pool, seed, per_seed = args
    rng = random.Random(seed)
    report = TypedReport()
    out = []
    w = _WORKER
    for tokens in seeds:
        for _ in range(per_seed):
            report.attempted += 1
            example, stats = _make_one_typed(
                tokens, context_pool, w["substitute"], w["featurizer"], rng,
                frames=w["frames"], frame_rate=w["frame_rate"],
                enclitic_rate=w["enclitic_rate"], reorder_inter=w["reorder_inter"],
                reorder_intra=w["reorder_intra"], attested=w["attested"], dense_rate=w.get("dense_rate", 0.0),
                ins_slot=w.get("ins_slot", 0.0), slot_words=w.get("slot_words"),
                copy_variants=w.get("copy_variants", 0.0), frameless_fill=w.get("frameless_fill", ""))
            if example is None:
                report.no_substitute += 1
                continue
            out.append(example)
            report.kept += 1
            for op in example.operations:
                report.op_counts[op] = report.op_counts.get(op, 0) + 1
            for op in example.fine_operations:
                if op != "INS":
                    report.fine_counts[op] = report.fine_counts.get(op, 0) + 1
            report.frames += stats["frame"]
            report.reorders += stats["reorder"]
            report.enclitics += stats["enclitic"]
            report.spelling += stats["spelling"]
            report.frameless += stats["frameless"]
    return out, report.__dict__


def generate_typed(seeds, context_pool, *, workers=24, per_seed=4, seed=1,
                   chunk_size=200, vectors_path=None, attested=None, accept=None,
                   frames=(), frame_rate=FRAME_RATE, enclitic_rate=ENCLITIC_RATE,
                   reorder_inter=0.0, reorder_intra=0.0, log=None, mlm_subst=False, subst_weight=None, mlm_model=None, cohypo=False, dense_rate=0.0,
                   norel_share=0.0, stem_subst=0.0, ins_slot=0.0, lexical_share=None, rel_share=None,
                   copy_variants=0.0, frameless_fill=""):
    """Generate in parallel, offline. Returns examples and a report."""
    import multiprocessing as mp

    chunks = [list(seeds[i:i + chunk_size]) for i in range(0, len(seeds), chunk_size)]
    context_pool = [list(p) for p in context_pool]
    frames = [list(f) for f in frames]
    tasks = [(c, context_pool, seed + i, per_seed) for i, c in enumerate(chunks)]
    if accept is None:
        # E4's calibration measures untyped coverage; the typed source realises
        # the same tags plus two rarer ones, so the same acceptance is close
        # enough, and the realism report says where it lands.
        accept, coverage = calibrate_acceptance(seeds, vectors_path, attested)
        if log:
            log(f"substitute coverage {coverage:.1%} -> accept {accept:.1%}")
    if mlm_subst:
        from retexo.datasets.mlm_subst import ContextualSubstituter
        _MLM_CACHE["cache"] = ContextualSubstituter.precompute(seeds, mlm_model or "bowphs/LaBerta", log=log)
    context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    out, totals = [], TypedReport()
    with context.Pool(workers, initializer=_init_worker,
                      initargs=(vectors_path, attested, accept, frames, frame_rate,
                                enclitic_rate, reorder_inter, reorder_intra, mlm_subst, subst_weight, mlm_model, cohypo, dense_rate,
                                norel_share, stem_subst, ins_slot, lexical_share, rel_share, copy_variants,
                                frameless_fill)) as pool:
        for examples, report in pool.imap_unordered(_chunk, tasks):
            out.extend(examples)
            totals.attempted += report["attempted"]; totals.kept += report["kept"]
            totals.no_substitute += report["no_substitute"]
            totals.frames += report["frames"]; totals.reorders += report["reorders"]
            totals.enclitics += report["enclitics"]
            totals.spelling += report["spelling"]; totals.frameless += report["frameless"]
            for k, v in report["op_counts"].items():
                totals.op_counts[k] = totals.op_counts.get(k, 0) + v
            for k, v in report["fine_counts"].items():
                totals.fine_counts[k] = totals.fine_counts.get(k, 0) + v
    return out, totals


# =============================================================================
# Evidence for every pair (E25)
# =============================================================================

_FEAT: dict = {}
#: E41: when True, featurize_pairs parses every passage with Stanza (parent, GPU) and
#: appends retexo.edit_typing.dep_features.DEP_FEATURES to every cell
DEP_FEATURES_ON = False
#: The Frame Support dry run (2026-09-27): when True, featurize_pairs appends one channel to every cell, the
#: in-order anchor support (``frame_support_channel``)
FRAME_CHANNEL_ON = False
#: One DependencyParser per process, so its parse cache survives repeat calls.
_DEP_PARSER: Optional[DependencyParser] = None


def _dep_parser() -> DependencyParser:
    global _DEP_PARSER
    if _DEP_PARSER is None:
        _DEP_PARSER = DependencyParser()
    return _DEP_PARSER


def _feat_init(vectors_path):
    from retexo.resources import Resources

    _FEAT["fz"] = LinkFeaturizer(Resources(offline=True, vectors_path=vectors_path))


def _feat_chunk(items):
    import numpy as np
    from retexo.edit_typing.link_features import N_FEATURES

    fz = _FEAT["fz"]
    out = []
    for item in items:
        source, target = item[0], item[1]
        dep = item[2] if len(item) > 2 else None                       # E41: (dep_s, dep_t)
        width = N_FEATURES + (len(DEP_FEATURES) if dep is not None else 0)
        arr = np.zeros((len(target), len(source), width), dtype=np.float16)
        n_s, n_t = len(source), len(target)
        for t, tw in enumerate(target):
            for s, sw in enumerate(source):
                row = fz(sw, tw, s, t, n_s, n_t)
                if dep is not None:
                    row = list(row) + DependencyParser.cell_features(source, target, dep[0], dep[1], s, t, lemma=fz.lemma)
                arr[t, s] = row
        out.append(arr)
    return out


def frame_support_channel(arr, window: int = 2, slack: int = 3):
    """The kept-frame support of every cell, from the evidence alone (no labels): the number of anchor cells
    (same form or same lemma) at reuse positions within ``window`` of the cell's reuse word, in the same order on the
    source side (the offsets agree in sign) and within ``slack`` of the cell's source offset, divided by
    ``2 * window``. The error analysis of 2026-09-26 measured the same count on links: sure substitutions sit in a
    kept frame (mean support 2.1), invented links do not (0.8)."""
    import numpy as np
    from retexo.edit_typing.link_features import FEATURE_NAMES

    anchor = (arr[..., FEATURE_NAMES.index("same_form")] > 0) | (arr[..., FEATURE_NAMES.index("same_lemma")] > 0)
    n_t, n_s = anchor.shape
    # a reuse neighbour counts once, however many source offsets match
    per_neighbour = np.zeros((n_t, n_s), dtype=np.float32)
    for dt in range(-window, window + 1):
        if dt == 0:
            continue
        hit = np.zeros((n_t, n_s), dtype=bool)
        for ds in range(-(abs(dt) + slack), abs(dt) + slack + 1):
            if ds == 0 or (ds > 0) != (dt > 0) or abs(ds - dt) > slack:
                continue
            t0, t1 = max(0, -dt), min(n_t, n_t - dt)
            s0, s1 = max(0, -ds), min(n_s, n_s - ds)
            if t0 < t1 and s0 < s1:
                hit[t0:t1, s0:s1] |= anchor[t0 + dt:t1 + dt, s0 + ds:s1 + ds]
        per_neighbour += hit
    return (per_neighbour / (2 * window)).astype(arr.dtype)


def featurize_pairs(examples, *, vectors_path=None, workers=24, chunk_size=48, log=None):
    """Attach ``pair_features`` -- the evidence for every (reuse, source) pair.

    The typed pointer needs the evidence inside the alignment decision, which
    means every candidate pair, not only the gold link. Per-word lookups are
    cached inside each worker, so the pair cost is string work; a 20k pool is
    ~8M pairs and takes well under a minute on forty workers.
    """
    import multiprocessing as mp

    items = [(list(e.source_tokens), list(e.target_tokens)) for e in examples]
    if DEP_FEATURES_ON:
        parses = _dep_parser().parse_all(
            [e.source_tokens for e in examples] + [e.target_tokens for e in examples], log=log)
        items = [(s, t, (parses[" ".join(s)], parses[" ".join(t)])) for s, t in items]
    tasks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    done = 0
    with context.Pool(workers, initializer=_feat_init, initargs=(vectors_path,)) as pool:
        for i, arrays in enumerate(pool.imap(_feat_chunk, tasks)):
            for e, a in zip(examples[i * chunk_size:(i + 1) * chunk_size], arrays):
                if FRAME_CHANNEL_ON:
                    import numpy as np
                    a = np.concatenate([a, frame_support_channel(a)[..., None]], axis=-1)
                # ChangeExample is frozen; this is the one field filled in after
                # construction, and it is filled exactly once
                object.__setattr__(e, "pair_features", a)
            done += len(arrays)
    if log:
        cells = sum(e.pair_features.shape[0] * e.pair_features.shape[1]
                    for e in examples if e.pair_features is not None)
        log(f"evidence for {done:,} pairs, {cells:,} cells")
    return examples


# =============================================================================
# Gold with fine labels and evidence
# =============================================================================

def _link_only_coarse(pair, t: int) -> str:
    """The coarse view of a link-only edge: ``NOP`` for the same normalised form, ``SUBST`` for any change."""
    from retexo.core.normalize import normalize

    s = pair.target_align[t]
    return "NOP" if normalize(pair.source_tokens[s]) == normalize(pair.target_tokens[t]) else "SUBST"


def fine_from_gold(pair, featurizer: LinkFeaturizer, *, gold_fine: bool = False):
    """A hand-labelled pair as a typed example.

    The silver labels carry NOP / MORPH / SUBST and FRAME spans. NOP and MORPH
    are exact fine tags; a silver SUBST is a *lexical* change whose relation
    the labeller did not name, so it is left unsupervised (-100 at training)
    rather than forced into a class -- the typer learns the fine relations from
    the constructed data and the coarse ones from the real data. With
    ``gold_fine`` (a V3 annotation, definition section 3.1: SYN, POS, NE-SUB,
    SUBST, SPLIT, MERGE named on the link) every operation in
    ``FINE_OPERATIONS`` is the fine target itself; the ``detail`` of a link
    (HYPER, ANT, the MORPH features) is never a label.
    """
    target_ops = list(pair.target_ops)          # fine: NOP / MORPH / SUBST / FRAME / INS
    fine, frame, features = [], [], []
    for t, op in enumerate(target_ops):
        s = pair.target_align[t] if pair.target_align and t < len(pair.target_align) else -1
        if op == "FRAME":
            fine.append("INS"); frame.append(1); features.append(None); continue
        frame.append(0)
        if s < 0:
            fine.append("INS"); features.append(None); continue
        if op == "NOP":
            fine.append("NOP")
        elif op == "MORPH":
            fine.append("MORPH")
        elif op == "LINK":
            fine.append("LINK")       # a link-only edge (self-training): the link trains, no type head does
        elif gold_fine and op in FINE_OPERATIONS:
            fine.append(op)           # the annotated V3 operation
        else:
            fine.append("?")          # unsupervised lexical change
        features.append(featurizer(pair.source_tokens[s], pair.target_tokens[t], s, t,
                                   len(pair.source_tokens), len(pair.target_tokens)))
    # The coarse head and the pointer see exactly what E4-era training saw:
    # COPY / SUBST / INS. Only the typer and the frame head see the fine view.
    # A link-only edge says nothing about its kind: the coarse head gets COPY where the two words are the same
    # form and SUBST (a change, INFLECT included) otherwise, never a lexical class it was not shown.
    coarse_ops = [coarse(_link_only_coarse(pair, t) if op == "LINK" else op) for t, op in enumerate(target_ops)]
    return ChangeExample(
        source_tokens=list(pair.source_tokens), target_tokens=list(pair.target_tokens),
        labels=[0 if op == "COPY" else 1 for op in coarse_ops],
        operations=coarse_ops,
        n_operations=sum(1 for op in coarse_ops if op != "COPY"),
        source_labels=list(pair.source_del),
        source_operations=["DEL" if d else "COPY" for d in pair.source_del],
        alignments=list(pair.target_align) if pair.target_align else None,
        fine_operations=fine, frame_labels=frame, link_features=features)


# =============================================================================
# A reusable generator object
# =============================================================================

@dataclass
class GeneratorConfig:
    """Every knob ``generate_typed`` takes, gathered so a caller can build one
    configuration once and reuse it across folds and pools."""

    vectors_path: Optional[str] = None
    attested: Optional[set] = None
    accept: Optional[float] = None
    frames: Sequence[Sequence[str]] = ()
    frame_rate: float = FRAME_RATE
    enclitic_rate: float = ENCLITIC_RATE
    reorder_inter: float = 0.0
    reorder_intra: float = 0.0
    mlm_subst: bool = False
    subst_weight: Optional[float] = None
    mlm_model: Optional[str] = None
    cohypo: bool = False
    dense_rate: float = 0.0
    workers: int = 24
    per_seed: int = 4
    seed: int = 1
    chunk_size: int = 200


class TypedGenerator:
    """The typed synthetic generator, as an object instead of free keywords.

    A thin wrapper over ``generate_typed``/``select_pool``/``frame_pool``: the
    functions do the multiprocess work (and stay directly importable, and
    directly unit-testable, for that reason); this class exists so a caller
    building several pools from one configuration -- one per fold, say --
    states the configuration once.

    Example:
        >>> generator = TypedGenerator(GeneratorConfig(
        ...     frames=frame_pool(gold_dir, folds_by_id, exclude_fold=4)))
        >>> pool, report = generator.generate(seeds, context_pool)
    """

    def __init__(self, config: Optional[GeneratorConfig] = None, **overrides):
        self.config = replace(config or GeneratorConfig(), **overrides)

    def generate(self, seeds, context_pool, *, log=None
                 ) -> Tuple[List[ChangeExample], TypedReport]:
        return generate_typed(seeds, context_pool, log=log, **asdict(self.config))

    @staticmethod
    def select_pool(pool, size: int, rng, rare_min: int = 1200):
        return select_pool(pool, size, rng, rare_min=rare_min)

    @staticmethod
    def frame_pool(gold_dir, folds_by_id: Dict[str, int], exclude_fold: int):
        return frame_pool(gold_dir, folds_by_id, exclude_fold)


__all__ = [
    "OPERATIONS", "TARGET", "FRAGMENTS", "FRAGMENT_LENGTHS", "COPY_VARIANT_SHARE",
    "attested_forms", "calibrate_acceptance", "coarse", "realism_report", "log_realism",
    "all_insert_baseline",
    "FINE_OPERATIONS", "TYPED_WEIGHTS", "RESIDUAL_MAX_COS", "FRAME_RATE",
    "ENCLITIC_RATE", "ENCLITICS", "RARE_TAGS",
    "select_pool", "balanced_subset", "fine_mix", "frame_pool",
    "TypedReport", "generate_typed", "featurize_pairs", "fine_from_gold",
    "GeneratorConfig", "TypedGenerator",
]
