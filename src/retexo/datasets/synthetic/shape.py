# datasets/synthetic/shape.py
"""The untyped shape (E4): the measured targets, the spelling variant, and the realism report."""

from __future__ import annotations

from retexo.core.normalize import normalize

# =============================================================================
# The untyped shape (E4): operations, and the measured targets
# =============================================================================

OPERATIONS = ("COPY", "SUBST", "INS")  # reuse side; DEL has its own head

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
FRAGMENT_LENGTHS = (
    (1, 0.62),
    (2, 0.18),
    (3, 0.03),
    (4, 0.05),
    (5, 0.02),
    (6, 0.02),
    (7, 0.02),
    (8, 0.01),
    (11, 0.02),
    (14, 0.02),
    (20, 0.01),
)

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


# ---------- The variant COPY carries ----------


def _variant(token: str, rng) -> str:
    """A differently-spelled token that still normalizes to the same string.

    Returns the token unchanged when no safe variant applies, so the caller can
    label it COPY either way.
    """
    body = token.rstrip(",.;:?!")
    if not body:
        return token
    kind = rng.random()
    if kind < 0.45:  # punctuation
        candidate = body + rng.choice(_TRAILING)
    elif kind < 0.75:  # capitalisation
        candidate = (body.lower() if body[:1].isupper() else body.capitalize()) + token[len(body) :]
    elif rng.random() < 0.5:  # aspiration, word-initial
        source, target = rng.choice(_INITIAL)
        if not body.lower().startswith(source):
            return token
        candidate = target + body[len(source) :] + token[len(body) :]
        if body[:1].isupper():
            candidate = candidate.capitalize()
    else:  # medial variation
        source, target = rng.choice(_ANYWHERE)
        if source not in body.lower():
            return token
        at = body.lower().index(source)
        candidate = body[:at] + target + body[at + len(source) :] + token[len(body) :]
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


# ---------- The coarse collapse ----------


def coarse(op: str) -> str:
    """The E4-era three-way collapse: ``COPY`` / ``SUBST`` / ``INS``.

    Not the same thing as ``retexo.baselines.labels``' canonical levels --
    this folds ``MORPH`` into ``SUBST`` (labels.py's V1 keeps them apart) -- and
    it is kept only because ``fine_from_gold`` uses it to give the untyped
    pointer head exactly the supervision E4-era training gave it. Do not reach
    for it as a general-purpose collapse; use ``retexo.baselines.labels``
    for that.
    """
    return {
        "NOP": "COPY",
        "REORDER": "COPY",
        "QUOTE": "COPY",
        "MORPH": "SUBST",
        "FRAME": "INS",
    }.get(op, op if op in OPERATIONS else "SUBST")


def _fragments(operations):
    """Contiguous runs of aligned (non-INS) tokens."""
    runs, current = [], 0
    for op in operations:
        if op != "INS":
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


# ---------- The realism report ----------


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
    gen_lengths = [length for f in gen_frag for length in f] or [0]
    ref_lengths = [length for f in ref_frag for length in f] or [0]

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

    substitutes = [
        t for e in examples for t, o in zip(e.target_tokens, e.operations) if o == "SUBST"
    ]
    attested = sum(1 for w in substitutes if normalize(w) in vocabulary)

    return {
        "COPY share": (gen.get("COPY", 0), ref.get("COPY", 0)),
        "SUBST share": (gen.get("SUBST", 0), ref.get("SUBST", 0)),
        "INS share": (gen.get("INS", 0), ref.get("INS", 0)),
        "DEL share": (gen_del / gen_src, ref_del / ref_src),
        "fragments/pair": (
            statistics.mean([len(f) for f in gen_frag] or [0]),
            statistics.mean([len(f) for f in ref_frag] or [0]),
        ),
        "fragment length": (statistics.mean(gen_lengths), statistics.mean(ref_lengths)),
        "source length": (
            statistics.mean([len(e.source_tokens) for e in examples]),
            statistics.mean([len(g.source_tokens) for g in gold]),
        ),
        "target length": (
            statistics.mean([len(e.target_tokens) for e in examples]),
            statistics.mean([len(g.target_tokens) for g in gold]),
        ),
        "COPY differing surface": (
            copy_variant_share(examples, gen_ops, lambda e: e.source_tokens),
            copy_variant_share(gold, gold_ops, lambda g: g.source_tokens),
        ),
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


# ---------- The degenerate baseline ----------


def all_insert_baseline(examples):
    """Predict INS for every reuse token, DEL for every source token.

    The degenerate script, and a strong one: it is 83% correct on the reuse side
    of real pairs.
    """
    return (
        [["INS"] * len(e.target_tokens) for e in examples],
        [[1] * len(e.source_tokens) for e in examples],
    )
