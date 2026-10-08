# datasets/synthetic/inventory.py
"""The typed inventory (E24): fine tags, substitution weights, event rates, and the rare tags."""

from __future__ import annotations

# =============================================================================
# The typed inventory (E24)
# =============================================================================

#: Fine tags a link can carry. NOP is a copy; SUBST is a lexical change that no
#: named relation fits -- a class the typer must be able to *say*, because on
#: real links the reader says it for two in five (adjudication, fold 4) and a
#: head that lacks it spends those on the nearest relation it knows.
FINE_OPERATIONS = (
    "NOP",
    "MORPH",
    "SYN",
    "HYPER",
    "HYPO",
    "ANT",
    "SYN-DIST",
    "NE-SUB",
    "POS",
    "SPLIT",
    "MERGE",
    "SUBST",
)

#: Tags a typed substitution source is asked for, in the mix the oracle finds on
#: real pairs (generation.DEFAULT_WEIGHTS), renormalised over what it can
#: realise. MORPH dominates because inflection dominates real substitution: the
#: hand labels carry 901 MORPH against 415 lexical SUBST. SUBST here is the
#: residual: a real word in the same slot with no attested relation and no
#: distributional closeness.
TYPED_WEIGHTS = {
    "MORPH": 0.50,
    "SYN": 0.14,
    "SYN-DIST": 0.07,
    "HYPER": 0.05,
    "HYPO": 0.04,
    "ANT": 0.03,
    "NE-SUB": 0.04,
    "POS": 0.04,
    "SUBST": 0.09,
}

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

#: Relation-bearing substitution tags (the generator's lexical tags other than the unrelated slot filler).
RELATED_TAGS = ("SYN", "SYN-DIST", "HYPER", "HYPO", "ANT", "NE-SUB", "POS")
