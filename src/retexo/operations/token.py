# retexo/operations/token.py
"""
Token operations: one source token becomes one reuse token.

They differ only in *why* the substitution is licensed — same lemma, attested
synonymy, a WordNet relation, distributional proximity — which is what the tag
records and what the cost prices. Applying any of them is identical work;
detecting them is not, and that is where they diverge.

Every one is its own inverse at the level of the tag, except the pair
``HYPER`` and ``HYPO``, which invert into each other: generalising a word and
specifying it are the same move in opposite directions.
"""

from __future__ import annotations

from typing import Optional

from retexo.operations.base import Level, Operation, Role

# =============================================================================
# Base
# =============================================================================


class _TokenOperation(Operation):
    """Shared behaviour for one-to-one substitutions."""

    level = Level.TOKEN
    role = Role.WRITING

    def inverse_tag(self) -> str:
        return self.tag


# =============================================================================
# Operations
# =============================================================================


class Nop(_TokenOperation):
    """Identical token after normalization."""

    tag = "NOP"
    default_cost = 0.0

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        from retexo.core.normalize import normalize

        if normalize(source) == normalize(target):
            return ""
        return None


class Morph(_TokenOperation):
    """Same lemma, different inflection."""

    tag = "MORPH"
    default_cost = 0.25
    requires = ("morphology",)

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        if not resources.has("morphology"):
            return None
        morph = resources.morphology
        lemma = morph.lemma(source)
        if not lemma or lemma != morph.lemma(target):
            return None
        # Naming the features that moved is what makes the operation readable
        # to a philologist: "sg.acc. -> pl.abl." says what "lemma=arma" cannot.
        # Collatinus cannot decline every lemma, so this is additive.
        before = morph.morpho_features(source)
        after = morph.morpho_features(target)
        if before and after and before != after:
            return f"lemma={lemma} {before}->{after}"
        return f"lemma={lemma}"


#: Similarity a wordnet-attested pair must still reach to be believed. Set well
#: below SYN_DIST_THRESHOLD because the wordnet is already evidence: this is a
#: veto on links the distributional record actively contradicts, not a second
#: independent test. A pair with no vector for either lemma is accepted on the
#: wordnet's word alone, so the gate never costs coverage where it cannot judge.
ATTESTED_SIMILARITY_FLOOR = 0.35


def _vectors_contradict(source: str, target: str, resources) -> bool:
    """Whether the vectors actively disagree with an attested relation.

    Shared by every operation whose evidence is a lexicon entry rather than a
    similarity score, so that one floor governs all of them.
    """
    if not resources.has("vectors"):
        return False
    similarity = resources.vectors.similarity(
        resources.lemma_or_surface(source), resources.lemma_or_surface(target)
    )
    return similarity is not None and similarity < ATTESTED_SIMILARITY_FLOOR


class _WordNetOperation(_TokenOperation):
    """A substitution licensed by a WordNet relation."""

    #: Key in the cached WordNet record.
    relation: str = ""
    requires = ("wordnet",)

    def _contradicted(self, source: str, target: str, resources) -> bool:
        """Whether the vectors actively disagree with an attested relation."""
        return _vectors_contradict(source, target, resources)

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        if not resources.has("wordnet"):
            return None
        lemma_src = resources.lemma_or_surface(source)
        lemma_tgt = resources.lemma_or_surface(target)
        for pos in resources.pos_candidates(source):
            record = resources.wordnet.lookup(lemma_src, pos)
            if lemma_tgt in record.get(self.relation, ()):
                if self._contradicted(source, target, resources):
                    return None
                return f"wn:{pos}"
        return None


class Syn(_WordNetOperation):
    """Attested synonym."""

    tag = "SYN"
    relation = "synonyms"
    default_cost = 0.50


class Hyper(_WordNetOperation):
    """Generalization: the reuse names a broader category."""

    tag = "HYPER"
    relation = "hypernyms"
    default_cost = 0.75

    def inverse_tag(self) -> str:
        return "HYPO"


class Hypo(_WordNetOperation):
    """Specification: the reuse names a narrower category."""

    tag = "HYPO"
    relation = "hyponyms"
    default_cost = 0.75

    def inverse_tag(self) -> str:
        return "HYPER"


class Ant(_WordNetOperation):
    """Antonym, usually accompanied by negation."""

    tag = "ANT"
    relation = "antonyms"
    default_cost = 1.00


#: Cosine floor for a distributional near-synonym, calibrated on the annotated
#: pairs rather than asserted. Sweeping thresholds against in-domain negatives
#: shows precision rising slowly to 0.68 at 0.60 and then stepping to 0.81 at
#: 0.65, against a chance baseline of 0.47. The step is where the measure stops
#: describing co-occurrence and starts describing similarity. Chosen for
#: precision: a false SYN-DIST invents a relation that is not there, while a
#: missed one costs an insertion and a deletion, which the residual reports.
SYN_DIST_THRESHOLD = 0.65


class SynDist(_TokenOperation):
    """Distributional near-synonym, above a cosine threshold.

    Costs more than an attested synonym on purpose: the cost measures how much
    the link is believed, not only how far the meaning moved.
    """

    tag = "SYN-DIST"
    default_cost = 0.60
    requires = ("vectors",)

    def __init__(self, threshold: float = SYN_DIST_THRESHOLD):
        self.threshold = threshold

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        if not resources.has("vectors"):
            return None
        similarity = resources.vectors.similarity(
            resources.lemma_or_surface(source), resources.lemma_or_surface(target)
        )
        if similarity is not None and similarity >= self.threshold:
            return f"cos={similarity:.2f}"
        return None


#: Similarity two names must reach before one is read as standing in for the
#: other. Unlike the wordnet operations, where the vectors only *veto* a link
#: the lexicon already attests, here they are the whole of the evidence: being
#: a name is a property of one token, not a relation between two. So this is a
#: requirement rather than a veto, and detection is off wherever vectors are
#: absent. Without it any two names in a window would pair at 0.75 against the
#: 2.00 of an insertion and a deletion, which would make unrelated passages
#: look related and damage exactly the detection claim the distance is for.
#:
#: PROVISIONAL. Unlike SYN_DIST_THRESHOLD this has not been calibrated against
#: in-domain negatives; it is set at the level where a distributional link is
#: merely credible rather than convincing, because the entity test already
#: carries independent evidence. Calibrate before relying on the number.
NE_SUB_SIMILARITY_FLOOR = 0.50


class NeSub(_TokenOperation):
    """One named entity substituted for another.

    Detection needs two things: that both tokens are names, and that they are
    plausibly the same name-slot filled differently.

    The first comes from CLTK's list of 39,725 Latin proper names, matched
    case-sensitively. The capitalization test an earlier prototype was rejected
    for asked only whether a token was capitalized, and so fired on every
    sentence-initial word; this asks whether a capitalized token is *also* a
    listed name, which *Arma* -- sentence-initial in the Aeneid -- is not.
    Folding case would undo this: 105 of the 300 commonest words in the
    benchmark then match, against 13 when the capital is required.

    The second comes from the vectors, and is required rather than advisory --
    see ``NE_SUB_SIMILARITY_FLOOR``.

    One known recall cost: the vectors are keyed on lemmas, and lemmatizing a
    proper name is exactly what the name list exists to avoid. *Italiam* and
    *Hesperiam* resolve correctly, but *Troiam* lemmatizes to the adjective
    *troius* and so finds no vector. The failure is silent and one-directional
    -- a wrong lemma yields no similarity and the operation declines -- so it
    costs recall rather than precision, which is the right way round.
    """

    tag = "NE-SUB"
    default_cost = 0.75
    requires = ("entities", "vectors")

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        if not resources.has("entities") or not resources.has("vectors"):
            return None
        entities = resources.entities
        if not (entities.is_name(source) and entities.is_name(target)):
            return None
        from retexo.core.normalize import normalize

        if normalize(source) == normalize(target):
            return None  # the same name, which is a NOP
        similarity = resources.vectors.similarity(
            resources.lemma_or_surface(source), resources.lemma_or_surface(target)
        )
        if similarity is None or similarity < NE_SUB_SIMILARITY_FLOOR:
            return None
        return f"ne cos={similarity:.2f}"


class Subst(_TokenOperation):
    """A lexical replacement that no named relation fits.

    The residual of the token inventory: the word changed, and neither the
    lemmatizer, the wordnet, the vectors nor the name list attests a relation
    between the two. On the hand-labelled pairs a reader says this for two in
    five lexical changes -- parallel replacements (*poenarum* -> *morborum*),
    textual variants (*consita* -> *concita*), allegorical substitutions
    (*filo* -> *Spiritu*). A script that cannot say it spends those links on
    the nearest relation it knows, which is worse than saying nothing.

    Priced above every attested relation and below an insertion plus a
    deletion: it is a change whose kind is unknown, not a change whose kind is
    known to be far. No detector, by definition -- it is what the detectors
    leave -- so it is emitted only by the learned typer.
    """

    tag = "SUBST"
    default_cost = 0.80
    detectable = False

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        return None


class Pos(_TokenOperation):
    """Derivational shift: same root, different formation.

    A true derivational pair crosses lemmas -- *amāre* to *amor* -- so
    lemmatization cannot find it. It was previously recorded here that no
    derivational lexicon for Latin had been identified, and that is wrong:
    Latin WordNet carries derivational relations on its *lemma* endpoint, under
    the symbol ``/``. It resolves *cano* to *carmen*, *fugio* to *fuga*, and
    *amo* to *amor* -- the very pair cited as proof that this could not be done.

    **One widening comes with this.** The relation returns a whole word family,
    which mixes a change of word class (*cano* to *carmen*) with a change of
    preverb at the same word class (*fugio* to *confugio*). Separating them
    needs a reliable part of speech for a surface form, which Collatinus does
    not give -- ``pos_candidates`` returns a guess ordering, not a decision. So
    what is detected is "derivationally related", which is broader than the
    operation's name. Either the name or the scope should eventually be
    reconciled; the cost is unchanged meanwhile.
    """

    tag = "POS"
    default_cost = 0.50
    requires = ("derivation",)

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        if not resources.has("derivation"):
            return None
        lemma_src = resources.lemma_or_surface(source)
        lemma_tgt = resources.lemma_or_surface(target)
        if not lemma_src or lemma_src == lemma_tgt:
            return None  # same lemma is MORPH, not a derivation
        for pos in resources.pos_candidates(source):
            record = resources.wordnet.lookup(lemma_src, pos)
            if lemma_tgt in record.get("derivatives", ()):
                # Derivational families reach a long way -- cano to bucina --
                # so the same veto the attested relations use applies here.
                if _vectors_contradict(source, target, resources):
                    return None
                return f"deriv:{pos}"
        return None


# =============================================================================
# Group-level back-offs (E31): "a change of this kind, the relation unclear"
# =============================================================================


class Form(_TokenOperation):
    """The same lemma or stem in another form, the exact operation not settled.

    The hierarchical typer's back-off for the form group (MORPH, POS, SPLIT,
    MERGE). Priced at the members' mean. Never detected; emitted only when the
    typer's confidence in a member falls short."""

    tag = "FORM"
    default_cost = 0.562  # mean of MORPH, POS, SPLIT, MERGE
    detectable = False

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        return None


class Sense(_TokenOperation):
    """A different word with a nameable relation, the relation not settled.

    The back-off for the sense group (SYN, SYN-DIST, HYPER, HYPO, ANT, NE-SUB).
    Priced at the members' mean; never detected."""

    tag = "SENSE"
    default_cost = 0.725  # mean of SYN, SYN-DIST, HYPER, HYPO, ANT, NE-SUB
    detectable = False

    def detect(self, source: str, target: str, resources) -> Optional[str]:
        return None
