# retexo/llm/llm_prompts.py
"""
E32: prompt variants for the two jobs we would trust an LLM with.

**type** -- name the relation between a reuse word and the source word it came
from. Scored against the reader's adjudicated verdicts on real links, beside
the trained typer's answer on the same links, so a prompt is only interesting
if it beats the model we already have.

**propose** -- for a Latin lemma the resources cannot serve, propose related
words (E32 stage 1). Scored where WordNet does answer, plus the generator's own
mechanical filters.

A variant is a :class:`PromptVariant` (system, template, parse) held in a
:class:`PromptRegistry`, so adding one is a dict entry on the registry.
Templates use ``{...}`` fields; ``parse`` names the reader in
``retexo.llm.llm_benchmark.PARSERS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

TAGS = (
    "NOP",
    "MORPH",
    "SYN",
    "SYN-DIST",
    "HYPER",
    "HYPO",
    "ANT",
    "NE-SUB",
    "POS",
    "SPLIT",
    "MERGE",
    "SUBST",
)

SYSTEM_EN = "You are a Latin philologist annotating literary reuse: one author has rewritten an earlier passage."
SYSTEM_LA = "Grammaticus es Latinus, qui imitationem poetarum annotas."

DEFINITIONS = """NOP      the same word, unchanged
MORPH    the same word in a different form (case, tense, number)
SYN      a synonym: another word with the same sense
SYN-DIST a word from the same field that is not a synonym (related, not equivalent)
HYPER    a more general word than the source word
HYPO     a more specific word than the source word
ANT      an opposite
NE-SUB   a different proper name
POS      the same stem in a different part of speech (noun for verb, and so on)
SPLIT    one source word rendered by two reuse words
MERGE    two source words rendered by one reuse word
SUBST    a different word with no semantic relation to the source word"""

RULES = """Rules the annotator follows:
- If the two words share a lemma and differ only in form, the answer is MORPH, never a sense relation.
- SYN only for words that could stand for each other; if they merely belong to the same field
  (grief and fear, sword and battle), the answer is SYN-DIST.
- SUBST is the answer whenever the replacement carries no semantic relation at all: a textual
  variant, a name swapped for a different referent, a word chosen for metre or allegory.
- Judge the two words as used in these passages, not their dictionary ranges."""

FEWSHOT = """Examples (invented, not from this corpus):
source word: ensem    reuse word: gladium   -> SYN
source word: canis    reuse word: animal    -> HYPER
source word: avis     reuse word: aquila    -> HYPO
source word: vitam    reuse word: mortem    -> ANT
source word: regis    reuse word: regia     -> POS
source word: Trojae   reuse word: Thebae    -> NE-SUB
source word: canebat  reuse word: cantat    -> MORPH
source word: luctus   reuse word: pavor     -> SYN-DIST
source word: navibus  reuse word: aratris   -> SUBST
source word: aurum    reuse word: ferrum    -> SUBST"""


#: What the reader's verdicts on this corpus actually look like: a lexical change
#: is as often *no* relation as a relation, and POS is commoner than any of the
#: wordnet relations. The models over-read: their commonest error is naming
#: SYN-DIST where the reader wrote SUBST.
PRIOR = """What these passages are like: when a later author changes a word, about half the
time there is no semantic relation at all -- a textual variant, a different name, a word
chosen for metre or for an allegory. Name a relation only when it is clear from the two
passages; otherwise the answer is SUBST.
Two pairs that are often confused:
- MORPH is the *same* word in another form (ponunt / ponens). POS is the *same stem* in a
  different part of speech (rex / regia, canere / cantus). If the part of speech changed,
  the answer is POS, not MORPH.
- SYN means the two words could stand for each other. Two words from one field that could
  not (grief / fear, ship / plough) are SYN-DIST at most, and often SUBST."""


ITEM = """SOURCE passage: {context_source}
REUSE passage:  {context_target}

source word: {source}
reuse word:  {target}"""

REASON_TAIL = """

Think in at most two sentences: do the words share a lemma, and if not, what is
the sense relation between them as used here? Then write the tag on the last
line, alone."""

PROPOSE_HEAD = """For the Latin word "{lemma}" ({pos}), give related Latin words.
Only words attested in classical or late Latin. Give the dictionary form."""

# =============================================================================
# Extra input: a word-by-word gloss produced by the model itself, and the
# typer's own lexical evidence for the pair (Julian's ask, 2026-09-09).
# =============================================================================

EVIDENCE_BLOCK = """
What the dictionaries and the lemmatiser say about this pair of words:
{flags}
"""

GLOSS_BLOCK = """
A word-by-word reading of the two passages:
{glosses}
"""

REASON_TAIL_2 = """

Think in at most two sentences, using the evidence above where it helps. Then write
the tag on the last line, alone."""


# =============================================================================
# PromptVariant and PromptRegistry
# =============================================================================


@dataclass(frozen=True)
class PromptVariant:
    """One prompt: a system message, a ``{...}``-templated user message, and the
    name of the reader in :data:`retexo.llm.llm_benchmark.PARSERS` that turns
    a reply into an answer."""

    system: str
    template: str
    parse: str


class PromptRegistry:
    """A named collection of :class:`PromptVariant` for one job (``type``,
    ``propose``, or ``gloss``).

    Behaves like a read-and-append mapping from variant name to
    :class:`PromptVariant`, plus :meth:`render`, which fills in the template's
    ``{...}`` fields and supplies the defaults every variant can rely on
    (``tags``, and the ``flags``/``glosses`` extra-input blocks when a variant
    does not use them).

    Example:
        ```python
        from retexo.llm.llm_prompts import TYPE

        system, user, parse = TYPE.render("T0", source="ensem", target="gladium",
                                          context_source="...", context_target="...")
        ```
    """

    def __init__(self, variants: Dict[str, PromptVariant] | None = None):
        self._variants: Dict[str, PromptVariant] = dict(variants or {})

    def __getitem__(self, name: str) -> PromptVariant:
        return self._variants[name]

    def __setitem__(self, name: str, variant: PromptVariant) -> None:
        self._variants[name] = variant

    def __contains__(self, name: str) -> bool:
        return name in self._variants

    def __iter__(self) -> Iterator[str]:
        return iter(self._variants)

    def __len__(self) -> int:
        return len(self._variants)

    def names(self) -> list[str]:
        """Every variant name, in insertion order."""
        return list(self._variants)

    def render(self, name: str, **fields) -> Tuple[str, str, str]:
        """The (system, user, parse) triple for ``name`` with ``fields`` filled in."""
        variant = self._variants[name]
        fields.setdefault("tags", ", ".join(TAGS))
        fields.setdefault("flags", "(not available)")  # the typer's evidence, when asked for
        fields.setdefault("glosses", "(not available)")  # the model's own word-by-word reading
        return variant.system, variant.template.format(**fields), variant.parse


# =============================================================================
# The type job's variants
# =============================================================================

TYPE = PromptRegistry(
    {
        "T0": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?
{tags}

Answer with the tag alone.""",
            "tag",
        ),
        "T1": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + """

Answer with the tag alone.""",
            "tag",
        ),
        "T2": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + """

Answer with the tag alone.""",
            "tag",
        ),
        "T3": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + "\n\n"
            + FEWSHOT
            + """

Answer with the tag alone.""",
            "tag",
        ),
        "T4": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + """

Think in at most two sentences: do the words share a lemma, and if not, what is
the sense relation between them as used here? Then write the tag on the last
line, alone.""",
            "last_tag",
        ),
        "T5": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Answer three questions about the two words, then give the tag.

1. Do they share a lemma (same word, different form)? yes / no
2. If not, is there a sense relation between them as used here? synonym / same field /
   more general / more specific / opposite / same stem other part of speech /
   different proper name / none
3. The tag from this list: """
            + ", ".join(TAGS)
            + """

Write the three answers on three lines, the tag alone on the last.""",
            "last_tag",
        ),
        "T6": PromptVariant(
            SYSTEM_LA,
            """Locus prior: {context_source}
Locus posterior: {context_target}

verbum prioris: {source}
verbum posterioris: {target}

Quomodo verbum posterius ad prius se habet? Elige unum:
"""
            + DEFINITIONS
            + """

Responde uno vocabulo.""",
            "tag",
        ),
        # T7-T9 add a prior over the class distribution (SUBST is the commonest answer)
        # and, for T9, a two-step framing that routes MORPH/POS out before naming a
        # sense relation.
        "T7": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + "\n\n"
            + PRIOR
            + REASON_TAIL,
            "last_tag",
        ),
        "T8": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + "\n\n"
            + PRIOR
            + "\n\n"
            + FEWSHOT
            + REASON_TAIL,
            "last_tag",
        ),
        "T9": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Answer in two steps.

Step 1. Is there any semantic relation between the two words as used in these passages,
or has the later author simply put a different word there? Remember that a different word
with no relation is the commonest case in this corpus.

Step 2. If there is no relation, the answer is SUBST. If the words share a lemma and
differ only in form, the answer is MORPH; if they share a stem across parts of speech,
POS. Otherwise choose the relation:

"""
            + DEFINITIONS
            + """

Write step 1 in one sentence, then the tag on the last line, alone.""",
            "last_tag",
        ),
        # T10 is T9 without the sentence that names the prior ("a different word with no
        # relation is the commonest case"). T9 beat every other prompt on the adjudicated
        # fine relations, where SUBST is 40% of the items, and lost badly on the coarse
        # MORPH-vs-SUBST test, where MORPH is 81% -- so the question is whether its gain
        # came from the two-step framing or from a prior that happened to match one test
        # set. T10 keeps the framing and drops the prior; T11 keeps neither step nor
        # prior but routes MORPH and POS out first, which is the other half of T9.
        "T10": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Answer in two steps.

Step 1. Is there any semantic relation between the two words as used in these passages,
or has the later author simply put a different word there?

Step 2. If there is no relation, the answer is SUBST. If the words share a lemma and
differ only in form, the answer is MORPH; if they share a stem across parts of speech,
POS. Otherwise choose the relation:

"""
            + DEFINITIONS
            + """

Write step 1 in one sentence, then the tag on the last line, alone.""",
            "last_tag",
        ),
        "T11": PromptVariant(
            SYSTEM_EN,
            ITEM
            + """

Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + """

Decide in this order, in at most two sentences: first, do the two words share a lemma and
differ only in form (MORPH)? If not, do they share a stem across parts of speech (POS)?
Only if neither, name the sense relation, or SUBST if there is none.

Then write the tag on the last line, alone.""",
            "last_tag",
        ),
        # T12-T15 add extra input: the typer's own lexical evidence (T12), the model's
        # own word-by-word gloss (T13), both (T14), and both on T9's two-step framing
        # (T15), so the effect of the extra input can be told apart from the framing.
        "T12": PromptVariant(
            SYSTEM_EN,
            ITEM
            + EVIDENCE_BLOCK
            + """
Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + REASON_TAIL_2,
            "last_tag",
        ),
        "T13": PromptVariant(
            SYSTEM_EN,
            ITEM
            + GLOSS_BLOCK
            + """
Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + REASON_TAIL_2,
            "last_tag",
        ),
        "T14": PromptVariant(
            SYSTEM_EN,
            ITEM
            + GLOSS_BLOCK
            + EVIDENCE_BLOCK
            + """
Which of these describes how the reuse word relates to the source word?

"""
            + DEFINITIONS
            + "\n\n"
            + RULES
            + REASON_TAIL_2,
            "last_tag",
        ),
        "T15": PromptVariant(
            SYSTEM_EN,
            ITEM
            + GLOSS_BLOCK
            + EVIDENCE_BLOCK
            + """
Answer in two steps.

Step 1. Is there any semantic relation between the two words as used in these passages,
or has the later author simply put a different word there?

Step 2. If there is no relation, the answer is SUBST. If the words share a lemma and
differ only in form, the answer is MORPH; if they share a stem across parts of speech,
POS. Otherwise choose the relation:

"""
            + DEFINITIONS
            + """

Write step 1 in one sentence, then the tag on the last line, alone.""",
            "last_tag",
        ),
    }
)


# =============================================================================
# The propose job's variants
# =============================================================================

PROPOSE = PromptRegistry(
    {
        "P0": PromptVariant(
            SYSTEM_EN,
            PROPOSE_HEAD
            + """

Answer as JSON only:
{{"SYN": [...], "HYPER": [...], "HYPO": [...], "ANT": [...]}}""",
            "json_relations",
        ),
        "P1": PromptVariant(
            SYSTEM_EN,
            PROPOSE_HEAD
            + """

SYN   words that could stand in its place with the same sense
HYPER more general words (its genus)
HYPO  more specific words (its species)
ANT   opposites

Up to three per relation, best first; leave a list empty rather than guessing.

Answer as JSON only:
{{"SYN": [...], "HYPER": [...], "HYPO": [...], "ANT": [...]}}""",
            "json_relations",
        ),
        "P2": PromptVariant(
            SYSTEM_EN,
            PROPOSE_HEAD
            + """

SYN   words that could stand in its place with the same sense
HYPER more general words (its genus)
HYPO  more specific words (its species)
ANT   opposites

Up to three per relation, best first; leave a list empty rather than guessing.

Example, for "gladius" (noun):
{{"SYN": ["ensis", "ferrum", "mucro"], "HYPER": ["telum", "arma"], "HYPO": [], "ANT": []}}

Answer as JSON only, in that shape.""",
            "json_relations",
        ),
        "P3": PromptVariant(
            SYSTEM_EN,
            PROPOSE_HEAD
            + """

SYN   words that could stand in its place with the same sense
HYPER more general words (its genus)
HYPO  more specific words (its species)
ANT   opposites

Up to three per relation, best first; leave a list empty rather than guessing.
Beside each word put a two-word gloss, so a reader can check it.

Answer as JSON only:
{{"SYN": [["ensis", "sword"], ...], "HYPER": [...], "HYPO": [...], "ANT": [...]}}""",
            "json_relations_glossed",
        ),
    }
)


# =============================================================================
# The gloss job's one variant (stage A of the evidence-fed prompts, T12-T15)
# =============================================================================

GLOSS = PromptRegistry(
    {
        "G0": PromptVariant(
            SYSTEM_EN,
            """Two Latin passages. The later one rewrites the earlier one.

EARLIER passage: {context_source}
LATER passage:   {context_target}

Render each passage word by word: every Latin word on its own line, followed by
its dictionary form and a short English gloss as used here.

EARLIER
<word> = <dictionary form> = <gloss>
...

LATER
<word> = <dictionary form> = <gloss>
...""",
            "gloss",
        ),
    }
)
