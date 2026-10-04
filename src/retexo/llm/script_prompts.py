# retexo/llm/script_prompts.py
"""The script-writing prompts of the E32 benchmark and the E40 rater, verbatim.

The texts are those of ``attic/scripts/run_e32_script.py`` (the output format,
the two operation inventories, the task, the inline-evidence note) and
``attic/scripts/generate_scripts.py`` (the corpus conventions, the reuse gate of
round 2, the closing instruction). ``PromptVariants`` assembles them into the
named variants P0 to P5; ``build_prompt`` fills one for a pair. Nothing here
calls a model.
"""

from __future__ import annotations

from typing import Dict, Tuple

from retexo.llm.llm_prompts import SYSTEM_EN

WORD_SCHEME = """OUTPUT FORMAT

Write exactly one line for every word of the LATER passage, in the order they appear.
{n_lines_note}Nothing else: no heading, no commentary, no blank lines.

Each line gives the later word, then the word of the earlier passage it came from, then
the operation:

  <later word> <- <earlier word>  <operation>

and, when it does not come from the earlier passage:

  <later word> <- -  <operation>

Copy both words exactly as they are printed, including any punctuation attached to them.
Do not use numbers.

The operations, exactly these words:

{operations}

Each word of the earlier passage may be used on at most one line.

Example of a well-formed answer, for a later passage of four words (not this one):

  Platonis <- -  FRAME
  uitam <- vita,  MORPH
  mortis. <- mortis  NOP
  sapienti <- -  INS

"""

COARSE_OPS = """  NOP    taken unchanged from that word of the earlier passage
  MORPH  that word of the earlier passage, in a different form
  SUBST  a different word, put where that word of the earlier passage stood
  INS    the later author's own word, from nowhere in the earlier passage
  FRAME  a word of the citing formula ("as Virgil says"), the later author's own"""

FINE_OPS = """  INS       the later author's own word, from nowhere in the earlier passage
  FRAME     a word of the citing formula ("as Virgil says"), the later author's own
  NOP       the same word, unchanged
  MORPH     the same word in a different form (case, tense, number)
  POS       the same stem in a different part of speech (rex / regia, canere / cantus)
  SYN       a synonym: a word that could stand in its place
  SYN-DIST  a word from the same field that could not stand in its place (grief / fear)
  HYPER     a more general word than the earlier one
  HYPO      a more specific word than the earlier one
  ANT       an opposite
  NE-SUB    a different proper name
  SPLIT     this word and the next together render one earlier word
  MERGE     this word alone renders two earlier words
  SUBST     a different word with no relation to the earlier one"""

TASK = """Below are two Latin passages. The later one may draw on the earlier one.
Decide, word by word, which words of the later passage were taken from the earlier one and
which the later author wrote himself, and write the edit script."""

INLINE_NOTE = """Each word is followed, in brackets, by what a lemmatiser and a Latin
dictionary say about it: its dictionary form, its part of speech, and any related words the
dictionary records. Those are facts about single words alone. They do not say which word
came from which -- that is what you decide. Two cautions from this corpus: two words with
the same dictionary form were written independently about three times in ten, especially
forms of "esse" and words of the citing formula; and a word the dictionary connects to
nothing may still be taken from the earlier passage through a paraphrase it does not
record."""

FIXES = """
Two conventions of this corpus:
- Punctuation, capitalisation and the spellings u/v and i/j are NOT a change of form.
  "haesit" and "haesit." are the same word (NOP); so are "non" and "Non", "iuuenum" and
  "iuvenum". MORPH is for a real change of case, number, tense or person.
- The citing formula -- the words with which the later author introduces or attributes the
  quotation, such as "ut ait Vergilius", "Plato's opinion is", "the poet says" -- is FRAME,
  not INS. Before you write the lines, find the citing formula if there is one.
"""

GATE = """
IMPORTANT -- there may be no reuse at all. The pairs you see were found by a search for
passages with two or three words in common, and in most of them the words coincide by
chance: both authors happen to write "et", "est", "quod", "in", "non", or a common noun.
Nothing about a pair guarantees that the later author knew the earlier passage.

A word is taken from the earlier passage only when it stands inside a reused stretch: a
run of the later passage in which at least two content words (nouns, verbs, adjectives,
names) reproduce the earlier passage close together and in the same order, or one rare
and unmistakable phrase. Inside such a stretch, a word that fills the place of an earlier
word is linked to it (NOP, MORPH, or a replacement). Outside any such stretch every word
is INS, even when the same word happens to occur in the earlier passage. Shared function
words, forms of "esse", and common verbs are never evidence on their own, and a verb is
not linked to an unrelated verb just because both are verbs.

If there is no reused stretch, every line is INS (or FRAME for a citing formula).
"""

GATE2 = """
Two identical words are not a stretch by themselves, nor are three scattered ones. Before
you link anything, quote to yourself the reused stretch from both passages. If you cannot
quote one that holds at least two content words close together, the pair is a coincidence
of vocabulary: write INS on every line (FRAME for a citing formula) and link nothing, not
even the words that happen to be identical. Ask: would a reader who knows the earlier
passage recognise it in the later one? If not, there is no reuse.
"""

SLOT = """
Inside a reused stretch, be complete: the later word that stands where an earlier word
stood is linked to that earlier word even when the two are unrelated (SUBST), and a word
that only changed its ending is MORPH. Outside the stretch, be strict: a word that merely
repeats a word of the earlier passage somewhere else is INS.
"""

STRETCH_LINE = """
Before the word lines, write one line

  REUSED: <the reused stretch, quoted from the earlier passage> => <the same stretch in the later passage>

with several stretches separated by " ; ", or  REUSED: none  when nothing is reused. Then the
word lines, one per word of the later passage.
"""

CLOSE = """
Work through the later passage in order. For each word ask first whether it lies inside a
reused stretch at all -- most words of a later passage are the author's own and get INS --
and only then which word it came from and how it changed.{relation_note}

Answer with the lines only."""

REL_FINE = """ Name the closest relation you can
defend from these two passages; when a word was simply replaced by an unrelated one, say
SUBST rather than reaching for a relation."""

USER_TURN = """EARLIER passage:
{source_annotated}

LATER passage:
{target_annotated}

{n_lines_note}Answer with the lines only."""


# =============================================================================
# The variants
# =============================================================================


class PromptVariants:
    """The named prompt variants: ``(static system text, user template)``.

    Everything that does not change from pair to pair sits in the system block,
    which the API caches; the user turn carries the two annotated passages.

    Example:
        ```python
        system, template = PromptVariants.get("P4")
        ```
    """

    @staticmethod
    def build(ops: str, stretch: bool, gate: str = GATE) -> Tuple[str, str]:
        static = (SYSTEM_EN + "\n\n" + TASK + "\n\n" + INLINE_NOTE + "\n" + FIXES + gate + "\n"
                  + WORD_SCHEME.replace("{operations}", ops).replace("{n_lines_note}", "")
                  + (STRETCH_LINE if stretch else "")
                  + CLOSE.format(relation_note=REL_FINE if ops is FINE_OPS else ""))
        return static, USER_TURN

    @classmethod
    def all(cls) -> Dict[str, Tuple[str, str]]:
        round_one = (TASK + "\n\n" + INLINE_NOTE + "\n" + FIXES + """
EARLIER passage:
{source_annotated}

LATER passage:
{target_annotated}

""" + WORD_SCHEME.replace("{operations}", FINE_OPS) + """
Work through the later passage in order. For each word ask first whether it comes from the
earlier passage at all -- most words of a later passage are the author's own and get INS --
and only then which word it came from and how it changed. Name the closest relation you can
defend from these two passages; when a word was simply replaced by an unrelated one, say
SUBST rather than reaching for a relation.

Answer with the lines only, one per word of the later passage.""")
        return {
            "P0": (SYSTEM_EN, round_one),                                  # round 1
            "P1": cls.build(FINE_OPS, stretch=True),                       # reuse gate + REUSED line, fine inventory
            "P2": cls.build(FINE_OPS, stretch=False),                      # reuse gate only
            "P3": cls.build(COARSE_OPS, stretch=True),                     # reuse gate + REUSED line, coarse inventory
            "P4": cls.build(FINE_OPS, stretch=False, gate=GATE + GATE2),   # gate + "identical words are not a stretch"
            "P5": cls.build(FINE_OPS, stretch=False, gate=GATE + GATE2 + SLOT),   # P4 + complete inside, strict outside
        }

    @classmethod
    def get(cls, name: str) -> Tuple[str, str]:
        variants = cls.all()
        if name not in variants:
            raise KeyError(f"unknown prompt variant {name!r}; expected one of {sorted(variants)}")
        return variants[name]


#: The rater's system text (E40): the coarse inventory, no reuse gate, the lines only.
RATER_SYSTEM = (SYSTEM_EN + "\n\n" + TASK + "\n\n" + INLINE_NOTE + "\n"
                + WORD_SCHEME.replace("{operations}", COARSE_OPS).replace("{n_lines_note}", "")
                + "\nWork through the later passage in order. Answer with the lines only.")

#: The rater's user turn (E40).
RATER_USER = "EARLIER passage:\n{src}\n\nLATER passage:\n{tgt}\n\nThat is {n} lines, one per word, in order. Answer with the lines only."
