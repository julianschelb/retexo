# retexo/edit_typing/glosses.py
"""
E35: the operations as label tokens (GLiNER-style).

Each operation gets a short gloss that the encoder reads on every pass, prefixed
to the pair as ``[LBL] gloss_1 [LBL] gloss_2 ... [SEP] source ... [SEP] reuse``.
The typer then scores a (reuse word, source word) pair against each label's
``[LBL]`` state by a dot product instead of through a fixed output layer, so a
starved relation has a description before it has examples.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


class GlossTable:
    """Per-operation glosses the typer's label tokens read, in a chosen language.

    Args:
        variant: ``"latin"``, ``"english"``, or ``"none"`` (the ``[LBL]`` token
            alone, GLiNER2's ``-short``).

    Example:
        ```python
        GlossTable("latin").words(["NOP", "SYN"])
        ```
    """

    LATIN: Dict[str, str] = {
        "NOP": "idem verbum",
        "MORPH": "idem vocabulum alia forma",
        "SYN": "verbum eiusdem sententiae",
        "SYN-DIST": "verbum simile sententia",
        "HYPER": "verbum generalius",
        "HYPO": "verbum specialius",
        "ANT": "verbum contrarium",
        "NE-SUB": "aliud nomen proprium",
        "POS": "eadem radix alia pars orationis",
        "SPLIT": "unum verbum in duo divisum",
        "MERGE": "duo verba in unum coniuncta",
        "SUBST": "aliud verbum",
    }

    ENGLISH: Dict[str, str] = {
        "NOP": "the same word",
        "MORPH": "same lemma different form",
        "SYN": "a synonym",
        "SYN-DIST": "a near synonym",
        "HYPER": "a more general word",
        "HYPO": "a more specific word",
        "ANT": "an opposite",
        "NE-SUB": "another proper name",
        "POS": "same stem different part of speech",
        "SPLIT": "one word split in two",
        "MERGE": "two words merged into one",
        "SUBST": "a different word",
    }

    VARIANTS = {"latin": LATIN, "english": ENGLISH}

    def __init__(self, variant: str):
        self.variant = variant

    def words(self, operations: Sequence[str]) -> List[List[str]]:
        """Per operation (in the model's order), the gloss as a word list."""
        if self.variant == "none":
            return [[] for _ in operations]
        table = self.VARIANTS[self.variant]
        out = []
        for op in operations:
            if op not in table:
                raise KeyError(f"no {self.variant} gloss for operation {op!r}")
            out.append(table[op].split())
        return out
