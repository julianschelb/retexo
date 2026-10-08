# retexo/datasets/composite.py
"""
Routing each operation to the source that can actually produce it.

No single substitution source realises the whole inventory. A distributional
model finds words close in meaning but cannot tell a hypernym from an antonym;
a morphological analyser inflects but does not substitute; WordNet names typed
relations but is also what the oracle detects with.

Restricting generation to what one source can do honestly (D-44) turned out to
be the wrong trade. Measured on the benchmark, the real pairs contain 6
``HYPER`` operations, 14 ``HYPO`` and 1 ``ANT`` across 1,211 training pairs,
with 2, 3 and 1 in the held-out set. Those operations cannot be learned from
real data at all, and refusing to generate them guarantees an F1 of zero rather
than risking a circular one.

This router therefore sends each tag to whichever source can express it:
morphology for ``MORPH``, the contextual model for ``SYN-DIST``, and WordNet
for the typed relations. The circularity that motivated D-44 is real but is
better handled by measuring it than by abstaining: ``held_out_lemmas``
optionally withholds part of the lexicon from generation, so that whether the
model learned the relation type or merely memorised the lexicon becomes a
question with an answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set

#: Which source handles which tag. Structural operations need no source.
DEFAULT_ROUTING: Dict[str, str] = {
    "MORPH": "morphology",
    "SYN-DIST": "contextual",
    "SYN": "wordnet",
    "HYPER": "wordnet",
    "HYPO": "wordnet",
    "ANT": "wordnet",
}


@dataclass
class CompositeSubstitutionSource:
    """Dispatches a substitution to whichever source can realise its tag.

    Example:
        ```python
        source = CompositeSubstitutionSource(
            contextual=contextual_source, lexical=lexical_source
        )
        source.replacement_in_context(words, 3, "HYPER", rng)
        ```
    """

    #: Handles SYN-DIST, and MORPH if no separate morphology source is given.
    contextual: object = None

    #: Handles the WordNet relations: SYN, HYPER, HYPO, ANT.
    lexical: object = None

    routing: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROUTING))

    #: Lemmas withheld from generation, so that transfer to unseen vocabulary
    #: can be measured rather than assumed. Empty means the whole lexicon is
    #: available, which is the faster and less rigorous setting.
    held_out_lemmas: Set[str] = field(default_factory=set)

    #: Counts per tag of what was asked for and what came back, so a run can
    #: report which operations the generator actually managed to produce
    #: instead of leaving it to be inferred from the resulting op mix.
    requested: Dict[str, int] = field(default_factory=dict)
    produced: Dict[str, int] = field(default_factory=dict)

    def supported(self) -> Set[str]:
        """Tags some configured source can realise."""
        out = set()
        for tag, which in self.routing.items():
            if (
                which == "wordnet"
                and self.lexical is not None
                or which in ("contextual", "morphology")
                and self.contextual is not None
            ):
                out.add(tag)
        return out

    def realises(self, tag: str) -> bool:
        """Whether a substitution under ``tag`` would mean what it says."""
        return tag in self.supported()

    def replacement(self, token: str, tag: str, rng: random.Random) -> Optional[str]:
        """Context-free substitution, deferred to the routed source."""
        source = self._source_for(tag)
        if source is None or not hasattr(source, "replacement"):
            return None
        return source.replacement(token, tag, rng)

    def replacement_in_context(
        self,
        words: Sequence[str],
        position: int,
        tag: str,
        rng: random.Random,
    ) -> Optional[str]:
        """A substitute for ``words[position]`` under ``tag``."""
        self.requested[tag] = self.requested.get(tag, 0) + 1
        source = self._source_for(tag)
        if source is None:
            return None

        if hasattr(source, "replacement_in_context"):
            result = source.replacement_in_context(words, position, tag, rng)
        else:
            result = source.replacement(words[position], tag, rng)

        # Withheld vocabulary is refused rather than substituted, so the
        # held-out half stays genuinely unseen during generation.
        if result and self.held_out_lemmas and result.lower() in self.held_out_lemmas:
            return None
        if result:
            self.produced[tag] = self.produced.get(tag, 0) + 1
        return result

    def yield_by_tag(self) -> Dict[str, str]:
        """How often each tag was asked for and how often it could be made."""
        return {
            tag: f"{self.produced.get(tag, 0)}/{count}"
            for tag, count in sorted(self.requested.items())
        }

    # ---------- Internals ----------

    def _source_for(self, tag: str):
        which = self.routing.get(tag)
        if which == "wordnet":
            return self.lexical
        if which in ("contextual", "morphology"):
            return self.contextual
        return None
