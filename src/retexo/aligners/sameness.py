# retexo/aligners/sameness.py
"""When two Latin words count as the same word.

Ported verbatim from ``run_e8.py`` (E8, the comparison policies), now under
``attic/scripts/``. :meth:`SamenessPolicy.same_current` is the rule the
pipeline uses today (equality under ``normalize``); the three wider rules fold
orthographic groups mined from the hand labels and were measured, not
adopted. :meth:`SamenessPolicy.operations` reads COPY / SUBST / INS off an
alignment under a chosen rule.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Dict, List

from retexo.core.normalize import normalize


class SamenessPolicy:
    """Four increasingly generous notions of "the same word", from today's
    rule to three measured (not adopted) extensions.

    Example:
        ```python
        SamenessPolicy.same_current("coelum", "caelum")     # False
        SamenessPolicy.same_extended("coelum", "caelum")     # True
        SamenessPolicy.RULES["+enclitic"]("tendens", "tendensque")  # True
        ```
    """

    #: Mined from the completed hand labels: every link labelled NOP whose two words
    #: `normalize` still reports as different. 138 links over 111 distinct forms,
    #: which fall into four groups of very different character.
    _ARCHAIC = (
        ("umus", "imus"),
        ("uma", "ima"),
        ("umum", "imum"),
        ("umis", "imis"),
        ("umae", "imae"),
        ("umo", "imo"),
    )

    @staticmethod
    def same_current(a: str, b: str) -> bool:
        """Today's rule."""
        return normalize(a) == normalize(b)

    @classmethod
    def _extended(cls, word: str) -> str:
        text = normalize(word).replace("oe", "ae")
        for archaic, classical in cls._ARCHAIC:
            if text.endswith(archaic):
                return text[: -len(archaic)] + classical
        return text

    @classmethod
    def same_extended(cls, a: str, b: str) -> bool:
        """Adds the two groups that are safe to fold outright.

        *coelum*/*caelum* (2% of the gap) and the archaic superlative *optuma*/
        *optima* (part of the tail). A general u->i fold would merge distinct
        lemmata, which `normalize.py` already warns about, so the archaic rule is
        restricted to those suffixes.
        """
        return cls._extended(a) == cls._extended(b)

    @staticmethod
    def _strip_enclitic(word: str) -> str:
        for enclitic in ("que", "ue", "ne"):
            if len(word) > len(enclitic) + 2 and word.endswith(enclitic):
                return word[: -len(enclitic)]
        return word

    @classmethod
    def same_enclitic(cls, a: str, b: str) -> bool:
        """Also treats *tendens* / *tendensque* as one word -- 38% of the gap.

        This is a labelling decision rather than an orthographic one: *-que* is a
        conjunction fused to the host word, so calling the pair identical is a
        convention, not a fact. It is the single largest group, so it is worth
        knowing what it is worth before deciding.
        """
        x, y = cls._extended(a), cls._extended(b)
        return x == y or cls._strip_enclitic(x) == cls._strip_enclitic(y)

    @staticmethod
    def _plural(word: str) -> str:
        return word[:-2] + "es" if word.endswith(("is", "es")) else word

    @classmethod
    def same_plural(cls, a: str, b: str) -> bool:
        """Also folds the archaic accusative plural *omnis* / *omnes* -- 13%.

        Unsafe in principle: *omnis* is also a nominative singular, so this merges
        two forms that a morphological analyser would keep apart. Included to
        measure, not to recommend.
        """
        x, y = cls._extended(a), cls._extended(b)
        return (
            x == y
            or cls._strip_enclitic(x) == cls._strip_enclitic(y)
            or cls._plural(cls._strip_enclitic(x)) == cls._plural(cls._strip_enclitic(y))
        )

    RULES: ClassVar[Dict[str, Callable[[str, str], bool]]] = {}

    @staticmethod
    def operations(examples, links, same: Callable[[str, str], bool]) -> List[List[str]]:
        """Read COPY/SUBST/INS off the alignment, under a given notion of sameness."""
        out = []
        for example, assigned in zip(examples, links):
            tags = []
            for word, token in enumerate(example.target_tokens):
                source = assigned[word] if word < len(assigned) else -1
                if source < 0 or source >= len(example.source_tokens):
                    tags.append("INS")
                elif same(token, example.source_tokens[source]):
                    tags.append("COPY")
                else:
                    tags.append("SUBST")
            out.append(tags)
        return out


SamenessPolicy.RULES = {
    "current": SamenessPolicy.same_current,
    "+folds": SamenessPolicy.same_extended,
    "+enclitic": SamenessPolicy.same_enclitic,
    "+plural": SamenessPolicy.same_plural,
}
#: Kept for callers that want the mapping without naming the class.
COMPARISON = SamenessPolicy.RULES
