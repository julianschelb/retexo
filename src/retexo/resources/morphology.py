# retexo/resources/morphology.py
"""
Lemmatization and inflectional morphology, via CLTK and Collatinus.

Backs the ``MORPH`` relation — same lemma, different surface form — and
supplies the lemma every lexical lookup is keyed on. Loading is lazy and both
lookups are memoised, because the oracle asks about the same token pairs
repeatedly and a relation between two lemmas does not depend on the passage
they came from.
"""

from __future__ import annotations

import functools
import re
from typing import List, Optional

from retexo.core.normalize import normalize

@functools.lru_cache(maxsize=1)
def _enclitic_exceptions() -> frozenset:
    """CLTK's curated list of words that only look as if they end in one."""
    try:
        import inspect

        from cltk.tokenizers.lat.lat import LatinWordTokenizer

        default = inspect.signature(
            LatinWordTokenizer.tokenize
        ).parameters["enclitics_exceptions"].default
        return frozenset(w.lower() for w in default)
    except Exception:
        return frozenset()


#: Enclitics that attach to a host word, longest first so *que* is tried
#: before a bare *ue* inside it.
ENCLITICS = ("que", "ue", "ve", "ne")

#: Shortest remainder worth believing. Below this almost any suffix match is
#: an accident of spelling rather than an enclitic.
MIN_ENCLITIC_STEM = 3

#: Collatinus tag positions, decoded. ``-`` means the slot does not apply.
_NUMBER = {"s": "sg", "p": "pl"}
_TENSE = {"p": "pres", "i": "impf", "f": "fut", "r": "perf", "l": "plup", "t": "futp"}
_MOOD = {"i": "ind", "s": "subj", "m": "imp", "n": "inf", "p": "part", "g": "ger",
         "d": "gerv", "u": "sup"}
_VOICE = {"a": "act", "p": "pass", "d": "dep"}
_CASE = {"n": "nom", "v": "voc", "a": "acc", "g": "gen", "d": "dat", "b": "abl",
         "l": "loc"}

def _decode_tag(label: Optional[str]) -> Optional[str]:
    """Decode one positional Collatinus tag into a readable gloss."""
    if not label or len(label) < 9:
        return None
    person, number, tense, mood, voice, _gender, case = (
        label[1], label[2], label[3], label[4], label[5], label[6], label[7]
    )
    parts = []
    if person in "123":
        parts.append(person)
    parts += [p for p in (
        _NUMBER.get(number), _TENSE.get(tense), _MOOD.get(mood),
        _VOICE.get(voice), _CASE.get(case),
    ) if p]
    return ".".join(parts) + "." if parts else None


# =============================================================================
# Backend
# =============================================================================


class Morphology:
    """CLTK lemmatizer with an optional Collatinus decliner.

    Example:
        ```python
        morphology = Morphology()
        morphology.lemma("armis")          # -> 'arma'
        morphology.same_lemma("arma", "armis")
        ```
    """

    def __init__(self) -> None:
        self._lemmatizer = None
        self._decliner = None

    # ---------- Backends ----------

    def _get_lemmatizer(self):
        if self._lemmatizer is None:
            from cltk.lemmatize.lat import LatinBackoffLemmatizer

            self._lemmatizer = LatinBackoffLemmatizer()
        return self._lemmatizer

    def _get_decliner(self):
        if self._decliner is None:
            from cltk.morphology.lat import CollatinusDecliner

            self._decliner = CollatinusDecliner()
        return self._decliner

    # ---------- Lookups ----------

    @functools.lru_cache(maxsize=50_000)
    def lemma(self, token: str) -> str:
        """The lemma of one surface token, normalized."""
        raw = re.sub(r"[^A-Za-z]", "", token)
        if not raw:
            return normalize(token)
        result = self._get_lemmatizer().lemmatize([raw])
        return normalize(result[0][1] if result else raw)

    def lemmas(self, tokens: List[str]) -> List[str]:
        """Lemmas for a whole passage, in one call to the backend."""
        raw = [re.sub(r"[^A-Za-z]", "", t) for t in tokens]
        return [normalize(lemma) for _, lemma in self._get_lemmatizer().lemmatize(raw)]

    def same_lemma(self, a: str, b: str) -> bool:
        """Whether two tokens share a lemma but differ on the surface.

        Identical surfaces are a copy, not an inflection, so they return
        ``False`` here and are caught by the identity relation first.
        """
        if normalize(a) == normalize(b):
            return False
        lemma_a = self.lemma(a)
        return bool(lemma_a) and lemma_a == self.lemma(b)

    @functools.lru_cache(maxsize=10_000)
    def morpho_label(self, token: str) -> Optional[str]:
        """A coarse morphological tag such as ``acc.sg.``, or ``None``.

        Best effort: Collatinus cannot decline every lemma, and a miss is not
        an error, only an absent justification on the operation.
        """
        target = normalize(token)
        try:
            forms = self._get_decliner().decline(self.lemma(token))
        except Exception:
            return None
        for surface, tag in forms:
            if normalize(surface) == target:
                return tag
        return None

    @functools.lru_cache(maxsize=10_000)
    def morpho_features(self, token: str) -> Optional[str]:
        """A readable gloss of a form's morphology, such as ``pl.dat.``.

        Collatinus tags are positional and nine characters wide: part of
        speech, person, number, tense, mood, voice, gender, case, degree, with
        ``-`` wherever a slot does not apply. ``--p----d-`` is a dative plural,
        ``v1spia---`` a first person singular present indicative active.

        This is what lets ``MORPH`` say *which* feature moved rather than only
        that something did. Best effort: a lemma Collatinus cannot decline
        yields ``None``, and the operation simply carries less justification.
        """
        return _decode_tag(self.morpho_label(token))

    @functools.lru_cache(maxsize=20_000)
    def pos_candidates(self, token: str) -> tuple:
        """Parts of speech to try in a lexical lookup, most likely first.

        Collatinus tags are positional, and only the first character carries
        the part of speech — ``v1spia---`` is a verb, while a nominal such as
        ``--s---mn-`` leaves it unmarked. An unmarked tag is genuinely
        ambiguous between noun and adjective, so rather than guessing we return
        an ordered list and let the caller try each. Returning a single guess
        here was silently sending every adjective to the noun index.
        """
        label = self.morpho_label(token) or ""
        head = label[:1]
        mapping = {"v": "v", "n": "n", "a": "a", "d": "r", "r": "r"}
        if head in mapping:
            primary = mapping[head]
            rest = tuple(p for p in ("n", "a", "v", "r") if p != primary)
            return (primary,) + rest
        # unmarked: nominal forms dominate, and adjectives are common in reuse
        return ("n", "a", "v", "r")

    def pos(self, token: str) -> str:
        """The single most likely part of speech. Prefer :meth:`pos_candidates`."""
        return self.pos_candidates(token)[0]

    # ---------- Enclitics ----------

    @functools.lru_cache(maxsize=20_000)
    def split_enclitic(self, token: str):
        """Split a trailing enclitic, as ``(stem, enclitic)``, or ``None``.

        Latin glues *-que*, *-ue* and *-ne* onto their host word, and separating
        them is the commonest cardinality change in verse: *uirumque* is *uirum*
        plus *que*.

        CLTK will not do this. ``LatinWordTokenizer`` advertises the feature and
        its own docstring promises ``abuterque -> ['abuter', '-que']``, but in
        1.1.7 the branch for ``que``/``ne``/``ue``/``ve`` appends the token
        unchanged while still marking it handled, so only ``n`` and ``st`` ever
        split. Its curated **exceptions** list is sound, though, and is reused
        here: 642 words that merely end in those letters -- *quoque*, *denique*,
        *neque*, *atque*, *usque* -- and must never be split.

        Past that veto the stem has to look like a real word, by either of two
        tests, because neither alone suffices. The lemmatizer catches inflected
        stems (*uirum* to *uir*) but not stems already in lemma form; Collatinus
        catches those (*arma*, *bonus*) but does not know every proper name.

        **Not wired into ``SPLIT`` detection.** That would need the oracle's
        alignment to go beyond one-to-one matching, which is a larger change
        than this. It is here to be measured against the residual first.
        """
        cleaned = re.sub(r"[^A-Za-z]", "", token)
        if not cleaned or cleaned.lower() in _enclitic_exceptions():
            return None
        for enclitic in ENCLITICS:
            if not cleaned.lower().endswith(enclitic):
                continue
            stem = cleaned[: -len(enclitic)]
            if len(stem) < MIN_ENCLITIC_STEM:
                continue
            inflected = self.lemma(stem) not in ("", normalize(stem))
            if inflected or self.morpho_label(stem) is not None:
                return stem, enclitic
        return None
