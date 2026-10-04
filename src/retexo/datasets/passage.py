# retexo/datasets/passage.py
"""
A passage with everything we know about each of its tokens.

The pipeline passes passages around as lists of bare strings, which means every
component that needs a lemma, a case or a part of speech asks for it again. This
class gathers that knowledge once, in one structure, so it can be inspected,
printed, or handed to a model as context.

**Nothing in the pipeline uses this.** It is deliberately standalone: the
oracle, the teacher and the formulations all still take token lists, and
``Passage.surfaces`` is the drop-in for them. Adding it to the pipeline would
change what the model sees, which is a decision to make on evidence rather than
in passing.

Annotation is staged by cost. Construction reads nothing. The local layer --
lemma, morphology, names, stopwords, syllables -- is computed on first access
and is fast. Lexical relations go to Latin WordNet, so they are opt-in via
``annotate(wordnet=True)`` and never happen by accident.

Every field degrades to ``None`` rather than raising when its resource is
missing, so a passage is still usable with no resources at all.

Example:
    ```python
    passage = Passage("arma virumque cano Troiae")
    passage.to_frame()          # a table, one row per token
    print(passage.render())     # the same thing as prompt context
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

from retexo.core.normalize import normalize
from retexo.resources import Resources

# =============================================================================
# One token
# =============================================================================


@dataclass(frozen=True)
class TokenInfo:
    """Everything known about a single token, in its passage.

    ``None`` means *not established* -- the resource was absent, or could not
    analyse this form -- and is deliberately distinct from an empty tuple,
    which means *asked, and the answer was nothing*.
    """

    index: int
    surface: str
    normalized: str

    #: Local layer: morphology, names, function words, sound.
    lemma: Optional[str] = None
    pos: Optional[str] = None
    morpho_tag: Optional[str] = None
    features: Optional[str] = None
    #: Every analysis the form admits, not only the first. Latin is richly
    #: ambiguous -- *oris* is dative or ablative plural of *ora*, and also
    #: genitive singular of *os* -- and a single reading hides that.
    readings: Optional[Tuple[str, ...]] = None
    is_name: bool = False
    is_stopword: bool = False
    enclitic: Optional[Tuple[str, str]] = None
    syllables: Optional[Tuple[str, ...]] = None

    #: Lexical layer: only populated by ``annotate(wordnet=True)``.
    synonyms: Optional[Tuple[str, ...]] = None
    hypernyms: Optional[Tuple[str, ...]] = None
    hyponyms: Optional[Tuple[str, ...]] = None
    antonyms: Optional[Tuple[str, ...]] = None
    derivatives: Optional[Tuple[str, ...]] = None
    has_vector: Optional[bool] = None

    def summary(self) -> str:
        """A one-line gloss, for a person reading a passage quickly."""
        bits = [self.surface]
        if self.lemma and self.lemma != self.normalized:
            bits.append(f"<{self.lemma}>")
        if self.features:
            bits.append(self.features)
        elif self.pos:
            bits.append(self.pos)
        if self.is_name:
            bits.append("NAME")
        if self.is_stopword:
            bits.append("func")
        return " ".join(bits)


# =============================================================================
# The passage
# =============================================================================


class Passage:
    """A Latin passage that knows about its own tokens.

    Indexing and iteration yield :class:`TokenInfo`. ``surfaces`` gives the
    plain token list the rest of the pipeline expects, and ``str()`` gives the
    original text back.
    """

    def __init__(
        self,
        text: str | Sequence[str],
        resources: Optional[Resources] = None,
    ) -> None:
        if isinstance(text, str):
            self.text = text
            self._surfaces: List[str] = text.split()
        else:
            self._surfaces = list(text)
            self.text = " ".join(self._surfaces)
        self.resources = resources if resources is not None else Resources()
        self._tokens: Optional[List[TokenInfo]] = None
        self._wordnet_done = False

    # ---------- Sequence behaviour ----------

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Passage({self.text[:40]!r}, {len(self._surfaces)} tokens)"

    def __len__(self) -> int:
        return len(self._surfaces)

    def __getitem__(self, index: int) -> TokenInfo:
        return self.tokens[index]

    def __iter__(self) -> Iterator[TokenInfo]:
        return iter(self.tokens)

    @property
    def surfaces(self) -> List[str]:
        """The bare token list, for code that expects one."""
        return list(self._surfaces)

    # ---------- Annotation ----------

    @property
    def tokens(self) -> List[TokenInfo]:
        """Annotated tokens, computing the local layer on first access."""
        if self._tokens is None:
            self._tokens = [self._annotate_local(i, s)
                            for i, s in enumerate(self._surfaces)]
        return self._tokens

    def _annotate_local(self, index: int, surface: str) -> TokenInfo:
        """Everything obtainable without going to the network."""
        resources = self.resources
        lemma = pos = morpho_tag = features = None
        enclitic = None
        readings = None
        if resources.has("morphology"):
            morphology = resources.morphology
            lemma = morphology.lemma(surface) or None
            pos = morphology.pos(surface)
            morpho_tag = morphology.morpho_label(surface)
            features = morphology.morpho_features(surface)
            enclitic = morphology.split_enclitic(surface)
            readings = _readings(surface, lemma)

        is_name = bool(resources.has("entities")
                       and resources.entities.is_name(surface))

        return TokenInfo(
            index=index,
            surface=surface,
            normalized=normalize(surface),
            lemma=lemma,
            pos=pos,
            morpho_tag=morpho_tag,
            features=features,
            readings=readings,
            is_name=is_name,
            is_stopword=_is_stopword(surface),
            enclitic=enclitic,
            syllables=_syllabify(surface),
        )

    def annotate(self, *, wordnet: bool = False, vectors: bool = False) -> "Passage":
        """Add the layers that cost something. Returns self, for chaining.

        ``wordnet`` goes to Latin WordNet, one lookup per distinct lemma, and
        is cached on disk between runs. ``vectors`` only records whether a
        lemma has a vector at all, which is what decides whether ``SYN-DIST``
        could ever fire on it.
        """
        if not (wordnet or vectors):
            return self
        updated: List[TokenInfo] = []
        for token in self.tokens:
            changes = {}
            if wordnet and self.resources.has("wordnet") and token.lemma:
                record = self.resources.wordnet.lookup(
                    token.lemma, token.pos or "n"
                )
                changes.update(
                    synonyms=tuple(record.get("synonyms", ())),
                    hypernyms=tuple(record.get("hypernyms", ())),
                    hyponyms=tuple(record.get("hyponyms", ())),
                    antonyms=tuple(record.get("antonyms", ())),
                    derivatives=tuple(record.get("derivatives", ())),
                )
            if vectors and token.lemma:
                changes["has_vector"] = bool(
                    self.resources.has("vectors")
                    and self.resources.vectors.contains(token.lemma)
                )
            updated.append(
                TokenInfo(**{**token.__dict__, **changes}) if changes else token
            )
        self._tokens = updated
        self._wordnet_done = self._wordnet_done or wordnet
        return self

    # ---------- Views ----------

    def rows(self) -> List[dict]:
        """One record per token, for tabular display."""
        out = []
        for token in self.tokens:
            row = {
                "i": token.index,
                "surface": token.surface,
                "lemma": token.lemma or "—",
                "pos": token.pos or "—",
                "features": token.features or "—",
                "tag": token.morpho_tag or "—",
                "readings": " | ".join(token.readings) if token.readings else "—",
                "name": "yes" if token.is_name else "",
                "func": "yes" if token.is_stopword else "",
                "enclitic": "+".join(token.enclitic) if token.enclitic else "—",
                "syllables": "-".join(token.syllables) if token.syllables else "—",
            }
            if self._wordnet_done:
                row["syn"] = len(token.synonyms or ())
                row["hyper"] = len(token.hypernyms or ())
                row["deriv"] = len(token.derivatives or ())
            out.append(row)
        return out

    def to_frame(self):
        """The passage as a DataFrame, one row per token."""
        import pandas as pd

        return pd.DataFrame(self.rows()).set_index("i")

    def render(self, *, relations: bool = False, max_relations: int = 6) -> str:
        """The passage as text, for use as model context.

        Written to be read by a model rather than rendered in a terminal: one
        token per line, fields named, and anything unestablished simply absent
        rather than filled with a placeholder that would have to be explained.
        """
        lines = [f"PASSAGE: {self.text}", f"TOKENS: {len(self)}", ""]
        for token in self.tokens:
            parts = [f"{token.index}", token.surface]
            if token.lemma and token.lemma != token.normalized:
                parts.append(f"lemma={token.lemma}")
            if token.pos:
                parts.append(f"pos={token.pos}")
            if token.readings and len(token.readings) > 1:
                parts.append(f"morph={' or '.join(token.readings)}")
            elif token.features:
                parts.append(f"morph={token.features}")
            if token.is_name:
                parts.append("proper-name")
            if token.is_stopword:
                parts.append("function-word")
            if token.enclitic:
                parts.append(f"enclitic={token.enclitic[0]}+{token.enclitic[1]}")
            lines.append("  ".join(parts))
            if relations:
                for label, values in (
                    ("syn", token.synonyms), ("hyper", token.hypernyms),
                    ("hypo", token.hyponyms), ("ant", token.antonyms),
                    ("deriv", token.derivatives),
                ):
                    if values:
                        shown = ", ".join(values[:max_relations])
                        more = f" (+{len(values) - max_relations})" if len(values) > max_relations else ""
                        lines.append(f"      {label}: {shown}{more}")
        return "\n".join(lines)

    def summary(self) -> dict:
        """Coverage counts — what is actually known about this passage."""
        tokens = self.tokens
        total = len(tokens) or 1
        return {
            "tokens": len(tokens),
            "lemmatized": sum(1 for t in tokens if t.lemma),
            "with_features": sum(1 for t in tokens if t.features),
            "proper_names": sum(1 for t in tokens if t.is_name),
            "function_words": sum(1 for t in tokens if t.is_stopword),
            "with_enclitic": sum(1 for t in tokens if t.enclitic),
            "feature_coverage": sum(1 for t in tokens if t.features) / total,
        }


# =============================================================================
# Optional backends
# =============================================================================


def _readings(surface: str, lemma: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Every morphological analysis Collatinus admits for this surface form.

    ``Morphology.morpho_features`` commits to the first match, which is right
    for a detector that needs one answer. Context for a model is better served
    by the ambiguity itself, so this collects them all.
    """
    if not lemma:
        return None
    try:
        from cltk.morphology.lat import CollatinusDecliner

        from retexo.resources.morphology import _decode_tag

        target = normalize(surface)
        found = []
        for form, tag in CollatinusDecliner().decline(lemma):
            if normalize(form) == target:
                decoded = _decode_tag(tag)
                if decoded and decoded not in found:
                    found.append(decoded)
        return tuple(found) or None
    except Exception:
        return None


def _is_stopword(token: str) -> bool:
    """Whether CLTK lists the token as a Latin function word."""
    try:
        from cltk.stops.lat import STOPS

        return normalize(token) in {normalize(w) for w in STOPS}
    except Exception:
        return False


def _syllabify(token: str) -> Optional[Tuple[str, ...]]:
    """Syllables, if CLTK's prosody module can divide the word."""
    try:
        from cltk.prosody.lat.scanner import Syllabifier

        return tuple(Syllabifier().syllabify(token))
    except Exception:
        return None


__all__ = ["Passage", "TokenInfo"]
