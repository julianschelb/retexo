# retexo/resources/__init__.py
"""
Lexical and morphological resources, loaded lazily and degrading gracefully.

Relations declare which resources they need; this bundle decides whether those
resources are actually available and loads them on first use. A relation whose
resource is missing is skipped rather than raising, so the oracle still runs
and simply produces a coarser analysis — which is also what makes a
resource-free run possible in tests.

Availability is checked without loading, so asking whether ``SYN-DIST`` is
possible does not pull a 300-dimensional model into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from retexo.core.normalize import normalize
from retexo.paths import home
from retexo.resources.entities import Entities
from retexo.resources.morphology import Morphology
from retexo.resources.vectors import Vectors
from retexo.resources.wordnet import LatinWordNet

#: Where cached WordNet responses live by default.
DEFAULT_CACHE = home() / "resources_cache" / "lwn"

#: Where the Latin word2vec model is expected, if present.
DEFAULT_VECTORS = home() / "data" / "raw" / "word2vec_models" / "latin_w2v_bamman_lemma300_100_1"

# =============================================================================
# Bundle
# =============================================================================


@dataclass
class Resources:
    """Lazily-loaded backends for morphology, WordNet and vectors.

    Example:
        ```python
        resources = Resources()
        resources.has("wordnet")      # True if the cache or the API is usable
        resources.lemma_or_surface("armis")
        ```
    """

    cache_dir: Path = DEFAULT_CACHE
    vectors_path: Optional[Path] = DEFAULT_VECTORS
    offline: bool = False
    enabled: Optional[Dict[str, bool]] = None

    _morphology: Optional[Morphology] = field(default=None, repr=False)
    _wordnet: Optional[LatinWordNet] = field(default=None, repr=False)
    _vectors: Optional[Vectors] = field(default=None, repr=False)
    _entities: Optional[Entities] = field(default=None, repr=False)

    # ---------- Availability ----------

    def has(self, name: str) -> bool:
        """Whether a named resource can be used, without loading it."""
        if self.enabled is not None and not self.enabled.get(name, True):
            return False
        if name == "morphology":
            try:
                import cltk  # noqa: F401

                return True
            except Exception:
                return False
        if name == "wordnet":
            return True  # a cache miss returns empty rather than failing
        if name == "derivation":
            # Derivational relations are carried by the wordnet, on its lemma
            # endpoint. The operation asks for "derivation" because that is
            # what it needs; where it comes from is this bundle's business.
            return self.has("wordnet")
        if name == "vectors":
            return self.vectors.available
        if name == "entities":
            return self.entities.available
        return False

    def status(self) -> Dict[str, object]:
        """A summary suitable for a run record."""
        return {
            "morphology": self.has("morphology"),
            "wordnet": self.has("wordnet"),
            "wordnet_cached": self.wordnet.cached_count(),
            "wordnet_offline": self.offline,
            "vectors": self.has("vectors"),
            "vectors_path": str(self.vectors_path) if self.vectors_path else None,
            "entities": self.has("entities"),
            "entities_count": self.entities.count() if self.has("entities") else 0,
        }

    # ---------- Backends ----------

    @property
    def morphology(self) -> Morphology:
        """CLTK lemmatization and Collatinus inflection."""
        if self._morphology is None:
            self._morphology = Morphology()
        return self._morphology

    @property
    def wordnet(self) -> LatinWordNet:
        """Latin WordNet, read through the local cache."""
        if self._wordnet is None:
            self._wordnet = LatinWordNet(self.cache_dir, offline=self.offline)
        return self._wordnet

    @property
    def vectors(self) -> Vectors:
        """Distributional lemma vectors."""
        if self._vectors is None:
            self._vectors = Vectors(self.vectors_path)
        return self._vectors

    @property
    def entities(self) -> Entities:
        """Latin proper names, from CLTK's list."""
        if self._entities is None:
            self._entities = Entities()
        return self._entities

    # ---------- Convenience ----------

    def lemma_or_surface(self, token: str) -> str:
        """The lemma if morphology is available, else the normalized surface."""
        if self.has("morphology"):
            return self.morphology.lemma(token)
        return normalize(token)

    def pos_of(self, token: str) -> str:
        """The single most likely part of speech for a lexical lookup."""
        return self.pos_candidates(token)[0]

    def pos_candidates(self, token: str) -> tuple:
        """Parts of speech to try, most likely first.

        Collatinus leaves the part of speech unmarked for nominal forms, so a
        lookup that committed to one guess would send every adjective to the
        noun index. Callers try these in order and stop at the first hit.
        """
        if self.has("morphology"):
            return self.morphology.pos_candidates(token)
        return ("n", "a", "v", "r")


__all__ = ["DEFAULT_CACHE", "DEFAULT_VECTORS", "LatinWordNet", "Morphology", "Resources", "Vectors"]
