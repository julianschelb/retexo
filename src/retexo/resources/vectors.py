# retexo/resources/vectors.py
"""
Distributional similarity over Latin lemmas.

Backs ``SYN-DIST``, the soft high-recall signal that complements WordNet's
precise but sparse relations. Vectors are lemma-trained, so lookups are keyed
on lemmas rather than surface forms.

The model file is not in the repository. When it is absent the resource
reports itself unavailable and the relations that depend on it are skipped,
rather than the oracle failing or, worse, quietly returning zero similarity for
everything and making every pair look unrelated.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

# =============================================================================
# Backend
# =============================================================================


class Vectors:
    """Gensim word vectors over Latin lemmas.

    Example:
        ```python
        vectors = Vectors(Path("data/raw/word2vec_models/latin_w2v_bamman"))
        vectors.available
        vectors.similarity("animus", "mens")
        ```
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else None
        self._vectors = None

    # ---------- Availability ----------

    @property
    def available(self) -> bool:
        """Whether a model file is present to load."""
        return bool(self.model_path and self.model_path.exists())

    def _get(self):
        if self._vectors is None:
            if not self.available:
                raise RuntimeError(
                    f"no word vectors at {self.model_path}; SYN-DIST is unavailable"
                )
            import gensim

            model = gensim.models.Word2Vec.load(str(self.model_path))
            self._vectors = getattr(model, "wv", model)
        return self._vectors

    # ---------- Lookup ----------

    def contains(self, lemma: str) -> bool:
        """Whether a lemma is in the vocabulary."""
        vectors = self._get()
        index = getattr(vectors, "key_to_index", None)
        return lemma in (index if index is not None else vectors.vocab)

    @functools.lru_cache(maxsize=100_000)
    def similarity(self, lemma_a: str, lemma_b: str) -> Optional[float]:
        """Cosine similarity, or ``None`` if either lemma is out of vocabulary.

        ``None`` rather than ``0.0``: an unknown word is not a dissimilar word,
        and collapsing the two would make missing coverage look like evidence
        of unrelatedness.
        """
        if lemma_a == lemma_b:
            return 1.0
        try:
            if not (self.contains(lemma_a) and self.contains(lemma_b)):
                return None
            return float(self._get().similarity(lemma_a, lemma_b))
        except Exception:
            return None
