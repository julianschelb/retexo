# retexo/resources/entities.py
"""
Latin proper names, for ``NE-SUB``.

Backs the one thing the operation needs that no lexicon supplies: whether a
token is a name at all. The list ships with CLTK -- 39,725 forms drawn from the
classical corpus, including inflected ones, so *Troiae* and *Italiam* are found
directly without lemmatization, which for proper names is unreliable (*Troiae*
lemmatizes to the adjective *Troius*).

**Matching is case-sensitive, and that is the whole design.** Every entry in the
list is capitalized. Folding case makes the test useless: 105 of the 300
commonest words in the benchmark corpus then match, including *in*, *non*,
*cum* and *me*. Requiring the capital cuts that to 13, of which 12 are real
names (*Catullus*, *Lesbia*, *Fabulle*).

This is not the capitalization test that the earlier prototype was rejected
for. That test asked only whether a token was capitalized, and so fired on
every sentence-initial word. This asks whether a capitalized token is *also* in
a curated name list, which *Arma* -- sentence-initial in the Aeneid -- is not.
One residual failure mode remains: a sentence-initial common word that happens
to be a listed name, such as *Non*. Callers should treat membership as
necessary evidence, not sufficient.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional, Set

#: Where CLTK keeps the Latin proper-name list once its models are installed.
DEFAULT_NAMES = Path(
    os.path.expanduser("~/cltk_data/lat/model/lat_models_cltk/ner/proper_names.txt")
)


class Entities:
    """Case-sensitive membership test against a list of Latin proper names.

    Example:
        ```python
        entities = Entities()
        entities.is_name("Troiae")    # True
        entities.is_name("Arma")      # False -- capitalized, but not a name
        ```
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_NAMES
        self._names: Optional[Set[str]] = None

    @property
    def available(self) -> bool:
        """Whether the name list is present, without reading it."""
        return self.path.exists()

    @property
    def names(self) -> Set[str]:
        """The name list, read once."""
        if self._names is None:
            if not self.path.exists():
                self._names = set()
            else:
                self._names = {
                    line.strip()
                    for line in self.path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
        return self._names

    @functools.lru_cache(maxsize=50_000)
    def is_name(self, token: str) -> bool:
        """Whether a token is capitalized and listed as a proper name."""
        cleaned = "".join(ch for ch in token if ch.isalpha())
        if not cleaned or not cleaned[:1].isupper():
            return False
        return cleaned in self.names

    def count(self) -> int:
        """How many names are loaded, for the run record."""
        return len(self.names)
