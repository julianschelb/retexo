# retexo/resources/wordnet.py
"""
Latin WordNet, read through a local cache.

Backs ``SYN``, ``HYPER``, ``HYPO`` and ``ANT``. Every response is written to
disk on first fetch so that bulk generation runs offline afterwards and a run
is reproducible against a pinned cache rather than against whatever the server
returns today.

Two details carried over deliberately. Lookups are restricted to a lemma's
primary synset rather than pooling every sense, because pooling produces
hundreds of spurious synonyms for any polysemous word. And the Exeter host
ships an expired certificate and redirects, so requests follow redirects with
verification disabled — without following redirects the endpoint answers 301
and every lookup silently returns nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

BASE = "https://latinwordnet.exeter.ac.uk/api"

#: Relation symbols used by the API, on the *synset* endpoint.
RELATIONS = {"@": "hypernyms", "~": "hyponyms", "!": "antonyms"}

#: Relation symbols on the *lemma* endpoint. Derivational relations hold
#: between lemmas rather than between senses -- *cano* to *carmen* is a fact
#: about the words, not about one of their meanings -- so they come from a
#: different endpoint and are fetched separately.
LEMMA_RELATIONS = {"/": "derivatives"}

EMPTY: Dict[str, List[str]] = {
    "synonyms": [],
    "hypernyms": [],
    "hyponyms": [],
    "antonyms": [],
    "derivatives": [],
}

#: Largest relation list we are willing to believe. Latin WordNet's synsets are
#: coarse, and a relation resolves to every lemma of the target synset, so a
#: broad synset yields hundreds of "synonyms" that share no sense: *populus*
#: paired with *glaeba*, *zona* with *aerarium*. Beyond this size the list is
#: describing a semantic field rather than a relation, and is discarded whole
#: rather than sampled, since there is no basis for choosing among its members.
#: Calibrated in ``calibrate_wordnet.py`` against the rate at which two lemmas
#: drawn at random are linked.
MAX_RELATION_SIZE = 24

# =============================================================================
# Client
# =============================================================================


class LatinWordNet:
    """Cached Latin WordNet client.

    Example:
        ```python
        wordnet = LatinWordNet(cache_dir=Path("cache/lwn"))
        wordnet.lookup("animus", "n")["synonyms"]
        ```
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        offline: bool = False,
        timeout: int = 25,
        max_synsets: int = 1,
        max_relation_size: int = MAX_RELATION_SIZE,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout = timeout
        self.max_synsets = max_synsets
        self.max_relation_size = max_relation_size
        self._http = None

    # ---------- Cache ----------

    def _path(self, lemma: str, pos: str) -> Path:
        return self.cache_dir / f"{pos}_{lemma.replace('/', '_')}.json"

    def cached_count(self) -> int:
        """How many lookups are already on disk."""
        return len(list(self.cache_dir.glob("*.json")))

    # ---------- Network ----------

    def _session(self):
        """One pooled HTTPS session per client.

        A lemma costs four or more requests, and ``requests.get`` opens a fresh
        TCP and TLS connection for every one of them. Reusing a session keeps
        the connection alive across all of them, which is most of the per-lemma
        cost on a link where the handshake dominates.
        """
        import requests
        import urllib3

        if getattr(self, "_http", None) is None:
            urllib3.disable_warnings()
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=16, pool_maxsize=32, max_retries=2
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.verify = False
            self._http = session
        return self._http

    def _get(self, path: str) -> dict:
        response = self._session().get(
            f"{BASE}/{path}",
            params={"format": "json"},
            timeout=self.timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response.json()

    def _synset_lemmas(self, pos: str, offset: str) -> List[str]:
        data = self._get(f"synsets/{pos}/{offset}/lemmas")
        out: List[str] = []
        for result in data.get("results", []):
            for group in result.get("lemmas", {}).values():
                out.extend(entry["lemma"] for entry in group)
        return out

    def _fetch(self, lemma: str, pos: str) -> Dict[str, List[str]]:
        record = {key: [] for key in EMPTY}
        synsets = self._get(f"lemmas/{lemma}/{pos}/synsets")

        offsets = []
        for result in synsets.get("results", []):
            for group in result.get("synsets", {}).values():
                offsets.extend((s["pos"], s["offset"]) for s in group)
        offsets = offsets[: self.max_synsets]

        for synset_pos, offset in offsets:
            for other in self._synset_lemmas(synset_pos, offset):
                if other != lemma:
                    record["synonyms"].append(other)
            relations = self._get(f"synsets/{synset_pos}/{offset}/relations")
            for result in relations.get("results", []):
                for symbol, targets in result.get("relations", {}).items():
                    field = RELATIONS.get(symbol)
                    if not field:
                        continue
                    for target in targets:
                        record[field].extend(self._synset_lemmas(target["pos"], target["offset"]))

        lemma_relations = self._get(f"lemmas/{lemma}/{pos}/relations")
        for result in lemma_relations.get("results", []):
            for symbol, targets in result.get("relations", {}).items():
                field = LEMMA_RELATIONS.get(symbol)
                if not field:
                    continue
                record[field].extend(target["lemma"] for target in targets if target.get("lemma"))

        return {k: sorted({x for x in v if x != lemma}) for k, v in record.items()}

    # ---------- Lookup ----------

    @staticmethod
    def _complete(record: Dict[str, List[str]]) -> bool:
        """Whether a cached record carries every relation we now ask for."""
        return all(key in record for key in EMPTY)

    def _filter(self, record: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Drop relation lists too large to be describing a relation.

        Applied on read rather than on write so the cache remains a faithful
        record of what the server returned, and so this policy can be changed
        and re-measured without re-fetching several thousand lookups.
        """
        if self.max_relation_size <= 0:
            return record
        out = dict(record)
        for key in EMPTY:
            values = record.get(key) or []
            out[key] = [] if len(values) > self.max_relation_size else list(values)
        return out

    def lookup(self, lemma: str, pos: str = "n") -> Dict[str, List[str]]:
        """Every relation for a lemma. Network only on a cache miss.

        A failed fetch is cached as empty with the error recorded, so a
        transient outage does not stall a bulk run and is visible afterwards.
        """
        if not lemma:
            return dict(EMPTY)
        path = self._path(lemma, pos)
        if path.exists():
            try:
                cached = json.loads(path.read_text())
            except (ValueError, OSError):
                cached = {}  # unreadable: treat as a miss and refetch
            # A record written before a relation was added lacks its key
            # entirely. Returning it as empty would silently disable whichever
            # operation depends on that relation, so an incomplete record is a
            # miss -- unless it failed, in which case refetching would only
            # repeat the failure.
            if self._complete(cached) or self.offline or "_error" in cached:
                return self._filter(cached)
        if self.offline:
            return dict(EMPTY)
        try:
            record = self._fetch(lemma, pos)
        except Exception as exc:  # noqa: BLE001 - a miss is cached, not raised
            record = {**EMPTY, "_error": str(exc)}
        # Written through a temporary file and renamed, so a concurrent reader
        # never sees a half-written record. Warming the cache in parallel while
        # a run is reading it is otherwise a torn-JSON crash waiting to happen.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False))
        temporary.replace(path)
        return self._filter(record)
