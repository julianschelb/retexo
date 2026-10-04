# retexo/pretraining/resource_pairs.py
"""The data of stage 0's objectives 2 and 3: resource pairs, lemma sentences, hard negatives.

Objective 2 (contrastive lemma pairs in context) needs lemma pairs the
resources call related and, for every lemma, sentences that contain it.
Objective 3 (pair identification) needs hard negatives beside the pool's
real pairs. All three builders write JSONL under ``data/pretrain/`` and
nothing is read off a gold link:

- ``ResourcePairBuilder``: the Latin WordNet cache (one JSON per lemma and
  part of speech, ``resources_cache/lwn/<pos>_<lemma>.json``) gives SYN pairs
  from ``synonyms`` and POS pairs from ``derivatives``; hypernyms, hyponyms
  and antonyms are the demoted relations of the paper definition and stay
  out; Bamman's lemma vectors add same-POS neighbours above a cosine
  threshold where the vectors are present. Name pairs are not built: the
  name list carries no classes.
- ``LemmaSentences``: the pool's passages lemmatised once (CLTK, cached) and
  indexed lemma -> sentences of 8 to 40 words; a lemma keeps its pairs only
  with at least ``min_sentences`` sentences.
- ``HardNegatives``: ``NegativeBuilder``'s three kinds over the benchmark.

    python -m retexo.pretraining.resource_pairs pairs --out data/pretrain/resource_pairs.jsonl
    python -m retexo.pretraining.resource_pairs sentences --pool data/pretrain/pairs.jsonl
    python -m retexo.pretraining.resource_pairs negatives --n 10000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

DEFAULT_CACHE = Path("resources_cache/lwn")
DEFAULT_OUT = Path("data/pretrain")

#: The cache file prefixes of the Latin WordNet and the part of speech they stand for.
POS_OF_PREFIX = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
_LEMMA_OK = re.compile(r"^[a-z]{3,}$")


@dataclass(frozen=True)
class ResourcePair:
    lemma_a: str
    lemma_b: str
    relation: str          # SYN, POS, VECTOR
    pos: str               # NOUN / VERB / ADJ / ADV (the part of speech both members share)
    weight: float

    def as_json(self) -> Dict[str, object]:
        return {"lemma_a": self.lemma_a, "lemma_b": self.lemma_b, "relation": self.relation, "pos": self.pos, "weight": self.weight}

    @classmethod
    def from_json(cls, obj: Dict[str, object]) -> "ResourcePair":
        return cls(str(obj["lemma_a"]), str(obj["lemma_b"]), str(obj["relation"]), str(obj["pos"]), float(obj["weight"]))


# =============================================================================
# Resource pairs
# =============================================================================


class ResourcePairBuilder:
    """Lemma pairs from the WordNet cache and, optionally, the lemma vectors.

    Example:
        ```python
        pairs = ResourcePairBuilder(Path("resources_cache/lwn")).build()
        ResourcePairBuilder.save(pairs, Path("data/pretrain/resource_pairs.jsonl"))
        ```
    """

    #: The demoted relations (paper definition section 3.1): never a contrastive positive.
    EXCLUDED = ("hypernyms", "hyponyms", "antonyms")

    def __init__(self, cache: Path = DEFAULT_CACHE, *, vectors=None, vector_threshold: float = 0.6, vector_top: int = 5):
        self.cache = Path(cache)
        self.vectors = vectors
        self.vector_threshold = vector_threshold
        self.vector_top = vector_top
        self.report: Counter = Counter()

    def entries(self) -> Iterator[Tuple[str, str, Dict[str, List[str]]]]:
        """``(lemma, pos, record)`` for every complete cache file."""
        for path in sorted(self.cache.glob("*.json")):
            prefix, _, lemma = path.stem.partition("_")
            pos = POS_OF_PREFIX.get(prefix)
            if pos is None or not _LEMMA_OK.match(lemma):
                self.report["skipped_name"] += 1
                continue
            try:
                record = json.loads(path.read_text())
            except (ValueError, OSError):
                self.report["skipped_unreadable"] += 1
                continue
            if "_error" in record:
                self.report["skipped_error"] += 1
                continue
            yield lemma, pos, record

    def wordnet_pairs(self) -> Iterator[ResourcePair]:
        for lemma, pos, record in self.entries():
            for other in record.get("synonyms", []) or []:
                other = str(other).lower()
                if _LEMMA_OK.match(other) and other != lemma:
                    self.report["SYN"] += 1
                    yield ResourcePair(lemma, other, "SYN", pos, 1.0)
            for other in record.get("derivatives", []) or []:
                other = str(other).lower()
                if _LEMMA_OK.match(other) and other != lemma:
                    self.report["POS"] += 1
                    yield ResourcePair(lemma, other, "POS", pos, 1.0)

    def vector_pairs(self, lemmas_by_pos: Dict[str, List[str]]) -> Iterator[ResourcePair]:
        """Same-POS nearest neighbours above the threshold, weighted by their cosine; only among the
        WordNet lemmas of the same part of speech (so the POS is known for both members)."""
        if self.vectors is None or not getattr(self.vectors, "available", False):
            return
        for pos, lemmas in lemmas_by_pos.items():
            present = [l for l in lemmas if self.vectors.contains(l)]
            for a in present:
                scored = []
                for b in present:
                    if b == a:
                        continue
                    sim = self.vectors.similarity(a, b)
                    if sim is not None and sim >= self.vector_threshold:
                        scored.append((sim, b))
                for sim, b in sorted(scored, reverse=True)[: self.vector_top]:
                    self.report["VECTOR"] += 1
                    yield ResourcePair(a, b, "VECTOR", pos, float(sim))

    def build(self, *, with_vectors: bool = True) -> List[ResourcePair]:
        pairs = list(self.wordnet_pairs())
        lemmas_by_pos: Dict[str, List[str]] = defaultdict(list)
        for lemma, pos, _ in self.entries():
            lemmas_by_pos[pos].append(lemma)
        if with_vectors:
            pairs.extend(self.vector_pairs(lemmas_by_pos))
        # one entry per unordered pair and relation, the higher weight kept
        best: Dict[Tuple[str, str, str], ResourcePair] = {}
        for p in pairs:
            key = (min(p.lemma_a, p.lemma_b), max(p.lemma_a, p.lemma_b), p.relation)
            if key not in best or p.weight > best[key].weight:
                best[key] = p
        out = list(best.values())
        self.report["pairs"] = len(out)
        return out

    @staticmethod
    def save(pairs: Sequence[ResourcePair], path: Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for p in pairs:
                handle.write(json.dumps(p.as_json(), ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def load(path: Path) -> List[ResourcePair]:
        return [ResourcePair.from_json(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# =============================================================================
# Lemma sentences
# =============================================================================


class LemmaSentences:
    """Sentences containing a lemma, from the pool's passages lemmatised once.

    Example:
        ```python
        index = LemmaSentences.build(pool_path, cache=Path("data/pretrain/corpus_lemmas.jsonl"))
        index.contexts("gladius")   # [{"tokens": [...], "index": 3}, ...]
        ```
    """

    def __init__(self, sentences: Dict[str, List[Dict[str, object]]]):
        self.sentences = sentences

    @staticmethod
    def lemmatise_pool(pool_path: Path, cache: Path, *, log=None) -> List[Dict[str, object]]:
        """Every distinct passage of the pool with its lemmas, cached as JSONL (``{"tokens", "lemmas"}``)."""
        cache = Path(cache)
        if cache.exists():
            return [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
        from retexo.pretraining.pool import PairPool
        from retexo.resources import Resources

        morphology = Resources(offline=True).morphology
        passages: Dict[str, List[str]] = {}
        for pair in PairPool.load(pool_path):
            for side in (pair.source_tokens, pair.reuse_tokens):
                passages.setdefault(" ".join(side), list(side))
        out = []
        for i, (key, tokens) in enumerate(passages.items()):
            out.append({"tokens": tokens, "lemmas": morphology.lemmas(tokens)})
            if log and i % 2000 == 0:
                log(f"[lemma_sentences] {i}/{len(passages)} passages lemmatised")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            for row in out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out

    @classmethod
    def build(cls, pool_path: Path, *, cache: Path, min_words: int = 8, max_words: int = 40,
              max_per_lemma: int = 40, seed: int = 1, log=None) -> "LemmaSentences":
        rows = cls.lemmatise_pool(pool_path, cache, log=log)
        rng = random.Random(seed)
        index: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            tokens, lemmas = row["tokens"], row["lemmas"]
            if not (min_words <= len(tokens) <= max_words):
                continue
            seen: Set[str] = set()
            for i, lemma in enumerate(lemmas):
                if lemma in seen or not _LEMMA_OK.match(lemma or ""):
                    continue
                seen.add(lemma)
                index[lemma].append({"tokens": tokens, "index": i})
        for lemma, items in index.items():
            if len(items) > max_per_lemma:
                index[lemma] = rng.sample(items, max_per_lemma)
        return cls(dict(index))

    def contexts(self, lemma: str) -> List[Dict[str, object]]:
        return self.sentences.get(lemma, [])

    def save(self, path: Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for lemma, items in self.sentences.items():
                handle.write(json.dumps({"lemma": lemma, "sentences": items}, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "LemmaSentences":
        out = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                out[obj["lemma"]] = obj["sentences"]
        return cls(out)

    def keep(self, pairs: Sequence[ResourcePair], *, min_sentences: int = 4) -> List[ResourcePair]:
        """The pairs whose both lemmas have enough sentences."""
        return [p for p in pairs if len(self.contexts(p.lemma_a)) >= min_sentences and len(self.contexts(p.lemma_b)) >= min_sentences]


# =============================================================================
# Hard negatives
# =============================================================================


class HardNegatives:
    """``NegativeBuilder``'s three kinds over the benchmark, as stage-0 pairs with label 0.

    Superseded for the paper's checkpoints (2026-09-24): these negatives pair the gold queries of the
    training folds, so a checkpoint that serves all five folds would have seen test passages. The
    stage-0 negatives come from ``python -m retexo.pretraining.pool --negatives N`` instead.

    Example:
        ```python
        HardNegatives.build(n=10000, held_out=4, out=Path("data/pretrain/negatives.jsonl"))
        ```
    """

    @staticmethod
    def build(*, n: int, held_out: int, out: Path, seed: int = 1, log=None) -> Path:
        from retexo.datasets.dataset import BenchmarkData
        from retexo.datasets.negatives import NegativeBuilder

        data = BenchmarkData.load()
        builder = NegativeBuilder(data, held_out=held_out)
        examples = builder.build(n, seed=seed)
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        kinds: Counter = Counter()
        with open(out, "w", encoding="utf-8") as handle:
            for i, ex in enumerate(examples):
                kind = getattr(ex, "negative_kind", "") or ""
                kinds[kind] += 1
                handle.write(json.dumps({"id": f"neg_{i:06d}", "label": 0, "kind": kind, "source_tokens": list(ex.source_tokens),
                                         "reuse_tokens": list(ex.target_tokens)}, ensure_ascii=False) + "\n")
        if log:
            log(f"[negatives] {len(examples)} negatives -> {out} {dict(kinds)}")
        return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("pairs", "sentences", "negatives"))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--pool", default="data/pretrain/pairs.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-vectors", action="store_true")
    ap.add_argument("--vector-threshold", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--held-out", type=int, default=4)
    args = ap.parse_args(argv)
    if args.what == "pairs":
        vectors = None
        if not args.no_vectors:
            from retexo.resources import Resources

            vectors = Resources(offline=True).vectors
        builder = ResourcePairBuilder(Path(args.cache), vectors=vectors, vector_threshold=args.vector_threshold)
        pairs = builder.build(with_vectors=not args.no_vectors)
        out = ResourcePairBuilder.save(pairs, Path(args.out or DEFAULT_OUT / "resource_pairs.jsonl"))
        print(f"{len(pairs)} resource pairs -> {out}; {dict(builder.report)}")
    elif args.what == "sentences":
        index = LemmaSentences.build(Path(args.pool), cache=DEFAULT_OUT / "corpus_lemmas.jsonl", log=print)
        out = index.save(Path(args.out or DEFAULT_OUT / "lemma_sentences.jsonl"))
        n_pairs = 0
        rp = DEFAULT_OUT / "resource_pairs.jsonl"
        if rp.exists():
            pairs = ResourcePairBuilder.load(rp); kept = index.keep(pairs)
            n_pairs = len(kept)
            print(f"resource pairs with >= 4 sentences on both sides: {n_pairs} of {len(pairs)}")
        print(f"{len(index.sentences)} lemmas with sentences -> {out}")
    else:
        HardNegatives.build(n=args.n, held_out=args.held_out, out=Path(args.out or DEFAULT_OUT / "negatives.jsonl"), log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
