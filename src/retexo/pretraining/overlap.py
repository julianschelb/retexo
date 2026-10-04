# retexo/pretraining/overlap.py
"""The all-overlap rung of the Data Scale ladder: every query-corpus pair sharing two content words.

E32's miner (``attic/scripts/mine_candidates.py``) kept the five best corpus segments per query
segment; the classifier then ranked those 250,000 pairs. This rung drops both filters: every
later-author segment against every earlier segment through the same inverted index (normalised
content tokens of at least four letters, no stop word, a token in more than 2,000 segments
discriminates nothing and is not indexed), kept when the two share at least ``min_shared``
content tokens. Every passage of the gold (any fold) is excluded on either side, as in
``pool.PoolBuilder``, and the pairs are written in the pool's format with stratum ``overlap``,
streamed, so millions of pairs never sit in memory.

    python -m retexo.pretraining.overlap --out data/pretrain/pairs_overlap.jsonl
    python -m retexo.pretraining.overlap --out /tmp/overlap_smoke.jsonl --max-pairs 2000     # a smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from retexo.pretraining.pool import DEFAULT_GOLD, PoolBuilder, PoolPair, _norm

STOP = set("""et in ad ab ex de cum non ut si sed aut atque ac vel nec neque quod quia quae qui quam
quo qua est sunt esse erat erant sit sint fuit erit enim autem nam tamen iam tum tunc hic haec hoc
ille illa illud ipse ipsa ipsum is ea id ego tu nos vos me te se sui sibi suus sua suum meus tuus noster
vester per pro sub super inter ante post contra sine ob propter apud iam etiam quoque quidem vero nunc
omnis omnes omnia unus una unum duo tres magnus magna magnum bonus bona bonum multus multa multum
quis quid quem cuius cui ne an num utrum dum donec quando ubi unde ibi inde ergo igitur itaque""".split())
#: A token in more segments than this is not indexed (E32's miner).
MAX_DF = 2000


def content(text: str) -> Set[str]:
    """Normalised content tokens: lower case, v to u, j to i, letters only, four letters or more, no stop word."""
    out = set()
    for token in text.split():
        t = re.sub(r"[^a-z]", "", token.lower().replace("v", "u").replace("j", "i"))
        if len(t) >= 4 and t not in STOP:
            out.add(t)
    return out


class OverlapMiner:
    """Every (query, corpus) pair with at least ``min_shared`` shared content tokens.

    Example:
        ```python
        miner = OverlapMiner(["arma uirumque cano troiae qui primus"], ["arma cano uirum atque deos"], set(), min_tokens=1)
        [p.reuse_tokens for p in miner.pairs()]      # [("arma", "cano", "uirum", "atque", "deos")]
        ```
    """

    def __init__(self, corpus: Sequence[str], queries: Sequence[str], gold_texts: Set[str], *, min_shared: int = 2,
                 max_df: int = MAX_DF, min_tokens: int = 6, max_tokens: int = 80):
        self.builder = PoolBuilder(gold_texts, min_tokens=min_tokens, max_tokens=max_tokens)
        self.corpus = list(dict.fromkeys(corpus))        # a text once: every pair below is then unique
        self.queries = list(dict.fromkeys(queries))
        self.min_shared = min_shared
        self.corpus_tokens = [content(t) for t in self.corpus]
        df = Counter(tok for toks in self.corpus_tokens for tok in toks)
        self.index: Dict[str, List[int]] = defaultdict(list)
        for i, toks in enumerate(self.corpus_tokens):
            for tok in toks:
                if df[tok] <= max_df:
                    self.index[tok].append(i)
        self.report = self.builder.report

    def pairs(self, max_pairs: int = 0, log=None) -> Iterator[PoolPair]:
        kept = 0
        for qi, query in enumerate(self.queries):
            shared: Counter = Counter()
            for tok in content(query):
                for i in self.index.get(tok, ()):
                    shared[i] += 1
            for i, n in shared.items():
                if n < self.min_shared or not self.builder._ok(self.corpus[i], query):
                    continue
                kept += 1
                self.report["kept_overlap"] += 1
                yield PoolPair(f"o_{kept:08d}", tuple(_norm(self.corpus[i]).split()), tuple(_norm(query).split()),
                               "overlap", float(n))
                if max_pairs and kept >= max_pairs:
                    return
            if log and qi % 10000 == 0:
                log(f"[overlap] {qi:,}/{len(self.queries):,} queries, {kept:,} pairs")


def load_texts(min_words: int, max_words: int) -> Tuple[List[str], List[str]]:
    """E32's corpus (earlier segments) and queries (later-author segments), by length."""
    from datasets import load_dataset

    from retexo.datasets.dataset import BenchmarkData

    data = BenchmarkData.load()
    corpus = [t for t in data.corpus_texts() if min_words <= len(t.split()) <= max_words]
    rows = load_dataset("julian-schelb/latin-classical-intertextuality-queries", split="train")
    queries = [r["text"] for r in rows if isinstance(r["text"], str) and min_words <= len(r["text"].split()) <= max_words]
    return corpus, queries


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold-records", default=str(DEFAULT_GOLD))
    ap.add_argument("--min-shared", type=int, default=2)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--max-pairs", type=int, default=0, help="stop after N pairs (0 = all)")
    ap.add_argument("--out", default="data/pretrain/pairs_overlap.jsonl")
    args = ap.parse_args(argv)
    corpus, queries = load_texts(args.min_words, args.max_words)
    print(f"[overlap] corpus segments {len(corpus):,}, query segments {len(queries):,}", flush=True)
    miner = OverlapMiner(corpus, queries, PoolBuilder.gold_texts(Path(args.gold_records)), min_shared=args.min_shared)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as handle:
        for pair in miner.pairs(max_pairs=args.max_pairs, log=lambda m: print(m, flush=True)):
            handle.write(json.dumps(pair.as_json(), ensure_ascii=False) + "\n")
            n += 1
    print(f"[overlap] {n:,} pairs -> {out}")
    for key, value in sorted(miner.report.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
