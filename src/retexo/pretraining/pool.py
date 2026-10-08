# retexo/pretraining/pool.py
"""The real-pairs level: reuse pairs without word labels, as stage-0 input.

Three sources, none of them annotated at the word level (the pretraining
definition, section 2):

- the scholar-graded Tesserae Lucan-Vergil parallels (grade 3 to 5) and the
  Valerius Flaccus intertext database, as selected for E32
  (``runs/e32_allusion/selected.jsonl``: ``earlier`` = source, ``later`` = reuse);
- the classifier-ranked corpus pairs of E32's mining (``runs/e32_mine/ranked.jsonl``,
  250k candidates sharing two content words, ``p_reuse`` from the reuse
  classifier), taken above a probability threshold with a cap per source
  passage and one pair per reuse passage;
- nothing from the benchmark: its 1,490 labelled pairs *are* the gold.

Every pair whose source or reuse text occurs in a gold pair (any fold) is
dropped, so no stage-0 checkpoint has seen a gold passage; the probe and the
later tables are then clean by construction. Tokens are the whitespace split
of the cleaned text, as everywhere in the harness.

    python -m retexo.pretraining.pool --out data/pretrain/pairs.jsonl
    python -m retexo.pretraining.pool --negatives 10000 --out data/pretrain/negatives.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

DEFAULT_ALLUSION = Path("runs/e32_allusion/selected.jsonl")
DEFAULT_RANKED = Path("runs/e32_mine/ranked.jsonl")
DEFAULT_GOLD = Path("data/gold_full/gold_2026-09-24.records.jsonl")


def _norm(text: str) -> str:
    return " ".join(str(text).split())


@dataclass(frozen=True)
class PoolPair:
    """One reuse pair of the pool: tokens of both sides, where it came from and how sure."""

    id: str
    source_tokens: Tuple[str, ...]
    reuse_tokens: Tuple[str, ...]
    stratum: str
    score: float
    ref_type: str = ""  # cit. / cf. where a human label exists (never here), "" otherwise

    def as_json(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "stratum": self.stratum,
            "score": self.score,
            "ref_type": self.ref_type,
            "source_tokens": list(self.source_tokens),
            "reuse_tokens": list(self.reuse_tokens),
        }

    @classmethod
    def from_json(cls, obj: Dict[str, object]) -> PoolPair:
        return cls(
            str(obj["id"]),
            tuple(obj["source_tokens"]),
            tuple(obj["reuse_tokens"]),
            str(obj.get("stratum", "")),
            float(obj.get("score", 0.0)),
            str(obj.get("ref_type", "") or ""),
        )


class PoolBuilder:
    """Assembles ``data/pretrain/pairs.jsonl`` from the E32 files with the gold excluded.

    Example:
        ```python
        builder = PoolBuilder(gold_texts=PoolBuilder.gold_texts(Path("data/gold_full/blind/....jsonl")))
        pairs = builder.build(min_p_reuse=0.9, cap=20000)
        PoolBuilder.save(pairs, Path("data/pretrain/pairs.jsonl"))
        ```
    """

    def __init__(
        self,
        gold_texts: Set[str],
        *,
        allusion: Path = DEFAULT_ALLUSION,
        ranked: Path = DEFAULT_RANKED,
        min_tokens: int = 6,
        max_tokens: int = 80,
        seed: int = 1,
    ):
        self.gold_texts = gold_texts
        self.allusion = Path(allusion)
        self.ranked = Path(ranked)
        self.min_tokens, self.max_tokens = min_tokens, max_tokens
        self.rng = random.Random(seed)
        self.report: Counter = Counter()

    @staticmethod
    def gold_texts(records_path: Path) -> Set[str]:
        """Every source and reuse passage of the gold, normalised, from a record file."""
        out: Set[str] = set()
        for line in Path(records_path).read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            out.add(_norm(" ".join(obj["source"]["tokens"])))
            out.add(_norm(" ".join(obj["reuse"]["tokens"])))
        return out

    def _ok(self, source: str, reuse: str) -> bool:
        s, r = _norm(source), _norm(reuse)
        if not s or not r or s == r:
            self.report["dropped_empty_or_identical"] += 1
            return False
        if s in self.gold_texts or r in self.gold_texts:
            self.report["dropped_gold_passage"] += 1
            return False
        for text in (s, r):
            n = len(text.split())
            if n < self.min_tokens or n > self.max_tokens:
                self.report["dropped_length"] += 1
                return False
        return True

    def allusion_pairs(self) -> Iterator[PoolPair]:
        """Tesserae (grade >= 3) and Valerius Flaccus pairs of E32's selection."""
        if not self.allusion.exists():
            return
        for line in self.allusion.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            source, reuse = obj["earlier"], obj["later"]
            if not self._ok(source, reuse):
                continue
            stratum = str(obj.get("stratum", "allusion"))
            score = float(obj.get("score", 0) or 0)
            self.report[f"kept_{stratum}"] += 1
            yield PoolPair(
                f"a_{obj.get('cand_id', self.report['allusion_seen'])}",
                tuple(_norm(source).split()),
                tuple(_norm(reuse).split()),
                stratum,
                score,
            )

    def ranked_pairs(
        self,
        *,
        min_p_reuse: float = 0.9,
        cap: int = 20000,
        per_source: int = 3,
        one_per_reuse: bool = True,
    ) -> Iterator[PoolPair]:
        """Classifier-ranked corpus pairs above ``min_p_reuse``: one per reuse passage (unless
        ``one_per_reuse`` is off), at most ``per_source`` per source passage, the most probable first,
        up to ``cap``; a cap or ``per_source`` of 0 is no limit (the Data Scale ladder's rungs)."""
        if not self.ranked.exists():
            return
        rows = []
        for line in self.ranked.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            p = float(obj.get("p_reuse", 0.0) or 0.0)
            if p >= min_p_reuse:
                rows.append((p, obj))
        rows.sort(key=lambda x: -x[0])
        taken_reuse: Set[str] = set()
        per_src: Counter = Counter()
        kept = 0
        for p, obj in rows:
            source, reuse = obj["earlier"], obj["later"]
            s, r = _norm(source), _norm(reuse)
            if (one_per_reuse and r in taken_reuse) or (per_source and per_src[s] >= per_source):
                self.report["dropped_ranked_cap"] += 1
                continue
            if not self._ok(source, reuse):
                continue
            taken_reuse.add(r)
            per_src[s] += 1
            kept += 1
            self.report["kept_ranked"] += 1
            yield PoolPair(f"m_{kept:06d}", tuple(s.split()), tuple(r.split()), "ranked", p)
            if cap and kept >= cap:
                break

    def build(
        self,
        *,
        min_p_reuse: float = 0.9,
        cap: int = 20000,
        per_source: int = 3,
        one_per_reuse: bool = True,
    ) -> List[PoolPair]:
        pairs = list(self.allusion_pairs()) + list(
            self.ranked_pairs(
                min_p_reuse=min_p_reuse, cap=cap, per_source=per_source, one_per_reuse=one_per_reuse
            )
        )
        seen: Set[Tuple[str, str]] = set()
        out = []
        for pair in pairs:
            key = (" ".join(pair.source_tokens), " ".join(pair.reuse_tokens))
            if key in seen:
                self.report["dropped_duplicate"] += 1
                continue
            seen.add(key)
            out.append(pair)
        self.rng.shuffle(out)
        self.report["total"] = len(out)
        return out

    def negative_pairs(
        self, n: int, *, max_p_reuse: float = 0.1
    ) -> List[Tuple[List[str], List[str], str]]:
        """Stage-0 negatives with no gold passage on either side (the pair identification's label 0): half
        classifier-rejected candidates (two shared content words, ``p_reuse`` at most ``max_p_reuse``: the
        lexical near-misses), half random re-pairings of those candidates' passages."""
        rows = []
        if self.ranked.exists():
            for line in self.ranked.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if float(obj.get("p_reuse", 1.0) or 0.0) <= max_p_reuse and self._ok(
                    obj["earlier"], obj["later"]
                ):
                    rows.append((_norm(obj["earlier"]).split(), _norm(obj["later"]).split()))
        self.rng.shuffle(rows)
        lexical = [(s, r, "lexical_low_p") for s, r in rows[: n - n // 2]]
        sources = [s for s, _ in rows]
        reuses = [r for _, r in rows]
        self.rng.shuffle(sources)
        random_pairs = [(s, r, "random") for s, r in zip(sources, reuses) if s != r][: n // 2]
        self.report["negatives_lexical_low_p"] = len(lexical)
        self.report["negatives_random"] = len(random_pairs)
        return lexical + random_pairs

    @staticmethod
    def save_negatives(negatives: Sequence[Tuple[List[str], List[str], str]], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for i, (source, reuse, kind) in enumerate(negatives):
                handle.write(
                    json.dumps(
                        {
                            "id": f"neg_{i:06d}",
                            "label": 0,
                            "kind": kind,
                            "source_tokens": list(source),
                            "reuse_tokens": list(reuse),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    @staticmethod
    def save(pairs: Sequence[PoolPair], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair.as_json(), ensure_ascii=False) + "\n")
        return path


class PairPool:
    """The saved pool, iterated as ``(source_tokens, reuse_tokens, is_real)`` in both orientations.

    Example:
        ```python
        pool = PairPool.load(Path("data/pretrain/pairs.jsonl"))
        len(pool)                      # pairs, one orientation
        for source, reuse, real in pool.both_orientations(): ...
        ```
    """

    def __init__(self, pairs: Sequence[PoolPair]):
        self.pairs = list(pairs)

    @classmethod
    def load(cls, path: Path, *, limit: Optional[int] = None) -> PairPool:
        pairs = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                pairs.append(PoolPair.from_json(json.loads(line)))
                if limit and len(pairs) >= limit:
                    break
        return cls(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[PoolPair]:
        return iter(self.pairs)

    def both_orientations(self) -> Iterator[Tuple[List[str], List[str], bool]]:
        for pair in self.pairs:
            yield list(pair.source_tokens), list(pair.reuse_tokens), True
            yield list(pair.reuse_tokens), list(pair.source_tokens), True

    def strata(self) -> Dict[str, int]:
        return dict(Counter(p.stratum for p in self.pairs))

    def to_records(self, *, limit: int = 0) -> List:
        """The pairs as records without links (level ``real_pairs``, fold -1): the corpus side as source, the
        later side as reuse; ``pair_label`` cf, marked as assumed. The Data Regimes' real pairs and the
        pairs-only aligners' extra bitext."""
        from retexo.baselines.record import Record

        out = []
        for pair in self.pairs[: limit or None]:
            out.append(
                Record(
                    id=f"real/{pair.id}",
                    level="real_pairs",
                    fold=-1,
                    source_work="",
                    source_tokens=list(pair.source_tokens),
                    reuse_work="",
                    reuse_tokens=list(pair.reuse_tokens),
                    pair_label="cf",
                    provenance={
                        "links": "none",
                        "pair_label": "assumed",
                        "stratum": pair.stratum,
                        "pool_score": pair.score,
                    },
                )
            )
        return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold-records", default=str(DEFAULT_GOLD))
    ap.add_argument("--allusion", default=str(DEFAULT_ALLUSION))
    ap.add_argument("--ranked", default=str(DEFAULT_RANKED))
    ap.add_argument("--min-p-reuse", type=float, default=0.9)
    ap.add_argument("--cap", type=int, default=20000, help="pairs at most (0 = no limit)")
    ap.add_argument(
        "--per-source", type=int, default=3, help="pairs per source passage at most (0 = no limit)"
    )
    ap.add_argument(
        "--many-per-reuse", action="store_true", help="drop the one-pair-per-reuse-passage rule"
    )
    ap.add_argument("--out", default="data/pretrain/pairs.jsonl")
    ap.add_argument(
        "--negatives",
        type=int,
        default=0,
        help="write N stage-0 negatives to --out instead of the pool (no gold passage on either side)",
    )
    args = ap.parse_args(argv)
    builder = PoolBuilder(
        PoolBuilder.gold_texts(Path(args.gold_records)),
        allusion=Path(args.allusion),
        ranked=Path(args.ranked),
    )
    if args.negatives:
        out = PoolBuilder.save_negatives(builder.negative_pairs(args.negatives), Path(args.out))
        print(
            f"{builder.report['negatives_lexical_low_p'] + builder.report['negatives_random']} negatives -> {out}"
        )
        for key, value in sorted(builder.report.items()):
            print(f"  {key}: {value}")
        return 0
    pairs = builder.build(
        min_p_reuse=args.min_p_reuse,
        cap=args.cap,
        per_source=args.per_source,
        one_per_reuse=not args.many_per_reuse,
    )
    out = PoolBuilder.save(pairs, Path(args.out))
    print(f"{len(pairs)} pairs -> {out}")
    for key, value in sorted(builder.report.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
