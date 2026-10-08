# retexo/datasets/dataset.py
"""
Loading the benchmark corpus, its labelled pairs, and negatives.

Three things come from here: the seed passages that synthetic variants are
built from, the real annotated pairs the oracle derives scripts for, and the
non-reuse pairs the final training phase needs.

**Folds need care.** The benchmark assigns folds by *query* segment, so all
source segments of a given query share a fold. A source passage reused by two
different authors can therefore appear in two folds — 149 of the 1,238 reused
passages do. Seeding from such a passage would place it on both sides of a
split, so seeds are drawn only from passages confined to the training folds.
Evaluation still uses the benchmark's own fold assignment, which keeps results
comparable with published numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from retexo.paths import home

#: Location of the processed benchmark: ``data/processed`` under :func:`retexo.paths.home`.
DEFAULT_DATA_DIR = home() / "data" / "processed"

# =============================================================================
# Records
# =============================================================================


@dataclass(frozen=True)
class LabelledPair:
    """One annotated (source, reuse) pair."""

    source: str
    target: str
    ref_type: str
    fold: int
    source_author: str
    query_author: str

    @property
    def is_positive(self) -> bool:
        """Whether this pair is an attested reuse."""
        return self.ref_type in ("cit.", "cf.")


@dataclass
class BenchmarkData:
    """Everything a run needs, loaded once.

    Example:
        ```python
        data = BenchmarkData.load()
        data.seeds(held_out=0)          # source passages safe to perturb
        data.pairs(held_out=0)          # (train, test) real pairs
        data.negatives(held_out=0, n=500)
        ```
    """

    labels: pd.DataFrame  # noqa: F821
    corpus: pd.DataFrame  # noqa: F821
    _cross_fold: set = field(default_factory=set, repr=False)

    # ---------- Construction ----------

    @classmethod
    def load(cls, data_dir: Optional[Path] = None) -> BenchmarkData:
        """Read the processed corpus and labels.

        Raises:
            FileNotFoundError: If the processed benchmark is not present, since
                silently falling back to toy data would make a run look real.
        """
        import pandas as pd

        base = Path(data_dir or DEFAULT_DATA_DIR)
        labels_path = base / "labels" / "labels.csv"
        corpus_path = base / "corpus" / "corpus.csv"
        for path in (labels_path, corpus_path):
            if not path.exists():
                raise FileNotFoundError(f"benchmark not found at {path}")

        labels = pd.read_csv(labels_path)
        corpus = pd.read_csv(corpus_path)
        spanning = labels.groupby("text_corpus_cleaned").fold_id.nunique()
        return cls(labels=labels, corpus=corpus, _cross_fold=set(spanning[spanning > 1].index))

    # ---------- Vocabulary ----------

    def corpus_texts(self) -> List[str]:
        """Every cleaned passage in the corpus.

        Used to build the substitution candidate pool, so that a generated
        substitute is a word this corpus actually uses rather than any word the
        lexicon happens to contain.
        """
        return [text for text in self.corpus.text_cleaned.dropna().tolist() if text]

    # ---------- Seeds ----------

    def seeds(self, held_out: int, *, min_tokens: int = 4) -> List[str]:
        """Source passages safe to perturb for a given held-out fold.

        Excludes any passage that also appears in the held-out fold, and any
        that appears in more than one fold at all, since the second case cannot
        be assigned to a side of the split without risking leakage.
        """
        held = set(self.labels.loc[self.labels.fold_id == held_out, "text_corpus_cleaned"])
        train = self.labels[self.labels.fold_id != held_out]
        passages = {
            text
            for text in train.text_corpus_cleaned.dropna().unique()
            if text not in held and text not in self._cross_fold
        }
        return sorted(p for p in passages if len(str(p).split()) >= min_tokens)

    # ---------- Real pairs ----------

    def pairs(self, held_out: int) -> Tuple[List[LabelledPair], List[LabelledPair]]:
        """Training and held-out annotated pairs, by the benchmark's folds."""

        def build(frame) -> List[LabelledPair]:
            out = []
            for row in frame.itertuples():
                source, target = row.text_corpus_cleaned, row.text_query_cleaned
                if not isinstance(source, str) or not isinstance(target, str):
                    continue
                out.append(
                    LabelledPair(
                        source=source,
                        target=target,
                        ref_type=row.ref_type,
                        fold=int(row.fold_id),
                        source_author=str(row.prefix_corpus_author),
                        query_author=str(row.prefix_query_author),
                    )
                )
            return out

        return (
            build(self.labels[self.labels.fold_id != held_out]),
            build(self.labels[self.labels.fold_id == held_out]),
        )

    # ---------- Negatives ----------

    def negatives(
        self, held_out: int, n: int, *, seed: int = 42, min_tokens: int = 4
    ) -> List[LabelledPair]:
        """Non-reuse pairs drawn from the same works as the positives.

        Pairing a real query passage with a source passage from an author it
        does cite, but a segment it does not, keeps the negatives in domain: a
        model must learn that these two are unrelated despite coming from the
        same corpus, rather than that unrelated text looks different.
        """
        rng = random.Random(seed)
        train = self.labels[self.labels.fold_id != held_out]
        attested = set(zip(train.text_corpus_cleaned, train.text_query_cleaned))

        queries = [
            t
            for t in train.text_query_cleaned.dropna().unique()
            if len(str(t).split()) >= min_tokens
        ]
        by_author: Dict[str, List[str]] = {}
        for row in train.itertuples():
            by_author.setdefault(str(row.prefix_corpus_author), []).append(row.text_corpus_cleaned)
        authors = [a for a, v in by_author.items() if v]
        if not queries or not authors:
            return []

        out: List[LabelledPair] = []
        attempts = 0
        while len(out) < n and attempts < n * 40:
            attempts += 1
            query = rng.choice(queries)
            author = rng.choice(authors)
            source = rng.choice(by_author[author])
            if not isinstance(source, str) or (source, query) in attested:
                continue
            if len(source.split()) < min_tokens:
                continue
            out.append(
                LabelledPair(
                    source=source,
                    target=query,
                    ref_type="no_match",
                    fold=-1,
                    source_author=author,
                    query_author="",
                )
            )
        return out

    # ---------- Reporting ----------

    def summary(self, held_out: int) -> Dict[str, int]:
        """Counts for the run record."""
        train, test = self.pairs(held_out)
        return {
            "seeds": len(self.seeds(held_out)),
            "train_pairs": len(train),
            "test_pairs": len(test),
            "cross_fold_excluded": len(self._cross_fold),
        }
