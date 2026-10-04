# retexo/datasets/negatives.py
"""
Negatives for the whether-decision, from the benchmark's own pair labels.

Renamed from the preliminary experiment module E27 (kept, unlike the fully
dead E-numbered modules, because the implementation notes plan to wire
``NegativeBuilder`` into the baselines harness once the training-regime code
lands); the class and its API are otherwise unchanged.

The typed pointer has never been shown a real Latin pair where the answer is
*no link anywhere*. Its whether-errors (135 false links against 49 declines on
one dumped model) are the shape of that gap. The benchmark supplies the
missing supervision in any quantity: a query segment paired with a source
segment it does not cite.

Three kinds, after the document-level paper's finding that the composition of
negatives, not their number, sets the false-positive rate:

    easy       a random passage of an author the query corpus cites
    lexical    the passage of the cited author with the most shared word forms
               that is *not* the cited one -- coincidental same-form words
    adjacent   the passage right before or after the cited one in the same
               work -- the right passage, the wrong sentence

Every negative is built from the *training* folds only (queries and cited
passages of the held-out fold are excluded on both sides), and no negative
pair is a labelled pair in any fold. A negative is an unlabelled example whose
every reuse word is null; its frames are unknown (-100).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from retexo.formulations.change_detector import ChangeExample
from retexo.core.normalize import normalize

KINDS = ("easy", "lexical", "adjacent")


class NegativeBuilder:
    """Negatives for one held-out fold, from the training folds and the corpus.

    Example:
        ```python
        builder = NegativeBuilder(data, held_out=4)
        negatives = builder.build(200)
        ```
    """

    def __init__(self, data, held_out, *, exclude_texts: Iterable[str] = (), min_tokens: int = 6, max_tokens: int = 60):
        """``held_out`` is one fold or several (the test fold, the dev fold); ``exclude_texts`` are
        further passages (the validation sample's) that no training negative may use on either side."""
        self.held_out = held_out
        folds = {int(held_out)} if isinstance(held_out, int) else {int(f) for f in held_out}
        labels = data.labels
        corpus = data.corpus
        self.min_tokens, self.max_tokens = min_tokens, max_tokens
        train = labels[~labels.fold_id.isin(folds)]
        held = labels[labels.fold_id.isin(folds)]
        # everything labelled, in any fold, is never a negative
        self.attested: Set[Tuple[str, str]] = set(zip(labels.text_corpus_cleaned.astype(str),
                                                      labels.text_query_cleaned.astype(str)))
        # held-out material stays out of the training negatives on both sides, compared normalised
        excluded = {normalize(str(t)) for t in exclude_texts}
        self.held_queries = {normalize(t) for t in held.text_query_cleaned.astype(str)} | excluded
        self.held_sources = {normalize(t) for t in held.text_corpus_cleaned.astype(str)} | excluded
        self.train_rows = [r for r in train.itertuples()
                           if isinstance(r.text_query_cleaned, str) and isinstance(r.text_corpus_cleaned, str)
                           and normalize(r.text_query_cleaned) not in self.held_queries
                           and normalize(r.text_corpus_cleaned) not in self.held_sources]
        self.held_rows = [r for r in held.itertuples()
                          if isinstance(r.text_query_cleaned, str) and isinstance(r.text_corpus_cleaned, str)]
        # the corpus, indexed for the three kinds
        ok = corpus.text_cleaned.notna()
        self.rows = corpus[ok].reset_index(drop=True)
        self.by_author: Dict[str, List[int]] = defaultdict(list)
        self.by_position: Dict[str, int] = {}
        self.forms: List[Set[str]] = []
        for i, r in enumerate(self.rows.itertuples()):
            self.by_author[str(r.prefix_author)].append(i)
            self.by_position[str(r.position)] = i
            self.forms.append({normalize(w) for w in str(r.text_cleaned).split()})
        self.cited_authors = sorted({str(r.prefix_corpus_author) for r in self.train_rows})

    # ---------- helpers ----------

    def _usable(self, source: str, query: str, *, for_training: bool) -> bool:
        if (source, query) in self.attested:
            return False
        n_s, n_q = len(source.split()), len(query.split())
        if not (self.min_tokens <= n_s <= self.max_tokens and self.min_tokens <= n_q <= self.max_tokens):
            return False
        if for_training and (normalize(query) in self.held_queries or normalize(source) in self.held_sources):
            return False
        return True

    def _text(self, i: int) -> str:
        return str(self.rows.text_cleaned.iloc[i])

    # ---------- the three kinds ----------

    def easy(self, row, rng: random.Random, *, for_training: bool) -> Optional[str]:
        for _ in range(40):
            author = rng.choice(self.cited_authors)
            i = rng.choice(self.by_author[author])
            src = self._text(i)
            if self._usable(src, row.text_query_cleaned, for_training=for_training):
                return src
        return None

    def lexical(self, row, rng: random.Random, *, for_training: bool) -> Optional[str]:
        author = str(row.prefix_corpus_author)
        cands = self.by_author.get(author) or self.by_author[rng.choice(self.cited_authors)]
        q_forms = {normalize(w) for w in row.text_query_cleaned.split()}
        cited = str(row.text_corpus_cleaned)
        best, best_n = None, -1
        # a sample keeps this linear in the corpus of one author
        for i in (cands if len(cands) <= 4000 else rng.sample(cands, 4000)):
            src = self._text(i)
            if src == cited:
                continue
            n = len(q_forms & self.forms[i])
            if n > best_n and self._usable(src, row.text_query_cleaned, for_training=for_training):
                best, best_n = src, n
        return best

    def adjacent(self, row, rng: random.Random, *, for_training: bool) -> Optional[str]:
        i = self.by_position.get(str(row.position_corpus))
        if i is None:
            return None
        prefix = str(self.rows.prefix.iloc[i])
        for j in rng.sample([i - 1, i + 1], 2):
            if 0 <= j < len(self.rows) and str(self.rows.prefix.iloc[j]) == prefix:
                src = self._text(j)
                if self._usable(src, row.text_query_cleaned, for_training=for_training):
                    return src
        return None

    # ---------- batches ----------

    @staticmethod
    def negative_example(source: str, target: str, *, kind: str) -> ChangeExample:
        s, t = source.split(), target.split()
        ex = ChangeExample(source_tokens=s, target_tokens=t, labels=[1] * len(t),
                           operations=["INS"] * len(t), n_operations=len(t),
                           source_labels=[1] * len(s), source_operations=["DEL"] * len(s),
                           alignments=[-1] * len(t), fine_operations=["INS"] * len(t),
                           frame_labels=None, link_features=[None] * len(t))
        object.__setattr__(ex, "negative_kind", kind)
        return ex

    def build(self, n: int, *, kinds: Sequence[str] = KINDS, seed: int = 0,
              for_training: bool = True) -> List[ChangeExample]:
        """``n`` negatives spread evenly over ``kinds``; training-fold queries
        (or held-out ones with ``for_training=False``, for evaluation)."""
        rng = random.Random(seed)
        rows = self.train_rows if for_training else self.held_rows
        out: List[ChangeExample] = []
        seen: Set[Tuple[str, str]] = set()
        per_kind = max(n // max(len(kinds), 1), 1)
        for kind in kinds:
            made, tries = 0, 0
            while made < per_kind and tries < per_kind * 30:
                tries += 1
                row = rng.choice(rows)
                src = getattr(self, kind)(row, rng, for_training=for_training)
                if src is None or (src, row.text_query_cleaned) in seen:
                    continue
                seen.add((src, row.text_query_cleaned))
                out.append(self.negative_example(src, row.text_query_cleaned, kind=kind))
                made += 1
        rng.shuffle(out)
        return out

    @staticmethod
    def density_by_type(pairs) -> Dict[str, float]:
        """Mean share of linked reuse words per reference type in the labels --
        the target of the density prior (route D)."""
        acc: Dict[str, List[float]] = defaultdict(list)
        for p in pairs:
            n = len(p.target_tokens)
            if n and p.target_align is not None:
                acc[p.ref_type].append(sum(1 for s in p.target_align if s >= 0) / n)
        return {k: sum(v) / len(v) for k, v in acc.items() if v}
