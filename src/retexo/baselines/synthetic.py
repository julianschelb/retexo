# retexo/baselines/synthetic.py
"""Note 03: the synthetic pool, the held-out synthetic set and the negatives as records.

A thin layer over ``retexo.datasets.synthetic`` (the typed generator,
``select_pool``, ``frame_pool``) and ``retexo.datasets.negatives``
(``NegativeBuilder``): the same call sequence ``train_aligner`` ran, with the
outputs written as harness records so that every row trains and scores on
files under ``data/records/`` instead of regenerating. Seeds are the corpus
when ``BenchmarkData`` is present and the training folds' passages otherwise
(the same fallback as note 15's ``TrainingData``); held-out passages and
held-out authors never seed a pool; the held-out synthetic set of a fold comes
from the pairs the pool did not select.

    python build_synthetic.py --fold 4 --size 20000 --workers 20
    python build_synthetic.py --fold 4 --dense --size 20000
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import labels
from retexo.baselines.record import Edge, Record, Span
from retexo.core.normalize import normalize

#: Seed passages are kept in this length band (``run_e28_score.py:154``).
SEED_LENGTHS = (8, 40)

#: The rare-tag floor of ``select_pool`` and the held-out set's per-class floor.
RARE_MIN = 1200
HELDOUT_PER_RARE = 60


# =============================================================================
# Examples to records
# =============================================================================


class SyntheticRecords:
    """Generated ``ChangeExample``s as records, with the replay check.

    Example:
        ```python
        records = SyntheticRecords.from_examples(pool, level="synthetic", fold=4)
        ```
    """

    @staticmethod
    def edges_of(example) -> Tuple[List[Edge], List[Span]]:
        links = list(example.alignments or [])
        fine = list(example.fine_operations or [])
        frames = list(example.frame_labels or [])
        edges: List[Edge] = []
        for t, s in enumerate(links):
            if s is None or s < 0:
                continue
            op = fine[t] if t < len(fine) else "SUBST"
            canon, detail = labels.canonical("SUBST" if op in ("?", None, "") else op)
            edges.append(Edge(t, int(s), canon or "SUBST", True, detail))
        spans: List[Span] = []
        start = None
        for t, flag in enumerate(frames + [0]):
            if flag and start is None:
                start = t
            elif not flag and start is not None:
                spans.append(Span(start, t, "FRAME"))
                start = None
        return edges, spans

    @staticmethod
    def pair_label(edges: Sequence[Edge]) -> str:
        """``cit`` when the links hold two in-order copies, ``cf`` otherwise."""
        copies = sorted((e.r, e.s) for e in edges if e.op == "COPY")
        for (r1, s1), (r2, s2) in zip(copies, copies[1:]):
            if r2 == r1 + 1 and s2 == s1 + 1:
                return "cit"
        return "cf"

    @classmethod
    def from_examples(
        cls, examples: Sequence, *, level: str = "synthetic", fold: int = -1, verify: bool = True
    ) -> List[Record]:
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.baselines.base import Prediction
        from retexo.core.scriba import Scriba

        scriba = Scriba()
        out = []
        for i, example in enumerate(examples):
            edges, spans = cls.edges_of(example)
            record = Record(
                id=f"{level}/f{fold}/{i}",
                level=level,
                fold=fold,
                source_work="",
                source_tokens=list(example.source_tokens),
                reuse_work="",
                reuse_tokens=list(example.target_tokens),
                pair_label=cls.pair_label(edges),
                links=edges,
                spans=spans,
                provenance={
                    "links": "construction",
                    "fine_ops": "construction",
                    "orientation": "swapped"
                    if getattr(example, "swapped_from", False)
                    else "forward",
                },
            )
            if verify:
                probe = Prediction(
                    links=[-1] * record.n_reuse,
                    tags=[""] * record.n_reuse,
                    frame=[0] * record.n_reuse,
                )
                for e in edges:
                    probe.links[e.r] = e.s
                    probe.tags[e.r] = e.op
                for span in spans:
                    for t in range(span.start, span.end):
                        probe.frame[t] = 1
                try:
                    script = PredictionAdapter.to_script(record, probe)
                    ok = script is not None and scriba.verify(
                        script, record.source_tokens, record.reuse_tokens
                    )
                except (IndexError, ValueError):
                    ok = False
                if not ok:
                    continue
            out.append(record)
        return out


# =============================================================================
# The pool builder
# =============================================================================


class PoolBuilder:
    """The synthetic pool, the held-out set and the negatives of one fold.

    Example:
        ```python
        builder = PoolBuilder(fold=4, train_records=train, held_records=test)
        pool, report = builder.build_pool(size=20000, workers=20)
        held = builder.heldout_synthetic(n=2000)
        ```
    """

    def __init__(
        self,
        fold: int,
        train_records: Sequence[Record],
        held_records: Sequence[Record] = (),
        *,
        seed: int = 1,
        log=None,
    ):
        self.fold = fold
        self.train_records = list(train_records)
        self.held_records = list(held_records)
        self.seed = seed
        self.log = log
        self._unselected: List = []

    # ---------- seeds ----------

    @staticmethod
    def benchmark():
        try:
            from retexo.datasets.dataset import BenchmarkData

            return BenchmarkData.load()
        except (FileNotFoundError, ImportError):
            return None

    def held_keys(self) -> Tuple[set, set]:
        """Normalised held-out passages and held-out authors, never seeds."""
        passages = {normalize(" ".join(r.source_tokens)) for r in self.held_records} | {
            normalize(" ".join(r.reuse_tokens)) for r in self.held_records
        }
        authors = {r.source_author for r in self.held_records if r.source_author} | {
            r.reuse_author for r in self.held_records if r.reuse_author
        }
        return passages, authors

    def seeds(self, *, exclude_authors: bool = True) -> List[List[str]]:
        """Seed passages: the corpus (held-out passages and authors excluded) or,
        without it, the training folds' own passages (held-out passages
        excluded; the author rule would empty a gold-only seed set, since the
        same three query authors sit in every fold)."""
        data = self.benchmark()
        held_passages, held_authors = self.held_keys()
        if data is not None:
            texts = [(t.split(), "") for t in data.corpus_texts()]
        else:
            texts = [(list(r.source_tokens), r.source_author) for r in self.train_records] + [
                (list(r.reuse_tokens), r.reuse_author) for r in self.train_records
            ]
            exclude_authors = False
        lo, hi = SEED_LENGTHS
        out = []
        for tokens, author in texts:
            if not (lo <= len(tokens) <= hi):
                continue
            if normalize(" ".join(tokens)) in held_passages:
                continue
            if exclude_authors and author and author in held_authors:
                continue
            out.append(tokens)
        random.Random(0).shuffle(out)
        return out

    def frames(self) -> List[List[str]]:
        out = []
        for record in self.train_records:
            for span in record.spans:
                if span.label == "FRAME" and span.end > span.start:
                    out.append(list(record.reuse_tokens[span.start : span.end]))
        return out

    # ---------- the pool ----------

    def build_pool(
        self,
        *,
        size: int = 20000,
        workers: int = 1,
        dense_rate: float = 0.0,
        mlm_subst: bool = False,
        rare_min: int = RARE_MIN,
        exclude_authors: bool = True,
        both_orientations: bool = True,
    ) -> Tuple[List[Record], Any]:
        from retexo.aligners.agreement import PairSwap
        from retexo.datasets import synthetic as syn

        seeds = self.seeds(exclude_authors=exclude_authors)
        if len(seeds) < 20:
            raise ValueError(
                f"fold {self.fold}: only {len(seeds)} seed passages; nothing to generate from"
            )
        contexts = seeds[:8000]
        examples, report = syn.generate_typed(
            seeds,
            contexts,
            per_seed=5,
            seed=self.seed,
            workers=workers,
            attested=syn.attested_forms(seeds, min_count=3),
            frames=self.frames(),
            frame_rate=syn.FRAME_RATE,
            enclitic_rate=syn.ENCLITIC_RATE,
            reorder_inter=0.0,
            reorder_intra=0.0,
            dense_rate=dense_rate,
            mlm_subst=mlm_subst,
            log=self.log,
        )
        chosen = syn.select_pool(examples, size, random.Random(7 + self.seed), rare_min=rare_min)
        taken = {id(x) for x in chosen}
        self._unselected = [x for x in examples if id(x) not in taken]
        if both_orientations:
            chosen = list(chosen) + [PairSwap.labelled(x) for x in chosen]
        records = SyntheticRecords.from_examples(chosen, level="synthetic", fold=self.fold)
        if self.log:
            self.log(
                f"[synthetic] fold {self.fold}: {len(records)} pool records from {len(seeds)} seeds "
                f"({len(examples)} generated, {len(self._unselected)} unselected)"
            )
        return records, report

    def heldout_synthetic(self, *, n: int = 2000, per_rare: int = HELDOUT_PER_RARE) -> List[Record]:
        """From the pairs the pool did not select, every fine class with enough links to score."""
        from retexo.datasets import synthetic as syn

        if not self._unselected:
            return []
        held = syn.balanced_subset(
            self._unselected, n, random.Random(36 + self.seed), per_rare=per_rare
        )
        return SyntheticRecords.from_examples(held, level="synthetic_heldout", fold=self.fold)

    def build_negatives(self, n: int, *, kinds: Optional[Sequence[str]] = None) -> List[Record]:
        """``n`` negatives spread over the kinds, or ``[]`` when the corpus is absent."""
        data = self.benchmark()
        if data is None:
            if self.log:
                self.log("[synthetic] negatives skipped: BenchmarkData not available")
            return []
        from retexo.datasets.negatives import KINDS, NegativeBuilder

        held_folds = (
            (self.fold, (self.fold + 1) % 5) if self.fold >= 0 else self.fold
        )  # the test and the dev fold
        examples = NegativeBuilder(data, held_out=held_folds).build(
            n, kinds=tuple(kinds or KINDS), seed=self.seed, for_training=True
        )
        records = SyntheticRecords.from_examples(
            examples, level="negative", fold=self.fold, verify=False
        )
        for record in records:
            record.links.clear()
            record.spans.clear()
            record.pair_label = "no_match"
        return records


# =============================================================================
# The coverage table
# =============================================================================


class Coverage:
    """How the pool's statistics compare with the gold's: the definition's
    coverage check, one line per statistic with the gap.

    Example:
        ```python
        print(Coverage.table(pool_records, gold_records))
        ```
    """

    STATISTICS = (
        "link_rate",
        "copy_share",
        "morph_share",
        "subst_share",
        "frame_share",
        "crossing_share",
    )

    @staticmethod
    def statistics(records: Sequence[Record]) -> Dict[str, float]:
        from retexo.baselines.record import RecordInterface

        n_words = n_links = copies = morphs = substs = frames = 0
        crossings = 0
        for record in records:
            links, tags, frame, _ = RecordInterface.links_of(record)
            n_words += len(links)
            n_links += sum(1 for s in links if s >= 0)
            copies += sum(1 for t in tags if t == "COPY")
            morphs += sum(1 for t in tags if t == "MORPH")
            substs += sum(1 for t, s in zip(tags, links) if s >= 0 and t not in ("COPY", "MORPH"))
            frames += sum(frame)
            linked = [s for s in links if s >= 0]
            crossings += int(any(b < a for a, b in zip(linked, linked[1:])))
        denom = max(n_links, 1)
        return {
            "link_rate": n_links / max(n_words, 1),
            "copy_share": copies / denom,
            "morph_share": morphs / denom,
            "subst_share": substs / denom,
            "frame_share": frames / max(n_words, 1),
            "crossing_share": crossings / max(len(records), 1),
        }

    @classmethod
    def table(
        cls, pool: Sequence[Record], gold: Sequence[Record], *, tolerance: float = 0.25
    ) -> str:
        a, b = cls.statistics(pool), cls.statistics(gold)
        lines = ["| statistic | pool | gold | gap | flag |", "|---|---|---|---|---|"]
        for name in cls.STATISTICS:
            gap = a[name] - b[name]
            flag = "gap" if b[name] and abs(gap) / max(b[name], 1e-9) > tolerance else ""
            lines.append(f"| {name} | {a[name]:.3f} | {b[name]:.3f} | {gap:+.3f} | {flag} |")
        return "\n".join(lines)
