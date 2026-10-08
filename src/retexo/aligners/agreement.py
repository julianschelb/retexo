# retexo/aligners/agreement.py
"""
Bidirectional-agreement decoding rules: score both directions, keep what both accept.

Renamed from the preliminary experiment module E29; kept under its own name,
unchanged, because ``retexo.baselines.decoder`` imports it directly and
several implementation notes say explicitly to reuse it verbatim.

Our pointer scores reuse -> source and its null is a scalar per reuse word.
awesome-align's null is a test on two directions: a link exists only if the
reuse word picks the source word *and* the source word picks it back.
SimAlign's mutual argmax is the same test without a threshold; its entropy
rule is a third signal. All three work on two sets of rows -- the forward
rows the pointer scores, and the reverse rows a swapped-sides pass produces --
and need no training.

Rows are the pointer's score-row shape: per reuse word, [(source, p) ...] best
first, with the null as source -1. Reverse rows are the same shape with the
roles swapped: per *source* word, [(reuse, p) ...].
"""

from __future__ import annotations

import math
from typing import ClassVar, Dict, List, Sequence, Tuple

from retexo.formulations.change_detector import ChangeExample

Row = List[Tuple[int, float]]


class PairSwap:
    """Building a pair with the roles reversed, for the reverse-direction pass
    and for symmetric training.

    Example:
        ```python
        reverse_pair = PairSwap.example(example)            # untrained: p(reuse word | source word)
        reverse_training_pair = PairSwap.labelled(example)   # a training pair with inverted labels
        ```
    """

    #: relations that change name when the sides change
    INVERT: ClassVar[Dict[str, str]] = {
        "HYPER": "HYPO",
        "HYPO": "HYPER",
        "SPLIT": "MERGE",
        "MERGE": "SPLIT",
    }

    @staticmethod
    def example(example: ChangeExample) -> ChangeExample:
        """The pair with the roles exchanged: the source passage sits in the reuse
        slot, so the model scores p(reuse word | source word)."""
        s, t = list(example.target_tokens), list(example.source_tokens)  # swapped on purpose
        return ChangeExample(
            source_tokens=s,
            target_tokens=t,
            labels=[0] * len(t),
            operations=["COPY"] * len(t),
            n_operations=0,
            source_labels=[0] * len(s),
            source_operations=["COPY"] * len(s),
            alignments=[-1] * len(t),
            fine_operations=["INS"] * len(t),
            frame_labels=[0] * len(t),
            link_features=[None] * len(t),
        )

    @classmethod
    def labelled(cls, example: ChangeExample) -> ChangeExample:
        """A *training* pair with the sides exchanged and its labels inverted.

        One-to-one links invert exactly; HYPER/HYPO and SPLIT/MERGE change name;
        a gold lexical change of unknown kind ("?") stays unknown; the former reuse
        side's inserted words become deleted source words; frames do not exist on
        a source passage, so the new reuse side carries none. Evidence grids are
        not transposed (the featurizer is not symmetric) -- recompute them."""
        old_s, old_t = list(example.source_tokens), list(example.target_tokens)
        links = example.alignments or [-1] * len(old_t)
        fine = example.fine_operations or ["INS"] * len(old_t)
        new_s, new_t = old_t, old_s
        new_links = [-1] * len(new_t)
        new_fine = ["INS"] * len(new_t)
        for t, s in enumerate(links):
            if s is not None and 0 <= s < len(new_t) and new_links[s] < 0:
                new_links[s] = t
                tag = fine[t] if t < len(fine) else "?"
                new_fine[s] = cls.INVERT.get(tag, tag)
        consumed = {t for t in new_links if t >= 0}
        new_source_labels = [0 if t in consumed else 1 for t in range(len(new_s))]
        coarse = ["COPY" if (k == "NOP") else ("INS" if k == "INS" else "SUBST") for k in new_fine]
        ex = ChangeExample(
            source_tokens=new_s,
            target_tokens=new_t,
            labels=[0 if op == "COPY" else 1 for op in coarse],
            operations=coarse,
            n_operations=sum(1 for op in coarse if op != "COPY"),
            source_labels=new_source_labels,
            source_operations=["DEL" if d else "COPY" for d in new_source_labels],
            alignments=new_links,
            fine_operations=new_fine,
            frame_labels=[0] * len(new_t),
            link_features=[None] * len(new_t),
        )
        object.__setattr__(ex, "swapped_from", True)
        return ex


class AgreementDecoder:
    """Bidirectional-agreement rules, each returning restricted forward rows for
    the Hungarian assignment.

    Example:
        ```python
        kept = AgreementDecoder.mutual_argmax(rows, reverse_rows)
        kept = AgreementDecoder.intersect(rows, reverse_rows, c=0.4)
        ```
    """

    @staticmethod
    def prob(row: Row, index: int) -> float:
        return dict(row).get(index, 0.0)

    @staticmethod
    def top(row: Row) -> int:
        """The best entry, the null (-1) included."""
        return row[0][0] if row else -1

    @classmethod
    def reverse_links(cls, rows_rev: Sequence[Row], n_reuse: int) -> List[int]:
        """The reverse direction read as an aligner of the reuse side: reuse word t
        links to the source word s whose best pick is t (first such s), else -1."""
        links = [-1] * n_reuse
        for s, row in enumerate(rows_rev):
            t = cls.top(row)
            if 0 <= t < n_reuse and links[t] < 0:
                links[t] = s
        return links

    @staticmethod
    def entropy(row: Row) -> float:
        ps = [p for _, p in row if p > 1e-12]
        z = sum(ps) or 1.0
        return -sum(p / z * math.log(p / z) for p in ps)

    @classmethod
    def intersect(cls, rows: Sequence[Row], rows_rev: Sequence[Row], c: float) -> List[Row]:
        """awesome-align: keep (t, s) iff p(s|t) > c and p(t|s) > c. A word whose
        every candidate fails keeps only its null."""
        out = []
        for t, row in enumerate(rows):
            kept = [
                (s, p)
                for s, p in row
                if s >= 0 and p > c and s < len(rows_rev) and cls.prob(rows_rev[s], t) > c
            ]
            out.append(sorted(kept + [(-1, cls.prob(row, -1))], key=lambda x: -x[1]))
        return out

    @classmethod
    def mutual_argmax(cls, rows: Sequence[Row], rows_rev: Sequence[Row]) -> List[Row]:
        """SimAlign Argmax: keep (t, s) iff s is t's best and t is s's best."""
        out = []
        for t, row in enumerate(rows):
            s = cls.top(row)
            kept = (
                [(s, cls.prob(row, s))]
                if s >= 0 and s < len(rows_rev) and cls.top(rows_rev[s]) == t
                else []
            )
            out.append(sorted(kept + [(-1, cls.prob(row, -1))], key=lambda x: -x[1]))
        return out

    @classmethod
    def entropy_filter(cls, rows: Sequence[Row], rows_rev: Sequence[Row], tau: float) -> List[Row]:
        """SimAlign's null: drop every link of a reuse word whose row entropy, and of
        a source word whose column entropy, are both above tau (normalised by log n)."""
        out = []
        h_col = [cls.entropy(r) / max(math.log(max(len(r), 2)), 1e-9) for r in rows_rev]
        for _t, row in enumerate(rows):
            h_row = cls.entropy(row) / max(math.log(max(len(row), 2)), 1e-9)
            kept = [
                (s, p)
                for s, p in row
                if s >= 0 and min(h_row, h_col[s] if s < len(h_col) else 1.0) <= tau
            ]
            out.append(sorted(kept + [(-1, cls.prob(row, -1))], key=lambda x: -x[1]))
        return out

    @classmethod
    def null_scale(cls, rows: Sequence[Row], k: float) -> List[Row]:
        """Our null re-weighted by k and renormalised (E18's move, at the raw pointer)."""
        out = []
        for row in rows:
            scaled = [(s, p * (k if s < 0 else 1.0)) for s, p in row]
            z = sum(p for _, p in scaled) or 1.0
            out.append(sorted([(s, p / z) for s, p in scaled], key=lambda x: -x[1]))
        return out

    @classmethod
    def compose(cls, *restricted: Sequence[Row]) -> List[Row]:
        """Keep a candidate only if every rule kept it; the null is the first rule's."""
        out = []
        for rows in zip(*restricted):
            keep = set.intersection(*[{s for s, _ in r if s >= 0} for r in rows])
            first = rows[0]
            out.append(
                sorted(
                    [(s, p) for s, p in first if s in keep] + [(-1, cls.prob(first, -1))],
                    key=lambda x: -x[1],
                )
            )
        return out
