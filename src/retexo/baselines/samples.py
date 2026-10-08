# retexo/baselines/samples.py
"""The hand-read part of a method note: which pairs to read, rendered so the
links can be checked, plus two cross-cutting views over the dumps.

``SampleSelector`` picks the best and the worst pairs of a dump by token
accuracy, among pairs with enough gold links that a "correct" pair means
something; ``MarkdownRenderer`` prints a pair with its indices, the gold
links and operations, the predicted ones, and the differences marked, so the
interpretation can be written under it. ``LeakageSplit`` scores a dump
separately on test pairs whose source passage also occurs in the training
folds and on clean pairs (the argument for or against regrouping the folds);
``DifficultyTable`` lines the dumps of every method up per pair and lists the
pairs every method fails.

    python -m retexo.baselines.samples runs/dry_typed_pointer_f4/predictions.jsonl --k 5
    python -m retexo.baselines.samples --leakage runs/dry_typed_pointer_f4/predictions.jsonl --fold 4
    python -m retexo.baselines.samples --difficulty runs/dry_*_f4/predictions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.baselines.base import Prediction
from retexo.baselines.record import Record, RecordInterface


@dataclass
class ScoredPair:
    """One pair of a dump with its per-pair token accuracy (``none`` counts as a class)."""

    record: Record
    pred: Prediction
    accuracy: float
    n_gold_links: int
    n_pred_links: int
    #: gold links that are not plain copies (MORPH, a lexical change, SPLIT, MERGE): what makes a correct pair informative
    n_hard: int = 0
    #: a tagger's prediction (no links): accuracy is then per-word label agreement at V1 (+ FRAME)
    tags_only: bool = False

    @staticmethod
    def of(record: Record, pred: Prediction) -> ScoredPair:
        from retexo.baselines import labels

        gold_links, gold_tags, gold_frame, _ = RecordInterface.links_of(record)
        n_gold = sum(1 for g in gold_links if g >= 0)
        n_hard = sum(1 for g, tag in zip(gold_links, gold_tags) if g >= 0 and tag != "COPY")
        pred_links = list(pred.links) + [-1] * (len(gold_links) - len(pred.links))
        tags_only = not any(s >= 0 for s in pred.links) and any(pred.tags)
        if tags_only:
            gold_labels = ScoredPair.v1_frame(gold_links, gold_tags, gold_frame)
            pred_labels = labels.tag_labels(
                list(pred.tags) + [""] * (len(gold_links) - len(pred.tags)), pred.frame, "V3"
            )
            pred_labels = [lab if lab == "FRAME" else labels.to_v1(lab) for lab in pred_labels]
            agree = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
            return ScoredPair(
                record, pred, agree / max(len(gold_labels), 1), n_gold, 0, n_hard, True
            )
        agree = sum(1 for g, p in zip(gold_links, pred_links) if g == p)
        return ScoredPair(
            record,
            pred,
            agree / max(len(gold_links), 1),
            n_gold,
            sum(1 for p in pred_links[: len(gold_links)] if p >= 0),
            n_hard,
        )

    @staticmethod
    def v1_frame(links, tags, frame) -> List[str]:
        from retexo.baselines import labels

        fine = labels.token_labels(links, tags, frame, "V3")
        return [lab if lab == "FRAME" else labels.to_v1(lab) for lab in fine]


class SampleSelector:
    """The ``k`` best and ``k`` worst pairs of a dump.

    Example:
        ```python
        best, worst = SampleSelector(min_links=3).select(rows, k=5)
        ```
    """

    def __init__(self, *, min_links: int = 3):
        self.min_links = min_links

    def scored(self, rows: Sequence[Tuple[Record, Prediction]]) -> List[ScoredPair]:
        return [ScoredPair.of(r, p) for r, p in rows]

    def select(
        self, rows: Sequence[Tuple[Record, Prediction]], *, k: int = 5
    ) -> Tuple[List[ScoredPair], List[ScoredPair]]:
        pairs = [s for s in self.scored(rows) if s.n_gold_links >= self.min_links]
        by_acc = sorted(pairs, key=lambda s: (s.accuracy, -s.n_gold_links))
        worst = by_acc[:k]
        # the best: perfect pairs first, among them those with the most non-copy links, so that "correct" is not
        # a verbatim quotation
        best = sorted(pairs, key=lambda s: (-s.accuracy, -s.n_hard, -s.n_gold_links))[:k]
        return best, worst


class MarkdownRenderer:
    """A pair as Markdown: both passages with indices, then one line per reuse
    word that carries a gold or a predicted link, differences marked with ``!``."""

    @staticmethod
    def render(sample: ScoredPair, *, title: str = "", compact: bool = True) -> str:
        from retexo.baselines import labels

        r, p = sample.record, sample.pred
        gold_links, gold_tags, gold_frame, gold_sure = RecordInterface.links_of(r)
        pred_links = list(p.links) + [-1] * (r.n_reuse - len(p.links))
        pred_tags = list(p.tags) + [""] * (r.n_reuse - len(p.tags))
        pred_frame = list(p.frame) + [0] * (r.n_reuse - len(p.frame))
        src = " ".join(f"{i}:{w}" for i, w in enumerate(r.source_tokens))
        reu = " ".join(f"{i}:{w}" for i, w in enumerate(r.reuse_tokens))
        if sample.tags_only:
            return MarkdownRenderer.render_tags(sample, title=title, compact=compact)
        lines = [
            f"**{title or r.id}** ({r.pair_label}., token accuracy {sample.accuracy:.2f}, "
            f"gold links {sample.n_gold_links}, predicted {sample.n_pred_links})",
            "",
            f"- source: {src}",
            f"- reuse: {reu}",
            "",
            "| reuse word | gold | predicted | |",
            "|---|---|---|---|",
        ]
        omitted = 0
        for t in range(r.n_reuse):
            g, pr = gold_links[t], pred_links[t]
            g_frame, p_frame = gold_frame[t], pred_frame[t]
            if g < 0 and pr < 0 and not g_frame and not p_frame:
                continue
            same_op = labels.canonical(gold_tags[t])[0] == labels.canonical(pred_tags[t])[0]
            agree = g == pr and (same_op or g < 0) and g_frame == p_frame
            if compact and agree and (gold_tags[t] == "COPY" or g_frame):
                omitted += 1  # a correct COPY or FRAME row says nothing; count it
                continue
            gold_cell = (
                f"{g}:{r.source_tokens[g]} {gold_tags[t]}{'' if gold_sure[t] else ' (possible)'}"
                if g >= 0
                else ("FRAME" if g_frame else "\u2014")
            )
            pred_cell = (
                f"{pr}:{r.source_tokens[pr]} {pred_tags[t]}"
                if 0 <= pr < r.n_source
                else ("FRAME" if p_frame else "\u2014")
            )
            lines.append(
                f"| {t}:{r.reuse_tokens[t]} | {gold_cell} | {pred_cell} | {'' if agree else '!'} |"
            )
        if omitted:
            lines.append(f"| *{omitted} correct COPY / FRAME rows omitted* | | | |")
        note = str(r.annotation.get("note", "") or r.provenance.get("note", "") or "")
        if note:
            lines += ["", f"- annotator's note: {note}"]
        lines += ["", "*Interpretation:* ", ""]
        return "\n".join(lines)

    @staticmethod
    def render_tags(sample: ScoredPair, *, title: str = "", compact: bool = True) -> str:
        """A tagger's pair: the gold label (V1 + FRAME, with the gold link for orientation) against the predicted tag."""
        from retexo.baselines import labels

        r, p = sample.record, sample.pred
        gold_links, gold_tags, gold_frame, _ = RecordInterface.links_of(r)
        gold_labels = ScoredPair.v1_frame(gold_links, gold_tags, gold_frame)
        pred_labels = labels.tag_labels(
            list(p.tags) + [""] * (r.n_reuse - len(p.tags)), p.frame, "V3"
        )
        pred_labels = [lab if lab == "FRAME" else labels.to_v1(lab) for lab in pred_labels]
        src = " ".join(f"{i}:{w}" for i, w in enumerate(r.source_tokens))
        reu = " ".join(f"{i}:{w}" for i, w in enumerate(r.reuse_tokens))
        lines = [
            f"**{title or r.id}** ({r.pair_label}., tag accuracy {sample.accuracy:.2f}, gold links {sample.n_gold_links}; a tagger, no links)",
            "",
            f"- source: {src}",
            f"- reuse: {reu}",
            "",
            "| reuse word | gold | predicted tag | |",
            "|---|---|---|---|",
        ]
        omitted = 0
        for t in range(r.n_reuse):
            g, pr = gold_labels[t], pred_labels[t]
            if g == "INS" and pr == "INS":
                continue
            if compact and g == pr and g in ("COPY", "FRAME"):
                omitted += 1
                continue
            link = (
                f" ({gold_links[t]}:{r.source_tokens[gold_links[t]]})" if gold_links[t] >= 0 else ""
            )
            lines.append(
                f"| {t}:{r.reuse_tokens[t]} | {g}{link} | {pr} | {'' if g == pr else '!'} |"
            )
        if omitted:
            lines.append(f"| *{omitted} correct COPY / FRAME rows omitted* | | | |")
        note = str(r.annotation.get("note", "") or r.provenance.get("note", "") or "")
        if note:
            lines += ["", f"- annotator's note: {note}"]
        lines += ["", "*Interpretation:* ", ""]
        return "\n".join(lines)

    @classmethod
    def section(cls, best: Sequence[ScoredPair], worst: Sequence[ScoredPair]) -> str:
        out = ["### Correct pairs", ""]
        out += [cls.render(s, title=f"{s.record.id} (correct)") for s in best]
        out += ["### Badly predicted pairs", ""]
        out += [cls.render(s, title=f"{s.record.id} (wrong)") for s in worst]
        return "\n".join(out)


class LeakageSplit:
    """Test pairs whose source passage also occurs in a training-fold pair, versus clean ones.

    Example:
        ```python
        leaky, clean = LeakageSplit(records, fold=4).split(rows)
        ```
    """

    def __init__(self, records: Sequence[Record], *, fold: int):
        self._train_sources = {" ".join(r.source_tokens) for r in records if r.fold != fold}

    def is_leaky(self, record: Record) -> bool:
        return " ".join(record.source_tokens) in self._train_sources

    def split(
        self, rows: Sequence[Tuple[Record, Prediction]]
    ) -> Tuple[List[Tuple[Record, Prediction]], List[Tuple[Record, Prediction]]]:
        leaky = [(r, p) for r, p in rows if self.is_leaky(r)]
        clean = [(r, p) for r, p in rows if not self.is_leaky(r)]
        return leaky, clean

    def scores(
        self, rows: Sequence[Tuple[Record, Prediction]], *, emits: str = "links"
    ) -> Dict[str, Dict[str, float]]:
        """``token_accuracy``, link F1 and V1 macro F1 on both halves."""
        from retexo.baselines.scorer import BaselineScorer

        out = {}
        for name, part in zip(("leaky", "clean"), self.split(rows)):
            if not part:
                out[name] = {"n": 0}
                continue
            result = BaselineScorer.score([r for r, _ in part], [p for _, p in part], emits=emits)
            out[name] = {
                "n": len(part),
                "token_accuracy": result.get("token_accuracy"),
                "link_f1": result["link"]["f1"] if result.get("link") else None,
                "op_macro_v1": result["ops"]["V1"]["macro_f1"],
            }
        return out


class DifficultyTable:
    """Per-pair token accuracy across the dumps of several methods.

    Example:
        ```python
        table = DifficultyTable({"typed_pointer": rows_a, "sultan": rows_b})
        print(table.markdown(worst=15))
        ```
    """

    def __init__(
        self, dumps: Dict[str, Sequence[Tuple[Record, Prediction]]], *, min_links: int = 1
    ):
        self.methods = list(dumps)
        self.by_pair: Dict[str, Dict[str, float]] = {}
        self.records: Dict[str, Record] = {}
        for method, rows in dumps.items():
            for r, p in rows:
                s = ScoredPair.of(r, p)
                if s.n_gold_links < min_links:
                    continue
                self.by_pair.setdefault(r.id, {})[method] = s.accuracy
                self.records[r.id] = r

    def hardest(self, n: int = 15) -> List[Tuple[str, float, Dict[str, float]]]:
        rows = []
        for pid, accs in self.by_pair.items():
            if len(accs) == len(self.methods):
                rows.append((pid, sum(accs.values()) / len(accs), accs))
        return sorted(rows, key=lambda x: x[1])[:n]

    def failed_by_all(self, threshold: float = 0.8) -> List[str]:
        return [
            pid
            for pid, accs in self.by_pair.items()
            if len(accs) == len(self.methods) and all(a < threshold for a in accs.values())
        ]

    def markdown(self, worst: int = 15) -> str:
        head = "| pair | label | mean | " + " | ".join(self.methods) + " | note |"
        lines = [head, "|" + "---|" * (len(self.methods) + 4)]
        for pid, mean, accs in self.hardest(worst):
            r = self.records[pid]
            note = str(r.annotation.get("note", "") or r.provenance.get("note", "") or "")[:80]
            lines.append(
                f"| {pid} | {r.pair_label} | {mean:.2f} | "
                + " | ".join(f"{accs[m]:.2f}" for m in self.methods)
                + f" | {note} |"
            )
        return "\n".join(lines)


def _read(path: Path):
    from retexo.baselines.adapters import PredictionAdapter

    return PredictionAdapter.read_dump(Path(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="best/worst pairs of a dump, leakage split, difficulty table"
    )
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-links", type=int, default=3)
    ap.add_argument("--leakage", action="store_true")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument(
        "--gold-records",
        default=None,
        help="record file the folds come from (default: the silver gold)",
    )
    ap.add_argument("--difficulty", action="store_true")
    ap.add_argument("--worst", type=int, default=15)
    args = ap.parse_args(argv)
    if args.difficulty:
        dumps = {Path(d).parent.name: _read(Path(d)) for d in args.dumps}
        table = DifficultyTable(dumps)
        print(table.markdown(args.worst))
        print(
            f"\nfailed by every method (accuracy < 0.8): {len(table.failed_by_all())} pairs: "
            f"{', '.join(table.failed_by_all()[:40])}"
        )
        return 0
    rows = _read(Path(args.dumps[0]))
    if args.leakage:
        from retexo.baselines.record import RecordCodec, load_records

        records = (
            load_records(Path(args.gold_records))
            if args.gold_records
            else RecordCodec.gold_records()
        )
        split = LeakageSplit(records, fold=args.fold)
        for name, scores in split.scores(rows).items():
            print(name, scores)
        return 0
    best, worst = SampleSelector(min_links=args.min_links).select(rows, k=args.k)
    print(MarkdownRenderer.section(best, worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
