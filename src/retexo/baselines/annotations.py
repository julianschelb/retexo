# retexo/baselines/annotations.py
"""Reading, checking and comparing annotation files (the blind LLM pass, the
expert corrections) on the record schema.

An annotation file holds one JSON object per pair with ``id``, ``links``,
``spans`` and ``note`` (the output shape of the Annotation Prompt) but no
tokens: ``AnnotationReader`` merges each line into the gold record of the same
id, so that the result is a ``Record`` with the annotator's edges and spans and
the gold's tokens, fold and pair label. ``AnnotationChecker`` applies the
validation rules of the prompt and the closed ``detail`` list (definition
section 3.1b) and adds the form checks the prompt cannot state (a COPY that is
not the same word, a MORPH that shares no stem). ``AnnotationAgreement``
scores one set of records against another: links, operations given a shared
link, frames.

    python -m retexo.baselines.annotations data/gold_full/corrections/claude_2026-09-15.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from retexo.baselines import labels
from retexo.baselines.record import Edge, Record, RecordCodec, RecordInterface, Span
from retexo.core.normalize import normalize

#: Provenance written on records that come from a blind annotation file.
BLIND_PROVENANCE = "llm-blind"


# =============================================================================
# Reading
# =============================================================================


class AnnotationReader:
    """Annotation lines merged into gold records.

    Example:
        ```python
        gold = {r.id: r for r in RecordCodec.gold_records()}
        records, skipped = AnnotationReader(gold).load(Path("data/gold_full/corrections/claude_2026-09-15.jsonl"))
        ```
    """

    def __init__(self, gold: Dict[str, Record], *, provenance: str = BLIND_PROVENANCE):
        self._gold = gold
        self._provenance = provenance

    def load(self, path: Path) -> Tuple[List[Record], List[str]]:
        """Records for every line whose id is in the gold, last line per id wins;
        the second value lists the ids that were not in the gold."""
        by_id: Dict[str, dict] = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                by_id[str(obj["id"])] = obj
        records, unknown = [], []
        for pair_id, obj in by_id.items():
            if pair_id not in self._gold:
                unknown.append(pair_id)
                continue
            records.append(self.merge(self._gold[pair_id], obj, source=Path(path).name))
        return records, unknown

    def merge(self, gold: Record, obj: dict, *, source: str = "") -> Record:
        """The gold record with the annotation's edges, spans and note.

        Operations are upper-cased and passed through ``Labels.canonical`` so
        that an old spelling (``NOP``, ``SYN-DIST`` as an op) lands on the
        record vocabulary; ``SUBST`` with ``detail: SYN-DIST`` (the placement
        of an earlier prompt) becomes ``SYN``.
        """
        if "words" in obj and "links" not in obj:
            obj = self._from_app(obj)
        links = []
        for e in obj.get("links", []):
            op, detail = labels.canonical(str(e["op"]).strip().upper())
            given = str(e.get("detail", "") or "").strip().upper()
            if op == "SUBST" and given.split("+")[0] == "SYN-DIST":
                op = "SYN"
            if op is None:
                continue
            links.append(Edge(int(e["r"]), int(e["s"]), op, bool(e.get("sure", True)), given or detail))
        spans = [Span(int(sp["start"]), int(sp["end"]), str(sp.get("label", "FRAME")).upper())
                 for sp in obj.get("spans", [])]
        return Record(
            id=gold.id, level=gold.level, fold=gold.fold,
            source_work=gold.source_work, source_tokens=list(gold.source_tokens),
            reuse_work=gold.reuse_work, reuse_tokens=list(gold.reuse_tokens),
            pair_label=gold.pair_label, links=links, spans=spans,
            provenance={"links": self._provenance, "fine_ops": f"{self._provenance}-V3", "file": source},
            annotation={"note": str(obj.get("note", "") or "")},
        )

    @staticmethod
    def _from_app(obj: dict) -> dict:
        """The annotation app's per-word shape (``words`` with ``t``, ``source``,
        ``op``, ``sure``, ``comment``) to the prompt's ``links`` / ``spans`` shape."""
        links, frame = [], []
        for w in obj["words"]:
            t, src, op = int(w["t"]), int(w.get("source", -1)), str(w.get("op", "INS")).upper()
            if op == "FRAME":
                frame.append(t)
            elif src >= 0 and op != "INS":
                links.append({"r": t, "s": src, "op": op, "sure": bool(w.get("sure", True)), "detail": w.get("detail", "")})
        spans = []
        for t in sorted(frame):
            if spans and spans[-1]["end"] == t:
                spans[-1]["end"] = t + 1
            else:
                spans.append({"start": t, "end": t + 1, "label": "FRAME"})
        return {"id": obj["id"], "links": links, "spans": spans, "note": obj.get("note", "")}


# =============================================================================
# Checking
# =============================================================================


@dataclass
class Problem:
    """One finding on one record; ``severity`` is ``error`` (the prompt's
    validation rules, the closed list) or ``warn`` (a form check that a
    legitimate reading can pass)."""

    record_id: str
    severity: str
    message: str


class AnnotationChecker:
    """The prompt's validation rules plus form checks.

    Example:
        ```python
        problems = AnnotationChecker().check(record)
        errors = [p for p in problems if p.severity == "error"]
        ```
    """

    #: Feature prefixes counted as "the same stem" for the MORPH check.
    STEM = 3

    def check(self, record: Record) -> List[Problem]:
        out: List[Problem] = []
        err = lambda m: out.append(Problem(record.id, "error", m))  # noqa: E731
        warn = lambda m: out.append(Problem(record.id, "warn", m))  # noqa: E731
        n_r, n_s = record.n_reuse, record.n_source
        seen_r: Dict[int, Edge] = {}
        by_s: Dict[int, List[Edge]] = defaultdict(list)
        for e in record.links:
            if not (0 <= e.r < n_r):
                err(f"r={e.r} outside the reuse ({n_r} words)"); continue
            if not (0 <= e.s < n_s):
                err(f"s={e.s} outside the source ({n_s} words)"); continue
            if e.op not in labels.EDGE_OPS:
                err(f"r={e.r}: op {e.op!r} is not an edge operation")
                continue
            if e.r in seen_r:
                err(f"r={e.r} linked twice")
            seen_r[e.r] = e
            by_s[e.s].append(e)
            try:
                lexical, morph = labels.parse_detail(e.op, e.detail)
            except ValueError as exc:
                err(f"r={e.r}: {exc}")
                lexical, morph = "", ()
            if e.op == "MERGE":
                if not e.detail:
                    err(f"r={e.r}: MERGE without the second source word in detail")
                elif not (0 <= int(e.detail[1:]) < n_s):
                    err(f"r={e.r}: MERGE detail {e.detail} outside the source")
            self._form_checks(record, e, warn)
        for s, edges in by_s.items():
            if len(edges) > 1 and not all(e.op == "SPLIT" for e in edges):
                err(f"s={s} linked by {len(edges)} reuse words ({', '.join(e.op for e in edges)}); only SPLIT may share")
        split_groups = Counter(e.s for e in record.links if e.op == "SPLIT")
        for s, count in split_groups.items():
            if count == 1:
                warn(f"s={s}: SPLIT with a single reuse word (allowed for a dropped enclitic; check)")
        for sp in record.spans:
            if sp.label != "FRAME":
                err(f"span label {sp.label!r}; FRAME is the only span label")
            if not (0 <= sp.start < sp.end <= n_r):
                err(f"span {sp.start}:{sp.end} outside the reuse ({n_r} words)")
            for t in range(max(0, sp.start), min(n_r, sp.end)):
                if t in seen_r:
                    err(f"reuse word {t} is inside a FRAME span and linked")
        return out

    def _form_checks(self, record: Record, e: Edge, warn) -> None:
        src, tgt = record.source_tokens[e.s], record.reuse_tokens[e.r]
        a, b = normalize(src), normalize(tgt)
        if e.op == "COPY" and a != b:
            if a[: self.STEM] == b[: self.STEM]:
                warn(f"r={e.r}: COPY {src!r} / {tgt!r} differ after folding (MORPH?)")
            else:
                warn(f"r={e.r}: COPY {src!r} / {tgt!r} are different words")
        if e.op == "MORPH":
            if a == b:
                warn(f"r={e.r}: MORPH {src!r} / {tgt!r} are the same word (COPY?)")
            elif a[: self.STEM] != b[: self.STEM]:
                warn(f"r={e.r}: MORPH {src!r} / {tgt!r} share no stem (SUBST?)")
        if e.op in labels.LEXICAL and a == b:
            warn(f"r={e.r}: {e.op} {src!r} / {tgt!r} are the same word (COPY?)")


# =============================================================================
# Agreement
# =============================================================================


@dataclass
class Agreement:
    """Scores of one annotation against another over the shared pairs."""

    pairs: int = 0
    link_precision: float = 0.0
    link_recall: float = 0.0
    link_f1: float = 0.0
    op_accuracy_v3: float = 0.0
    op_accuracy_v1: float = 0.0
    op_kappa_v3: float = 0.0
    shared_links: int = 0
    frame_precision: float = 0.0
    frame_recall: float = 0.0
    frame_f1: float = 0.0
    per_op: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def table(self) -> str:
        lines = [f"pairs {self.pairs}   links P {self.link_precision:.3f} R {self.link_recall:.3f} F1 {self.link_f1:.3f}   "
                 f"op given a shared link ({self.shared_links}): V3 acc {self.op_accuracy_v3:.3f} kappa {self.op_kappa_v3:.3f}, "
                 f"V1 acc {self.op_accuracy_v1:.3f}   frame tokens P {self.frame_precision:.3f} R {self.frame_recall:.3f} F1 {self.frame_f1:.3f}",
                 f"{'op':8s} {'n_ref':>6s} {'n_sys':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}"]
        for op, row in self.per_op.items():
            lines.append(f"{op:8s} {int(row['n_ref']):6d} {int(row['n_sys']):6d} {row['P']:6.3f} {row['R']:6.3f} {row['F1']:6.3f}")
        return "\n".join(lines)


class AnnotationAgreement:
    """``system`` scored against ``reference`` on the pairs both hold.

    Links are (r, s) pairs; operations are compared where both annotations
    hold the same link; frames are compared per reuse token.

    Example:
        ```python
        agreement = AnnotationAgreement().score(blind_records, silver_records)
        print(agreement.table())
        ```
    """

    def score(self, system: Sequence[Record], reference: Sequence[Record]) -> Agreement:
        ref = {r.id: r for r in reference}
        out = Agreement()
        tp = fp = fn = 0
        ftp = ffp = ffn = 0
        pairs_v3: List[Tuple[str, str]] = []
        per_op_counts: Dict[str, Counter] = defaultdict(Counter)
        for sys_rec in system:
            if sys_rec.id not in ref:
                continue
            ref_rec = ref[sys_rec.id]
            out.pairs += 1
            s_links = {(e.r, e.s): e.op for e in sys_rec.links}
            r_links = {(e.r, e.s): e.op for e in ref_rec.links}
            shared = s_links.keys() & r_links.keys()
            tp += len(shared); fp += len(s_links.keys() - shared); fn += len(r_links.keys() - shared)
            for key in shared:
                pairs_v3.append((s_links[key], r_links[key]))
            for key, op in s_links.items():
                per_op_counts[op]["n_sys"] += 1
                if key in r_links and r_links[key] == op:
                    per_op_counts[op]["tp"] += 1
            for op in r_links.values():
                per_op_counts[op]["n_ref"] += 1
            s_frame = set(self._frame_tokens(sys_rec)); r_frame = set(self._frame_tokens(ref_rec))
            ftp += len(s_frame & r_frame); ffp += len(s_frame - r_frame); ffn += len(r_frame - s_frame)
        out.link_precision, out.link_recall, out.link_f1 = self._prf(tp, fp, fn)
        out.frame_precision, out.frame_recall, out.frame_f1 = self._prf(ftp, ffp, ffn)
        out.shared_links = len(pairs_v3)
        if pairs_v3:
            out.op_accuracy_v3 = sum(a == b for a, b in pairs_v3) / len(pairs_v3)
            out.op_accuracy_v1 = sum(labels.to_v1(a) == labels.to_v1(b) for a, b in pairs_v3) / len(pairs_v3)
            out.op_kappa_v3 = self._kappa(pairs_v3)
        out.confusion = dict(Counter(pairs_v3))
        for op in labels.EDGE_OPS:
            c = per_op_counts[op]
            p, r, f = self._prf(c["tp"], c["n_sys"] - c["tp"], c["n_ref"] - c["tp"])
            out.per_op[op] = {"n_ref": c["n_ref"], "n_sys": c["n_sys"], "P": p, "R": r, "F1": f}
        return out

    @staticmethod
    def _frame_tokens(record: Record) -> Iterable[int]:
        for sp in record.spans:
            if sp.label == "FRAME":
                yield from range(sp.start, sp.end)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    @staticmethod
    def _kappa(pairs: Sequence[Tuple[str, str]]) -> float:
        n = len(pairs)
        observed = sum(a == b for a, b in pairs) / n
        ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
        return (observed - expected) / (1 - expected) if expected < 1 else 1.0


# =============================================================================
# Report
# =============================================================================


class AnnotationReport:
    """Everything the checker and the agreement say about one file, printed."""

    def __init__(self, gold: Optional[Dict[str, Record]] = None):
        self._gold = gold if gold is not None else {r.id: r for r in RecordCodec.gold_records()}

    def run(self, path: Path, *, show: int = 12) -> Tuple[List[Record], List[Problem], Agreement]:
        records, unknown = AnnotationReader(self._gold).load(path)
        checker = AnnotationChecker()
        problems = [p for r in records for p in checker.check(r)]
        errors = [p for p in problems if p.severity == "error"]
        warns = [p for p in problems if p.severity == "warn"]
        silver = [self._gold[r.id] for r in records]
        agreement = AnnotationAgreement().score(records, silver)
        ops = Counter(e.op for r in records for e in r.links)
        details = Counter((e.op, e.detail) for r in records for e in r.links if e.detail and e.op != "MERGE")
        print(f"{path}: {len(records)} records ({len(unknown)} unknown ids), "
              f"{sum(len(r.links) for r in records)} links, {sum(1 for r in records for e in r.links if not e.sure)} possible, "
              f"{sum(len(r.spans) for r in records)} frame spans, {sum(1 for r in records if not r.links)} without links")
        print("ops:", dict(ops.most_common()))
        print("details:", dict(details.most_common(20)))
        print(f"errors {len(errors)}, warnings {len(warns)}")
        for p in errors[:show]:
            print(f"  ERROR {p.record_id}: {p.message}")
        kinds = Counter(p.message.split(":")[1].split("'")[0].strip() if ":" in p.message else p.message for p in warns)
        for p in warns[:show]:
            print(f"  warn  {p.record_id}: {p.message}")
        print("against the silver labels:")
        print(agreement.table())
        top = sorted(((a, b), n) for (a, b), n in agreement.confusion.items() if a != b)
        top = sorted(top, key=lambda kv: -kv[1])[:10]
        print("confusions (blind, silver):", ", ".join(f"{a}/{b} {n}" for (a, b), n in top))
        return records, problems, agreement


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 1
    for path in args:
        AnnotationReport().run(Path(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
