# retexo/baselines/record.py
"""The record of section 1.3: one object per pair, read and written once.

The paper stores every pair, at every level of supervision, as one JSONL line
whose finest annotation is a list of edges (reuse word ``r`` comes from source
word ``s`` by operation ``op``); insertions and deletions are read off the
absence of an edge and never stored, and sourceless runs that carry a label
(citing formulas, ``FRAME``) sit in ``spans``. This module is the only place
that knows the nesting of that line: on disk ``source`` and ``reuse`` are
objects with ``work`` and ``tokens``; in code the ``Record`` is flat
(``record.source_tokens``), so that no method ever spells a JSON path.

The gold is not stored in this shape yet (``data/gold_full/labels.json`` keeps the
exception format that ``e3.load_gold`` expands), so ``gold_to_records`` is the
bridge, and ``record_to_example`` is the bridge back to the ``ChangeExample``
every trainable formulation in ``retexo`` consumes. Interface (II) of the
harness, the per-token triple ``(links, tags, frame)``, is derived from the
edges by ``links_of`` and turned back into edges by ``edges_from``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import labels

# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class Edge:
    """One link: reuse word ``r`` comes from source word ``s`` by ``op``.

    Attributes:
        r: Reuse (target) index.
        s: Source index.
        op: The finest operation on the edge, one of ``labels.EDGE_OPS``.
        sure: ``False`` for a *possible* link from adjudication.
        detail: A demoted relation (HYPER, ANT) or a morphological feature.
        p: The model's probability, on predicted edges only.
        sure_by: Who judged sure against possible (``llm-blind``, ``raters``, ``expert``).
    """

    r: int
    s: int
    op: str
    sure: bool = True
    detail: str = ""
    p: Optional[float] = None
    sure_by: str = ""


@dataclass
class Span:
    """A sourceless run of the reuse that carries a label (``FRAME``).

    ``start`` and ``end`` are reuse indices, ``end`` exclusive; the fields are
    two integers, not a tuple, so that ``Edge.r`` and ``Span.start`` cannot be
    confused, and the label field is not called ``type``.
    """

    start: int
    end: int
    label: str = "FRAME"


@dataclass
class Record:
    """One pair at any level of supervision, flat in code, nested on disk.

    Attributes:
        id: ``<level>/<n>`` or the gold's ``p0421``.
        level: ``gold`` | ``synthetic`` | ``real_pairs`` | ``external``.
        fold: The test fold the pair belongs to (gold), else ``-1``.
        source_work, reuse_work: Citations of the two passages.
        source_tokens, reuse_tokens: Whitespace tokens, punctuation attached.
        pair_label: ``cit`` | ``cf`` | ``no_match``.
        links: The edges.
        spans: Labelled sourceless runs.
        provenance: Who produced what (``links``, ``fine_ops``, silences).
        annotation: Cached lookups (``lemma``, ``pos``, ``regime`` ...).
        split: ``train`` | ``dev`` | ``test`` on external sets.
        pred: The ``pred`` block of a dump, when the record was read from one.
        benchmark_id: The row of the benchmark's labels file (gold and real pairs).
        source_meta, reuse_meta: The benchmark's fields of each side besides ``work``
            and ``tokens``: ``author``, ``citation``, ``text_original``, ``text``,
            ``text_english``.

    On disk a file in the paper format (``provenance.format`` set) spells the
    operations as the paper does (INFLECT, SUBST with the relation in ``detail``)
    and splits ``annotation`` into ``note`` and ``lookup``; in code every record
    uses the code spelling (MORPH, SYN, POS, NE-SUB) and one ``annotation`` dict,
    so no method sees the difference.

    Example:
        ```python
        from retexo.baselines.record import Record, Edge, links_of

        record = Record(id="demo/1", level="gold", fold=4,
                        source_work="Verg. Aen. 1.1", source_tokens=["arma", "uirumque", "cano"],
                        reuse_work="Ov. Am. 1.1.1", reuse_tokens=["arma", "graui", "cano"],
                        pair_label="cit", links=[Edge(0, 0, "COPY"), Edge(2, 2, "COPY")])
        links, tags, frame, sure = links_of(record)
        print(links)   # [0, -1, 2]
        print(tags)    # ['COPY', 'INS', 'COPY']
        ```
    """

    id: str
    level: str
    fold: int
    source_work: str
    source_tokens: List[str]
    reuse_work: str
    reuse_tokens: List[str]
    pair_label: str
    links: List[Edge] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    annotation: Dict[str, Any] = field(default_factory=dict)
    split: Optional[str] = None
    pred: Optional[Dict[str, Any]] = None
    benchmark_id: Optional[int] = None
    source_meta: Dict[str, Any] = field(default_factory=dict)
    reuse_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_author(self) -> str:
        return str(self.source_meta.get("author") or self.source_work)

    @property
    def reuse_author(self) -> str:
        return str(self.reuse_meta.get("author") or self.reuse_work)

    @property
    def n_reuse(self) -> int:
        return len(self.reuse_tokens)

    @property
    def n_source(self) -> int:
        return len(self.source_tokens)


# =============================================================================
# JSON in and out, and the gold bridge
# =============================================================================


class RecordCodec:
    """Reads and writes ``Record`` as the nested JSON of section 1.3.

    Also the bridge from the gold's exception format (``gold_to_records``) and
    to the model-side ``ChangeExample`` (``to_example``).

    Example:
        ```python
        records = RecordCodec.load(Path("data/records.jsonl"))
        RecordCodec.save(records, Path("runs/copy.jsonl"))
        ```
    """

    _SPLIT_KEY = re.compile(r'"split": "(train|dev|test)"')

    _SIDE_KEYS = ("work", "tokens")

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> Record:
        """One parsed JSON line to a ``Record`` (the only reader of the nesting).

        Reads both formats: the code spelling of the dry runs and the paper
        format of the frozen gold, whose operations ``Labels.from_paper`` maps
        back (INFLECT to MORPH, SUBST with a SYN, POS or NE-SUB detail to that
        label) and whose ``note`` and ``lookup`` join ``annotation``.
        """
        links = []
        for e in obj.get("links", []):
            op, detail = labels.Labels.from_paper(str(e["op"]), e.get("detail", "") or "")
            links.append(Edge(int(e["r"]), int(e["s"]), op, bool(e.get("sure", True)),
                              detail, e.get("p"), str(e.get("sure_by", "") or "")))
        spans = [Span(int(sp["start"]), int(sp["end"]), str(sp.get("label", "FRAME")))
                 for sp in obj.get("spans", [])]
        annotation = dict(obj.get("annotation", {}))
        annotation.update(obj.get("lookup", {}) or {})
        if obj.get("note"):
            annotation["note"] = str(obj["note"])
        source, reuse = obj.get("source", {}), obj.get("reuse", {})
        return Record(
            id=str(obj["id"]), level=str(obj.get("level", "gold")), fold=int(obj.get("fold", -1)),
            source_work=str(source.get("work", "")),
            source_tokens=list(source["tokens"]),
            reuse_work=str(reuse.get("work", "")),
            reuse_tokens=list(reuse["tokens"]),
            pair_label=str(obj.get("pair_label", "cit")),
            links=links, spans=spans,
            provenance=dict(obj.get("provenance", {})), annotation=annotation,
            split=obj.get("split"), pred=obj.get("pred"),
            benchmark_id=obj.get("benchmark_id"),
            source_meta={k: v for k, v in source.items() if k not in RecordCodec._SIDE_KEYS},
            reuse_meta={k: v for k, v in reuse.items() if k not in RecordCodec._SIDE_KEYS},
        )

    @staticmethod
    def paper_format(record: Record) -> bool:
        """Whether the record is written in the paper format (its provenance names a ``format``)."""
        return bool(record.provenance.get("format"))

    @classmethod
    def to_json(cls, record: Record) -> Dict[str, Any]:
        """A ``Record`` to the nested object of section 1.3 (the only writer).

        A record whose provenance names a ``format`` is written in the paper
        format (the Storage Format of the Data and Splits note): the operations
        in the paper's spelling, ``note`` and ``lookup`` in place of
        ``annotation``; every other record in the code spelling, as before.
        """
        paper = cls.paper_format(record)
        head: Dict[str, Any] = {"id": record.id, "level": record.level, "fold": record.fold}
        if record.benchmark_id is not None:
            head["benchmark_id"] = record.benchmark_id
        out: Dict[str, Any] = {
            **head,
            "source": cls._side_json(record.source_meta, record.source_work, record.source_tokens),
            "reuse": cls._side_json(record.reuse_meta, record.reuse_work, record.reuse_tokens),
            "pair_label": record.pair_label,
            "links": [cls._edge_json(e, paper=paper) for e in record.links],
            "spans": [{"start": sp.start, "end": sp.end, "label": sp.label} for sp in record.spans],
            "provenance": dict(record.provenance),
        }
        if paper:
            lookup = {k: v for k, v in record.annotation.items() if k != "note"}
            out["note"] = str(record.annotation.get("note", "") or "")
            out["lookup"] = lookup
        else:
            out["annotation"] = dict(record.annotation)
        if record.split is not None:
            out["split"] = record.split
        if record.pred is not None:
            out["pred"] = record.pred
        return out

    @staticmethod
    def _side_json(meta: Dict[str, Any], work: str, tokens: Sequence[str]) -> Dict[str, Any]:
        side: Dict[str, Any] = {}
        if "author" in meta:
            side["author"] = meta["author"]
        side["work"] = work
        side.update({k: v for k, v in meta.items() if k != "author"})
        side["tokens"] = list(tokens)
        return side

    @staticmethod
    def _edge_json(edge: Edge, *, paper: bool = False) -> Dict[str, Any]:
        op, detail = labels.Labels.to_paper(edge.op, edge.detail) if paper else (edge.op, edge.detail)
        out: Dict[str, Any] = {"r": edge.r, "s": edge.s, "op": op, "sure": bool(edge.sure)}
        if edge.sure_by:
            out["sure_by"] = edge.sure_by
        if detail:
            out["detail"] = detail
        if edge.p is not None:
            out["p"] = round(float(edge.p), 5)
        return out

    @classmethod
    def load(cls, path: Path, *, max_per_split: Optional[Dict[str, int]] = None) -> List[Record]:
        """Every line of a JSONL record file; ``max_per_split`` caps a split while streaming (the 1.1 M en-fr train lines)."""
        out = []
        seen: Dict[str, int] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if max_per_split:
                    m = cls._SPLIT_KEY.search(line)
                    split = m.group(1) if m else None
                    if split in max_per_split:
                        seen[split] = seen.get(split, 0) + 1
                        if seen[split] > max_per_split[split]:
                            continue
                out.append(cls.from_json(json.loads(line)))
        return out

    @classmethod
    def save(cls, records: Sequence[Record], path: Path) -> None:
        """Write records as JSONL, one nested object per line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(cls.to_json(record), ensure_ascii=False) + "\n")

    @staticmethod
    def gold_records(gold_dir: Path = Path("data/gold_full")) -> List[Record]:
        """The 1,490 hand-labelled pairs as records, fold from ``sample_pairs.json``.

        Goes through ``GoldPair.load`` so that the expansion of the exception
        format stays in one place; operation names pass through
        ``Labels.canonical`` (NOP becomes COPY), frames become ``spans``, and
        the provenance says the links and the V1 operations are human.
        """
        from retexo.datasets.gold import GoldPair

        gold_dir = Path(gold_dir)
        pairs = {p["id"]: p for p in json.loads((gold_dir / "sample_pairs.json").read_text())}
        raw_labels = json.loads((gold_dir / "labels.json").read_text())
        out = []
        for pair in GoldPair.load(gold_dir):
            meta = pairs[pair.id]
            label = raw_labels[pair.id]
            edges = []
            for src, tgt, op in label["alignments"]:
                if 0 <= tgt < len(pair.target_tokens) and 0 <= src < len(pair.source_tokens):
                    canon, detail = labels.canonical(op)
                    if canon is not None:
                        edges.append(Edge(int(tgt), int(src), canon, True, detail))
            spans = [Span(int(a), min(int(b) + 1, len(pair.target_tokens)), "FRAME")
                     for a, b in label.get("frame", []) if a <= b]
            out.append(Record(
                id=pair.id, level="gold", fold=int(meta["fold"]),
                source_work=str(meta.get("source_author", "")), source_tokens=list(pair.source_tokens),
                reuse_work=str(meta.get("query_author", "")), reuse_tokens=list(pair.target_tokens),
                pair_label=pair.ref_type.rstrip("."),
                links=edges, spans=spans,
                # the gold_full labels were read and corrected by an LLM from the E7 model's proposals
                # (data/gold_full/README.md); no human has checked them yet, so they are silver
                provenance={"links": "llm-silver", "fine_ops": "llm-silver-V1", "note": pair.note},
                annotation={},
            ))
        return out

    @staticmethod
    def swapped(record: Record) -> Record:
        """The record with source and reuse exchanged and its links inverted (SPLIT and MERGE trade
        places; frames do not exist on a source passage). Every method's second orientation."""
        from dataclasses import replace

        inverted = {"SPLIT": "MERGE", "MERGE": "SPLIT"}
        edges = [Edge(e.s, e.r, inverted.get(e.op, e.op), e.sure, e.detail) for e in record.links]
        return replace(record, id=record.id + "~", source_tokens=list(record.reuse_tokens),
                       reuse_tokens=list(record.source_tokens), links=edges, spans=[],
                       source_meta=dict(record.reuse_meta), reuse_meta=dict(record.source_meta))

    #: Relations the record keeps in ``detail`` that the generator emits as fine operations of their own.
    CONSTRUCTED_RELATIONS = ("HYPER", "HYPO", "ANT", "SYN-DIST")

    @classmethod
    def constructed_fine_op(cls, op: str, detail: str) -> str:
        """The generator's own fine operation behind a stored edge (``SUBST`` + ``HYPER`` is HYPER)."""
        head = (detail or "").split("+")[0]
        return head if op in ("SUBST", "SYN") and head in cls.CONSTRUCTED_RELATIONS else op

    @staticmethod
    def _as_goldpair(record: Record):
        """A ``GoldPair`` view of the record (private: two consumers need one)."""
        from retexo.datasets.gold import GoldPair

        links, tags, frame, _ = RecordInterface.links_of(record)
        constructed = str(record.provenance.get("fine_ops", "")).startswith("construction")
        detail_of = {e.r: e.detail for e in record.links} if constructed else {}
        ops = []
        for t, tag in enumerate(tags):
            if links[t] < 0:
                ops.append("FRAME" if frame[t] else "INS")
            else:
                if constructed:
                    tag = RecordCodec.constructed_fine_op(tag, detail_of.get(t, ""))
                ops.append(labels.TO_LINK_TAG.get(tag, tag))
        used = {s for s in links if s >= 0}
        return GoldPair(
            id=record.id, ref_type=record.pair_label,
            source_tokens=list(record.source_tokens), target_tokens=list(record.reuse_tokens),
            target_ops=ops, source_del=[0 if s in used else 1 for s in range(record.n_source)],
            target_align=list(links), note=str(record.provenance.get("note", "")),
        )

    @classmethod
    def to_example(cls, record: Record, featurizer=None, *, supervise_v3: Optional[bool] = None):
        """The record as the ``ChangeExample`` the formulations train on.

        With a featurizer the example carries the evidence vectors of its gold
        links (``fine_from_gold``); without one it is the plain
        ``GoldPair.as_example`` view. ``supervise_v3`` makes the annotated V3
        operations the fine targets; by default (``None``) that happens exactly
        when the record's ``provenance["fine_ops"]`` names a V3 annotation
        (``llm-blind-V3``, ``human-V3``), never for the silver's V1 labels.
        """
        pair = cls._as_goldpair(record)
        if featurizer is None:
            return pair.as_example()
        from retexo.datasets.synthetic import fine_from_gold

        if supervise_v3 is None:
            supervise_v3 = cls.annotated_at_v3(record)
        return fine_from_gold(pair, featurizer, gold_fine=bool(supervise_v3))

    @staticmethod
    def annotated_at_v3(record: Record) -> bool:
        """Whether the record's operations were annotated at V3 (``provenance["fine_ops"]`` ends in ``-V3``)."""
        return str(record.provenance.get("fine_ops", "")).upper().endswith("-V3")


# =============================================================================
# Interface (II): the per-token triple
# =============================================================================


class RecordInterface:
    """The per-token ``(links, tags, frame)`` view of a record's edges.

    Example:
        ```python
        links, tags, frame, sure = RecordInterface.links_of(record)
        record.links = RecordInterface.edges_from(links, tags, frame)
        ```
    """

    @staticmethod
    def links_of(record: Record) -> Tuple[List[int], List[str], List[int], List[bool]]:
        """Edges to the per-token triple plus the sure flags.

        Returns ``(links, tags, frame, sure)``: per reuse word the source index or
        ``-1``, the edge operation or ``""`` where unlinked, the frame flag, and
        whether the edge is sure. A reuse word with two edges (a MERGE) keeps its
        first edge here; the second is recovered by ``edges_from`` through
        ``extra``. Two reuse words sharing one source word (a SPLIT) both keep
        their link, since the list can hold that.
        """
        n = record.n_reuse
        links = [-1] * n
        tags = [""] * n
        sure = [True] * n
        for edge in sorted(record.links, key=lambda e: (e.r, e.s)):
            if 0 <= edge.r < n and links[edge.r] < 0:
                links[edge.r] = edge.s
                tags[edge.r] = edge.op
                sure[edge.r] = bool(edge.sure)
        frame = [0] * n
        for span in record.spans:
            if span.label == "FRAME":
                for t in range(max(0, span.start), min(n, span.end)):
                    frame[t] = 1
        return links, tags, frame, sure

    @staticmethod
    def extra_edges(record: Record) -> List[Edge]:
        """The edges ``links_of`` cannot hold: a MERGE's second source edge."""
        seen = set()
        out = []
        for edge in sorted(record.links, key=lambda e: (e.r, e.s)):
            if edge.r in seen:
                out.append(edge)
            else:
                seen.add(edge.r)
        return out

    @staticmethod
    def edges_from(links: Sequence[int], tags: Sequence[str], frame: Sequence[int],
                   extra: Sequence[Edge] = ()) -> List[Edge]:
        """The per-token triple back to edges; ``frame`` is not an edge and is ignored."""
        out = [Edge(t, int(s), tags[t] if t < len(tags) and tags[t] else "SUBST")
               for t, s in enumerate(links) if s is not None and s >= 0]
        out.extend(Edge(e.r, e.s, e.op, e.sure, e.detail, e.p) for e in extra)
        return out

    @staticmethod
    def frame_spans_of(frame: Sequence[int]) -> List[Span]:
        """Maximal runs of the frame mask as ``Span`` objects."""
        out = []
        start = None
        for t, flag in enumerate(list(frame) + [0]):
            if flag and start is None:
                start = t
            elif not flag and start is not None:
                out.append(Span(start, t, "FRAME"))
                start = None
        return out


#: Backward-compatible module-level aliases.
record_from_json = RecordCodec.from_json
record_to_json = RecordCodec.to_json
load_records = RecordCodec.load
save_records = RecordCodec.save
gold_to_records = RecordCodec.gold_records
record_to_example = RecordCodec.to_example
links_of = RecordInterface.links_of
extra_edges = RecordInterface.extra_edges
edges_from = RecordInterface.edges_from
frame_spans_of = RecordInterface.frame_spans_of
