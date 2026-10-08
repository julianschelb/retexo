# demo/annotate.py
"""Gradio annotation app: correct the silver token labels, model confidence beside them.

Loads the 1,490 pairs of ``data/gold_full`` with their current (LLM-read, silver)
token labels and, where a dump covers a pair, the model's own link, type and
confidence per reuse word (the champion's fold-4 dump by default). Each pair is
one editable table: the reuse word, its source word (index), its operation from
the extended inventory, whether the link is sure, a comment; the model's
proposal and its confidence stand beside them, and cells the model is unsure
about or disagrees with are flagged. A pair can be shown blind (no model
columns) for the random arm of the annotation protocol (``Next Steps`` §4a).

Corrections are written per annotator to ``data/gold_full/corrections/<annotator>.jsonl``,
one line per saved pair (the whole table, with a timestamp); the last line per
pair wins. Nothing here changes the gold files.

    python demo/annotate.py --annotator julian --dump runs/champion_fold4/predictions.jsonl
    python demo/annotate.py --annotator expert1 --blind --sample data/gold_full/sample_round1.json
"""

from __future__ import annotations

import argparse
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from retexo.baselines import labels  # noqa: E402
from retexo.paths import home  # noqa: E402

HERE = home()
from retexo.baselines.record import Record, RecordCodec, RecordInterface  # noqa: E402

#: The extended inventory the annotators choose from (the fine types of the definition plus the nulls).
OPERATIONS = (
    "COPY",
    "MORPH",
    "SYN",
    "SYN-DIST",
    "HYPER",
    "HYPO",
    "ANT",
    "NE-SUB",
    "POS",
    "SPLIT",
    "MERGE",
    "SUBST",
    "INS",
    "FRAME",
)

#: What each operation means, shown in the app.
GLOSSES = {
    "COPY": "the same word, unchanged (spelling variants count as unchanged)",
    "MORPH": "the same word in another form (case, number, tense, person)",
    "SYN": "a synonym: a word that could stand in its place",
    "SYN-DIST": "a word of the same field that could not stand in its place",
    "HYPER": "a more general word",
    "HYPO": "a more specific word",
    "ANT": "an opposite",
    "NE-SUB": "a different proper name",
    "POS": "the same stem as another part of speech",
    "SPLIT": "this word and its neighbour together render one source word",
    "MERGE": "this word alone renders two source words (name the first; note the second)",
    "SUBST": "a different word with no relation to the source word it replaces",
    "INS": "the later author's own word, from nowhere in the source",
    "FRAME": 'a word of the citing formula ("as Virgil says")',
}

#: Below this confidence a model cell is flagged.
UNSURE = 0.7

TABLE_COLUMNS = [
    "#",
    "reuse word",
    "source #",
    "operation",
    "sure",
    "comment",
    "model source #",
    "model op",
    "model p",
    "flag",
]


# =============================================================================
# The model's proposals, from a dump
# =============================================================================


@dataclass
class Proposal:
    """The model's view of one pair: link, type and confidence per reuse word."""

    links: List[int]
    tags: List[str]
    p: List[Optional[float]]
    frame: List[int]

    @staticmethod
    def from_legacy(row: Dict[str, Any]) -> Proposal:
        """The preliminary round's dump (``links``, ``gated_tags``, ``frames``, ``top``)."""
        links = [int(s) for s in row.get("links", [])]
        tags = list(row.get("gated_tags") or row.get("model_tags") or ["INS"] * len(links))
        frames = [int(f) for f in row.get("frames", [0] * len(links))]
        ps: List[Optional[float]] = []
        for t, top in enumerate(row.get("top") or []):
            chosen = links[t] if t < len(links) else -1
            prob = dict((int(s), float(p)) for s, p in top).get(chosen)
            ps.append(
                prob if prob is not None else (dict((int(s), float(p)) for s, p in top).get(-1))
            )
        ps += [None] * (len(links) - len(ps))
        return Proposal(links, tags, ps, frames)

    @staticmethod
    def from_harness(pred) -> Proposal:
        n = len(pred.links)
        return Proposal(
            list(pred.links), list(pred.tags), list(pred.link_p or [None] * n), list(pred.frame)
        )

    def op_at(self, t: int) -> str:
        if t >= len(self.links):
            return ""
        if self.links[t] < 0:
            return "FRAME" if t < len(self.frame) and self.frame[t] else "INS"
        canon, detail = labels.canonical(self.tags[t] or "SUBST")
        return detail or canon or "SUBST"


class Proposals:
    """Every dump's proposals, keyed by pair id.

    Example:
        ```python
        proposals = Proposals([Path("runs/champion_fold4/predictions.jsonl")])
        proposals.get("p0010")          # Proposal or None
        ```
    """

    def __init__(self, paths: Sequence[Path]):
        self.by_id: Dict[str, Proposal] = {}
        self.sources: List[str] = []
        for path in paths:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        if not path.exists():
            print(f"[annotate] no dump at {path}", flush=True)
            return
        first = next(
            (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            {},
        )
        n = 0
        if isinstance(first.get("pred"), dict):  # the harness dump
            from retexo.baselines.adapters import PredictionAdapter

            for record, pred in PredictionAdapter.read_dump(path):
                if not pred.invalid:
                    self.by_id[record.id] = Proposal.from_harness(pred)
                    n += 1
        else:  # the preliminary round's dump
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("id") and "links" in row:
                    self.by_id[row["id"]] = Proposal.from_legacy(row)
                    n += 1
        self.sources.append(f"{path} ({n} pairs)")
        print(f"[annotate] {n} proposals from {path}", flush=True)

    def get(self, pair_id: str) -> Optional[Proposal]:
        return self.by_id.get(pair_id)


# =============================================================================
# Corrections on disk
# =============================================================================


class CorrectionStore:
    """One JSONL per annotator; the last line per pair wins.

    Example:
        ```python
        store = CorrectionStore(Path("data/gold_full/corrections"), "julian")
        store.save("p0010", rows, note="")
        store.get("p0010")              # the saved rows, or None
        ```
    """

    def __init__(self, directory: Path, annotator: str):
        self.path = Path(directory) / f"{annotator}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.annotator = annotator
        self.saved: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.saved[row["id"]] = row

    def save(
        self,
        pair_id: str,
        rows: List[Dict[str, Any]],
        *,
        note: str = "",
        blind: bool = False,
        seconds: float = 0.0,
    ) -> None:
        entry = {
            "id": pair_id,
            "annotator": self.annotator,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blind": blind,
            "seconds": round(seconds, 1),
            "note": note,
            "words": rows,
        }
        with self.path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.saved[pair_id] = entry

    def get(self, pair_id: str) -> Optional[Dict[str, Any]]:
        return self.saved.get(pair_id)

    def done(self) -> int:
        return len(self.saved)


# =============================================================================
# The pairs
# =============================================================================


@dataclass
class Session:
    """The records in annotation order and the annotator's place in them."""

    records: List[Record]
    proposals: Proposals
    store: CorrectionStore
    blind: bool = False
    index: int = 0
    opened_at: float = field(default_factory=time.time)

    @property
    def current(self) -> Record:
        return self.records[self.index]

    def find(self, pair_id: str) -> Optional[int]:
        for i, record in enumerate(self.records):
            if record.id == pair_id:
                return i
        return None


def silver_rows(
    record: Record, proposal: Optional[Proposal], saved: Optional[Dict[str, Any]], *, blind: bool
) -> List[List[Any]]:
    """The editable table: the saved correction if there is one, else the silver
    labels, with the model's columns beside them."""
    links, tags, frame, sure = RecordInterface.links_of(record)
    detail = {e.r: e.detail for e in record.links if e.detail}
    rows = []
    for t, word in enumerate(record.reuse_tokens):
        if saved is not None and t < len(saved["words"]):
            w = saved["words"][t]
            src, op, is_sure, comment = (
                w.get("source", -1),
                w.get("op", "INS"),
                w.get("sure", True),
                w.get("comment", ""),
            )
        else:
            src = links[t]
            op = (detail.get(t) or tags[t]) if src >= 0 else ("FRAME" if frame[t] else "INS")
            is_sure, comment = bool(sure[t]) if src >= 0 else True, ""
        m_src, m_op, m_p, flag = "", "", "", ""
        if proposal is not None and not blind and t < len(proposal.links):
            m_src = proposal.links[t] if proposal.links[t] >= 0 else -1
            m_op = proposal.op_at(t)
            prob = proposal.p[t] if t < len(proposal.p) else None
            m_p = f"{prob:.2f}" if prob is not None else ""
            disagree = (m_src != src) or (labels.canonical(m_op)[0] != labels.canonical(op)[0])
            flag = ("disagrees " if disagree else "") + (
                "unsure" if prob is not None and prob < UNSURE else ""
            )
        rows.append(
            [
                t,
                word,
                int(src),
                op,
                "yes" if is_sure else "possible",
                comment,
                m_src,
                m_op,
                m_p,
                flag.strip(),
            ]
        )
    return rows


def render_pair(record: Record, rows: List[List[Any]]) -> str:
    """The source with indices and the reuse coloured by operation, links as arrows."""
    colours = {
        "COPY": "#2a7",
        "MORPH": "#27a",
        "INS": "#999",
        "FRAME": "#a63",
        "SPLIT": "#a2a",
        "MERGE": "#a2a",
    }
    src = " ".join(
        f"<span style='white-space:nowrap'><sup style='color:#999'>{i}</sup>{html.escape(w)}</span>"
        for i, w in enumerate(record.source_tokens)
    )
    parts = []
    for row in rows:
        _t, word, s, op = row[0], row[1], int(row[2]), str(row[3])
        colour = colours.get(op, "#c33")
        arrow = f"<sup style='color:#999'>&rarr;{s}</sup>" if s >= 0 else ""
        flag = " style='outline:2px solid #e90'" if row[9] else ""
        parts.append(
            f"<span{flag} title='{html.escape(op)}'><span style='color:{colour};font-weight:600'>{html.escape(word)}</span>{arrow}</span>"
        )
    legend = (
        " ".join(f"<span style='color:{c}'>{op}</span>" for op, c in colours.items())
        + " <span style='color:#c33'>lexical types</span>"
    )
    return (
        f"<div style='font-family:Georgia,serif;font-size:16px;line-height:1.9'>"
        f"<div style='color:#666;font-size:12px'>SOURCE ({record.source_work})</div><div>{src}</div>"
        f"<div style='color:#666;font-size:12px;margin-top:8px'>REUSE ({record.reuse_work}, {record.pair_label}.)</div>"
        f"<div>{' '.join(parts)}</div><div style='font-size:11px;color:#666;margin-top:6px'>{legend}; "
        f"orange outline = the model disagrees or is unsure</div></div>"
    )


def validate(rows: List[List[Any]], record: Record) -> List[str]:
    problems = []
    used: Dict[int, int] = {}
    for row in rows:
        t, s, op = int(row[0]), row[2], str(row[3]).strip().upper()
        try:
            s = int(s)
        except (TypeError, ValueError):
            problems.append(f"word {t}: source # must be an integer or -1")
            continue
        if op not in OPERATIONS:
            problems.append(f"word {t}: operation {op!r} is not one of {', '.join(OPERATIONS)}")
        if s >= record.n_source:
            problems.append(
                f"word {t}: source # {s} is beyond the source ({record.n_source} words)"
            )
        if s >= 0 and op in ("INS", "FRAME"):
            problems.append(f"word {t}: {op} cannot point at a source word")
        if s < 0 and op not in ("INS", "FRAME"):
            problems.append(f"word {t}: {op} needs a source #")
        if s >= 0 and op != "SPLIT":
            if s in used:
                problems.append(
                    f"word {t}: source # {s} is already used by word {used[s]} (only SPLIT may share a source word)"
                )
            used[s] = t
    return problems


# =============================================================================
# The app
# =============================================================================


def build(session: Session):
    import gradio as gr

    def show(index: int):
        session.index = max(0, min(index, len(session.records) - 1))
        session.opened_at = time.time()
        record = session.current
        saved = session.store.get(record.id)
        rows = silver_rows(record, session.proposals.get(record.id), saved, blind=session.blind)
        status = (
            f"pair {session.index + 1} of {len(session.records)}: **{record.id}** (fold {record.fold}, {record.pair_label}.) "
            f"| {'corrected ' + saved['time'] if saved else 'silver labels'} | {session.store.done()} pairs saved by {session.store.annotator}"
            + (" | blind" if session.blind else "")
        )
        return record.id, status, render_pair(record, rows), rows, (saved or {}).get("note", ""), ""

    def save(pair_id: str, table, note: str):
        record = session.current
        rows = table.values.tolist() if hasattr(table, "values") else list(table)
        rows = [list(r) for r in rows]
        problems = validate(rows, record)
        if problems:
            return "**Not saved:** " + "; ".join(problems), render_pair(record, rows)
        words = [
            {
                "t": int(r[0]),
                "word": str(r[1]),
                "source": int(r[2]),
                "op": str(r[3]).strip().upper(),
                "sure": str(r[4]).strip().lower() != "possible",
                "comment": str(r[5] or ""),
            }
            for r in rows
        ]
        session.store.save(
            record.id,
            words,
            note=note or "",
            blind=session.blind,
            seconds=time.time() - session.opened_at,
        )
        return (
            f"Saved {record.id} ({session.store.done()} pairs by {session.store.annotator}).",
            render_pair(record, rows),
        )

    def goto(pair_id: str):
        index = session.find(pair_id.strip())
        return show(index if index is not None else session.index)

    with gr.Blocks(title="Edit-script annotation", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "## Correct the token labels\n"
            "Every reuse word has a source word (its index, or -1) and an operation. Edit the **source #**, "
            "**operation**, **sure** (yes / possible) and **comment** cells; the model's columns are read-only "
            "context. Save writes the whole pair; the last save per pair wins."
        )
        with gr.Row():
            pair_box = gr.Textbox(label="pair id", scale=1)
            go = gr.Button("go", scale=0)
            prev_btn = gr.Button("previous", scale=0)
            next_btn = gr.Button("next", scale=0)
            unsaved_btn = gr.Button("next unsaved", scale=0)
        status = gr.Markdown()
        view = gr.HTML()
        table = gr.Dataframe(
            headers=TABLE_COLUMNS,
            datatype=["number", "str", "number", "str", "str", "str", "str", "str", "str", "str"],
            interactive=True,
            wrap=True,
            label="words (edit source #, operation, sure, comment)",
        )
        with gr.Row():
            note = gr.Textbox(
                label="note on the pair (frame boundary, doubts, anything the labels cannot say)",
                scale=3,
            )
            save_btn = gr.Button("save pair", variant="primary", scale=1)
        message = gr.Markdown()
        with gr.Accordion("operations", open=False):
            gr.Markdown("\n".join(f"- **{op}**: {GLOSSES[op]}" for op in OPERATIONS))
        outputs = [pair_box, status, view, table, note, message]
        app.load(lambda: show(session.index), None, outputs)
        go.click(goto, [pair_box], outputs)
        prev_btn.click(lambda: show(session.index - 1), None, outputs)
        next_btn.click(lambda: show(session.index + 1), None, outputs)

        def next_unsaved():
            for offset in range(1, len(session.records) + 1):
                index = (session.index + offset) % len(session.records)
                if session.store.get(session.records[index].id) is None:
                    return show(index)
            return show(session.index)

        unsaved_btn.click(next_unsaved, None, outputs)
        save_btn.click(save, [pair_box, table, note], [message, view])
    return app


def select_records(
    records: List[Record], *, sample: Optional[Path], folds: Optional[Sequence[int]], limit: int
) -> List[Record]:
    if sample is not None and sample.exists():
        ids = json.loads(sample.read_text())
        wanted = {i: k for k, i in enumerate(ids)}
        records = sorted([r for r in records if r.id in wanted], key=lambda r: wanted[r.id])
    if folds:
        records = [r for r in records if r.fold in folds]
    return records[:limit] if limit else records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--gold-dir", default="data/gold_full")
    ap.add_argument(
        "--dump",
        action="append",
        default=[],
        help="a predictions.jsonl with the model's proposals (repeatable)",
    )
    ap.add_argument("--sample", default=None, help="a JSON list of pair ids, in annotation order")
    ap.add_argument("--folds", default=None, help="e.g. 4 or 0,1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--blind", action="store_true", help="hide the model's columns (the random arm)"
    )
    ap.add_argument("--corrections", default="data/gold_full/corrections")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    records = RecordCodec.gold_records(HERE / args.gold_dir)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    records = select_records(
        records, sample=Path(args.sample) if args.sample else None, folds=folds, limit=args.limit
    )
    dumps = [HERE / d for d in (args.dump or ["runs/champion_fold4/predictions.jsonl"])]
    session = Session(
        records,
        Proposals(dumps),
        CorrectionStore(HERE / args.corrections, args.annotator),
        blind=args.blind,
    )
    print(
        f"[annotate] {len(records)} pairs, {len(session.proposals.by_id)} with proposals, annotator {args.annotator}"
        f"{', blind' if args.blind else ''}",
        flush=True,
    )
    build(session).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
