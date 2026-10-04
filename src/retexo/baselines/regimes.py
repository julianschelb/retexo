# retexo/baselines/regimes.py
"""Note 33: the training regimes of Table 4, a registry over one architecture.

Every row of Table 4 is the plain typed pointer of note 15 with only the data
changed: synthetic only, gold only, synthetic then gold (the default), without
negatives, without dense rewrites (the standard pool against the dense pool),
plus real pairs with LLM-written links, plus self-training rounds, plus
distillation from the full system, the gold learning curve, and the Mesham
test (train at V3 against V1). ``REGIMES`` maps a name to the ``cfg.extra``
overrides and the extra data of that row; ``resolve`` fills a fold in;
``curve_sample`` draws the nested, stratified learning-curve samples;
``SelfTraining`` filters a model's own edges on real pairs (mutual decoded
links above ``tau`` whose script replays) into link-only records;
``Distillation`` turns the full system's labels into gold-shaped records.

The driver reads ``--regime NAME[,key=value]`` and merges the resolved
overrides into ``cfg.extra`` before ``fit``; the loops of ``run_regime_loops.py``
call the driver once per round.

    python run_baseline.py --method typed_pointer --fold 4 --regime gold_only
    python run_baseline.py --method typed_pointer --fold 4 --regime curve,n=300,pretrain=0
    python run_regime_loops.py --fold 4 --regime self_train --rounds 3
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from retexo.baselines.base import Prediction
from retexo.baselines.record import Edge, Record, RecordCodec

#: The rows of Table 4 as ``cfg.extra`` overrides (``{fold}`` filled by ``resolve``).
REGIMES: Dict[str, Dict[str, Any]] = {
    "default": {},
    "synthetic_only": {"gold_passes": 0},
    "gold_only": {"size": 0},
    "active": {"size": 0, "ids": ""},     # the active-learning dry run: the acquired pairs (``ids=<file>``), warm-started
    "no_negatives": {"neg_ratio": 0.0, "negatives": "none"},
    "dense": {"dense_rate": 0.3, "mlm_subst": 1, "pool_file": "data/records/synthetic_dense_f{fold}.jsonl"},
    "llm_links": {"extra_pairs": "runs/e32_allusion/extra_pairs_nosub.json,runs/e32_mine/extra_pairs_nosub.json",
                  "links_only": 1},
    "self_train": {"rounds": 1, "tau": 0.9, "extra_records": "runs/self_train_f{fold}/round{round}.jsonl", "links_only": 1},
    "distill": {"extra_records": "runs/distill_f{fold}/labels.jsonl"},
    "curve": {"n": "all", "pretrain": 1},
    "v1_types": {"fine_operations": "v1"},
}

#: The learning curve's sizes; ``all`` is every training pair.
CURVE_SIZES = (100, 300, 600, "all")

#: Self-training's threshold grid on the dev fold.
TAU_GRID = (0.8, 0.85, 0.9, 0.95)


# =============================================================================
# The registry
# =============================================================================


def resolve(name: str, fold: int, **params) -> Dict[str, Any]:
    """The ``cfg.extra`` overrides of a regime for one fold, with the row's
    parameters (``n``, ``pretrain``, ``rounds``, ``tau`` ...) applied.

    Example:
        ```python
        resolve("gold_only", 4)["size"]          # 0
        resolve("curve", 4, n=300)["n"]           # 300
        ```
    """
    if name not in REGIMES:
        raise KeyError(f"unknown regime {name!r}; known: {sorted(REGIMES)}")
    out = dict(REGIMES[name])
    out.update(params)
    if name == "curve" and not int(out.get("pretrain", 1)):
        out["size"] = 0
    if name == "v1_types":
        out["fine_operations"] = "v1"
    fill = {"fold": fold, "round": out.get("round", out.get("rounds", 1))}
    for key, value in list(out.items()):
        if isinstance(value, str) and "{" in value:
            out[key] = value.format(**fill)
    out["regime"] = name
    return out


def parse_regime(text: str) -> Tuple[str, Dict[str, Any]]:
    """``"curve,n=300,pretrain=0"`` to ``("curve", {"n": 300, "pretrain": 0})``."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    name, params = parts[0], {}
    for item in parts[1:]:
        key, value = item.split("=", 1)
        try:
            params[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            params[key] = value
    return name, params


# =============================================================================
# The learning curve
# =============================================================================


def curve_sample(records: Sequence[Record], n, seed: int = 0) -> List[Record]:
    """A nested, stratified sample of ``n`` training records: the 100 lie inside
    the 300 inside the 600, the cit./cf. ratio of the corpus kept within a pair
    or two, deterministic under ``seed``. ``"all"`` returns every record.

    Example:
        ```python
        small, larger = curve_sample(train, 100), curve_sample(train, 300)
        assert {r.id for r in small} <= {r.id for r in larger}
        ```
    """
    if n == "all" or n is None:
        return list(records)
    n = int(n)
    by_label: Dict[str, List[Record]] = {}
    for record in records:
        by_label.setdefault(record.pair_label, []).append(record)
    rng = random.Random(seed)
    ordered: Dict[str, List[Record]] = {}
    for label, items in sorted(by_label.items()):
        items = sorted(items, key=lambda r: r.id)
        rng.shuffle(items)
        ordered[label] = items
    total = len(records)
    chosen: List[Record] = []
    for label, items in ordered.items():
        share = int(round(n * len(items) / max(total, 1)))
        chosen.extend(items[:share])
    # rounding can leave the sample a pair short or long: fix at the largest stratum
    largest = max(ordered, key=lambda k: len(ordered[k]))
    taken = {r.id for r in chosen}
    while len(chosen) < min(n, total):
        nxt = next(r for r in ordered[largest] if r.id not in taken)
        chosen.append(nxt); taken.add(nxt.id)
    while len(chosen) > n:
        chosen.pop()
    return chosen


# =============================================================================
# Extra data as records
# =============================================================================


class ExtraPairs:
    """Gold-shaped pairs from a JSON file (the E32 LLM-scripted pairs) as records.

    Example:
        ```python
        records = ExtraPairs.load(Path("runs/e32_allusion/extra_pairs_nosub.json"), links_only=True)
        ```
    """

    @staticmethod
    def load(path: Path, *, links_only: bool = False, level: str = "real_pairs") -> List[Record]:
        from retexo.baselines import labels

        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        out = []
        for row in rows:
            source, target = list(row["source_tokens"]), list(row["target_tokens"])
            ops, align = row.get("target_ops") or [], row.get("target_align") or []
            edges, spans = [], []
            frame_start = None
            for t, (op, s) in enumerate(zip(ops, align)):
                if op == "FRAME":
                    frame_start = t if frame_start is None else frame_start
                    continue
                if frame_start is not None:
                    spans.append(_frame_span(frame_start, t)); frame_start = None
                if s is None or s < 0:
                    continue
                canon, detail = labels.canonical(op)
                if canon is None:
                    canon = "SUBST"
                if links_only and canon not in ("COPY", "MORPH"):
                    canon, detail = "LINK", ""
                edges.append(Edge(t, int(s), canon, True, detail))
            if frame_start is not None:
                spans.append(_frame_span(frame_start, len(target)))
            out.append(Record(id=f"{level}/{row['id']}", level=level, fold=-1, source_work="", source_tokens=source,
                              reuse_work="", reuse_tokens=target, pair_label=str(row.get("ref_type", "llm")).rstrip("."),
                              links=edges, spans=spans, provenance={"links": "llm", "fine_ops": "llm" if not links_only else "none"}))
        return out


def _frame_span(start: int, end: int):
    from retexo.baselines.record import Span

    return Span(start, end, "FRAME")


def link_only_example(record: Record, featurizer=None):
    """The record as a training example whose ``LINK`` edges carry no type target at any level (fine tag
    ``LINK``, masked by the heads), exact COPY/MORPH stay; the coarse head sees COPY for the same form and
    SUBST for a change. So a self-training or LLM-linked pair trains the pointer's location and not its typer.
    (Until 2026-09-25 a LINK edge became the lexical group target, which taught every self-trained
    inflection as a lexical change: the reduced test's INFLECT F1 fell from .61 to .30 over two rounds.)
    ``RecordCodec.to_example`` does this for every record; this is the explicit name for it."""
    from retexo.baselines.adapters import PredictionAdapter
    from retexo.baselines.record import RecordCodec

    return RecordCodec.to_example(record, featurizer or PredictionAdapter._stub_featurizer())


# =============================================================================
# Self-training and distillation
# =============================================================================


class SelfTraining:
    """One round: the model's own edges on real pairs, filtered.

    An edge ``(t, s)`` is kept iff ``s`` is the decoded link of ``t`` forward and
    ``t`` the decoded link of ``s`` backward, the averaged probability is at least
    ``tau``, and the pair's decoded script replays under ``Scriba``; every other
    reuse word becomes a null target; the record enters the gold stage link-only.

    Example:
        ```python
        kept = SelfTraining.filter(records, forward_preds, reverse_preds, tau=0.9, decode=decoder.decode_default)
        ```
    """

    @staticmethod
    def agreed_edges(record: Record, forward: Prediction, reverse: Optional[Prediction], *, tau: float,
                     decode: Callable, tau_changed: Optional[float] = None) -> List[Edge]:
        """``tau_changed`` (the changed-form teacher, 2026-09-27): links between different forms pass at this
        threshold instead of ``tau``; identical forms keep ``tau``."""
        from retexo.core.normalize import normalize

        rows, rev = forward.scores, forward.rev_scores if reverse is None else reverse.scores
        if not rows:
            return []
        links_f = decode(rows, n_source=record.n_source)
        links_b = decode(rev, n_source=record.n_reuse) if rev else None
        p_f = [dict(row) for row in rows]
        p_b = [dict(row) for row in rev] if rev else None
        out = []
        for t, s in enumerate(links_f):
            if s is None or s < 0:
                continue
            if links_b is not None and (s >= len(links_b) or links_b[s] != t):
                continue
            p = p_f[t].get(s, 0.0)
            if p_b is not None and s < len(p_b):
                p = 0.5 * (p + p_b[s].get(t, 0.0))
            changed = normalize(record.reuse_tokens[t]) != normalize(record.source_tokens[int(s)])
            if p >= (tau_changed if changed and tau_changed is not None else tau):
                out.append(Edge(t, int(s), "LINK", True, "", round(float(p), 4)))
        return out

    @staticmethod
    def framed(record: Record, edges: Sequence[Edge], *, window: int = 2, slack: int = 3, min_links: int = 3) -> List[Edge]:
        """The changed-form teacher's pair filter: a changed-form edge stays only in a pair with at least
        ``min_links`` edges and with another kept edge in order within ``window`` reuse words (source offset in
        the same direction, within ``slack``) -- the kept frame of the error analysis. Identical forms always stay."""
        from retexo.core.normalize import normalize

        out = []
        for e in edges:
            if normalize(record.reuse_tokens[e.r]) == normalize(record.source_tokens[e.s]):
                out.append(e); continue
            if len(edges) < min_links:
                continue
            support = sum(1 for o in edges if o is not e and 0 < abs(o.r - e.r) <= window
                          and (o.s - e.s > 0) == (o.r - e.r > 0) and o.s != e.s and abs((o.s - e.s) - (o.r - e.r)) <= slack)
            if support >= 1:
                out.append(e)
        return out

    @classmethod
    def filter(cls, records: Sequence[Record], forward: Sequence[Prediction], reverse: Optional[Sequence[Prediction]] = None,
               *, tau: float = 0.9, decode: Optional[Callable] = None, tau_changed: Optional[float] = None,
               frame_filter: bool = False) -> List[Record]:
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.baselines.decoder import BaselineDecoder
        from retexo.core.scriba import Scriba

        decode = decode or (lambda rows, n_source: BaselineDecoder.decode_default(rows, theta=0.45, n_source=n_source))
        scriba = Scriba()
        out = []
        for i, record in enumerate(records):
            fwd = forward[i]
            rev = reverse[i] if reverse is not None else None
            edges = cls.agreed_edges(record, fwd, rev, tau=tau, decode=decode, tau_changed=tau_changed)
            if frame_filter:
                edges = cls.framed(record, edges)
            if not edges:
                continue
            probe = Prediction(links=[-1] * record.n_reuse, tags=[""] * record.n_reuse, frame=[0] * record.n_reuse)
            for edge in edges:
                probe.links[edge.r] = edge.s
                probe.tags[edge.r] = "SUBST"
            script = PredictionAdapter.to_script(record, probe)
            if script is None or not scriba.verify(script, record.source_tokens, record.reuse_tokens):
                continue
            out.append(replace(record, id=f"self/{record.id}", level="real_pairs", links=edges, spans=[],
                               provenance={"links": "self-training", "fine_ops": "none", "tau": tau,
                                           "tau_changed": tau_changed, "frame_filter": frame_filter}))
        return out

    @staticmethod
    def type_edges(records: Sequence[Record], predictions: Sequence[Prediction], *, mode: str, featurizer) -> List[Record]:
        """Give the kept link-only edges a type (the INFLECT follow-up of 2026-09-26). ``rule``: the shared rule
        typer's operation for every edge; ``own``: the model's own tag (its confident prediction, so the type head
        distils itself); ``agree``: identical forms stay COPY, a changed-form edge keeps the rule's type only where
        the model's own tag falls in the same class (MORPH or lexical), other changed-form edges stay link-only
        (no type target), never dropped. A lexical tag becomes SUBST, which the heads read as a lexical change of
        unknown kind. (The first ``agree`` run of 2026-09-26 dropped the identical-form edges, which then trained
        as null targets and cost link F1.)"""
        from retexo.baselines.typer import RuleTyper

        def kind(tag: str) -> str:
            tag = "COPY" if tag == "NOP" else tag          # the head says NOP where the rule says COPY
            return tag if tag in ("COPY", "MORPH") else ("LEX" if tag else "")

        out = []
        for record, pred in zip(records, predictions):
            links = [-1] * record.n_reuse
            for edge in record.links:
                links[edge.r] = edge.s
            rule, _, _ = RuleTyper.rule_type(record, links, featurizer)
            own = list(pred.tags or [])
            edges = []
            for edge in record.links:
                tag = rule[edge.r] or "SUBST"
                mine = own[edge.r] if edge.r < len(own) else ""
                if mode == "own":
                    tag = "COPY" if kind(mine) == "COPY" else mine if mine == "MORPH" else ("SUBST" if mine else "LINK")
                elif mode == "agree":
                    if kind(tag) == "COPY":
                        tag = "COPY"
                    elif kind(mine) != kind(tag):
                        tag = "LINK"                # no agreement: the link trains, no type head does
                edges.append(replace(edge, op=tag if tag in ("COPY", "MORPH", "LINK") else "SUBST"))
            if edges:
                out.append(replace(record, links=edges, provenance={**record.provenance, "fine_ops": f"rule-{mode}"}))
        return out

    @classmethod
    def tune_tau(cls, dev: Sequence[Record], forward: Sequence[Prediction], reverse: Optional[Sequence[Prediction]] = None,
                 *, grid: Sequence[float] = TAU_GRID) -> float:
        """The threshold whose kept edges are most precise against the dev gold,
        breaking ties toward more edges (the note's stand-in for a full retrain per tau)."""
        from retexo.baselines.record import RecordInterface

        best = (-1.0, 0, grid[0])
        for tau in grid:
            right = total = 0
            for record, fwd, rev in zip(dev, forward, reverse or [None] * len(dev)):
                gold = RecordInterface.links_of(record)[0]
                for edge in cls.agreed_edges(record, fwd, rev, tau=tau, decode=lambda rows, n_source: [
                        row[0][0] if row and row[0][0] >= 0 else -1 for row in rows]):
                    total += 1; right += int(gold[edge.r] == edge.s)
            precision = right / max(total, 1)
            if (precision, total) > (best[0], best[1]):
                best = (precision, total, tau)
        return best[2]


class Distillation:
    """A teacher's labels on real pairs as gold-shaped records.

    The paper's teacher is the two-view merge (ours plus the span aligner): its decoded, rule-typed
    prediction dump on the real pairs (``from_dump``); ``records`` reads the older full system's
    ``pred`` blocks.

    Example:
        ```python
        records = Distillation.from_dump(Path("runs/drd_teacher_f4/real.jsonl"))
        ```
    """

    @staticmethod
    def from_dump(path: Path, *, teacher: str = "merged_view") -> List[Record]:
        """Every decoded link of the dump as an edge with its typed operation, the FRAME runs as spans;
        pairs where the teacher links nothing are kept (they teach the null)."""
        from retexo.baselines.adapters import read_dump

        out = []
        for record, pred in read_dump(Path(path)):
            edges = [Edge(t, int(s), (pred.tags[t] if t < len(pred.tags) and pred.tags[t] else "SUBST"))
                     for t, s in enumerate(pred.links) if s is not None and s >= 0]
            spans, start = [], None
            for t, flag in enumerate(list(pred.frame or []) + [0]):
                if flag and start is None:
                    start = t
                elif not flag and start is not None:
                    spans.append(_frame_span(start, t)); start = None
            out.append(replace(record, id=f"distill/{record.id}", level="real_pairs", links=edges, spans=spans, pred=None,
                               provenance={"links": teacher, "fine_ops": f"{teacher}-rule-typer"}))
        return out

    @staticmethod
    def records(labelled: Sequence[Record]) -> List[Record]:
        out = []
        for record in labelled:
            pred = record.pred or {}
            edges = [e if isinstance(e, Edge) else Edge(int(e["r"]), int(e["s"]), str(e.get("op", "SUBST")), True, str(e.get("detail", "")))
                     for e in pred.get("edges", [])]
            frame = pred.get("frame") or []
            spans = []
            start = None
            for t, flag in enumerate(list(frame) + [0]):
                if flag and start is None:
                    start = t
                elif not flag and start is not None:
                    spans.append(_frame_span(start, t)); start = None
            out.append(replace(record, id=f"distill/{record.id}", level="real_pairs", links=edges, spans=spans, pred=None,
                               provenance={"links": "full_system", "fine_ops": "full_system-gated"}))
        return out


# =============================================================================
# Applying a regime to a run
# =============================================================================


class RegimeApplier:
    """What the driver does with a resolved regime: the extra overrides, the
    learning-curve sample, the extra records appended to the training set.

    Example:
        ```python
        overrides, train = RegimeApplier.apply("curve,n=300", fold=4, train=train)
        ```
    """

    @staticmethod
    def apply(text: str, *, fold: int, train: Sequence[Record], log=None) -> Tuple[Dict[str, Any], List[Record]]:
        name, params = parse_regime(text)
        overrides = resolve(name, fold, **params)
        records = list(train)
        if name == "curve":
            records = curve_sample(records, overrides.get("n", "all"), seed=0)
        if str(overrides.get("ids", "")).strip():          # an explicit set of training records, one id per line
            keep = {line.strip() for line in Path(str(overrides["ids"])).read_text().splitlines() if line.strip()}
            records = [r for r in records if r.id in keep]
            if log:
                log(f"[regime {name}] {len(records)} of {len(keep)} listed records found in the training folds")
        links_only = bool(int(overrides.get("links_only", 0)))
        for key in ("extra_pairs", "extra_records"):
            for path in [p for p in str(overrides.get(key, "")).split(",") if p.strip()]:
                path = Path(path.strip())
                if not path.exists():
                    if log:
                        log(f"[regime {name}] {path} not found; that data is left out")
                    continue
                extra = ExtraPairs.load(path, links_only=links_only) if path.suffix == ".json" else RecordCodec.load(path)
                records.extend(extra)
                if log:
                    log(f"[regime {name}] + {len(extra)} records from {path}")
        if log:
            log(f"[regime {name}] {len(records)} training records; overrides {dict((k, v) for k, v in overrides.items() if k != 'regime')}")
        return overrides, records
