# retexo/baselines/adapters.py
"""The bridges between the three interfaces the project already uses.

Interface (I) is a score row per reuse word, consumed by every decoder;
interface (II) is the per-token triple ``(links, tags, frame)`` consumed by
``Repairer.apply``, ``decode.decode_script`` and ``metrics.evaluate``;
interface (III) is the prediction dump ``runs/<exp>/predictions.jsonl`` that
the preliminary readers (``attic/scripts/ensemble_dump.py``, ``rescore_dump.py``) read. This module converts
between them and nothing else: ``decode_prediction`` and ``type_prediction``
are the two steps of the default ``Baseline.postprocess``; ``to_script`` and
``from_script`` cross the tag boundary to the decoder, which spells the copy
``NOP`` and lists neither ``COPY`` nor ``SUBST``; ``write_dump`` and
``read_dump`` carry the ``pred`` block of section 1.3 beside the legacy keys
the old readers need.

An invalid prediction (``raw`` set, no links) has no script: ``to_script``
returns ``None`` and the scorer counts it, as ``ScriptModel.predict`` already
does for an unreadable output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import labels
from retexo.baselines.base import Prediction
from retexo.baselines.record import (Edge, Record, edges_from, extra_edges, frame_spans_of, links_of,
                                         record_from_json, record_to_json)

# =============================================================================
# The two steps of the default hook
# =============================================================================


class PredictionAdapter:
    """Bridges interface (I) scores, (II) the per-token triple, and (III) the dump.

    Example:
        ```python
        pred = PredictionAdapter.decode_prediction(pred, "default", {}, record)
        pred = PredictionAdapter.type_prediction(pred, record, "rule")
        script = PredictionAdapter.to_script(record, pred)
        ```
    """

    #: Score-row entries kept per reuse word in a dump.
    TOP_K_IN_DUMP = 6

    @staticmethod
    def decode_prediction(pred: Prediction, decoder_name: str, dials: Dict[str, Any], record: Record) -> Prediction:
        """Fill ``links`` from ``scores`` with the named decoder; a no-op without scores."""
        from retexo.baselines import decoder as dec

        if pred.scores is None or decoder_name == "none" or pred.invalid:
            return pred
        theta = float(dials.get("theta", dec.DEFAULT_THETA))
        links, extra = dec.decode(decoder_name, pred.scores, theta=theta, rev_rows=pred.rev_scores,
                                  n_source=record.n_source, record=record, dials=dials)
        n = record.n_reuse
        pred.links = [int(s) for s in links[:n]] + [-1] * (n - len(links))
        pred.extra = [Edge(t, s, "") for t, s in extra] + [e for e in pred.extra if e.op]
        prob = [dict(row) for row in pred.scores]
        pred.link_p = [float(prob[t].get(s, prob[t].get(-1, 0.0))) if t < len(prob) else 0.0
                       for t, s in enumerate(pred.links)]
        return pred

    @classmethod
    def type_prediction(cls, pred: Prediction, record: Record, typer_name: str, featurizer=None,
                        *, frame_rule: str = "keyword", head=None, keywords=None, templates=()) -> Prediction:
        """Fill ``tags`` (and ``frame`` for alignment-only rows) with the shared typer."""
        from retexo.baselines import typer as typ

        if typer_name == "own" or pred.invalid:
            return pred
        featurizer = featurizer or cls._stub_featurizer()
        if typer_name == "trained" and head is not None:
            tags = typ.type_with_head(head, record, pred.links, featurizer)
        else:
            tags, outcomes, details = typ.rule_type(record, pred.links, featurizer)
            pred.meta["outcomes"] = outcomes
            pred.meta["details"] = details
        pred.tags = [tag if s >= 0 else "" for tag, s in zip(tags, pred.links)]
        for edge in pred.extra:
            if not edge.op:
                t_tags, _, _ = typ.rule_type(record, [edge.s if t == edge.r else -1 for t in range(record.n_reuse)],
                                             featurizer)
                edge.op = t_tags[edge.r] or "SUBST"
        if frame_rule != "none" and not any(pred.frame):
            pred.frame = typ.frame_rule(record, pred.links, templates, keywords or typ.load_keywords())
        pred.frame = [f if s < 0 else 0 for f, s in zip(pred.frame, pred.links)]
        return pred

    @staticmethod
    def _stub_featurizer():
        """A featurizer for records without resources: identity and enclitic tests only."""
        from retexo.edit_typing.link_features import FEATURE_NAMES
        from retexo.core.normalize import normalize
        from retexo.edit_typing.repair import ENCLITICS

        def phi(source: str, target: str, s: int = 0, t: int = 0, n_s: int = 1, n_t: int = 1) -> List[float]:
            f = {name: 0.0 for name in FEATURE_NAMES}
            a, b = normalize(source), normalize(target)
            f["same_form"] = float(a == b)
            f["lemma_missing"] = 1.0
            f["wn_missing"] = 1.0
            f["cos_missing"] = 1.0
            enc_s = next((e for e in ENCLITICS if a.endswith(e) and len(a) - len(e) >= 3), None)
            enc_t = next((e for e in ENCLITICS if b.endswith(e) and len(b) - len(e) >= 3), None)
            f["enclitic_src"] = float(enc_s is not None)
            f["enclitic_tgt"] = float(enc_t is not None)
            stem_s = a[: -len(enc_s)] if enc_s else a
            stem_t = b[: -len(enc_t)] if enc_t else b
            f["enclitic_stem_match"] = float(bool(enc_s) != bool(enc_t) and stem_s == stem_t and a != b)
            f["same_case"] = float(source[:1].isupper() == target[:1].isupper())
            f["len_ratio"] = min(len(a), len(b)) / max(len(a), len(b), 1)
            f["rel_position"] = min(max(0.5 + ((t / max(n_t, 1)) - (s / max(n_s, 1))) / 2.0, 0.0), 1.0)
            return [f[name] for name in FEATURE_NAMES]

        return phi

    # ---------- interface (II) to the decoder's script and back ----------

    @staticmethod
    def to_edges(pred: Prediction) -> List[Edge]:
        if pred.invalid:
            return []
        out = edges_from(pred.links, pred.tags, pred.frame, pred.extra)
        if pred.link_p:
            for e in out:
                if e.p is None and e.r < len(pred.link_p) and e.s == (pred.links[e.r] if e.r < len(pred.links) else None):
                    e.p = pred.link_p[e.r]
        return out

    @staticmethod
    def to_script(record: Record, pred: Prediction):
        """The decoded ``EditScript``; ``None`` for an invalid prediction.

        Maps COPY to NOP on the way in; ``decode_script`` itself forces NOP
        where the two forms are equal and rewrites any tag outside
        ``LINK_TAGS`` to SUBST.
        """
        from retexo.aligners.decode import ScriptDecoder

        if pred is None or pred.invalid:
            return None
        n = record.n_reuse
        links = [int(s) if s is not None else -1 for s in pred.links[:n]] + [-1] * (n - len(pred.links))
        tags = [labels.TO_LINK_TAG.get(t, t) or "SUBST" for t in list(pred.tags[:n]) + [""] * (n - len(pred.tags))]
        frame = [int(bool(f)) for f in list(pred.frame[:n]) + [0] * (n - len(pred.frame))]
        dels = pred.dels
        return ScriptDecoder.decode_script(record.source_tokens, record.reuse_tokens, links, tags, frame, dels)

    @staticmethod
    def from_script(script) -> Tuple[List[int], List[str], List[int]]:
        """``decode.per_token_view`` back to the record's vocabulary (no NOP survives)."""
        from retexo.aligners.decode import ScriptDecoder

        view = ScriptDecoder.per_token_view(script)
        tags = []
        for t, tag in enumerate(view["tags"]):
            if view["link"][t] < 0:
                tags.append("")
            else:
                op, _ = labels.canonical(tag)
                tags.append(op or "SUBST")
        return list(view["link"]), tags, [int(f) for f in view["frame"]]

    # ---------- interface (III): the dump ----------

    @classmethod
    def _pred_json(cls, record: Record, pred: Prediction) -> Dict[str, Any]:
        edges = cls.to_edges(pred)
        block: Dict[str, Any] = {
            "links": [{"r": e.r, "s": e.s, "op": e.op, **({"p": round(float(e.p), 5)} if e.p is not None else {})}
                      for e in edges],
            "frame_spans": [{"start": sp.start, "end": sp.end, "label": sp.label} for sp in frame_spans_of(pred.frame)],
            "top": [[(int(s), round(float(p), 5)) for s, p in row[:cls.TOP_K_IN_DUMP]] for row in pred.scores] if pred.scores else None,
            "link_p": [round(float(x), 5) for x in pred.link_p] if pred.link_p else None,
            "frame_p": [None if x is None else round(float(x), 5) for x in pred.frame_p] if pred.frame_p else None,
            "dels": [int(x) for x in pred.dels] if pred.dels else None,
            "meta": pred.meta, "raw": pred.raw,
            # per-token tags travel beside the edges so that a types-only row (no links) keeps them
            "tags": list(pred.tags) if any(pred.tags) else None,
        }
        if pred.rev_scores:
            block["rev_top"] = [[(int(s), round(float(p), 5)) for s, p in row[:cls.TOP_K_IN_DUMP]] for row in pred.rev_scores]
        return block

    @classmethod
    def _legacy_json(cls, record: Record, pred: Prediction) -> Dict[str, Any]:
        """The keys ``dump_predictions.py`` writes, so the old readers keep working."""
        from retexo.aligners.decode import ScriptDecoder

        g_links, g_tags, g_frame, _ = links_of(record)
        gold_ops = [("FRAME" if g_frame[t] else "INS") if s < 0 else labels.TO_LINK_TAG.get(g_tags[t], g_tags[t])
                    for t, s in enumerate(g_links)]
        used = {s for s in g_links if s >= 0}
        n = record.n_reuse
        if pred.invalid:
            links, tags, frame = [-1] * n, ["INS"] * n, [0] * n
        else:
            links = [int(s) if s is not None else -1 for s in pred.links[:n]] + [-1] * (n - len(pred.links))
            tags = [("INS" if links[t] < 0 else labels.TO_LINK_TAG.get(x, x) or "SUBST")
                    for t, x in enumerate(list(pred.tags[:n]) + [""] * (n - len(pred.tags)))]
            frame = [int(bool(f)) for f in list(pred.frame[:n]) + [0] * (n - len(pred.frame))]
        script = cls.to_script(record, pred)
        view = ScriptDecoder.per_token_view(script) if script is not None else {"tags": tags, "frame": frame, "reorder": [0] * n,
                                                                    "quote": [0] * n, "link": links}
        regimes = record.annotation.get("regime") or ["none"] * n
        return {
            "id": record.id, "ref_type": record.pair_label + ("." if record.pair_label in ("cit", "cf") else ""),
            "note": str(record.provenance.get("note", "")),
            "source": list(record.source_tokens), "target": list(record.reuse_tokens),
            "gold_ops": gold_ops, "gold_align": list(g_links), "gold_del": [0 if s in used else 1 for s in range(record.n_source)],
            "links": links, "frames": frame, "dels": list(pred.dels) if pred.dels else [0 if s in set(links) else 1 for s in range(record.n_source)],
            "gated_tags": tags, "view": view, "regime": list(regimes),
            "top": [[(int(s), round(float(p), 4)) for s, p in row[:4]] for row in pred.scores] if pred.scores else [[] for _ in range(n)],
            "frame_p": list(pred.frame_p) if pred.frame_p else [None] * n,
        }

    @classmethod
    def write_dump(cls, records: Sequence[Record], preds: Sequence[Prediction], path: Path) -> None:
        """One JSON object per pair: the record, its ``pred`` block, and the legacy keys."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record, pred in zip(records, preds):
                # the legacy keys ``links``, ``source`` and ``target`` collide with the
                # record's own keys, so the section 1.3 object travels under ``record``
                rec = record_to_json(record)
                rec.pop("pred", None)
                obj = cls._legacy_json(record, pred)
                obj["record"] = rec
                obj["pred"] = cls._pred_json(record, pred)
                handle.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def read_dump(path: Path) -> List[Tuple[Record, Prediction]]:
        """The dump back to ``(Record, Prediction)`` pairs, from the ``pred`` block."""
        out = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                record = record_from_json(obj["record"] if "record" in obj else obj)
                block = obj.get("pred") or {}
                n = record.n_reuse
                pred = Prediction.empty(n)
                pred.raw = block.get("raw")
                edges = block.get("links", [])
                if pred.raw is not None and not edges:
                    pred.links = []
                    pred.tags = []
                else:
                    seen = set()
                    for e in edges:
                        r, s = int(e["r"]), int(e["s"])
                        if r in seen or not (0 <= r < n):
                            pred.extra.append(Edge(r, s, str(e.get("op", "")), True, "", e.get("p")))
                            continue
                        seen.add(r)
                        pred.links[r] = s
                        pred.tags[r] = str(e.get("op", ""))
                    for sp in block.get("frame_spans", []):
                        for t in range(int(sp["start"]), min(n, int(sp["end"]))):
                            pred.frame[t] = 1
                if block.get("tags") and not pred.invalid:
                    stored = list(block["tags"])[:n]
                    pred.tags = [stored[t] if t < len(stored) and stored[t] else pred.tags[t] for t in range(n)]
                if block.get("top") is not None:
                    pred.scores = [[(int(s), float(p)) for s, p in row] for row in block["top"]]
                if block.get("rev_top") is not None:
                    pred.rev_scores = [[(int(s), float(p)) for s, p in row] for row in block["rev_top"]]
                pred.link_p = block.get("link_p")
                pred.frame_p = block.get("frame_p")
                pred.dels = block.get("dels")
                pred.meta = dict(block.get("meta") or {})
                out.append((record, pred))
        return out

    @staticmethod
    def examples_from_records(records: Sequence[Record], featurizer=None):
        from retexo.baselines.record import record_to_example

        return [record_to_example(r, featurizer) for r in records]


# =============================================================================
# The legacy rescoring reader
# =============================================================================


class LegacyRescorer:
    """Rescores a prediction dump from its legacy keys, with optional repairs.

    A verbatim port of ``rescore_dump.rescore`` (now ``attic/scripts/``).
    """

    #: The hand labels' four-way vocabulary, as the preliminary runners spelt it.
    FOUR = ["NOP", "MORPH", "SUBST", "INS"]

    @classmethod
    def _to_four(cls, tag: str) -> str:
        """A fine tag onto the hand labels' vocabulary (``run_e25.to_four``)."""
        if tag in ("INS", "FRAME"):
            return "INS"
        if tag == "NOP":
            return "NOP"
        if tag in ("MORPH", "SPLIT", "MERGE"):
            return "MORPH"          # the labeller called enclitic changes MORPH
        return "SUBST"

    @staticmethod
    def _mask_prf(gold_masks, pred_masks):
        from retexo.datasets.gold import TypedScorer

        tp = fp = fn = 0
        for g, p in zip(gold_masks, pred_masks):
            for a, b in zip(g, p):
                if a and b:
                    tp += 1
                elif b:
                    fp += 1
                elif a:
                    fn += 1
        return TypedScorer.prf(tp, fp, fn)

    @staticmethod
    def _span_prf(gold_spans, pred_spans):
        from retexo.datasets.gold import TypedScorer

        tp = fp = fn = 0
        for g, p in zip(gold_spans, pred_spans):
            g, p = set(g), set(p)
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        return TypedScorer.prf(tp, fp, fn)

    @classmethod
    def rescore(cls, rows, rules=(), verbose_name="arm", transform=None, **kw):
        """Reads ``source``, ``target``, ``links``, ``gated_tags``, ``view.frame``,
        ``gold_ops`` and ``gold_align`` from each row, applies the named repairs
        from ``retexo.edit_typing.repair`` in order, prints the twelve-column
        line the preliminary experiments used, and returns the four-way score,
        the frame mask and span scores, and the reorder score. Kept so that
        every dump the driver writes stays readable by the old procedure.
        """
        from retexo.aligners.decode import ScriptDecoder
        from retexo.datasets.gold import TypedScorer
        from retexo.edit_typing.repair import Repairer
        from retexo.aligners.sameness import SamenessPolicy

        gold4, pred4, goldc, predc = [], [], [], []
        gold_fr, pred_fr, gold_re, pred_re = [], [], [], []
        declined = false = 0
        for r in rows:
            src, tgt = r["source"], r["target"]
            L = list(r["links"]); T = list(r["gated_tags"]); F = list(r["view"]["frame"])
            if transform is not None:
                L, T, F = transform(r)
            for rule in rules:
                if rule == "spelling":
                    T = Repairer.spelling(src, tgt, L, T)
                elif rule.startswith("frame"):
                    F = Repairer.frame_adjacency(L, F, max_gap=kw.get("max_gap", 2), one_span=kw.get("one_span", True))
                elif rule == "extend":
                    F = Repairer.frame_extend(tgt, L, F, colon_rule=kw.get("colon", True), extend_left=kw.get("left", True))
                elif rule == "geometry":
                    L, T = Repairer.subst_geometry(L, T, allow_size_mismatch=True)
                elif rule == "geometry-strict":
                    L, T = Repairer.subst_geometry(L, T, allow_size_mismatch=False)
            # frame tokens are INS in the four-way view
            p4 = ["INS" if L[t] < 0 else cls._to_four(T[t]) for t in range(len(tgt))]
            g4 = [cls._to_four(o) for o in r["gold_ops"]]
            gold4.append(g4); pred4.append(p4)
            goldc.append(["INS" if o in ("INS", "FRAME") else ("SUBST" if o == "SUBST" else "COPY") for o in r["gold_ops"]])
            predc.append(["INS" if L[t] < 0 else ("COPY" if SamenessPolicy.same_current(tgt[t], src[L[t]]) else "SUBST") for t in range(len(tgt))])
            gold_fr.append([1 if o == "FRAME" else 0 for o in r["gold_ops"]]); pred_fr.append(F)
            G = r["gold_align"]
            gset = ScriptDecoder.reordered_targets(G); pset = ScriptDecoder.reordered_targets(L)
            gold_re.append([1 if t in gset else 0 for t in range(len(G))]); pred_re.append([1 if t in pset else 0 for t in range(len(G))])
            declined += sum(1 for t in range(len(G)) if G[t] >= 0 and L[t] < 0)
            false += sum(1 for t in range(len(G)) if G[t] < 0 and L[t] >= 0)
        four = TypedScorer.score_typed(gold4, pred4, cls.FOUR)
        fr = cls._mask_prf(gold_fr, pred_fr); fs = cls._span_prf([ScriptDecoder.frame_spans(g) for g in gold_fr], [ScriptDecoder.frame_spans(p) for p in pred_fr])
        re_ = cls._mask_prf(gold_re, pred_re)
        f = four["per_operation"]
        print(f"{verbose_name:<28}{four['macro_f1']:>7.3f}{f['NOP']['f1']:>7.3f}{f['MORPH']['f1']:>7.3f}{f['SUBST']['f1']:>7.3f}{f['INS']['f1']:>7.3f}"
              f"{fr['f1']:>7.3f}{fs['f1']:>7.3f}{re_['f1']:>7.3f}{declined:>6}{false:>6}"
              f"   MORPH p/r {f['MORPH']['precision']:.2f}/{f['MORPH']['recall']:.2f}  SUBST p/r {f['SUBST']['precision']:.2f}/{f['SUBST']['recall']:.2f}")
        return four, fr, fs, re_


#: Backward-compatible module-level aliases.
TOP_K_IN_DUMP = PredictionAdapter.TOP_K_IN_DUMP
decode_prediction = PredictionAdapter.decode_prediction
type_prediction = PredictionAdapter.type_prediction
to_edges = PredictionAdapter.to_edges
to_script = PredictionAdapter.to_script
from_script = PredictionAdapter.from_script
write_dump = PredictionAdapter.write_dump
read_dump = PredictionAdapter.read_dump
examples_from_records = PredictionAdapter.examples_from_records
LEGACY_FOUR = LegacyRescorer.FOUR
rescore_legacy = LegacyRescorer.rescore
