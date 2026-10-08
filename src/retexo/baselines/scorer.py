# retexo/baselines/scorer.py
"""One scorer for every table, reading dumps and never a model.

The numbers the paper reports do not exist as one function yet:
``metrics.evaluate`` scores ``EditScript`` objects as operation sets,
``e3.score_typed`` scores per-token labels, ``run_e25`` has the mask and span
F1, ``run_e26.score_arm`` prints a twelve-column row that needs a warm
featurizer. This module computes, from interface (III) dumps alone: token
accuracy with "none" as a class (the headline of Table 1), link precision,
recall and F(alpha) under sure/possible scoring (Och and Ney 2003; Fraser and
Marcu 2007) with AER stored for comparability, per-class operation F1 at
every level of the inventory by collapsing (V0, V1, mode, group, V3) with the
ERRANT convention that a MORPH with the wrong source is both a false positive
and a false negative, FRAME span F1 with exact boundaries, the structural
flags (REORDER, QUOTE at n = 2, DISPERSE, DEL), the replay rate, invented
links on hard negatives, the breakdowns by reference type, regime and run
length, fold aggregation, the split-half test, and rater agreement.

A prediction with ``raw`` set and no links is an invalid output: it is scored
as the empty prediction on every metric and counted under ``invalid``. A row
whose ``emits`` is ``types`` (the taggers) has its alignment columns blanked
and its operation scores computed without the source.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from retexo.aligners.decode import ScriptDecoder
from retexo.baselines import labels
from retexo.baselines.base import Prediction
from retexo.baselines.record import Record, extra_edges, links_of
from retexo.baselines.typer import coverage  # noqa: F401  (re-exported: defined once in 30 Typer)


class BaselineScorer:
    """One scorer for every table, reading dumps and never a model.

    Example:
        ```python
        result = BaselineScorer.score_dump(Path("runs/copy/predictions.jsonl"))
        BaselineScorer.print_row(result, method="copy_input")
        ```
    """

    QUOTE_MIN = 2  # definition 3.1: n = 2 fixed; decode.QUOTE_MIN (4) is not changed
    LEVELS = labels.LEVELS

    # ---------- small helpers ----------

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}

    @staticmethod
    def _links(pred: Optional[Prediction], n: int) -> List[int]:
        if pred is None or pred.invalid or not pred.links:
            return [-1] * n
        out = [int(x) if x is not None else -1 for x in pred.links[:n]]
        return out + [-1] * (n - len(out))

    @staticmethod
    def _gold_sets(record: Record) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """``(S, P)``: the sure links and the possible links (``P`` contains ``S``)."""
        sure = {(e.r, e.s) for e in record.links if e.sure}
        possible = {(e.r, e.s) for e in record.links}
        return sure, possible

    @classmethod
    def predicted_edges(cls, pred: Optional[Prediction], n: int) -> Set[Tuple[int, int]]:
        links = cls._links(pred, n)
        out = {(t, s) for t, s in enumerate(links) if s >= 0}
        if pred is not None and not pred.invalid:
            out |= {(e.r, e.s) for e in pred.extra}
        return out

    # ---------- alignment metrics ----------

    @staticmethod
    def token_accuracy_from_links(
        pred_links: Sequence[Sequence[int]], gold_links: Sequence[Sequence[int]]
    ) -> float:
        """Plain token accuracy on link lists (sure links only, for tuning)."""
        right = total = 0
        for p, g in zip(pred_links, gold_links):
            for t, s in enumerate(g):
                total += 1
                q = p[t] if t < len(p) else -1
                right += int((q if q is not None else -1) == (s if s is not None else -1))
        return right / total if total else 0.0

    @classmethod
    def link_prf_from_links(
        cls, pred_links: Sequence[Sequence[int]], gold_links: Sequence[Sequence[int]]
    ) -> Dict[str, float]:
        a = {
            (i, t, s)
            for i, p in enumerate(pred_links)
            for t, s in enumerate(p)
            if s is not None and s >= 0
        }
        g = {
            (i, t, s)
            for i, p in enumerate(gold_links)
            for t, s in enumerate(p)
            if s is not None and s >= 0
        }
        tp = len(a & g)
        out = cls._prf(tp, len(a) - tp, len(g) - tp)
        return {"precision": out["P"], "recall": out["R"], "f1": out["F1"]}

    @classmethod
    def token_accuracy(
        cls, records: Sequence[Record], preds: Sequence[Optional[Prediction]]
    ) -> float:
        """One decision per reuse token, "none" a class; possible links count as right."""
        right = total = 0
        for record, pred in zip(records, preds):
            sure, possible = cls._gold_sets(record)
            links = cls._links(pred, record.n_reuse)
            for t, q in enumerate(links):
                total += 1
                has_sure = any(r == t for r, _ in sure)
                if q >= 0:
                    right += int((t, q) in possible)
                else:
                    right += int(not has_sure)
        return right / total if total else 0.0

    @classmethod
    def link_prf(
        cls, records: Sequence[Record], preds: Sequence[Optional[Prediction]], *, alpha: float = 0.5
    ) -> Dict[str, float]:
        """Sure/possible precision, recall, F(alpha), F1, AER and the counts."""
        n_pa = n_sa = n_a = n_s = n_p = 0
        for _i, (record, pred) in enumerate(zip(records, preds)):
            sure, possible = cls._gold_sets(record)
            a = cls.predicted_edges(pred, record.n_reuse)
            n_pa += len(possible & a)
            n_sa += len(sure & a)
            n_a += len(a)
            n_s += len(sure)
            n_p += len(possible)
        precision = n_pa / n_a if n_a else 0.0
        recall = n_sa / n_s if n_s else 0.0
        f_alpha = 1.0 / (alpha / precision + (1 - alpha) / recall) if precision and recall else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aer = 1.0 - (n_pa + n_sa) / (n_s + n_a) if n_s + n_a else 1.0
        return {
            "precision": precision,
            "recall": recall,
            "f": f_alpha,
            "f1": f1,
            "alpha": alpha,
            "aer": aer,
            "n_sure": n_s,
            "n_possible": n_p,
            "n_pred": n_a,
        }

    # ---------- operation metrics ----------

    @staticmethod
    def items_of(
        links: Sequence[int],
        tags: Sequence[str],
        frame: Sequence[int],
        level: str,
        n_source: Optional[int] = None,
        *,
        require_source: bool = True,
    ) -> Set[Tuple]:
        """The item set of one pair at ``level``: ``(t, s, label)`` per reuse token, ``("s", s, label)`` per source token."""
        out: Set[Tuple] = set()
        # a tagger (``require_source=False``) is read from its tags: an unlinked word keeps its tag's label
        per_token = (
            labels.token_labels(links, tags, frame, level)
            if require_source
            else labels.tag_labels(list(tags) + [""] * (len(links) - len(tags)), frame, level)
        )
        for t, label in enumerate(per_token):
            s = links[t] if t < len(links) else -1
            s = s if (s is not None and s >= 0) else None
            out.add((t, s if require_source else None, label))
        if n_source is not None and require_source:
            for s, label in enumerate(labels.source_labels(links, n_source)):
                if label == "DEL":
                    out.add(("s", s, labels.collapse(level, "DEL")))
        return out

    @classmethod
    def _pred_triple(
        cls, pred: Optional[Prediction], record: Record
    ) -> Tuple[List[int], List[str], List[int]]:
        n = record.n_reuse
        links = cls._links(pred, n)
        if pred is None or pred.invalid:
            return links, [""] * n, [0] * n
        tags = list(pred.tags[:n]) + [""] * (n - len(pred.tags))
        frame = [int(bool(f)) for f in pred.frame[:n]] + [0] * (n - len(pred.frame))
        for t, s in enumerate(links):
            if s >= 0 and not tags[t]:
                raise ValueError(
                    f"{record.id}: linked reuse word {t} has an empty tag; the typer did not run"
                )
        return links, tags, frame

    @classmethod
    def op_scores(
        cls,
        records: Sequence[Record],
        preds: Sequence[Optional[Prediction]],
        level: str,
        *,
        require_source: bool = True,
    ) -> Dict[str, Any]:
        """Per-class P/R/F1 and macro F1 at ``level``, ERRANT-style item matching."""
        tp: Dict[str, int] = {}
        fp: Dict[str, int] = {}
        fn: Dict[str, int] = {}
        # the fixed-denominator view: sure gold items per class, and how many of them were found
        sure_total: Dict[str, int] = {}
        sure_found: Dict[str, int] = {}
        confusion: Dict[str, Dict[str, int]] = {}
        right = total = 0
        for record, pred in zip(records, preds):
            g_links, g_tags, g_frame, g_sure = links_of(record)
            sure_items = cls.items_of(
                [s if g_sure[t] or s < 0 else -1 for t, s in enumerate(g_links)],
                g_tags,
                g_frame,
                level,
                record.n_source,
                require_source=require_source,
            )
            all_items = cls.items_of(
                g_links, g_tags, g_frame, level, record.n_source, require_source=require_source
            )
            possible_only = all_items - sure_items
            # a reuse token whose only edge is possible must not count as a sure INS
            possible_tokens = {item[0] for item in possible_only if item[0] != "s"}
            sure_items = {
                item for item in sure_items if not (item[0] in possible_tokens and item[1] is None)
            }
            p_links, p_tags, p_frame = cls._pred_triple(pred, record)
            pred_items = cls.items_of(
                p_links, p_tags, p_frame, level, record.n_source, require_source=require_source
            )
            dels = list(getattr(pred, "dels", None) or []) if pred is not None else []
            if not require_source and dels:
                # a tagger with a source head (``pred.dels``) is scored on DEL too, against the gold's unlinked
                # source words; a tagger without one is not asked for DEL at all
                del_label = labels.collapse(level, "DEL")
                for s, label in enumerate(labels.source_labels(g_links, record.n_source)):
                    if label == "DEL":
                        all_items.add(("s", s, del_label))
                        sure_items.add(("s", s, del_label))
                for s, flag in enumerate(dels[: record.n_source]):
                    if flag:
                        pred_items.add(("s", s, del_label))
            matched = set()
            for item in pred_items:
                op_cls = item[2]
                if item in possible_only:
                    tp[op_cls] = tp.get(op_cls, 0) + 1
                elif item in sure_items:
                    tp[op_cls] = tp.get(op_cls, 0) + 1
                    matched.add(item)
                else:
                    fp[op_cls] = fp.get(op_cls, 0) + 1
            for item in sure_items - matched:
                fn[item[2]] = fn.get(item[2], 0) + 1
            for item in sure_items:
                sure_total[item[2]] = sure_total.get(item[2], 0) + 1
            for item in matched:
                sure_found[item[2]] = sure_found.get(item[2], 0) + 1
            gold_by_token = {item[0]: item[2] for item in all_items if item[0] != "s"}
            pred_by_token = {item[0]: item[2] for item in pred_items if item[0] != "s"}
            for t in range(record.n_reuse):
                g = gold_by_token.get(t)
                p = pred_by_token.get(t)
                if g is None:
                    continue
                total += 1
                right += int(
                    g == p and (not require_source or (g_links[t] == p_links[t]) or g_links[t] < 0)
                )
                confusion.setdefault(g, {})[p] = confusion.setdefault(g, {}).get(p, 0) + 1
        classes = [c for c in labels.classes(level) if (tp.get(c, 0) + fn.get(c, 0)) > 0]
        per_class = {
            c: cls._prf(tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)) for c in labels.classes(level)
        }
        for c in labels.classes(level):
            # the counts a reader can compare across rows: sure gold items (a fixed denominator), how many were found,
            # and how many items the row predicted (found + wrong, possible links found included in the predicted count)
            per_class[c]["sure"] = sure_total.get(c, 0)
            per_class[c]["sure_found"] = sure_found.get(c, 0)
            per_class[c]["predicted"] = tp.get(c, 0) + fp.get(c, 0)
        macro = sum(per_class[c]["F1"] for c in classes) / len(classes) if classes else 0.0
        out: Dict[str, Any] = {
            "macro_f1": macro,
            "per_class": per_class,
            "accuracy": right / total if total else 0.0,
            "confusion": confusion,
            "classes_present": classes,
        }
        if level == "V1":
            ins_del = cls._prf(
                tp.get("INS", 0) + tp.get("DEL", 0),
                fp.get("INS", 0) + fp.get("DEL", 0),
                fn.get("INS", 0) + fn.get("DEL", 0),
            )
            out["ins_del_f1"] = ins_del["F1"]
        return out

    # ---------- structure, replay, negatives ----------

    @classmethod
    def _mask_prf(cls, gold_masks, pred_masks) -> Dict[str, float]:
        tp = fp = fn = 0
        for g, p in zip(gold_masks, pred_masks):
            for a, b in zip(g, p):
                if a and b:
                    tp += 1
                elif b:
                    fp += 1
                elif a:
                    fn += 1
        return cls._prf(tp, fp, fn)

    @classmethod
    def _span_prf(cls, gold_spans, pred_spans) -> Dict[str, float]:
        tp = fp = fn = 0
        for g, p in zip(gold_spans, pred_spans):
            g, p = set(g), set(p)
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        return cls._prf(tp, fp, fn)

    @classmethod
    def frame_span_prf(
        cls, records: Sequence[Record], preds: Sequence[Optional[Prediction]]
    ) -> Dict[str, float]:
        gold, pred = [], []
        for record, p in zip(records, preds):
            _, _, g_frame, _ = links_of(record)
            _, _, p_frame = cls._pred_triple(p, record)
            gold.append(ScriptDecoder.frame_spans(g_frame))
            pred.append(ScriptDecoder.frame_spans(p_frame))
        return cls._span_prf(gold, pred)

    @classmethod
    def structure_scores(
        cls,
        records: Sequence[Record],
        preds: Sequence[Optional[Prediction]],
        *,
        quote_min: int = QUOTE_MIN,
    ) -> Dict[str, Dict[str, float]]:
        """REORDER token F1, QUOTE span F1 (n = 2), DISPERSE pair F1, DEL mask F1."""
        g_re, p_re, g_q, p_q, g_d, p_d, g_del, p_del = [], [], [], [], [], [], [], []
        for record, pred in zip(records, preds):
            g_links, g_tags, _, _ = links_of(record)
            p_links, p_tags, _ = cls._pred_triple(pred, record)
            n = record.n_reuse
            g_set, p_set = (
                ScriptDecoder.reordered_targets(g_links),
                ScriptDecoder.reordered_targets(p_links),
            )
            g_re.append([int(t in g_set) for t in range(n)])
            p_re.append([int(t in p_set) for t in range(n)])
            g_link_tags = [labels.TO_LINK_TAG.get(x, x) for x in g_tags]
            p_link_tags = [labels.TO_LINK_TAG.get(x, x) for x in p_tags]
            g_q.append(ScriptDecoder.quote_spans(g_links, g_link_tags, minimum=quote_min))
            p_q.append(ScriptDecoder.quote_spans(p_links, p_link_tags, minimum=quote_min))
            g_d.append([int(ScriptDecoder.dispersed(record.reuse_tokens, g_links))])
            p_d.append([int(ScriptDecoder.dispersed(record.reuse_tokens, p_links))])
            g_del.append(
                [1 if x == "DEL" else 0 for x in labels.source_labels(g_links, record.n_source)]
            )
            p_del.append(
                [1 if x == "DEL" else 0 for x in labels.source_labels(p_links, record.n_source)]
            )
        return {
            "reorder_token": cls._mask_prf(g_re, p_re),
            "quote_span": cls._span_prf(g_q, p_q),
            "disperse_pair": cls._mask_prf(g_d, p_d),
            "del": cls._mask_prf(g_del, p_del),
        }

    @staticmethod
    def replay_rate(records: Sequence[Record], preds: Sequence[Optional[Prediction]]) -> float:
        """Share of pairs whose decoded script reproduces the reuse under ``Scriba.verify``."""
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.core.scriba import Scriba

        scriba = Scriba()
        hits = 0
        for record, pred in zip(records, preds):
            script = PredictionAdapter.to_script(record, pred) if pred is not None else None
            hits += int(
                script is not None
                and scriba.verify(script, record.source_tokens, record.reuse_tokens)
            )
        return hits / len(records) if records else 0.0

    @classmethod
    def invented_links(
        cls, records: Sequence[Record], preds: Sequence[Optional[Prediction]]
    ) -> Dict[str, float]:
        """Links predicted on hard negatives (``pair_label == "no_match"``)."""
        pairs = links = with_links = 0
        for record, pred in zip(records, preds):
            if record.pair_label != "no_match":
                continue
            pairs += 1
            n = len(cls.predicted_edges(pred, record.n_reuse))
            links += n
            with_links += int(n > 0)
        return {
            "pairs": pairs,
            "links": links,
            "links_per_pair": links / pairs if pairs else 0.0,
            "pairs_with_links": with_links / pairs if pairs else 0.0,
        }

    # ---------- breakdowns ----------

    @staticmethod
    def _subset(records, preds, keep) -> Tuple[List[Record], List[Optional[Prediction]]]:
        rs, ps = [], []
        for r, p in zip(records, preds):
            if keep(r):
                rs.append(r)
                ps.append(p)
        return rs, ps

    @classmethod
    def by_ref_type(cls, records, preds, fn=None) -> Dict[str, Any]:
        fn = fn or cls.token_accuracy
        out = {}
        for label in sorted({r.pair_label for r in records}):
            rs, ps = cls._subset(records, preds, lambda r, label=label: r.pair_label == label)
            out[label + ("." if label in ("cit", "cf") else "")] = fn(rs, ps)
        return out

    @classmethod
    def by_regime(cls, records, preds, fn=None) -> Dict[str, Any]:
        """Per attestation regime of the gold link (``annotation["regime"]``, per reuse
        word; ``RuleTyper.annotate_regimes``): the share of those reuse words whose
        predicted link is the gold link, and the share whose predicted operation
        is the gold operation, with the word count. Unlinked words (``none``) are
        left out; a record without the annotation contributes nothing."""
        from retexo.baselines.record import links_of

        hits: Dict[str, List[int]] = {}
        for record, pred in zip(records, preds):
            regimes = record.annotation.get("regime")
            if not regimes or pred is None:
                continue
            gold_links, gold_tags, _, _ = links_of(record)
            for t, regime in enumerate(regimes):
                if regime == "none" or t >= len(gold_links) or gold_links[t] < 0:
                    continue
                pred_link = pred.links[t] if t < len(pred.links) else -1
                pred_tag = labels.canonical(pred.tags[t] if t < len(pred.tags) else "")[0] or ""
                gold_tags = (
                    [labels.canonical(g)[0] or "" for g in gold_tags] if t == 0 else gold_tags
                )
                bucket = hits.setdefault(regime, [0, 0, 0])
                bucket[0] += 1
                if any(s >= 0 for s in pred.links):
                    bucket[1] += int(pred_link == gold_links[t])
                    bucket[2] += int(pred_link == gold_links[t] and pred_tag == gold_tags[t])
                else:  # a tagger: no links, the tag alone is judged
                    bucket[2] += int(pred_tag == gold_tags[t])
        return {
            k: {"n": v[0], "link_acc": v[1] / v[0], "op_acc": v[2] / v[0]}
            for k, v in sorted(hits.items())
            if v[0]
        }

    @classmethod
    def by_run_length(
        cls, records: Sequence[Record], preds: Sequence[Optional[Prediction]]
    ) -> Dict[str, Dict[str, float]]:
        """Link P/R/F1 restricted to gold links inside runs of one word against runs of two or more."""

        def runs(links: Sequence[int]) -> Dict[int, int]:
            length: Dict[int, int] = {}
            start = None
            for t in range(len(links) + 1):
                ok = (
                    t < len(links)
                    and links[t] >= 0
                    and (start is None or links[t] == links[t - 1] + 1)
                )
                if ok and start is None:
                    start = t
                elif not ok:
                    if start is not None:
                        for u in range(start, t):
                            length[u] = t - start
                    start = t if (t < len(links) and links[t] >= 0) else None
            return length

        out = {}
        for name, keep in (("single", lambda n: n == 1), ("multi", lambda n: n >= 2)):
            tp = fp = fn = 0
            for record, pred in zip(records, preds):
                g_links, _, _, _ = links_of(record)
                lengths = runs(g_links)
                gold = {(t, s) for t, s in enumerate(g_links) if s >= 0 and keep(lengths.get(t, 1))}
                tokens = {t for t, _ in gold}
                pred_set = {
                    (t, s) for t, s in cls.predicted_edges(pred, record.n_reuse) if t in tokens
                }
                tp += len(gold & pred_set)
                fp += len(pred_set - gold)
                fn += len(gold - pred_set)
            out[name] = cls._prf(tp, fp, fn)
        return out

    # ---------- the whole score, aggregation, split-half, agreement ----------

    @classmethod
    def score(
        cls,
        records: Sequence[Record],
        preds: Sequence[Optional[Prediction]],
        *,
        negatives=None,
        levels: Sequence[str] = None,
        alpha: float = 0.5,
        emits: str = "scores",
        dials: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Every number of the JSON schema for one run."""
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.metrics import ScriptScorer

        levels = levels if levels is not None else cls.LEVELS
        types_only = emits == "types"
        n_invalid = sum(1 for p in preds if p is None or p.invalid)
        out: Dict[str, Any] = {
            "n": len(records),
            "invalid": {"n": n_invalid, "rate": n_invalid / len(records) if records else 0.0},
        }
        positives, pos_preds = cls._subset(records, preds, lambda r: r.pair_label != "no_match")
        if types_only:
            out["token_accuracy"] = None
            out["link"] = None
        else:
            out["token_accuracy"] = cls.token_accuracy(positives, pos_preds)
            out["link"] = cls.link_prf(positives, pos_preds, alpha=alpha)
        out["ops"] = {
            level: cls.op_scores(positives, pos_preds, level, require_source=not types_only)
            for level in levels
        }
        out["frame_span"] = cls.frame_span_prf(positives, pos_preds)
        out["structure"] = None if types_only else cls.structure_scores(positives, pos_preds)
        out["replay_rate"] = cls.replay_rate(positives, pos_preds)
        if negatives is not None:
            neg_records, neg_preds = negatives
            out["negatives"] = cls.invented_links(neg_records, neg_preds)
        else:
            out["negatives"] = cls.invented_links(records, preds)
        out["by_ref_type"] = cls.by_ref_type(
            positives,
            pos_preds,
            cls.token_accuracy
            if not types_only
            else (lambda r, p: cls.op_scores(r, p, "V1", require_source=False)["macro_f1"]),
        )
        out["by_regime"] = cls.by_regime(positives, pos_preds)
        out["by_run_length"] = None if types_only else cls.by_run_length(positives, pos_preds)
        out["coverage"] = coverage(positives)
        gold_scripts, pred_scripts = [], []
        for record, pred in zip(positives, pos_preds):
            g_links, g_tags, g_frame, _ = links_of(record)
            gold_scripts.append(
                PredictionAdapter.to_script(
                    record,
                    Prediction(
                        links=g_links,
                        tags=[t or "" for t in g_tags],
                        frame=g_frame,
                        extra=extra_edges(record),
                    ),
                )
            )
            pred_scripts.append(
                PredictionAdapter.to_script(record, pred) if pred is not None else None
            )
        legacy = ScriptScorer.evaluate(pred_scripts, gold_scripts, generative=False)
        out["legacy"] = {
            "alignment_f1": legacy.alignment.f1,
            "typed_f1": legacy.typed.f1,
            "link_strict": legacy.link_strict,
            "link_lenient": legacy.link_lenient,
            "macro_typed_f1": legacy.macro_typed_f1,
            "invalid": legacy.invalid,
        }
        out["dials"] = dict(dials or {})
        return out

    @classmethod
    def score_dump(
        cls, path: Path, *, negatives: Optional[Path] = None, emits: str = "scores", **kw
    ) -> Dict[str, Any]:
        """Score a ``predictions.jsonl`` (interface III); ``negatives`` is a second dump on hard negatives."""
        from retexo.baselines.adapters import PredictionAdapter

        rows = PredictionAdapter.read_dump(Path(path))
        records = [r for r, _ in rows]
        preds = [p for _, p in rows]
        neg = None
        if negatives is not None:
            neg_rows = PredictionAdapter.read_dump(Path(negatives))
            neg = ([r for r, _ in neg_rows], [p for _, p in neg_rows])
        return cls.score(records, preds, negatives=neg, emits=emits, **kw)

    @staticmethod
    def print_row(result: Dict[str, Any], method: str = "?", where: str = "") -> str:
        """The summary line the driver prints last and the mail quotes."""
        acc = result.get("token_accuracy")
        link = result.get("link") or {}
        v1 = result.get("ops", {}).get("V1", {})
        parts = [
            f"[baseline {method}]{(' ' + where) if where else ''}:",
            f"token_acc {acc:.3f}" if acc is not None else "token_acc -",
            f"link_f1 {link.get('f1', 0.0):.3f}" if link else "link_f1 -",
            f"op_macro {v1.get('macro_f1', 0.0):.3f}",
        ]
        if link:
            parts.append(f"aer {link.get('aer', 0.0):.3f}")
        line = " ".join(parts)
        print(line)
        return line

    @staticmethod
    def flat(result: Dict[str, Any], prefix: str = "test_") -> Dict[str, float]:
        """Scalar leaves with dotted keys and the repository's metric prefix."""
        out: Dict[str, float] = {}

        def walk(node, key):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{key}.{k}" if key else str(k))
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                out[prefix + key] = float(node)

        walk({k: v for k, v in result.items() if k not in ("dials",)}, "")
        return out

    @classmethod
    def aggregate_folds(cls, per_fold: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """``mean``, ``min``, ``max``, ``std`` for every scalar leaf over the folds."""
        flats = [cls.flat(r, prefix="") for r in per_fold]
        keys = sorted(set().union(*[set(f) for f in flats]))
        out: Dict[str, Any] = {"n_folds": len(per_fold), "metrics": {}}
        for key in keys:
            values = [f[key] for f in flats if key in f]
            mean = sum(values) / len(values)
            std = (
                math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                if len(values) > 1
                else 0.0
            )
            out["metrics"][key] = {
                "mean": mean,
                "min": min(values),
                "max": max(values),
                "std": std,
                "n": len(values),
            }
        return out

    @classmethod
    def _per_pair_metric(cls, record: Record, pred: Optional[Prediction], metric: str) -> float:
        if metric == "token_accuracy":
            return cls.token_accuracy([record], [pred])
        if metric == "link_f1":
            return cls.link_prf([record], [pred])["f1"]
        if metric.startswith("op_macro"):
            level = metric.split(":")[1] if ":" in metric else "V1"
            return cls.op_scores([record], [pred], level)["macro_f1"]
        raise ValueError(f"unknown split-half metric {metric!r}")

    @classmethod
    def split_half(
        cls,
        dump_a: Path,
        dump_b: Path,
        metric: str = "token_accuracy",
        *,
        n_splits: int = 10,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Whether A beats B on both halves of every random split (20 of 20 confirms)."""
        from retexo.baselines.adapters import PredictionAdapter

        a = {r.id: (r, p) for r, p in PredictionAdapter.read_dump(Path(dump_a))}
        b = {r.id: (r, p) for r, p in PredictionAdapter.read_dump(Path(dump_b))}
        ids = sorted(set(a) & set(b))
        per = {
            i: (
                cls._per_pair_metric(a[i][0], a[i][1], metric),
                cls._per_pair_metric(b[i][0], b[i][1], metric),
            )
            for i in ids
        }
        rng = random.Random(seed)
        agree = 0
        total = 2 * n_splits
        sign = None
        for _ in range(n_splits):
            order = list(ids)
            rng.shuffle(order)
            halves = (order[: len(order) // 2], order[len(order) // 2 :])
            for half in halves:
                if not half:
                    continue
                diff = sum(per[i][0] - per[i][1] for i in half) / len(half)
                s = 1 if diff > 0 else (-1 if diff < 0 else 0)
                sign = s if sign is None else sign
                agree += int(s != 0 and s == sign)
        mean_diff = sum(per[i][0] - per[i][1] for i in ids) / len(ids) if ids else 0.0
        return {
            "agree": agree,
            "of": total,
            "confirmed": agree == total and total > 0,
            "n_pairs": len(ids),
            "mean_diff": mean_diff,
            "metric": metric,
        }

    @staticmethod
    def agreement(
        links_a: Sequence[Tuple[int, int]],
        links_b: Sequence[Tuple[int, int]],
        cells: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        """IAA ``2 |A1 and A2| / (|A1| + |A2|)`` and Cohen's kappa over the cells."""
        a, b = set(links_a), set(links_b)
        iaa = 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0
        cells = list(cells) if cells is not None else sorted(a | b)
        if not cells:
            return {"iaa": iaa, "kappa": 1.0}
        va = [int(c in a) for c in cells]
        vb = [int(c in b) for c in cells]
        n = len(cells)
        po = sum(int(x == y) for x, y in zip(va, vb)) / n
        pa1, pb1 = sum(va) / n, sum(vb) / n
        pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        return {"iaa": iaa, "kappa": kappa}


#: Backward-compatible module-level aliases; ``sc.score_dump(...)`` etc. still work.
QUOTE_MIN = BaselineScorer.QUOTE_MIN
LEVELS = BaselineScorer.LEVELS
predicted_edges = BaselineScorer.predicted_edges
token_accuracy_from_links = BaselineScorer.token_accuracy_from_links
link_prf_from_links = BaselineScorer.link_prf_from_links
token_accuracy = BaselineScorer.token_accuracy
link_prf = BaselineScorer.link_prf
items_of = BaselineScorer.items_of
op_scores = BaselineScorer.op_scores
frame_span_prf = BaselineScorer.frame_span_prf
structure_scores = BaselineScorer.structure_scores
replay_rate = BaselineScorer.replay_rate
invented_links = BaselineScorer.invented_links
by_ref_type = BaselineScorer.by_ref_type
by_regime = BaselineScorer.by_regime
by_run_length = BaselineScorer.by_run_length
score = BaselineScorer.score
score_dump = BaselineScorer.score_dump
print_row = BaselineScorer.print_row
flat = BaselineScorer.flat
aggregate_folds = BaselineScorer.aggregate_folds
split_half = BaselineScorer.split_half
agreement = BaselineScorer.agreement


# =============================================================================
# Command line
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Score a predictions dump, aggregate folds, or run the split-half test."
    )
    ap.add_argument("dump", nargs="?", help="runs/<exp>/predictions.jsonl")
    ap.add_argument("--negatives", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--emits", default="scores")
    ap.add_argument("--folds", nargs="*", default=None, help="dumps of five folds to aggregate")
    ap.add_argument("--split-half", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--metric", default="token_accuracy")
    ap.add_argument(
        "--out", default=None, help="JSON path; default runs/<exp>.json beside the dump folder"
    )
    args = ap.parse_args(argv)
    if args.split_half:
        result = BaselineScorer.split_half(
            Path(args.split_half[0]), Path(args.split_half[1]), args.metric
        )
        print(
            f"[scorer] split-half {args.metric}: agree {result['agree']} of {result['of']} "
            f"(mean diff {result['mean_diff']:+.4f}, {result['n_pairs']} pairs)"
        )
        return 0
    if args.folds:
        per = [
            BaselineScorer.score_dump(Path(p), alpha=args.alpha, emits=args.emits)
            for p in args.folds
        ]
        agg = BaselineScorer.aggregate_folds(per)
        for key in ("token_accuracy", "link.f1", "ops.V1.macro_f1"):
            m = agg["metrics"].get(key)
            if m:
                print(
                    f"[scorer] {key}: mean {m['mean']:.3f} min {m['min']:.3f} max {m['max']:.3f} std {m['std']:.3f}"
                )
        if args.out:
            Path(args.out).write_text(json.dumps(agg, indent=1))
        return 0
    if not args.dump:
        ap.error("a dump path is required")
    result = BaselineScorer.score_dump(
        Path(args.dump),
        negatives=Path(args.negatives) if args.negatives else None,
        alpha=args.alpha,
        emits=args.emits,
    )
    out = Path(args.out) if args.out else Path(args.dump).parent.with_suffix(".json")
    out.write_text(json.dumps(result, indent=1, default=str))
    BaselineScorer.print_row(result, method=Path(args.dump).parent.name)
    print(f"[scorer] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
