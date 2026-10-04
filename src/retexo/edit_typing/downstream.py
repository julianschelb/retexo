# retexo/edit_typing/downstream.py
"""
The edit script as a classifier of no match / cf. / cit.

Renamed from the preliminary experiment module E28; kept, not archived,
because ``ScriptFeaturizer.tier_grid`` is the shape ``typed_pointer.py``'s
``refine_mode == "chain"`` training state reads (planned for note 22's
refinement row) and ``ScriptFeaturizer.features``/``DownstreamScorer``'s
methods are the protocol note 35 plans to wrap in the not-yet-built
``retexo/baselines/downstream.py`` -- a harness-facing module of that name
will sit alongside this one the way ``retexo.baselines.synthetic`` will
sit alongside ``retexo.datasets.synthetic``.

What is here is everything the benchmark paper's classification protocol needs,
so the numbers land in the same table as the cross-encoders and the lexical
baselines:

    EvaluationPool.load(fold_dir, labels_csv)    the paper's per-fold all-pairs pool
    ScriptFeaturizer.tier_grid(pair_features)    attestation tiers for a whole pair, in numpy
    ScriptFeaturizer().features(...)             one row per pair, read off the script
    DownstreamScorer.macro_micro(qids, gold, pred)   the paper's aggregator, copied
                                                      verbatim in substance from the
                                                      lexical-baseline script
    DownstreamScorer.find_threshold(labels, probs)   the paper's plateau_high rule

The aggregator and the threshold rule are *copied*, not imported, so this
module has no dependency on the classification notebooks and the numbers are
still computed the same way.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from retexo.edit_typing.link_features import FEATURE_NAMES

_F = {n: i for i, n in enumerate(FEATURE_NAMES)}
LABELS = {"no_match": 0, "cit.": 1, "cf.": 2}

# =============================================================================
# The paper's pools
# =============================================================================


class EvaluationPool:
    """The evaluation pool of one fold: queries, sources, and the gold labels.

    Attributes:
        queries: Rows of ``query_document.csv``.
        sources: Rows of ``source_document.csv``.
        gold: ``(query_id, source_id) -> ref_type`` ("cit." or "cf.").
        unmatched: References in ``ground_truth.csv`` whose type could not be
            joined from ``labels.csv`` (counted as "cit.", reported here).

    Example:
        ```python
        pool = EvaluationPool.load(fold_dir, labels_csv)
        ```
    """

    def __init__(self, queries: List[Dict[str, str]], sources: List[Dict[str, str]],
                gold: Dict[Tuple[str, str], str], unmatched: int):
        self.queries = queries
        self.sources = sources
        self.gold = gold
        self.unmatched = unmatched

    @staticmethod
    def read_csv_rows(path: Path) -> List[Dict[str, str]]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @classmethod
    def load(cls, fold_dir: Path, labels_csv: Path) -> "EvaluationPool":
        """``ground_truth.csv`` says which pairs are references; the reference *type*
        is joined from ``labels.csv`` on the cleaned text pair (the way the BERT
        3-class script does it; cit. wins when a pair carries both)."""
        queries = cls.read_csv_rows(fold_dir / "query_document.csv")
        sources = cls.read_csv_rows(fold_dir / "source_document.csv")
        gt = cls.read_csv_rows(fold_dir / "ground_truth.csv")
        by_text: Dict[Tuple[str, str], str] = {}
        priority = {"cit.": 0, "cf.": 1}
        for row in cls.read_csv_rows(labels_csv):
            key = (row["text_query_cleaned"].strip(), row["text_corpus_cleaned"].strip())
            rt = row["ref_type"]
            if key not in by_text or priority.get(rt, 9) < priority.get(by_text[key], 9):
                by_text[key] = rt
        q_text = {r["seg_id"]: r["text"] for r in queries}
        s_text = {r["seg_id"]: r["text"] for r in sources}
        gold: Dict[Tuple[str, str], str] = {}
        unmatched = 0
        for r in gt:
            key = (r["query_id"], r["source_id"])
            rt = by_text.get((q_text.get(key[0], "").strip(), s_text.get(key[1], "").strip()))
            if rt is None:
                unmatched += 1
                rt = "cit."          # a reference of unknown type; counted, reported
            gold[key] = rt
        return cls(queries, sources, gold, unmatched)


# =============================================================================
# Attestation over a whole pair, and features read off one script
# =============================================================================

NONE, RELATION, ENCLITIC, LEMMA, FORM = 0, 1, 2, 3, 4

FINE_LEXICAL = ("SYN", "HYPER", "HYPO", "ANT", "SYN-DIST", "NE-SUB", "POS")

SCRIPT_FEATURES = [
    "n_q", "n_s", "n_links", "link_rate", "src_cov", "longest_run", "n_runs",
    "nop_share", "morph_share", "subst_share", "lex_rel_share", "split_merge_share",
    "ins_share", "del_share",
    "quote_longest", "quote_n", "quote_share", "reorder_n", "reorder_share",
    "frame_n", "adapt_n", "disperse_n",
    "mean_link_p", "min_link_p", "mean_null_p_unlinked",
    "form1_link_share", "lemma_link_share", "none_link_share",
    "shared_forms", "shared_lemmas", "jaccard_lemma", "len_ratio",
]


class ScriptFeaturizer:
    """Attestation tiers over a whole pair, and the per-pair feature row read
    off a decoded script -- what the downstream classifier trains on.

    Example:
        ```python
        tiers = ScriptFeaturizer.tier_grid(pair_features)
        row = ScriptFeaturizer.features(view, script, n_source, link_p, null_p, tiers)
        ```
    """

    FEATURES = SCRIPT_FEATURES
    FINE_LEXICAL = FINE_LEXICAL

    @staticmethod
    def tier_grid(pf) -> np.ndarray:
        """[n_t, n_s] tiers from the evidence grid (same rule as ``Attester.pair_tier``)."""
        pf = np.asarray(pf, dtype=np.float32)
        form = pf[..., _F["same_form"]] > 0
        lemma = pf[..., _F["same_lemma"]] > 0
        enclitic = (pf[..., _F["enclitic_stem_match"]] > 0) & \
                   (pf[..., _F["enclitic_src"]] != pf[..., _F["enclitic_tgt"]])
        relation = (pf[..., _F["wn_any"]] > 0) | (pf[..., _F["both_names"]] > 0)
        out = np.zeros(pf.shape[:2], dtype=np.int8)
        out[relation] = RELATION
        out[enclitic] = ENCLITIC
        out[lemma] = LEMMA
        out[form] = FORM
        return out

    @staticmethod
    def features(view, script, n_s: int, link_p: Sequence[float],
                null_p: Sequence[float], tiers: np.ndarray) -> List[float]:
        """One row per pair. ``view`` is decode.per_token_view(script); ``tiers``
        the [n_t, n_s] tier grid; ``link_p`` the model's probability of the chosen
        link (0 where unlinked); ``null_p`` its null probability per word."""
        tags, link = view["tags"], view["link"]
        n_t = max(len(tags), 1)
        linked = [t for t, s in enumerate(link) if s >= 0]
        n_links = len(linked)
        # runs of consecutive linked reuse words
        longest = run = n_runs = 0
        prev = False
        for s in link:
            on = s >= 0
            if on:
                run += 1
                if not prev:
                    n_runs += 1
            else:
                run = 0
            longest = max(longest, run)
            prev = on
        counts = Counter(tags)
        lex_rel = sum(counts.get(k, 0) for k in FINE_LEXICAL)
        split_merge = counts.get("SPLIT", 0) + counts.get("MERGE", 0)
        subst = counts.get("SUBST", 0)
        quote_spans, adapt_n, disperse_n, del_n = [], 0, 0, 0
        for op in script.operations:
            if op.tag == "QUOTE":
                quote_spans.append(len(op.target_indices))
            elif op.tag == "ADAPT":
                adapt_n += 1
            elif op.tag == "DISPERSE":
                disperse_n += 1
            elif op.tag == "DEL":
                del_n += 1
        reorder_n = sum(view["reorder"])
        # regimes of the linked words
        best = tiers.max(axis=1) if tiers.size else np.zeros(n_t, dtype=np.int8)
        n_best = (tiers == best[:, None]).sum(axis=1) if tiers.size else np.zeros(n_t)
        form1 = sum(1 for t in linked if best[t] == FORM and n_best[t] == 1)
        lemma_any = sum(1 for t in linked if best[t] == LEMMA)
        none_any = sum(1 for t in linked if best[t] == NONE)
        shared_forms = int((best == FORM).sum())
        shared_lemmas = int((best >= LEMMA).sum())
        denom = max(n_links, 1)
        lp = [link_p[t] for t in linked]
        npu = [null_p[t] for t in range(len(link)) if link[t] < 0]
        return [
            n_t, n_s, n_links, n_links / n_t, n_links / max(n_s, 1), longest, n_runs,
            counts.get("NOP", 0) / denom, counts.get("MORPH", 0) / denom, subst / denom,
            lex_rel / denom, split_merge / denom,
            counts.get("INS", 0) / n_t, del_n / max(n_s, 1),
            max(quote_spans, default=0), len(quote_spans), sum(quote_spans) / n_t,
            reorder_n, reorder_n / denom,
            sum(view["frame"]), adapt_n, disperse_n,
            float(np.mean(lp)) if lp else 0.0, float(min(lp)) if lp else 0.0,
            float(np.mean(npu)) if npu else 1.0,
            form1 / denom, lemma_any / denom, none_any / denom,
            shared_forms, shared_lemmas,
            shared_lemmas / max(n_t + n_s - shared_lemmas, 1), n_t / max(n_s, 1),
        ]


# =============================================================================
# The paper's aggregator and threshold rule
# =============================================================================


class DownstreamScorer:
    """The paper's classification-protocol scores: per-query macro/micro P/R/F1,
    the plateau-high threshold rule, and recall@k.

    Example:
        ```python
        macro, micro, rows = DownstreamScorer.macro_micro(qids, gold, pred)
        threshold = DownstreamScorer.find_threshold(labels, probs)
        ```
    """

    @staticmethod
    def macro_micro(qids: Sequence[str], gold: np.ndarray, pred: np.ndarray):
        """Per-query P/R/F1 macro-averaged over queries with a gold positive;
        FPR / FNR / SMR over all queries; micro pools the match-only counts."""
        rows = []
        for qi, qid in enumerate(qids):
            g, p = gold[qi], pred[qi]
            tp = int(((g == 1) & (p == 1)).sum()); fp = int(((g == 0) & (p == 1)).sum())
            fn = int(((g == 1) & (p == 0)).sum()); tn = int(((g == 0) & (p == 0)).sum())
            n = tp + fp + fn + tn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append(dict(query_id=qid, precision=prec, recall=rec, f1=f1,
                             accuracy=(tp + tn) / n if n else 0.0, tp=tp, fp=fp, fn=fn, tn=tn,
                             fpr=fp / n if n else 0.0, fnr=fn / n if n else 0.0,
                             smr=(fp + fn) / n if n else 0.0))
        match = [r for r in rows if r["tp"] + r["fn"] > 0]
        mean = lambda key, rs: float(np.mean([r[key] for r in rs])) if rs else 0.0
        macro = {"precision": mean("precision", match), "recall": mean("recall", match),
                 "f1": mean("f1", match), "accuracy": mean("accuracy", match),
                 "fpr": mean("fpr", rows), "fnr": mean("fnr", rows), "smr": mean("smr", rows),
                 "tp": sum(r["tp"] for r in rows), "fp": sum(r["fp"] for r in rows),
                 "fn": sum(r["fn"] for r in rows), "tn": sum(r["tn"] for r in rows)}
        tp_m = sum(r["tp"] for r in match); fp_m = sum(r["fp"] for r in match)
        fn_m = sum(r["fn"] for r in match); tn_m = sum(r["tn"] for r in match)
        p = tp_m / (tp_m + fp_m) if tp_m + fp_m else 0.0
        r = tp_m / (tp_m + fn_m) if tp_m + fn_m else 0.0
        tot_a = macro["tp"] + macro["fp"] + macro["fn"] + macro["tn"]
        micro = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                 "accuracy": (tp_m + tn_m) / (tp_m + fp_m + fn_m + tn_m) if match else 0.0,
                 "fpr": macro["fp"] / tot_a if tot_a else 0.0,
                 "fnr": macro["fn"] / tot_a if tot_a else 0.0,
                 "smr": (macro["fp"] + macro["fn"]) / tot_a if tot_a else 0.0}
        return macro, micro, rows

    @staticmethod
    def find_threshold(labels: np.ndarray, probs: np.ndarray, *, method: str = "plateau_high",
                       tolerance: float = 0.01) -> float:
        """The paper's rule: over thresholds 0.01..0.99, the largest one whose
        binary F1 is within ``tolerance`` of the best (favours precision)."""
        labels = np.asarray(labels).astype(int); probs = np.asarray(probs, dtype=float)
        grid = np.arange(0.01, 1.00, 0.01)
        f1s = []
        for t in grid:
            pred = probs >= t
            tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
            fn = int((~pred & (labels == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        f1s = np.asarray(f1s)
        if method == "max_f1":
            return float(grid[int(f1s.argmax())])
        plateau = grid[f1s >= f1s.max() - tolerance]
        return float(plateau.max() if method == "plateau_high" else plateau.min())

    @staticmethod
    def recall_at_k(qids: Sequence[str], gold: np.ndarray, score: np.ndarray, ks=(10, 100, 1000)):
        """Per query with a gold positive: share of its positives in the top-k
        sources by score; averaged over queries."""
        out = {}
        for k in ks:
            vals = []
            for qi in range(len(qids)):
                pos = np.flatnonzero(gold[qi] == 1)
                if not len(pos):
                    continue
                top = np.argsort(-score[qi])[:k]
                vals.append(len(set(pos) & set(top.tolist())) / len(pos))
            out[k] = float(np.mean(vals)) if vals else 0.0
        return out
