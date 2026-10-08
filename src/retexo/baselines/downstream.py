# retexo/baselines/downstream.py
"""Note 35: the script as explanation, Table 3.

A logistic regression over **script features only** (mode shares, operation
shares at V1 and V3, link density, run lengths, crossings, the model's own
link confidence; no lexicon, no overlap counts) predicts the pair label; its
coefficients are the explanation; a gradient-boosting ceiling goes in a
footnote. Two numbers per script source: **reference versus no-match** macro
F1 on the hard-negative pool under the benchmark's own per-query aggregation
and plateau threshold rule, and **cit. versus cf.** macro F1 on the gold
positives. The features are read off the same dumps every other table uses
(``predictions.jsonl``), the ceiling row off the gold script.

``retexo/edit_typing/downstream.py`` supplies the feature row, the
aggregator and the threshold rule (copied from the benchmark); this module
restricts the features to the script-only columns, adds the four mode shares,
and holds the two classifiers and their protocol.

    python run_table3.py --rows gold,typed_pointer --folds 0-4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import labels
from retexo.baselines.base import Prediction
from retexo.baselines.record import Record
from retexo.edit_typing.downstream import SCRIPT_FEATURES, DownstreamScorer, ScriptFeaturizer

#: Lexicon and overlap columns, excluded: the row must be explained by the script alone.
EXCLUDED_FEATURES = (
    "form1_link_share",
    "lemma_link_share",
    "none_link_share",
    "shared_forms",
    "shared_lemmas",
    "jaccard_lemma",
)

#: The four mode shares over reuse tokens (``labels.to_mode``).
MODE_FEATURES = tuple(f"mode_{m.lower()}" for m in labels.MODES)

#: The columns Table 3 trains on.
SCRIPT_ONLY_FEATURES: Tuple[str, ...] = (
    tuple(f for f in SCRIPT_FEATURES if f not in EXCLUDED_FEATURES) + MODE_FEATURES
)

#: E34's pair head (the Downstream pair-head row): [p(no match), p(cit.), p(cf.)] from ``Prediction.meta``, as extra
#: columns when every row of a dump carries them.
PAIR_HEAD_FEATURES: Tuple[str, ...] = ("head_none", "head_cit", "head_cf")

#: The label columns carried beside the features.
META_COLUMNS = ("id", "query_id", "fold", "pair_label", "is_reference", "is_cit")

#: The protocol's constants (``attic/scripts/run_e28_classify.py``).
NEG_RATIO = 10
SEED = 42
C = 1.0
MAX_ITER = 2000
GBDT = {"max_iter": 300, "learning_rate": 0.05}
RECALL_KS = (10, 100, 1000)


# =============================================================================
# Features
# =============================================================================


class ScriptOnlyFeatures:
    """One feature row per pair from a prediction (or the gold), script columns only.

    Example:
        ```python
        row = ScriptOnlyFeatures.row(record, pred)              # dict of SCRIPT_ONLY_FEATURES + META_COLUMNS
        frame = ScriptOnlyFeatures.from_dump(Path("runs/typed_pointer_f4/predictions.jsonl"))
        ```
    """

    @staticmethod
    def query_id(record: Record) -> str:
        return str(
            record.annotation.get("query_id")
            or record.annotation.get("query")
            or " ".join(record.reuse_tokens)
        )

    @classmethod
    def row(
        cls, record: Record, pred: Optional[Prediction], *, gold: bool = False
    ) -> Optional[Dict[str, Any]]:
        import numpy as np

        from retexo.aligners.decode import ScriptDecoder
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.baselines.record import RecordInterface

        if gold:
            links, tags, frame, _ = RecordInterface.links_of(record)
            pred = Prediction(links=list(links), tags=list(tags), frame=list(frame))
            pred.link_p = [1.0 if s >= 0 else 0.0 for s in links]
        if pred is None or pred.invalid:
            return None
        script = PredictionAdapter.to_script(record, pred)
        if script is None:
            return None
        view = ScriptDecoder.per_token_view(script)
        n = record.n_reuse
        link_p = list(pred.link_p or [0.0] * n) + [0.0] * (n - len(pred.link_p or []))
        null_p = (
            [1.0 - p if s >= 0 else 1.0 for p, s in zip(link_p, pred.links)]
            if not gold
            else [1.0] * n
        )
        tiers = np.zeros((n, record.n_source), dtype=np.int8)
        values = ScriptFeaturizer.features(view, script, record.n_source, link_p, null_p, tiers)
        row = {
            name: float(v)
            for name, v in zip(SCRIPT_FEATURES, values)
            if name not in EXCLUDED_FEATURES
        }
        # the view's tags are the script's (NOP for a copy); the mode reads the record vocabulary
        modes = [
            labels.to_mode((labels.canonical(tag)[0] or "SUBST") if s >= 0 else "INS", bool(f))
            for tag, s, f in zip(view["tags"], view["link"], view["frame"])
        ]
        for name, mode in zip(MODE_FEATURES, labels.MODES):
            row[name] = modes.count(mode) / max(n, 1)
        head = (pred.meta or {}).get("pair_head") if not gold else None
        if head and len(head) == 3:
            row.update(dict(zip(PAIR_HEAD_FEATURES, map(float, head))))
        label = record.pair_label.rstrip(".")
        row.update(
            {
                "id": record.id,
                "query_id": cls.query_id(record),
                "fold": record.fold,
                "pair_label": label,
                "is_reference": int(label in ("cit", "cf")),
                "is_cit": int(label == "cit"),
            }
        )
        return row

    @classmethod
    def from_pairs(
        cls, pairs: Sequence[Tuple[Record, Optional[Prediction]]], *, gold: bool = False
    ):
        import pandas as pd

        rows = [cls.row(r, p, gold=gold) for r, p in pairs]
        frame = pd.DataFrame([r for r in rows if r is not None])
        head = [
            c
            for c in PAIR_HEAD_FEATURES
            if len(frame) and c in frame.columns and frame[c].notna().all()
        ]
        return (
            frame[list(SCRIPT_ONLY_FEATURES) + head + list(META_COLUMNS)] if len(frame) else frame
        )

    @classmethod
    def from_dump(cls, path: Path):
        from retexo.baselines.adapters import PredictionAdapter

        return cls.from_pairs(PredictionAdapter.read_dump(Path(path)))

    @classmethod
    def from_gold(cls, records: Sequence[Record]):
        return cls.from_pairs([(r, None) for r in records], gold=True)

    @staticmethod
    def dedupe(
        pairs: Sequence[Tuple[Record, Optional[Prediction]]],
    ) -> Tuple[List[Tuple[Record, Optional[Prediction]]], List[Tuple[str, str]]]:
        """One record per distinct text pair (note 35: 1,467 of the gold's 1,490):
        the first by id wins, the later duplicates' ids come back as
        ``(kept, dropped)`` pairs; a label conflict is reported, not resolved."""
        seen: Dict[Tuple[str, str], str] = {}
        kept, dropped = [], []
        for record, pred in sorted(pairs, key=lambda x: x[0].id):
            key = (" ".join(record.source_tokens), " ".join(record.reuse_tokens))
            if key in seen:
                dropped.append((seen[key], record.id))
                continue
            seen[key] = record.id
            kept.append((record, pred))
        return kept, dropped


# =============================================================================
# The two classifiers
# =============================================================================


class SimilarityScore:
    """The similarity row: the benchmark cross-encoder's reference probability (one minus its no-match
    probability) as the single feature, on the same positives and pools as the script rows
    (``tools/build_downstream_pools.py`` writes both scores).

    Example:
        ```python
        positives = SimilarityScore.positives(gold_records, {4: json.load(open("data/downstream/reference_scores_f4.json"))})
        pool = SimilarityScore.pool(RecordCodec.load(Path("data/downstream/pool_f4.records.jsonl")))
        ```
    """

    FEATURES = ("similarity",)

    @staticmethod
    def _row(record: Record, score: float) -> Dict[str, Any]:
        label = record.pair_label
        return {
            "similarity": float(score),
            "id": record.id,
            "query_id": ScriptOnlyFeatures.query_id(record),
            "fold": record.fold,
            "pair_label": label,
            "is_reference": int(label in ("cit", "cf")),
            "is_cit": int(label == "cit"),
        }

    @classmethod
    def positives(cls, records: Sequence[Record], scores_by_fold: Dict[int, Dict[str, float]]):
        """The gold pairs with a cross-encoder score (keyed ``query text<TAB>source text``); the missing ones
        are reported by the caller through the frame's length."""
        import pandas as pd

        rows = []
        for record in records:
            key = f"{' '.join(record.reuse_tokens)}\t{' '.join(record.source_tokens)}"
            score = scores_by_fold.get(record.fold, {}).get(key)
            if score is not None:
                rows.append(cls._row(record, score))
        return pd.DataFrame(rows)

    @classmethod
    def pool(cls, records: Sequence[Record]):
        import pandas as pd

        return pd.DataFrame(
            [
                cls._row(r, 1.0 - float(r.provenance["prob_no_match"]))
                for r in records
                if r.provenance.get("prob_no_match") is not None
            ]
        )


class LengthOnly:
    """The length control of the Downstream table: the two passages' lengths in words and their ratio, no script.
    If it comes close to the script rows, the script features separate references from the pool by how the
    passages were cut, not by how the reuse was made.

    Example:
        ```python
        positives = LengthOnly.frame(gold_records)
        pool = LengthOnly.frame(RecordCodec.load(Path("data/downstream/pool_f4.records.jsonl")))
        ```
    """

    FEATURES = ("source_len", "reuse_len", "len_ratio")

    @staticmethod
    def _row(record: Record) -> Dict[str, Any]:
        label = record.pair_label.rstrip(".")
        return {
            "source_len": float(record.n_source),
            "reuse_len": float(record.n_reuse),
            "len_ratio": record.n_reuse / max(record.n_source, 1),
            "id": record.id,
            "query_id": ScriptOnlyFeatures.query_id(record),
            "fold": record.fold,
            "pair_label": label,
            "is_reference": int(label in ("cit", "cf")),
            "is_cit": int(label == "cit"),
        }

    @classmethod
    def frame(cls, records: Sequence[Record]):
        import pandas as pd

        return pd.DataFrame([cls._row(r) for r in records])


class ReferenceClassifier:
    """Reference versus no-match on the hard-negative pool, the benchmark's protocol.

    Example:
        ```python
        model, threshold = ReferenceClassifier.fit(train_frame)
        scores = ReferenceClassifier.evaluate(model, threshold, test_frame)
        ```
    """

    @staticmethod
    def sample_negatives(frame, *, neg_ratio: int = NEG_RATIO, seed: int = SEED):
        import pandas as pd

        positives = frame[frame["is_reference"] == 1]
        negatives = frame[frame["is_reference"] == 0]
        n_neg = min(len(negatives), neg_ratio * max(len(positives), 1))
        return (
            pd.concat([positives, negatives.sample(n=n_neg, random_state=seed)])
            if n_neg
            else positives
        )

    @staticmethod
    def build(kind: str = "logreg"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if kind == "gbdt":
            return HistGradientBoostingClassifier(random_state=SEED, **GBDT)
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER))

    @classmethod
    def fit(
        cls,
        train,
        *,
        kind: str = "logreg",
        neg_ratio: int = NEG_RATIO,
        seed: int = SEED,
        target: str = "is_reference",
        threshold_on: str = "train",
        features: Sequence[str] = SCRIPT_ONLY_FEATURES,
    ):
        """The classifier on positives plus sampled negatives; the plateau
        threshold on the training rows' probabilities (``cv`` for the boosted ceiling). ``features``
        is the similarity row's single score column instead of the script columns."""
        import numpy as np
        from sklearn.model_selection import cross_val_predict

        sample = (
            cls.sample_negatives(train, neg_ratio=neg_ratio, seed=seed)
            if target == "is_reference"
            else train
        )
        X = sample[list(features)].to_numpy(dtype=float)
        y = sample[target].to_numpy(dtype=int)
        model = cls.build(kind).fit(X, y)
        if threshold_on == "cv" and len(np.unique(y)) > 1 and len(y) >= 10:
            probs = cross_val_predict(cls.build(kind), X, y, cv=5, method="predict_proba")[:, 1]
        else:
            probs = model.predict_proba(X)[:, 1]
        threshold = DownstreamScorer.find_threshold(y, probs) if len(np.unique(y)) > 1 else 0.5
        return model, threshold

    @staticmethod
    def evaluate(
        model,
        threshold: float,
        test,
        *,
        target: str = "is_reference",
        features: Sequence[str] = SCRIPT_ONLY_FEATURES,
    ) -> Dict[str, Any]:
        """The paper's per-query macro/micro F1 at the threshold, plus recall@k."""

        X = test[list(features)].to_numpy(dtype=float)
        probs = model.predict_proba(X)[:, 1]
        gold_all = test[target].to_numpy(dtype=int)
        pred_all = (probs >= threshold).astype(int)
        qids = sorted(set(test["query_id"]))
        gold_rows, pred_rows, score_rows = [], [], []
        for qid in qids:
            mask = (test["query_id"] == qid).to_numpy()
            gold_rows.append(gold_all[mask])
            pred_rows.append(pred_all[mask])
            score_rows.append(probs[mask])
        macro, micro, _ = DownstreamScorer.macro_micro(qids, gold_rows, pred_rows)
        recall = DownstreamScorer.recall_at_k(qids, gold_rows, score_rows, ks=RECALL_KS)
        return {
            "macro": macro,
            "micro": micro,
            "recall_at_k": {str(k): v for k, v in recall.items()},
            "threshold": float(threshold),
            "n": int(len(test)),
            "n_queries": len(qids),
        }


class TypeClassifier:
    """cit. versus cf. on the gold positives, five folds, macro F1 over the two classes.

    Example:
        ```python
        result = TypeClassifier.fit_and_evaluate(positives_frame)      # {"macro_f1": ..., "per_fold": [...]}
        ```
    """

    @staticmethod
    def fit_and_evaluate(
        positives,
        *,
        kind: str = "logreg",
        folds: Sequence[int] = (0, 1, 2, 3, 4),
        features: Sequence[str] = SCRIPT_ONLY_FEATURES,
    ) -> Dict[str, Any]:
        import numpy as np
        from sklearn.metrics import f1_score

        per_fold, models = [], {}
        for fold in folds:
            train = positives[positives["fold"] != fold]
            test = positives[positives["fold"] == fold]
            if not len(train) or not len(test) or train["is_cit"].nunique() < 2:
                continue
            model = ReferenceClassifier.build(kind).fit(
                train[list(features)].to_numpy(dtype=float), train["is_cit"].to_numpy(dtype=int)
            )
            pred = model.predict(test[list(features)].to_numpy(dtype=float))
            per_fold.append(
                float(f1_score(test["is_cit"].to_numpy(dtype=int), pred, average="macro"))
            )
            models[fold] = model
        return {
            "macro_f1": float(np.mean(per_fold)) if per_fold else 0.0,
            "per_fold": per_fold,
            "models": models,
        }


class Explanation:
    """The standardised coefficients of the logistic regressions, one column per
    fold, and how many folds agree on each sign.

    Example:
        ```python
        table = Explanation.coefficients({4: model_fold4, 3: model_fold3})
        ```
    """

    @staticmethod
    def coefficients(models: Dict[int, Any], features: Sequence[str] = SCRIPT_ONLY_FEATURES):
        import numpy as np
        import pandas as pd

        columns = {}
        for fold, model in sorted(models.items()):
            estimator = model[-1] if hasattr(model, "steps") else model
            coef = getattr(estimator, "coef_", None)
            if coef is None:
                continue
            columns[f"fold {fold}"] = np.asarray(coef).ravel()
        table = pd.DataFrame(columns, index=list(features))
        if len(table.columns):
            signs = np.sign(table.to_numpy())
            table["mean"] = table[list(columns)].mean(axis=1)
            table["sign agreement"] = [int(max((row > 0).sum(), (row < 0).sum())) for row in signs]
            table = table.reindex(table["mean"].abs().sort_values(ascending=False).index)
        return table
