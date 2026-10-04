# retexo/baselines/early_stopping.py
"""Early stopping for every trained row, on a validation sample of the training folds.

The paper's ground rules (``Paper Experiments/Experiment Ground Rules``) fix one rule
for every method: train at least ``min_epochs``; from then on, score a validation
sample after every epoch and stop when the score has not improved for ``patience``
epochs, or at ``max_epochs``; then restore the best epoch's weights. The validation
sample is 20 % of the training records, drawn once per run and grouped by the reusing
passage, so no passage sits on both sides; it is never trained on. The dev fold stays
the threshold's fold and the test fold stays unseen.

The score is L3 macro F1 through the same path a table row takes: the method's own
prediction, the shared decoder with the null threshold tuned on the validation sample,
the method's typer (the rule typer for alignment-only rows), and the shared scorer. A
typing-only row is scored at its own level (the mode tagger at the modes); the NMT
aligner, whose alignments exist only after its translation stage, stops that stage on
the validation pairs' translation loss instead.

The best weights are copied to CPU memory, never to disk (the no-weights rule).

Every evaluation also keeps every metric of the metric note (``Paper Experiments/Evaluation
Metrics``) that a single model's prediction defines, read from the same prediction
(``TrainingMonitor``): after every epoch, and ``synthetic_evals`` times across the synthetic
stage, which is one long epoch. The invented links are counted on a fixed random re-pairing
of the validation passages, never on the test fold. The run record keeps the whole series
(``training_output.evaluations``), so the tables' metrics can be read over training time.

    stopper = EarlyStopping.for_method(self, log=log)        # None without a validation sample
    for epoch in range(1, stopper.max_epochs + 1 if stopper else fixed + 1):
        train_one_epoch()
        if stopper and not stopper.step(epoch, modules):
            break
    if stopper:
        stopper.restore(modules)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from retexo.baselines.record import Record, links_of

#: The per-method rule (minimum epochs, patience, maximum epochs); ``cfg.extra`` overrides
#: with ``es_min``, ``es_patience``, ``es_max``. Every method on ours' schedule (``SharedSchedule``) has the
#: same epoch, one pass over the same examples, and so the same rule: the maximum is only a cap, the
#: validation score decides, and the learning-rate schedules that span the maximum decay alike. Two
#: exceptions: the translation-model aligner counts translation epochs (another unit), and the fine-tuned
#: language model's epoch costs hours.
SHARED_RULE = (4, 2, 16)
RULES: Dict[str, Tuple[int, int, int]] = {
    "typed_pointer": SHARED_RULE,
    "span_aligner": SHARED_RULE,
    "tagger_v1": SHARED_RULE,
    "tagger_mode": SHARED_RULE,
    "span_pair": SHARED_RULE,
    "sim_aligner_ft": SHARED_RULE,
    "nmt_aligner": (2, 1, 6),
    "llm_qlora": (1, 1, 3),
}
#: The share of the training records held out as the validation sample.
VALIDATION_SHARE = 0.2

Modules = Dict[str, Any]          # name -> anything with state_dict() / load_state_dict()


# =============================================================================
# The validation sample
# =============================================================================


class ValidationSplit:
    """20 % of the training records, grouped by the reusing passage.

    Example:
        ```python
        fit_records, valid = ValidationSplit.split(train, share=0.2, seed=1)
        ```
    """

    @staticmethod
    def group_key(record: Record) -> str:
        return str(record.reuse_meta.get("citation") or " ".join(record.reuse_tokens))

    @classmethod
    def split(cls, records: Sequence[Record], *, share: float = VALIDATION_SHARE, seed: int = 1
              ) -> Tuple[List[Record], List[Record]]:
        """``(fit, valid)``: whole groups go to the validation side until it holds ``share`` of the records."""
        if share <= 0 or len(records) < 2:
            return list(records), []
        groups: Dict[str, List[Record]] = {}
        for record in records:
            groups.setdefault(cls.group_key(record), []).append(record)
        keys = sorted(groups)
        random.Random(1000 + seed).shuffle(keys)
        target = share * len(records)
        valid_keys, n_valid = set(), 0
        for key in keys:
            if n_valid >= target:
                break
            valid_keys.add(key)
            n_valid += len(groups[key])
        fit = [r for r in records if cls.group_key(r) not in valid_keys]
        valid = [r for r in records if cls.group_key(r) in valid_keys]
        return fit, valid


# =============================================================================
# The validation score
# =============================================================================


class ValidationScorer:
    """L3 macro F1 of a method on the validation sample, through the table's own path."""

    @classmethod
    def _decoded(cls, method, records: List[Record]):
        """The method's prediction on ``records``, decoded and typed at a null threshold tuned on them, by the
        criterion and over the grid the driver tunes the dev fold with (``method.tune_criterion``,
        ``method.theta_grid``; token accuracy over the default grid when the driver set neither)."""
        from retexo.baselines import decoder as dec

        preds = method.predict(records)
        dials: Dict[str, Any] = {"theta": dec.DEFAULT_THETA, "frame_rule": "keyword"}
        if "split_repair" in method.cfg.extra:
            dials["split_repair"] = float(method.cfg.extra["split_repair"])
        if any(p.scores for p in preds):
            dials["theta"] = cls.tune_theta(method, records, preds, dials, grid=getattr(method, "theta_grid", None),
                                            criterion=getattr(method, "tune_criterion", "token_accuracy"))
        return [method.postprocess(r, p, dials) for r, p in zip(records, preds)], dials

    @staticmethod
    def tune_theta(method, records: Sequence[Record], preds: Sequence[Any], dials: Dict[str, Any], *,
                   grid: Optional[Sequence[float]] = None, criterion: str = "token_accuracy", level: str = "V1") -> float:
        """The null threshold on ``records`` by ``criterion``; ties go to the larger theta.

        ``token_accuracy`` and ``link_f1`` score the decoded links (``dec.tune_null_threshold``); ``l3``
        decodes and types every pair through ``method.postprocess`` and scores L3 macro F1 (the scorer's
        V1), the tables' headline. Token accuracy is flat near its top and gains by dropping the
        low-similarity INFLECT and SUBST links: on the fine-tuned similarity aligner it chose theta 0.84
        to 0.90 on the dev fold where the dev fold's L3 peaks at 0.78 to 0.83 (2026-09-28). ``preds`` are
        not changed.
        """
        from dataclasses import replace

        from retexo.baselines import decoder as dec
        from retexo.baselines.scorer import BaselineScorer

        if criterion != "l3":
            return dec.tune_null_threshold(
                [p.scores or [] for p in preds], [links_of(r)[0] for r in records],
                rev_per_pair=[p.rev_scores for p in preds], n_source_per_pair=[r.n_source for r in records],
                grid=grid, criterion=criterion, decoder=method.decoder, records=records, dials=dials)
        grid = list(grid) if grid is not None else [round(0.05 * i, 2) for i in range(1, 20)]   # the decoder's default
        require_source = bool(getattr(method, "validation_require_source", True))
        best_theta, best_value = grid[0], -1.0
        for theta in grid:
            trial = {**dials, "theta": theta}
            done = [method.postprocess(r, replace(p, links=list(p.links), tags=list(p.tags), frame=list(p.frame),
                                                  extra=list(p.extra), meta=dict(p.meta)), trial)
                    for r, p in zip(records, preds)]
            value = float(BaselineScorer.op_scores(records, done, level, require_source=require_source)["macro_f1"])
            if value >= best_value:
                best_theta, best_value = theta, value
        return best_theta

    @classmethod
    def score(cls, method, records: Sequence[Record], *, level: str = "V1") -> float:
        from retexo.baselines.scorer import BaselineScorer

        records = list(records)
        if not records:
            return 0.0
        done, _ = cls._decoded(method, records)
        require_source = bool(getattr(method, "validation_require_source", True))
        return float(BaselineScorer.op_scores(records, done, level, require_source=require_source)["macro_f1"])

    @classmethod
    def evaluate(cls, method, records: Sequence[Record], *, level: str = "V1",
                 negatives: Sequence[Record] = ()) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """``(headline, full scorer result, dials)`` from one prediction: the headline is exactly ``score``'s
        value (what early stopping decides on); the full result holds every metric of the table's scorer,
        with the invented links on ``negatives`` decoded at the same dials."""
        from retexo.baselines.scorer import BaselineScorer

        records = list(records)
        done, dials = cls._decoded(method, records)
        require_source = bool(getattr(method, "validation_require_source", True))
        headline = float(BaselineScorer.op_scores(records, done, level, require_source=require_source)["macro_f1"])
        neg = None
        if negatives:
            negatives = list(negatives)
            neg = (negatives, [method.postprocess(r, p, dials) for r, p in zip(negatives, method.predict(negatives))])
        full = BaselineScorer.score(records, done, negatives=neg, emits=getattr(method, "emits", "scores"))
        return headline, full, dials


class NoteMetrics:
    """The metric note's per-row metrics, as one flat record from a scorer result (paper names: INFLECT for
    the code's MORPH; L3 is the scorer's V1 level, L4 its V3, G1 the modes, G2 the groups). The downstream
    and agreement metrics are not properties of one prediction and stay out.

    Example:
        ```python
        NoteMetrics.flat(BaselineScorer.score(records, preds))["L3_macro_f1"]
        ```
    """

    PAPER = {"MORPH": "INFLECT"}

    @staticmethod
    def _f1(block: Optional[Dict[str, Any]]) -> Optional[float]:
        return None if not block else block.get("F1", block.get("f1"))

    @classmethod
    def flat(cls, result: Dict[str, Any]) -> Dict[str, Optional[float]]:
        link = result.get("link") or {}
        ops = result.get("ops") or {}
        out: Dict[str, Optional[float]] = {
            "token_accuracy": result.get("token_accuracy"),
            "link_precision": link.get("precision"), "link_recall": link.get("recall"), "link_f1": link.get("f1"),
            "invented_links_per_pair": (result.get("negatives") or {}).get("links_per_pair"),
        }
        for level, name in (("V1", "L3"), ("V3", "L4")):
            block = ops.get(level) or {}
            out[f"{name}_macro_f1"] = block.get("macro_f1")
            out[f"{name}_micro_f1"] = block.get("accuracy")        # pooled over the labelled reuse words
            for op, scores in (block.get("per_class") or {}).items():
                out[f"{name}_f1_{cls.PAPER.get(op, op)}"] = cls._f1(scores)
        per_class = (ops.get("V1") or {}).get("per_class") or {}
        for op in ("SUBST", "MORPH"):
            scores = per_class.get(op) or {}
            out[f"sure_recall_{cls.PAPER.get(op, op)}"] = (scores["sure_found"] / scores["sure"]
                                                             if scores.get("sure") else None)
        out["G1_mode_macro_f1"] = (ops.get("mode") or {}).get("macro_f1")
        out["G2_group_macro_f1"] = (ops.get("group") or {}).get("macro_f1")
        structure = result.get("structure") or {}
        out["frame_span_f1"] = cls._f1(result.get("frame_span"))
        out["quote_span_f1"] = cls._f1(structure.get("quote_span"))
        out["reorder_token_f1"] = cls._f1(structure.get("reorder_token"))
        out["disperse_pair_f1"] = cls._f1(structure.get("disperse_pair"))
        out["resource_silent_link_accuracy"] = ((result.get("by_regime") or {}).get("no_rel_found") or {}).get("link_acc")
        out["replay_rate"] = result.get("replay_rate")
        out["invalid_output_rate"] = (result.get("invalid") or {}).get("rate")
        return {k: (round(float(v), 5) if isinstance(v, (int, float)) else v) for k, v in out.items()}


class TrainingMonitor:
    """The metric note's metrics on the validation sample over training time.

    ``evaluate`` scores the current model (after the synthetic stage, after every epoch);
    ``progress(stage, evals)`` is a batch callback ``(done, total)`` that evaluates ``evals - 1`` times
    inside a long epoch at even fractions (the last point is the epoch's own). An evaluation leaves the
    training untouched: the modules' train/eval flags are restored and no training randomness is drawn.

    Example:
        ```python
        monitor = TrainingMonitor.for_method(method, log=log)       # None without a validation sample
        model.fit(pool, on_batch=monitor.progress("synthetic", 4))
        monitor.evaluate("synthetic")
        ```
    """

    def __init__(self, method, records: Sequence[Record], *, level: str = "V1", log=None, repairings: bool = True):
        import time

        self.method, self.records, self.level, self.log = method, list(records), level, log
        self.points: List[Dict[str, Any]] = []
        self.started = time.time()
        self.negatives = self.repairings(self.records) if repairings else []
        featurizer = getattr(method, "featurizer", None)
        if featurizer is not None:
            # the lookup column (resource-silent link accuracy), as the driver adds it to the dev and test folds
            from retexo.baselines.typer import RuleTyper

            RuleTyper.annotate_regimes(self.records, featurizer)

    @classmethod
    def for_method(cls, method, *, log=None) -> Optional["TrainingMonitor"]:
        existing = getattr(method, "monitor", None)
        if existing is not None:
            return existing
        valid = getattr(method, "validation", None)
        if not valid:
            return None
        monitor = cls(method, valid, level=getattr(method, "validation_level", "V1"), log=log,
                      repairings=bool(int(method.cfg.extra.get("monitor_negatives", 1))))
        method.monitor = monitor
        return monitor

    @staticmethod
    def repairings(records: Sequence[Record], seed: int = 1) -> List[Record]:
        """Each validation reuse passage with the source passage of another validation pair (a derangement,
        seeded): pairs that share no reuse, where every link is invented."""
        from dataclasses import replace

        if len(records) < 2:
            return []
        order = list(range(len(records)))
        random.Random(seed).shuffle(order)
        shifted = order[1:] + order[:1]
        out = []
        for i, j in zip(order, shifted):
            a, b = records[i], records[j]
            out.append(replace(a, id=f"repair/{a.id}/{b.id}", source_tokens=list(b.source_tokens),
                               source_work=b.source_work, links=[], spans=[], pair_label="no_match", pred=None))
        return out

    def _flags(self):
        modules = self.method.modules() if hasattr(self.method, "modules") else {}
        flags = []
        for module in modules.values():
            if module is not None and hasattr(module, "modules"):
                flags.extend((m, m.training) for m in module.modules())
        return flags

    def evaluate(self, stage: str, *, epoch: Optional[int] = None, fraction: Optional[float] = None) -> float:
        import time

        flags = self._flags()
        try:
            headline, full, dials = ValidationScorer.evaluate(self.method, self.records, level=self.level,
                                                              negatives=self.negatives)
            for module, _ in flags:                  # the validation loss as validation: no dropout
                module.train(False)
            valid_loss = self.method.validation_loss(self.records) if hasattr(self.method, "validation_loss") else None
        finally:
            for module, training in flags:
                module.train(training)
        point = {"stage": stage, "epoch": epoch, "fraction": fraction,
                 "seconds": round(time.time() - self.started, 1), "theta": dials.get("theta"),
                 "headline": round(headline, 5),
                 "validation_loss": round(valid_loss, 5) if valid_loss is not None else None, **NoteMetrics.flat(full)}
        self.points.append(point)
        if self.log:
            where = f"epoch {epoch}" if epoch is not None else f"{stage} {fraction:.2f}" if fraction is not None else stage
            self.log(f"[monitor] {where}: link F1 {point['link_f1'] if point['link_f1'] is not None else '-'}, "
                     f"L3 macro F1 {point['L3_macro_f1']}, invented links per pair {point['invented_links_per_pair']}")
        return headline

    def progress(self, stage: str, evals: int) -> Optional[Callable[[int, int], None]]:
        """A ``(done, total)`` batch callback evaluating at 1/evals, 2/evals ... (evals - 1)/evals of the epoch."""
        if evals <= 1:
            return None
        marks = iter(range(1, evals))
        state = {"next": next(marks, None)}

        def on_batch(done: int, total: int) -> None:
            while state["next"] is not None and total and done >= total * state["next"] / evals:
                self.evaluate(stage, fraction=state["next"] / evals)
                state["next"] = next(marks, None)

        return on_batch


# =============================================================================
# The rule
# =============================================================================


@dataclass
class EarlyStopping:
    """Minimum epochs, patience, maximum epochs, and the best epoch's weights in memory.

    Example:
        ```python
        stopper = EarlyStopping(min_epochs=4, patience=2, max_epochs=16, score=lambda: scorer(), log=print)
        for epoch in range(1, stopper.max_epochs + 1):
            train_one_epoch()
            if not stopper.step(epoch, {"encoder": encoder}):
                break
        stopper.restore({"encoder": encoder})
        ```
    """

    min_epochs: int
    patience: int
    max_epochs: int
    score: Callable[[], float]
    log: Optional[Callable[[str], None]] = None
    higher_is_better: bool = True
    metric: str = "L3 macro F1"
    best_score: Optional[float] = None
    best_epoch: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    monitor: Optional[TrainingMonitor] = None
    _best_state: Optional[Dict[str, Any]] = None

    # ---------- construction ----------

    @classmethod
    def for_method(cls, method, *, log=None, score: Optional[Callable[[], float]] = None,
                   higher_is_better: bool = True, metric: str = "L3 macro F1") -> Optional["EarlyStopping"]:
        """The method's rule, or ``None`` when the run has no validation sample."""
        valid = getattr(method, "validation", None)
        if not valid:
            return None
        lo, patience, hi = RULES.get(method.name, (2, 2, 10))
        extra = method.cfg.extra
        lo, patience, hi = int(extra.get("es_min", lo)), int(extra.get("es_patience", patience)), int(extra.get("es_max", hi))
        if method.cfg.smoke:
            lo, patience, hi = 1, 1, 2
        monitor = None
        if score is None:
            level = getattr(method, "validation_level", "V1")
            score = lambda: ValidationScorer.score(method, valid, level=level)  # noqa: E731
            metric = {"V1": "L3 macro F1", "V3": "L4 macro F1", "mode": "mode (G1) macro F1"}.get(level, f"{level} macro F1")
            monitor = TrainingMonitor.for_method(method, log=log)       # every note metric from the same prediction
            if monitor is not None:
                monitor.level = level
        stopper = cls(min_epochs=max(1, lo), patience=max(1, patience), max_epochs=max(lo, hi), score=score, log=log,
                      higher_is_better=higher_is_better, metric=metric, monitor=monitor)
        method.early_stopping = stopper
        if log:
            log(f"[early stopping] {method.name}: {len(valid)} validation pairs, min {stopper.min_epochs}, "
                f"patience {stopper.patience}, max {stopper.max_epochs} epochs, on {metric}")
        return stopper

    # ---------- the loop ----------

    def _better(self, value: float) -> bool:
        if self.best_score is None:
            return True
        return value > self.best_score + 1e-9 if self.higher_is_better else value < self.best_score - 1e-9

    def step(self, epoch: int, modules: Modules) -> bool:
        """Score the epoch, keep the best weights; ``False`` once training should stop."""
        value = float(self.monitor.evaluate("epoch", epoch=epoch) if self.monitor is not None else self.score())
        improved = self._better(value)
        if improved:
            self.best_score, self.best_epoch = value, epoch
            self._best_state = self.snapshot(modules)
        self.history.append({"epoch": epoch, "valid": round(value, 5)})
        if self.log:
            self.log(f"[early stopping] epoch {epoch}: valid {self.metric} {value:.4f}"
                     f"{' (best)' if improved else f' (best {self.best_score:.4f} at epoch {self.best_epoch})'}")
        if epoch >= self.max_epochs:
            return False
        return not (epoch >= self.min_epochs and epoch - self.best_epoch >= self.patience)

    def restore(self, modules: Modules) -> None:
        """Load the best epoch's weights back into the modules."""
        if self._best_state is None:
            return
        for name, module in modules.items():
            if module is not None and name in self._best_state:
                module.load_state_dict(self._best_state[name])
        if self.log:
            self.log(f"[early stopping] restored epoch {self.best_epoch} (valid {self.metric} {self.best_score:.4f}, "
                     f"{len(self.history)} epochs run)")

    @staticmethod
    def snapshot(modules: Modules) -> Dict[str, Any]:
        """A CPU copy of every module's state, never written to disk."""
        import torch

        def copy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().to("cpu", copy=True)
            if isinstance(value, dict):
                return {k: copy(v) for k, v in value.items()}
            return value

        return {name: copy(module.state_dict()) for name, module in modules.items() if module is not None}

    def release(self) -> None:
        """Drop the in-memory copy once training is over."""
        self._best_state = None

    def summary(self) -> Dict[str, Any]:
        """What the run record keeps: the chosen epoch, its score, and the whole validation curve."""
        return {"metric": self.metric, "min_epochs": self.min_epochs, "patience": self.patience,
                "max_epochs": self.max_epochs, "best_epoch": self.best_epoch, "best_score": self.best_score,
                "epochs_run": len(self.history), "history": self.history}


class EarlyStoppingGroup:
    """Several stoppers of one method (the NMT aligner's two directions) as one record entry."""

    def __init__(self, stoppers: Dict[str, Optional[EarlyStopping]]):
        self.stoppers = {k: v for k, v in stoppers.items() if v is not None}

    def summary(self) -> Dict[str, Any]:
        return {name: stopper.summary() for name, stopper in self.stoppers.items()}
