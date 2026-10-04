# retexo/baselines/tagger.py
"""Note 20: the "typing without alignment" baseline, in two rows.

A tag-and-realise tagger in the GECToR, LaserTagger and PIE tradition, over
the concatenated pair `[CLS] source [SEP] reuse [SEP]`: one label per reuse
word, one binary KEEP/DEL label per source word, no source position -- the
row that asks whether *knowing which word it came from* matters at all, since
this row never says. Built thinly on `ChangeDetector` (`pointer=False`, an
n-way operation head, the optional source head), which already carries the
loss, the class weights and the inference-time logit bias GECToR's own dials
need; this module supplies the two label spaces, the label derivation, the
staged training schedule and the min-change-probability gate `ChangeDetector`
does not have.

Two rows, one class: `TaggerV1` ({COPY, MORPH, SUBST, INS, FRAME} on the
reuse side, {KEEP, DEL} on the source side) and `TaggerMode` ({VERBATIM,
ALLUSION, FRAME, NOMATCH}, no source head) -- `Tagger` carries the shared
logic, each subclass only names its `LabelSet`.

Two simplifications from the implementation note, disclosed here rather than
found later: the synthetic pretraining stage is not implemented (stage I of
the note's three-stage schedule), and the mixed-negatives stage (stage III)
is opportunistic rather than guaranteed -- both need `BenchmarkData`, the
preliminary round's raw-corpus table, which is not present on either machine
this was built on (`data/processed/labels/labels.csv` does not exist here or
on the training server). `fit` tries `NegativeBuilder` and logs a skip, not a silent
gap, when the data is absent; every check in this module runs gold-stage
only until that table is restored.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction
from retexo.baselines.labels import Labels
from retexo.baselines.record import Record, RecordInterface
from retexo.baselines.scorer import BaselineScorer
from retexo.formulations.change_detector import ChangeDetector, ChangeDetectorConfig, ChangeExample

#: The V1 tagger's own label space: V1 with FRAME kept distinct (so the FRAME
#: span column exists at all), collapsed to plain V1 -- FRAME to INS -- by
#: the shared scorer at scoring time (``Labels.to_v1``, unchanged).
V1_FRAME = ("COPY", "MORPH", "SUBST", "INS", "FRAME")

#: Keep-bias grid and minimum-change-probability grid for `tune`.
KEEP_BIAS_GRID = (0.0, 0.1, 0.2, 0.35, 0.5)
MIN_CHANGE_P_GRID = (0.0, 0.3, 0.5, 0.66)


def _v1_frame_labels(links: Sequence[int], tags: Sequence[str], frame: Sequence[int]) -> List[str]:
    """V1 with FRAME kept distinct: the finest level collapsed to V1 everywhere except FRAME."""
    fine = Labels.token_labels(links, tags, frame, "V3")
    return [op if op == "FRAME" else Labels.to_v1(op) for op in fine]


# =============================================================================
# Label sets
# =============================================================================


#: The iSTS chunk-type vocabulary (SemEval-2016 Task 2), read from
#: ``record.annotation["ists_type"]`` -- a converter-supplied per-reuse-token
#: field, not derived from edge operations like every other label set.
ISTS_TYPES = ("EQUI", "OPPO", "SPE1", "SPE2", "SIMI", "REL", "NOALI")


@dataclass(frozen=True)
class LabelSet:
    """One tagger row's label space: its classes, its "nothing happened" class, the
    level the shared scorer tunes and scores it at, and whether it has a source head."""

    name: str
    classes: Tuple[str, ...]
    keep: str
    score_level: Optional[str]
    source_head: bool

    def token_labels(self, record: Record, *, train_on: str = "all") -> List[str]:
        if self.name == "ists":
            fixed = list(record.annotation.get("ists_type") or [])
            return (fixed + ["NOALI"] * record.n_reuse)[:record.n_reuse]
        links, tags, frame, sure = RecordInterface.links_of(record)
        if train_on == "sure":
            links = [s if sure[t] else -1 for t, s in enumerate(links)]
        if self.name == "V1":
            return _v1_frame_labels(links, tags, frame)
        return Labels.token_labels(links, tags, frame, "mode")


LABEL_SETS: Dict[str, LabelSet] = {
    "V1": LabelSet("V1", V1_FRAME, keep="COPY", score_level="V1", source_head=True),
    "mode": LabelSet("mode", Labels.MODES, keep="VERBATIM", score_level="mode", source_head=False),
    "ists": LabelSet("ists", ISTS_TYPES, keep="EQUI", score_level=None, source_head=False),
}


# =============================================================================
# Examples
# =============================================================================


def examples_from_records(records: Sequence[Record], label_set: LabelSet, *, train_on: str = "all"
                          ) -> List[ChangeExample]:
    """One `ChangeExample` per record, labelled at `label_set`'s level.

    A possible-only edge (``train_on="sure"``) is dropped to a null link before
    the label is read, so its reuse word is labelled as unlinked (INS or
    NOMATCH), per the note's own rule; the iSTS label set ignores `train_on`
    (its labels come from the converter's chunk types, not from edges).
    """
    out = []
    for record in records:
        links, _, _, _ = RecordInterface.links_of(record)
        reuse_labels = label_set.token_labels(record, train_on=train_on)
        source_labels_str = Labels.source_labels(links, record.n_source)
        out.append(ChangeExample(
            source_tokens=list(record.source_tokens), target_tokens=list(record.reuse_tokens),
            labels=[0 if op == label_set.keep else 1 for op in reuse_labels],
            operations=reuse_labels, n_operations=sum(1 for op in reuse_labels if op != label_set.keep),
            source_labels=[0 if op == "KEEP" else 1 for op in source_labels_str],
            source_operations=source_labels_str))
    return out


def _class_weights(examples: Sequence[ChangeExample], classes: Sequence[str], *, cap: float = 20.0
                   ) -> Tuple[Tuple[str, float], ...]:
    """Inverse class frequency over `examples`, capped at `cap`, normalised so the
    most frequent class has weight 1."""
    counts = {c: 0 for c in classes}
    for example in examples:
        for op in example.operations:
            if op in counts:
                counts[op] += 1
    biggest = max(counts.values()) if counts else 0
    weights = {c: min(biggest / n, cap) if n else cap for c, n in counts.items()}
    return tuple(weights.items())


# =============================================================================
# Tagger
# =============================================================================


class Tagger(Baseline):
    """A tag-and-realise tagger: `ChangeDetector` with `pointer=False`, staged and dialled.

    ``cfg.extra`` keys: ``pooling`` (``mean``), ``focal_gamma`` (0),
    ``cold_passes`` (2), ``cold_lr`` (1e-3), ``gold_passes`` (4),
    ``negatives`` (1: try the mixed-negatives stage; 0: gold only),
    ``neg_ratio`` (0.5), ``keep_bias`` (tuned), ``min_change_p`` (tuned),
    ``train_on`` (``all`` | ``sure``).

    Subclasses name the row: `TaggerV1` (``label_set_name = "V1"``),
    `TaggerMode` (``label_set_name = "mode"``).
    """

    emits = "types"
    trainable = True
    typer = "own"
    decoder = "none"
    label_set_name: str = ""

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        #: ``--extra label_set=ists`` runs the iSTS check through this same
        #: class (the note's own driver recipe); the subclass default applies
        #: otherwise.
        self.label_set = LABEL_SETS[str(cfg.extra.get("label_set", self.label_set_name))]
        self.pooling = str(cfg.extra.get("pooling", "mean"))
        self.focal_gamma = float(cfg.extra.get("focal_gamma", 0.0))
        self.cold_passes = int(cfg.extra.get("cold_passes", 2))
        self.cold_lr = float(cfg.extra.get("cold_lr", 1e-3))
        self.gold_passes = int(cfg.extra.get("gold_passes", 4))
        self.use_negatives = bool(int(cfg.extra.get("negatives", 1)))
        self.neg_ratio = float(cfg.extra.get("neg_ratio", 0.5))
        self.keep_bias = float(cfg.extra.get("keep_bias", 0.0))
        self.min_change_p = float(cfg.extra.get("min_change_p", 0.0))
        self.train_on = str(cfg.extra.get("train_on", cfg.train_on))
        self.model: Optional[ChangeDetector] = None

    def _build_model(self, class_weights: Tuple[Tuple[str, float], ...]) -> ChangeDetector:
        return ChangeDetector(ChangeDetectorConfig(
            base_model=self.cfg.base_model, pooling=self.pooling, max_length=self.cfg.max_length,
            epochs=1, batch_size=int(self.cfg.extra.get("batch_size", self.cfg.batch_size)),
            learning_rate=self.cfg.learning_rate, device=self.cfg.device, seed=self.cfg.seed,
            source_head=self.label_set.source_head, operations=self.label_set.classes,
            class_weights=class_weights, logit_bias=(), focal_gamma=self.focal_gamma))

    # ---------- training ----------

    def _negatives(self, held_folds, n: int) -> List[ChangeExample]:
        """The mixed-negatives stage, opportunistic: skipped with a log line, not
        silently, when `BenchmarkData` is not present."""
        try:
            from retexo.datasets.dataset import BenchmarkData
            from retexo.datasets.negatives import NegativeBuilder

            data = BenchmarkData.load()
        except (FileNotFoundError, ImportError):
            return []
        held_texts = [" ".join(tokens) for r in self.validation for tokens in (r.source_tokens, r.reuse_tokens)]
        return NegativeBuilder(data, held_out=held_folds, exclude_texts=held_texts).build(n, for_training=True)

    def validation_loss(self, records: Sequence[Record]) -> Optional[float]:
        """The tagger's training loss on ``records`` (no gradient)."""
        if self.model is None:
            return None
        return self.model.evaluation_loss(examples_from_records(records, self.label_set, train_on=self.train_on))

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
           ) -> "Tagger":
        import torch

        gold = examples_from_records(train, self.label_set, train_on=self.train_on)
        if not gold:
            return self
        self.model = self._build_model(_class_weights(gold, self.label_set.classes))
        from retexo.baselines.early_stopping import EarlyStopping
        from retexo.baselines.schedule import SharedSchedule

        self.validation_level = self.label_set.score_level or "V1"
        stopper = EarlyStopping.for_method(self, log=log)
        gold_passes = stopper.max_epochs if stopper is not None else max(self.gold_passes, 1)
        rng = random.Random(self.cfg.seed)

        def one_pass(examples, frozen: bool, label: str, on_batch=None) -> None:
            for p in self.model._encoder.parameters():
                p.requires_grad = not frozen
            order = list(examples)
            rng.shuffle(order)
            self.model.fit(order, log=log, on_batch=on_batch)
            if log:
                log(f"[tagger] pass {label} ({'cold' if frozen else 'warm'}, {len(examples)} examples)")

        def stop_after(epoch: int) -> bool:
            return stopper is not None and not stopper.step(epoch, self.modules())

        if SharedSchedule.applies(self):
            # ours' schedule (the shared training set): one synthetic epoch in both orientations, its first
            # tenth with the encoder frozen (GECToR's cold steps for the fresh heads); then real passes of
            # the real pairs in both orientations, one negative per real pair and 600 fresh synthetic pairs
            schedule = SharedSchedule(self, train)
            synthetic = examples_from_records(schedule.synthetic_epoch(), self.label_set)
            cold = len(synthetic) // 10
            one_pass(synthetic[:cold], True, "synthetic, cold")
            monitor = stopper.monitor if stopper is not None else None
            one_pass(synthetic[cold:], False, "synthetic", monitor.progress("synthetic", int(self.cfg.extra.get("synthetic_evals", 4))) if monitor else None)
            if monitor is not None:
                monitor.evaluate("synthetic", fraction=1.0)
            if log:
                log(f"[tagger] shared schedule {schedule.summary()}")
            for i in range(gold_passes):
                one_pass(examples_from_records(schedule.real_pass(), self.label_set, train_on=self.train_on),
                         False, f"{i + 1}/{gold_passes}")
                if stop_after(i + 1):
                    break
        else:
            # the method's own recipe (external sets, runs without --shared-data)
            held_folds = tuple(f for f in (self.cfg.fold, self.cfg.dev_fold) if f is not None and f >= 0)
            n_negatives = int(len(gold) * self.neg_ratio)
            negatives = self._negatives(held_folds, n_negatives) if self.use_negatives and held_folds else []
            if self.use_negatives and log:
                log(f"[tagger] negatives: {len(negatives)} ({'BenchmarkData found' if negatives else 'not available, skipped'})")
            cold = min(self.cold_passes, gold_passes)   # cold passes are a prefix of gold_passes, never more
            epoch = 0
            for i in range(gold_passes):
                one_pass(gold, i < cold, f"{i + 1}/{gold_passes}")
                epoch += 1
                if stop_after(epoch):
                    break
            if stopper is not None:
                stopper.restore(self.modules())
            if negatives:
                # GECToR's last stage: one pass with negatives mixed in, kept only if the validation sample agrees
                one_pass(gold + negatives, False, "with negatives")
                if stopper is not None:
                    stopper.step(epoch + 1, self.modules())
        if stopper is not None:
            stopper.restore(self.modules())
            stopper.release()
        for p in self.model._encoder.parameters():
            p.requires_grad = True
        return self

    def modules(self) -> Dict[str, Any]:
        """The encoder and the two heads."""
        if self.model is None:
            return {}
        return {"encoder": self.model._encoder, "head": self.model._head, "source_head": self.model._source_head}

    # ---------- dev-fold dials ----------

    def _dev_macro_f1(self, dev: List[Record], preds: List[Prediction]) -> float:
        """The V1/mode rows use the shared scorer's macro F1; the iSTS label set has
        no level there (a converter-supplied field, not an edge collapse), so its
        dev criterion is a plain per-class F1 over `record.annotation["ists_type"]`."""
        if self.label_set.score_level is not None:
            return BaselineScorer.op_scores(dev, preds, self.label_set.score_level,
                                            require_source=False)["macro_f1"]
        tp: Dict[str, int] = {}; fp: Dict[str, int] = {}; fn: Dict[str, int] = {}
        for record, pred in zip(dev, preds):
            gold = self.label_set.token_labels(record)
            for g, p in zip(gold, pred.tags):
                if g == p:
                    tp[g] = tp.get(g, 0) + 1
                else:
                    fn[g] = fn.get(g, 0) + 1
                    fp[p] = fp.get(p, 0) + 1
        f1s = []
        for c in self.label_set.classes:
            t, fpc, fnc = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
            if t + fpc + fnc == 0:
                continue
            precision = t / (t + fpc) if (t + fpc) else 0.0
            recall = t / (t + fnc) if (t + fnc) else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        if self.model is None or not dev:
            return {}
        best_bias, best_p, best_f1 = self.keep_bias, self.min_change_p, -1.0
        for bias in KEEP_BIAS_GRID:
            for min_p in MIN_CHANGE_P_GRID:
                self.keep_bias, self.min_change_p = bias, min_p
                preds = [self.postprocess(r, p, {}) for r, p in zip(dev, self.predict(dev))]
                f1 = self._dev_macro_f1(dev, preds)
                if f1 > best_f1:
                    best_bias, best_p, best_f1 = bias, min_p, f1
        self.keep_bias, self.min_change_p = best_bias, best_p
        if log:
            log(f"[tagger] tuned keep_bias={best_bias} min_change_p={best_p} (dev macro F1 {best_f1:.3f})")
        return {"keep_bias": best_bias, "min_change_p": best_p}

    # ---------- inference ----------

    @staticmethod
    def gate(logits, keep_index: int, keep_bias: float, min_change_p: float) -> List[int]:
        """GECToR's two inference dials, pure: `keep_bias` on the keep class's logit
        before the softmax, then `min_change_p` as a pair-level gate -- if no
        word's non-keep probability clears it, every word in the pair is kept.

        Split out from `_predict_one_pass` so the dials are testable on hand-built
        logits, without a model (the project's own style check forbids
        `from_pretrained` in test files).
        """
        import torch

        bias = torch.zeros(logits.shape[-1], device=logits.device)
        bias[keep_index] = keep_bias
        probs = torch.softmax(logits + bias, dim=-1)
        non_keep = 1.0 - probs[:, keep_index]
        if not bool((non_keep > min_change_p).any()):
            return [keep_index] * logits.shape[0]
        return probs.argmax(dim=-1).tolist()

    def _predict_one_pass(self, examples: Sequence[ChangeExample]) -> List[List[str]]:
        """`ChangeDetector.predict_operations` plus `gate`, which needs per-token
        softmax probabilities `predict_operations` does not expose."""
        import torch

        model = self.model
        model._encoder.eval(); model._head.eval()
        classes = model._classes
        keep_index = classes.index(self.label_set.keep)
        out: List[List[str]] = []
        batch_size = model.config.batch_size
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                chunk = list(examples[start:start + batch_size])
                if not chunk:
                    continue
                batch, spans = model._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk], model.config.max_length)
                batch_on = {k: v.to(model.config.device) for k, v in batch.items()}
                hidden = model._encoder(**batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    tags = [self.label_set.keep] * len(example.target_tokens)
                    usable = [(w, a, b) for w, (a, b) in enumerate(spans[row]) if w < len(example.target_tokens)]
                    if usable:
                        st = torch.tensor([a for _, a, _ in usable])
                        en = torch.tensor([max(b, a + 1) for _, a, b in usable])
                        rw = torch.full_like(st, row)
                        vectors = model._word_vectors(hidden, rw, st, en)
                        logits = model._head(vectors)
                        chosen = self.gate(logits, keep_index, self.keep_bias, self.min_change_p)
                        for (word, _, _), guess in zip(usable, chosen):
                            tags[word] = classes[guess]
                    out.append(tags)
        model._encoder.train(); model._head.train()
        return out

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        if self.model is None:
            return [Prediction.empty(r.n_reuse) for r in records]
        examples = examples_from_records(records, self.label_set, train_on="all")
        tags_per_record = self._predict_one_pass(examples)
        dels_per_record = self.model.predict_source(examples) if self.label_set.source_head else None
        for i, record in enumerate(records):
            pred = Prediction.empty(record.n_reuse)
            tags = tags_per_record[i]
            pred.tags = list(tags)
            pred.frame = [1 if tag == "FRAME" else 0 for tag in tags]
            if dels_per_record is not None:
                pred.dels = list(dels_per_record[i])
            out.append(pred)
        return out

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """No decoder, no shared typer: the tags are the output. Returned unchanged."""
        return pred

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            return
        torch.save({"encoder_state": self.model._encoder.state_dict(), "head_state": self.model._head.state_dict(),
                   "source_head_state": self.model._source_head.state_dict() if self.model._source_head else None,
                   "keep_bias": self.keep_bias, "min_change_p": self.min_change_p}, path / "tagger.pt")

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> "Tagger":
        import torch

        method = cls(cfg)
        checkpoint = torch.load(Path(path) / "tagger.pt", map_location=cfg.device)
        method.model = method._build_model(())
        method.model._encoder.load_state_dict(checkpoint["encoder_state"])
        method.model._head.load_state_dict(checkpoint["head_state"])
        if method.model._source_head is not None and checkpoint["source_head_state"] is not None:
            method.model._source_head.load_state_dict(checkpoint["source_head_state"])
        method.keep_bias = checkpoint["keep_bias"]
        method.min_change_p = checkpoint["min_change_p"]
        return method


@BaselineRegistry.register
class TaggerV1(Tagger):
    """The V1 tagger: {COPY, MORPH, SUBST, INS, FRAME} on the reuse side, {KEEP, DEL} on the source side.

    Example:
        ```python
        method = TaggerV1(BaselineConfig(device="cpu"))
        method.fit(train, dev)
        pred = method.predict([record])[0]
        # python run_baseline.py --method tagger_v1 --fold 4 --extra gold_passes=4
        ```
    """

    name = "tagger_v1"
    early_stopping_capable = True
    validation_require_source = False
    label_set_name = "V1"


@BaselineRegistry.register
class TaggerMode(Tagger):
    """The direct mode tagger: {VERBATIM, ALLUSION, FRAME, NOMATCH}, no source head.

    Example:
        ```python
        method = TaggerMode(BaselineConfig(device="cpu"))
        method.fit(train, dev)
        # python run_baseline.py --method tagger_mode --fold 4 --extra gold_passes=4
        ```
    """

    name = "tagger_mode"
    early_stopping_capable = True
    validation_require_source = False
    label_set_name = "mode"
