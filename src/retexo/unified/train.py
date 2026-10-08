# retexo/unified/train.py
"""The feasibility trainer for the unified pointer: one arm, one fold, every pass monitored.

Trains ``UnifiedPointer`` on the gold training folds (both orientations, no
synthetic stage: the "gold only" regime of Table 4, the honest floor for a
feasibility test; the dev fold is held out of training here, unlike the
harness's ``split_gold``, so that its curve measures generalisation) and after
every pass scores the dev fold twice -- through the structured decoder and
through the shared default decoder on the same rows -- on alignment (token
accuracy, sure/possible link P/R/F1, AER) and typing (V1 macro F1, per-class
F1, frame token F1). Loss components (the parent's location + name + frame
loss, the gate loss, the structured margin) are logged per pass, everything to
``metrics.jsonl`` under the run directory, so a run can be watched while it
trains. The test fold is scored every pass too, so the end can report both
the final test number and the one at the best dev pass (selection by dev);
every test pair is dumped with gold and both predictions for the error
analysis.

The training loop mirrors the parent's ``fit`` (same parameter groups and
learning rates) rather than calling it, so one optimiser persists across
passes and the per-pass evaluation can run between them.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.baselines.base import Prediction
from retexo.baselines.labels import Labels
from retexo.baselines.record import Record, RecordCodec, RecordInterface
from retexo.baselines.scorer import BaselineScorer
from retexo.unified.decoder import DecoderConfig
from retexo.unified.model import UnifiedPointer

#: The arms of the brainstorm note, section 5: (use_gate, structured_margin_weight).
ARMS: Dict[str, Tuple[bool, float]] = {
    "plain": (False, 0.0),  # arm 0 through the default decoder, arm 2a through the structured one
    "gate": (True, 0.0),  # arm 1 (+ 2a)
    "margin": (False, 0.5),  # arm 2b
    "gate_margin": (True, 0.5),  # arm 3
}


@dataclass(frozen=True)
class TrainConfig:
    """One run's settings; the defaults are the note-15 recipe minus the synthetic stage."""

    arm: str = "gate_margin"
    fold: int = 4
    base_model: str = "ashleygong03/bamman-burns-latin-bert"
    passes: int = 8
    batch_size: int = 32
    learning_rate: float = 2e-5
    typer_lr: float = 1e-3
    null_pointer_weight: float = 0.2
    max_length: int = 256
    seed: int = 1
    device: str = "cuda"
    both_orientations: bool = True
    theta: float = 0.45
    smoke: int = 0
    out: Path = Path("runs/unified")
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    gold_dir: Path = Path("data/gold_full")
    #: a fold-carrying record file (the blind V3 annotation) instead of the silver labels of ``gold_dir``
    gold_records: Optional[Path] = None
    #: the champion's 23-dimensional evidence vector in the typer and the locator
    #: (lemma equality, WordNet relation, vector cosine, name, morphology, position),
    #: from the Latin resources; off = the architecture-only regime of note 15
    evidence: bool = False
    featurizer_workers: int = 16


# =============================================================================
# Trainer
# =============================================================================


class UnifiedTrainer:
    """Build, train pass by pass, score after every pass, dump at the end.

    Example:
        ```python
        trainer = UnifiedTrainer(TrainConfig(arm="gate_margin", fold=4, out=Path("runs/unified_f4")))
        trainer.run(log=print)
        # python run_unified.py --arm gate_margin --fold 4 --out runs/unified_f4_gate_margin
        ```
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.model: Optional[UnifiedPointer] = None
        self.metrics: List[Dict] = []
        self._featurizer = None

    # ---------- data ----------

    def records(self) -> Tuple[List[Record], List[Record], List[Record]]:
        from retexo.baselines.splits import split_gold

        records = (
            RecordCodec.load(Path(self.config.gold_records))
            if self.config.gold_records
            else RecordCodec.gold_records(self.config.gold_dir)
        )
        dev_fold = (self.config.fold + 1) % 5
        train, dev, test = split_gold(records, self.config.fold, dev_fold)
        # the harness keeps the dev fold inside train (it only tunes a threshold on it);
        # here dev has to measure generalisation, so it is held out of training
        train = [r for r in train if r.fold != dev_fold]
        if self.config.smoke:
            train, dev, test = (
                train[: self.config.smoke],
                dev[: self.config.smoke],
                test[: self.config.smoke],
            )
        return train, dev, test

    def featurizer(self):
        """The evidence source: the Latin resources when ``evidence`` is on,
        else the identity-only stub (whose vector the model then does not read)."""
        if self._featurizer is None:
            if self.config.evidence:
                from retexo.edit_typing.link_features import LinkFeaturizer
                from retexo.resources import Resources

                self._featurizer = LinkFeaturizer(Resources(offline=True))
            else:
                from retexo.baselines.adapters import PredictionAdapter

                self._featurizer = PredictionAdapter._stub_featurizer()
        return self._featurizer

    def examples(self, records: Sequence[Record], *, both_orientations: bool = False):
        """Gold records as typed examples (fine types and frame labels from the
        gold's own labels, the evidence vector from ``featurizer``) -- and
        their swapped twins when asked."""
        from retexo.aligners.agreement import PairSwap
        from retexo.datasets.synthetic import fine_from_gold

        featurizer = self.featurizer()
        out = []
        for record in records:
            pair = RecordCodec._as_goldpair(record)
            example = fine_from_gold(pair, featurizer)
            out.append(example)
            if both_orientations:
                out.append(PairSwap.labelled(example))
        if self.config.evidence:
            # the typed pointer reads the evidence at *every* (reuse, source) cell
            # (``pair_features``), not only at the gold link that fine_from_gold fills;
            # each example, a swapped twin included, is featurised from its own tokens
            from retexo.datasets.synthetic import featurize_pairs

            featurize_pairs(out, workers=self.config.featurizer_workers)
        return out

    # ---------- model ----------

    def build(self) -> UnifiedPointer:
        from retexo.datasets import synthetic as syn
        from retexo.edit_typing.link_features import FEATURE_NAMES
        from retexo.formulations.change_detector import ChangeDetectorConfig

        c = self.config
        use_gate, margin = ARMS[c.arm]
        fine_ops = tuple(syn.FINE_OPERATIONS)
        fine_weights = tuple(
            (op, 1.0 if op == "NOP" else 2.0 if op == "MORPH" else 4.0) for op in fine_ops
        )
        model = UnifiedPointer(
            ChangeDetectorConfig(
                base_model=c.base_model,
                pooling="mean",
                device=c.device,
                epochs=1,
                batch_size=c.batch_size,
                learning_rate=c.learning_rate,
                seed=42 + c.seed,
                source_head=False,
                operations=(),
                pointer=True,
                pointer_style="dot",
                null_pointer_weight=c.null_pointer_weight,
                max_length=c.max_length,
                fine_operations=fine_ops,
                feature_dim=len(FEATURE_NAMES) if c.evidence else 0,
                use_link_features=c.evidence,
                fine_class_weights=fine_weights,
                frame_head=True,
                frame_positive_weight=3.0,
                typer_lr=c.typer_lr,
                typed_pointer=True,
            ),
            use_gate=use_gate,
            structured_margin_weight=margin,
            decoder=c.decoder,
        )
        model.frame_null_weight = 1.75
        model.name_loss_weight = 0.0
        self.model = model
        return model

    # ---------- evaluation ----------

    def predictions(
        self, records: Sequence[Record]
    ) -> Tuple[List[Prediction], List[Prediction], List[Dict]]:
        """Two predictions per record on the same rows -- structured decoder and
        default decoder -- plus a diagnostic row (gate probabilities, rows)."""
        from retexo.baselines.decoder import BaselineDecoder

        model = self.model
        examples = self.examples(records)
        rows_per = model.predict_alignment_scores(examples)
        structured = model.predict_structured(examples)
        gates = model.gate_probabilities(examples) if model.use_gate else [None] * len(examples)
        preds_struct, preds_default, diagnostics = [], [], []
        for record, example, rows, (links_s, frame_s), gate in zip(
            records, examples, rows_per, structured, gates
        ):
            links_d = BaselineDecoder.decode_default(rows, theta=self.config.theta)
            tags_s = model.predict_typed([example], [links_s])[0]
            tags_d = model.predict_typed([example], [links_d])[0]
            frame_d = model.predict_frames([example], [links_d])[0]
            preds_struct.append(self._prediction(record, links_s, tags_s, frame_s, rows))
            preds_default.append(self._prediction(record, links_d, tags_d, frame_d, rows))
            # the full rows (every candidate, four decimals), so the structured decoder
            # can be re-run offline on a finished run without the model
            diagnostics.append(
                {
                    "gate": [[round(x, 4) for x in g] for g in gate] if gate is not None else None,
                    "rows": [[[s, round(p, 4)] for s, p in row] for row in rows],
                }
            )
        return preds_struct, preds_default, diagnostics

    @staticmethod
    def _prediction(
        record: Record, links: Sequence[int], tags: Sequence[str], frame: Sequence[int], rows
    ) -> Prediction:
        n = record.n_reuse
        pred = Prediction.empty(n)
        pred.links = [int(s) if s is not None else -1 for s in list(links)[:n]] + [-1] * (
            n - len(links)
        )
        canon = []
        for t, tag in enumerate(list(tags)[:n]):
            op, _ = Labels.canonical(tag)
            canon.append(
                op if pred.links[t] >= 0 and op else ("" if pred.links[t] < 0 else "SUBST")
            )
        pred.tags = canon + [""] * (n - len(canon))
        pred.frame = [
            int(f) if s < 0 else 0
            for f, s in zip(list(frame)[:n] + [0] * (n - len(frame)), pred.links)
        ]
        pred.scores = rows
        return pred

    @staticmethod
    def score(records: Sequence[Record], preds: Sequence[Prediction]) -> Dict[str, float]:
        gold_links = [RecordInterface.links_of(r)[0] for r in records]
        link = BaselineScorer.link_prf(records, preds)
        ops = BaselineScorer.op_scores(records, preds, "V1")
        per = {c: round(ops["per_class"][c]["F1"], 4) for c in ("COPY", "MORPH", "SUBST", "INS")}
        frame = BaselineScorer.frame_span_prf(records, preds)
        tp = fp = fn = 0
        for r, p in zip(records, preds):
            g = RecordInterface.links_of(r)[2]
            for gf, pf in zip(g, p.frame):
                tp += int(gf and pf)
                fp += int(pf and not gf)
                fn += int(gf and not pf)
        frame_tok = 2 * tp / max(2 * tp + fp + fn, 1)
        return {
            "token_acc": round(
                BaselineScorer.token_accuracy_from_links([p.links for p in preds], gold_links), 4
            ),
            "link_p": round(link["precision"], 4),
            "link_r": round(link["recall"], 4),
            "link_f1": round(link["f1"], 4),
            "aer": round(link["aer"], 4),
            "op_macro_v1": round(ops["macro_f1"], 4),
            **{f"f1_{k.lower()}": v for k, v in per.items()},
            "frame_span_f1": round(frame["F1"], 4),
            "frame_token_f1": round(frame_tok, 4),
        }

    # ---------- training ----------

    def run(self, *, log=print) -> Dict:
        import torch

        c = self.config
        c.out.mkdir(parents=True, exist_ok=True)
        (c.out / "config.json").write_text(
            json.dumps(
                {
                    **asdict(c),
                    "out": str(c.out),
                    "gold_dir": str(c.gold_dir),
                    "gold_records": str(c.gold_records),
                },
                indent=1,
            )
        )
        train, dev, test = self.records()
        log(
            f"[unified] arm {c.arm} fold {c.fold}: train {len(train)} dev {len(dev)} test {len(test)}"
            f"{' (smoke)' if c.smoke else ''}; both orientations {c.both_orientations}; evidence {c.evidence}"
        )
        examples = self.examples(train, both_orientations=c.both_orientations)
        model = self.build()
        params_encoder = (
            list(model._encoder.parameters())
            + list(model._pointer_source.parameters())
            + list(model._pointer_target.parameters())
            + [model._pointer_null, model._frame_null]
        )
        fresh = (
            list(model._typer.parameters())
            + list(model._loc_mlp.parameters())
            + list(model._pair_head.parameters())
            + list(model._frame_head.parameters())
            + list(model._extra_fresh_parameters())
        )
        optimizer = torch.optim.AdamW(
            [{"params": params_encoder, "lr": c.learning_rate}, {"params": fresh, "lr": c.typer_lr}]
        )
        rng = random.Random(c.seed)
        metrics_path = c.out / "metrics.jsonl"
        metrics_path.write_text("")
        started = time.time()
        for pass_no in range(1, c.passes + 1):
            model._encoder.train()
            model._typer.train()
            model._loc_mlp.train()
            order = list(examples)
            rng.shuffle(order)
            totals = {"loss": 0.0, "gate": 0.0, "margin": 0.0}
            n = 0
            for start in range(0, len(order), c.batch_size):
                chunk = order[start : start + c.batch_size]
                loss = model._typed_loss(chunk)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                parts = getattr(model, "last_loss_parts", {})
                totals["loss"] += float(loss.item())
                totals["gate"] += parts.get("gate", 0.0)
                totals["margin"] += parts.get("margin", 0.0)
                n += 1
            mean = {k: round(v / max(n, 1), 4) for k, v in totals.items()}
            preds_s, preds_d, _ = self.predictions(dev)
            row = {
                "pass": pass_no,
                "elapsed_s": round(time.time() - started, 1),
                **{f"loss_{k}": v for k, v in mean.items()},
                "dev_structured": self.score(dev, preds_s),
                "dev_default": self.score(dev, preds_d),
            }
            # the test fold scored every pass as well, so the end can report the test
            # number at the best dev pass (selection by dev, never by test)
            test_s, test_d, _ = self.predictions(test)
            row["test_structured"], row["test_default"] = (
                self.score(test, test_s),
                self.score(test, test_d),
            )
            self.metrics.append(row)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            s, d = row["dev_structured"], row["dev_default"]
            log(
                f"[unified] pass {pass_no}/{c.passes} loss {mean['loss']} (gate {mean['gate']}, margin {mean['margin']}) | "
                f"dev structured: acc {s['token_acc']} f1 {s['link_f1']} macro {s['op_macro_v1']} subst {s['f1_subst']} frame {s['frame_token_f1']} | "
                f"default: acc {d['token_acc']} f1 {d['link_f1']} macro {d['op_macro_v1']} subst {d['f1_subst']} | {row['elapsed_s']}s"
            )
        preds_s, preds_d, diagnostics = self.predictions(test)
        best = (
            max(self.metrics, key=lambda r: r["dev_structured"]["op_macro_v1"])
            if self.metrics
            else None
        )
        final = {
            "arm": c.arm,
            "fold": c.fold,
            "passes": c.passes,
            "n_train": len(train),
            "n_dev": len(dev),
            "n_test": len(test),
            "test_structured": self.score(test, preds_s),
            "test_default": self.score(test, preds_d),
            "best_dev_pass": best["pass"] if best else None,
            "test_structured_at_best_dev": best["test_structured"] if best else None,
            "test_default_at_best_dev": best["test_default"] if best else None,
            "dev_curve": self.metrics,
            "wall_clock_s": round(time.time() - started, 1),
        }
        (c.out / "result.json").write_text(json.dumps(final, indent=1))
        self.dump(test, preds_s, preds_d, diagnostics, c.out / "predictions.jsonl")
        t_s, t_d = final["test_structured"], final["test_default"]
        log(
            f"[unified] TEST structured: acc {t_s['token_acc']} link_f1 {t_s['link_f1']} macro {t_s['op_macro_v1']} "
            f"subst {t_s['f1_subst']} frame {t_s['frame_token_f1']} | default: acc {t_d['token_acc']} link_f1 {t_d['link_f1']} "
            f"macro {t_d['op_macro_v1']} subst {t_d['f1_subst']} | {final['wall_clock_s']}s"
        )
        if best:
            b = best["test_structured"]
            log(
                f"[unified] TEST at best dev pass {best['pass']}: structured acc {b['token_acc']} link_f1 {b['link_f1']} "
                f"macro {b['op_macro_v1']} subst {b['f1_subst']} frame {b['frame_token_f1']}"
            )
        return final

    @staticmethod
    def dump(
        records: Sequence[Record],
        preds_s: Sequence[Prediction],
        preds_d: Sequence[Prediction],
        diagnostics: Sequence[Dict],
        path: Path,
    ) -> None:
        with path.open("w") as handle:
            for record, ps, pd, diag in zip(records, preds_s, preds_d, diagnostics):
                g_links, g_tags, g_frame, _ = RecordInterface.links_of(record)
                handle.write(
                    json.dumps(
                        {
                            "id": record.id,
                            "source": record.source_tokens,
                            "reuse": record.reuse_tokens,
                            "gold": {"links": g_links, "tags": g_tags, "frame": g_frame},
                            "structured": {"links": ps.links, "tags": ps.tags, "frame": ps.frame},
                            "default": {"links": pd.links, "tags": pd.tags, "frame": pd.frame},
                            "gate": diag["gate"],
                            "rows": diag["rows"],
                        }
                    )
                    + "\n"
                )
