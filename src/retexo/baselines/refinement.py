# retexo/baselines/refinement.py
"""Note 22: iterative refinement, a loop around the typed pointer.

Table 2's "refinement with k passes" row. Pass k sees the script of pass k - 1
as a per-token state (undecided, linked with the linked source word's vector,
declined, frame; source side consumed or free) through the pointer's own
``refine_state`` hook, trained on gold-derived states (random corruption, or
an easy-to-hard chain) with the model's own first pass rolled in with
probability ``rollin``; inference repeats until the script stops changing or a
cap is reached. The second design, the **grid refiner** (a small 2-D
convolution over the frozen base's logit grid, the evidence grid and the
previous script; ``retexo.refinement.grid_refiner``), is the ``grid``
variant. Both were built and closed in E36; this row measures them again in
the harness.

The class overrides ``postprocess``: decode pass one with the shared decoder,
then for k = 2 .. ``passes`` set the state from the previous links, re-score,
decode again; every pass's rows go to ``meta["passes"]``; the last pass names
the links and the frames.

    python run_baseline.py --method refined_pointer --fold 4 --extra refine=random,rollin=0.3,passes=2
    python run_baseline.py --method refined_pointer --fold 4 --extra passes=3,until_stable=1
    python run_baseline.py --method refined_pointer --fold 4 --extra design=grid,evidence=1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Record
from retexo.baselines.typed_pointer import TypedPointerBaseline

#: The dials of the two designs, E36's values.
REFINE_DEFAULTS: Dict[str, Any] = {
    "design": "state", "refine": "random", "rollin": 0.3, "flip": 0.2, "add": 0.2, "passes": 2, "until_stable": 0,
    "refiner_epochs": 3, "refiner_lr": 1e-3, "hidden": 32, "layers": 2, "state_dropout": 0.2, "refiner_flip": 0.1,
    "refiner_null_weight": 0.2, "rollin_rounds": 0,
}


@BaselineRegistry.register
class RefinedPointer(Baseline):
    """The typed pointer with a second look at its own script.

    ``cfg.extra`` (``REFINE_DEFAULTS``): ``design`` (``state`` | ``grid``),
    ``refine`` (``random`` | ``chain`` | ``gold``, the training states),
    ``rollin``, ``passes``, ``until_stable``, the grid refiner's dials, plus
    every dial of note 15 for the base (``load_base`` skips its training).

    Example:
        ```python
        method = RefinedPointer(cfg).fit(train, dev, log=print)
        pred = method.postprocess(record, method.predict([record])[0], {"theta": 0.45})
        pred.meta["settled_at"], len(pred.meta["passes"])
        ```
    """

    name = "refined_pointer"
    emits = "scores"
    trainable = True
    typer = "own"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {**REFINE_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in REFINE_DEFAULTS}}
        self.design = str(self.dials["design"])
        base_extra = {k: v for k, v in cfg.extra.items() if k not in REFINE_DEFAULTS}
        if self.design == "state":
            base_extra["refine"] = str(self.dials["refine"])
        from dataclasses import replace

        self.base = TypedPointerBaseline(replace(cfg, extra=base_extra))
        self.refiner = None                      # the grid design's convolution
        self._grids: Dict[str, Any] = {}

    # ---------- training ----------

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()) -> "RefinedPointer":
        self.base.fit(train, dev, log=log)
        model = self.base.model
        if model is None:
            return self
        if self.design == "state":
            model.refine_rollin = float(self.dials["rollin"])
            return self
        from retexo.refinement.grid_refiner import GridRefiner
        from retexo.baselines.record import RecordInterface

        grids = [self.grid_of(r) for r in train]
        golds = [(RecordInterface.links_of(r)[0], RecordInterface.links_of(r)[2]) for r in train]
        self.refiner = GridRefiner(grids[0].phi.shape[-1] if grids else 0, hidden=int(self.dials["hidden"]),
                                   layers=int(self.dials["layers"]), device=self.cfg.device)
        self.refiner.fit(grids, golds, device=self.cfg.device, epochs=int(self.dials["refiner_epochs"]),
                         lr=float(self.dials["refiner_lr"]), seed=self.cfg.seed, dropout=float(self.dials["state_dropout"]),
                         flip=float(self.dials["refiner_flip"]), null_weight=float(self.dials["refiner_null_weight"]),
                         rollin_rounds=int(self.dials["rollin_rounds"]), log=log)
        return self

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """``passes`` by token accuracy on the dev fold (1, 2, 3), at the fixed theta."""
        from retexo.baselines.record import RecordInterface

        if not dev or self.base.model is None or self.cfg.smoke or "passes" in self.cfg.extra:
            return {"passes": int(self.dials["passes"])} if "passes" in self.cfg.extra else {}   # an explicit pass count is kept
        theta = float(self.cfg.extra.get("theta", 0.45))
        best = (-1.0, int(self.dials["passes"]))
        preds = self.predict(dev)
        for passes in (1, 2, 3):
            right = total = 0
            for record, pred in zip(dev, preds):
                done = self.refine(record, Prediction(links=list(pred.links), tags=list(pred.tags), frame=list(pred.frame),
                                                      scores=pred.scores, rev_scores=pred.rev_scores),
                                   {"theta": theta}, passes=passes)
                gold = RecordInterface.links_of(record)[0]
                right += sum(int(a == b) for a, b in zip(done.links, gold)); total += len(gold)
            acc = right / max(total, 1)
            if acc > best[0]:
                best = (acc, passes)
        self.dials["passes"] = best[1]
        if log:
            log(f"[refined_pointer] dev: passes {best[1]} (token accuracy {best[0]:.4f})")
        return {"passes": best[1]}

    # ---------- the grid design's substrate ----------

    def grid_of(self, record: Record):
        from retexo.refinement.grid_refiner import PairGrid

        if record.id not in self._grids:
            example = self.base.examples_of([record])[0]
            words = self.base.model.predict_cells([example])[0]
            phi = getattr(example, "pair_features", None)
            self._grids[record.id] = PairGrid.from_cells(words, record.n_source, phi)
        return self._grids[record.id]

    # ---------- inference ----------

    def predict(self, records: List[Record]) -> List[Prediction]:
        """Pass one: the base's rows with an undecided state."""
        model = self.base.model
        if model is not None:
            for example in self.base.examples_of(records):
                object.__setattr__(example, "refine_state", None)
        return self.base.predict(records)

    def rows_at_pass(self, record: Record, links: Sequence[int], frame: Sequence[int]) -> Rows:
        """The base's rows with the state built from a previous pass's script."""
        from retexo.refinement.refine import State

        example = self.base.examples_of([record])[0]
        model = self.base.model
        if self.design == "grid":
            from retexo.refinement.grid_refiner import PairGrid

            grid = self.grid_of(record)
            state = State.from_script(links, frame, record.n_source)
            previous = grid.scores()
            loc, nulls = self.refiner.correct(grid, state, PairGrid.confidence(previous), self.cfg.device)
            return [[(int(s), float(p)) for s, p in row] for row in grid.scores(loc, nulls)]
        object.__setattr__(example, "refine_state", State.from_script(links, frame, record.n_source))
        rows = model.predict_alignment_scores([example])[0]
        object.__setattr__(example, "refine_state", None)
        return [[(int(s), float(p)) for s, p in row] for row in rows]

    def refine(self, record: Record, pred: Prediction, dials: Dict[str, float], *, passes: Optional[int] = None) -> Prediction:
        from retexo.baselines.adapters import PredictionAdapter

        model = self.base.model
        passes = int(self.dials["passes"]) if passes is None else passes
        until_stable = bool(int(self.dials["until_stable"]))
        pred = PredictionAdapter.decode_prediction(pred, self.decoder, dials, record)
        if model is None or pred.scores is None:
            return pred
        example = self.base.examples_of([record])[0]
        history: List[Rows] = [pred.scores]
        links = list(pred.links)
        frame = model.predict_frames([example], [links])[0] if self.base.recipe.typed else model.predict_frames([example])[0]
        frame = [int(f) if s < 0 else 0 for f, s in zip(frame, links)]
        settled = 0
        for k in range(2, passes + 1):
            rows = self.rows_at_pass(record, links, frame)
            probe = Prediction(links=[-1] * record.n_reuse, tags=[""] * record.n_reuse, frame=[0] * record.n_reuse,
                               scores=rows, rev_scores=None)
            probe = PredictionAdapter.decode_prediction(probe, self.decoder, dials, record)
            new_links = list(probe.links)
            new_frame = model.predict_frames([example], [new_links])[0] if self.base.recipe.typed else frame
            new_frame = [int(f) if s < 0 else 0 for f, s in zip(new_frame, new_links)]
            history.append(rows)
            if until_stable and new_links == links and new_frame == frame:
                settled = k
                links, frame = new_links, new_frame
                break
            links, frame = new_links, new_frame
        pred.scores = history[-1]
        pred.links = links
        pred.frame = frame
        pred.link_p = [float(dict(row).get(s, 0.0)) if s >= 0 else 0.0 for row, s in zip(history[-1], links)]
        pred.meta["passes"] = [[[int(s), round(float(p), 4)] for s, p in row[:4]] for row in history[-1]] if len(history) > 1 else []
        pred.meta["n_passes"] = len(history)
        pred.meta["settled_at"] = settled
        return pred

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """The passes, then the base's own naming at the final links."""
        from retexo.baselines.adapters import PredictionAdapter

        pred = self.refine(record, pred, dials)
        model = self.base.model
        if model is None or pred.scores is None:
            return pred
        example = self.base.examples_of([record])[0]
        if self.base.recipe.typed:
            pred.tags = [tag if s >= 0 else "" for tag, s in zip(model.predict_typed([example], [pred.links])[0], pred.links)]
            pred.frame_p = self.base.frame_probabilities(example)
        else:
            pred = PredictionAdapter.type_prediction(pred, record, self.base.typer, self.featurizer,
                                                     frame_rule=str(dials.get("frame_rule", "keyword")))
            pred.frame = [int(f) if s < 0 else 0 for f, s in zip(pred.frame, pred.links)]
        used = {s for s in pred.links if s >= 0}
        pred.dels = [0 if s in used else 1 for s in range(record.n_source)]
        return pred

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        self.base.save(Path(path) / "base")
        if self.refiner is not None:
            import torch

            torch.save({str(i): m.state_dict() for i, m in enumerate(self.refiner._submodules())}, Path(path) / "refiner.pt")

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> "RefinedPointer":
        method = cls(cfg)
        method.base.load_into(Path(path) / "base")
        if method.design == "state":
            method.base.model.refine_rollin = float(method.dials["rollin"])
        elif (Path(path) / "refiner.pt").exists():
            import torch
            from retexo.refinement.grid_refiner import GridRefiner

            state = torch.load(Path(path) / "refiner.pt", map_location=cfg.device)
            n_evidence = method.base.model.config.feature_dim
            method.refiner = GridRefiner(n_evidence, hidden=int(method.dials["hidden"]), layers=int(method.dials["layers"]),
                                         device=cfg.device)
            for i, m in enumerate(method.refiner._submodules()):
                m.load_state_dict(state[str(i)])
        return method
