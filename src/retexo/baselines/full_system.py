# retexo/baselines/full_system.py
"""Note 16: the preliminary champion as a method row, one switch per Table 2 row.

The typed pointer of note 15 **with** the 23-dimensional evidence vector (aligner
A), an **ensemble** with a second pointer trained on dense rewrites (aligner B,
cell probabilities averaged), a **bonus** ``beta`` on every cell a QLoRA script
writer (the rater of note 23) also proposes, one-to-one decoding at ``theta``,
**gated types** (the evidence's own name where a resource attests, the model's
head where it is silent), the frame head, and the **three rule repairs**
(spelling, substitution geometry, frame adjacency).

Table 2 strips one component at a time: ``beta=0`` (minus the rater),
``ensemble=0`` (A only), ``repairs=0``, ``evidence=0`` (A without evidence,
named by the rule typer); all four together is note 15 up to the decoder.

The class overrides ``postprocess``: decode the ensemble rows (the rater's
bonus is already in them), then the ensemble tag rule over A's and B's gated
tags, frames from A's head on unlinked words, deletions, and the repairs last.

    python run_baseline.py --method full_system --fold 4
    python run_baseline.py --method full_system --fold 4 --extra beta=0
    python run_baseline.py --method full_system --set multimwa_mtref --base-model bert-base-cased \\
        --extra evidence=0,gate=0,rater=none,typed=0,size=0,gold_passes=6,negatives=none
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry, labels
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Record
from retexo.baselines.typed_pointer import TypedPointerBaseline

#: The champion's dials (``attic/scripts/ensemble_dump.py``, ``run_e36.py``, ``repair.py``).
SYSTEM_DEFAULTS: Dict[str, Any] = {
    "ensemble": 1,
    "beta": 0.3,
    "theta": 0.45,
    "weight_a": 0.5,
    "repairs": 1,
    "evidence": 1,
    "gate": 1,
    "rater": "qlora",
    "b_dense_rate": 0.3,
    "b_mlm": 1,
    "b_seed_offset": 1,
    "top_k": 6,
    "frame_threshold": 0.2,
    "load_a": "",
    "load_b": "",
    "load_rater": "",
}

#: Dials of the rater's ``beta`` and the decoder's ``theta`` searched on the dev fold.
BETA_GRID = (0.0, 0.15, 0.3, 0.45)
THETA_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(7))

#: Keys of ``cfg.extra`` handed on to the two pointers (note 15's recipe dials).
POINTER_KEYS = (
    "typed",
    "size",
    "gold_passes",
    "negatives",
    "neg_ratio",
    "null_weight",
    "swap",
    "directions",
    "train_on",
    "pool_per_pass",
    "rare_min",
)


# =============================================================================
# The ensemble
# =============================================================================


class Ensemble:
    """Averaged cells, the rater's bonus, and the tag rule of ``ensemble_dump.py``.

    Example:
        ```python
        rows = Ensemble.rows(rows_a, rows_b, llm_links=[0, -1], weight_a=0.5, beta=0.3)
        ```
    """

    @staticmethod
    def rows(
        rows_a: Rows,
        rows_b: Optional[Rows],
        llm_links: Optional[Sequence[int]] = None,
        *,
        weight_a: float = 0.5,
        beta: float = 0.0,
        top_k: int = 6,
    ) -> Rows:
        """Per reuse word ``weight_a . p_A + (1 - weight_a) . p_B`` over the top
        entries of each, plus ``beta`` on the rater's cell; A alone when B is
        absent. The null keeps its averaged mass; the bonus never lands on it."""
        out: Rows = []
        for t, row_a in enumerate(rows_a):
            cand: Dict[int, float] = {}
            if rows_b is None:
                for s, p in row_a:
                    cand[int(s)] = cand.get(int(s), 0.0) + float(p)
            else:
                row_b = rows_b[t] if t < len(rows_b) else []
                for s, p in row_a:
                    cand[int(s)] = cand.get(int(s), 0.0) + weight_a * float(p)
                for s, p in row_b:
                    cand[int(s)] = cand.get(int(s), 0.0) + (1.0 - weight_a) * float(p)
            if (
                beta
                and llm_links is not None
                and t < len(llm_links)
                and llm_links[t] is not None
                and llm_links[t] >= 0
            ):
                cell = int(llm_links[t])
                cand[cell] = cand.get(cell, 0.0) + beta
            ranked = sorted(cand.items(), key=lambda x: -x[1])
            kept = ranked[:top_k]
            if not any(s == -1 for s, _ in kept):
                kept.append((-1, cand.get(-1, 0.0)))
            out.append([(int(s), round(float(v), 6)) for s, v in kept])
        return out

    @staticmethod
    def greedy_links(rows: Rows, theta: float) -> List[int]:
        """``ensemble_dump.py:64-69``: words in descending order of their best
        cell, a link iff that cell beats ``theta`` and its source is unused. The
        row itself decodes through the shared decoder (Hungarian, the null as a
        competing column); this rule is kept for the fold-4 regression against
        the stored champion dump."""
        scored = []
        for t, row in enumerate(rows):
            best = [(float(p), int(s)) for s, p in row if s >= 0]
            if best:
                v, s = max(best)
                scored.append((v, t, s))
        links, used = [-1] * len(rows), set()
        for v, t, s in sorted(scored, reverse=True):
            if v <= theta or s in used:
                continue
            used.add(s)
            links[t] = s
        return links

    @staticmethod
    def tag(
        t: int,
        s: int,
        source: Sequence[str],
        reuse: Sequence[str],
        links_a: Sequence[int],
        tags_a: Sequence[str],
        links_b: Optional[Sequence[int]],
        tags_b: Optional[Sequence[str]],
        same_lemma: bool,
    ) -> str:
        """A's tag where A made the same link, B's where only B did, NOP for
        identical or same-spelling forms, MORPH where the lemmas agree, SUBST
        otherwise (``ensemble_dump.py:71-80``)."""
        from retexo.aligners.sameness import SamenessPolicy
        from retexo.edit_typing.repair import Repairer

        if t < len(links_a) and links_a[t] == s and tags_a[t]:
            return tags_a[t]
        if (
            links_b is not None
            and tags_b is not None
            and t < len(links_b)
            and links_b[t] == s
            and tags_b[t]
        ):
            return tags_b[t]
        if SamenessPolicy.same_current(reuse[t], source[s]) or Repairer.spelling_key(
            reuse[t]
        ) == Repairer.spelling_key(source[s]):
            return "NOP"
        return "MORPH" if same_lemma else "SUBST"


class GatedTyper:
    """The evidence's own name where a resource attests, the model's where it is silent.

    Example:
        ```python
        typer = GatedTyper(featurizer)          # None: no evidence, the model's tags pass through
        tags = typer.gate(record, links, model_tags)
        ```
    """

    def __init__(self, featurizer=None):
        self.featurizer = featurizer

    def phi(self, record: Record, t: int, s: int) -> Optional[List[float]]:
        if self.featurizer is None:
            return None
        return list(
            self.featurizer(
                record.source_tokens[s],
                record.reuse_tokens[t],
                s,
                t,
                record.n_source,
                record.n_reuse,
            )
        )

    def same_lemma(self, record: Record, t: int, s: int) -> bool:
        from retexo.edit_typing.link_features import FEATURE_NAMES

        phi = self.phi(record, t, s)
        return bool(phi is not None and phi[FEATURE_NAMES.index("same_lemma")] > 0.5)

    def gate(self, record: Record, links: Sequence[int], model_tags: Sequence[str]) -> List[str]:
        from retexo.edit_typing.attest import Attester

        out = list(model_tags)
        if self.featurizer is None:
            return out
        for t, s in enumerate(links):
            if s is None or s < 0:
                continue
            said, attested = Attester.attest_type(self.phi(record, t, s))
            if attested:
                out[t] = said
        return out


# =============================================================================
# The method row
# =============================================================================


@BaselineRegistry.register
class FullSystem(Baseline):
    """ "Full system, ours": the pointer with evidence, the ensemble, the rater's
    bonus, gated types and the repairs.

    ``cfg.extra`` (``SYSTEM_DEFAULTS``): ``ensemble``, ``beta``, ``theta``,
    ``weight_a``, ``repairs``, ``evidence``, ``gate``, ``rater`` (``qlora`` |
    ``none``), ``b_dense_rate``, ``b_mlm``, ``frame_threshold``, ``load_a``,
    ``load_b``, ``load_rater`` (saved pieces to skip training); plus note 15's
    recipe dials, handed on to both pointers.

    Example:
        ```python
        method = FullSystem(cfg).fit(train, dev, log=print)
        dials = {"theta": 0.45, **method.tune(dev)}
        preds = [method.postprocess(r, p, dials) for r, p in zip(test, method.predict(test))]
        # python run_baseline.py --method full_system --fold 4 --extra rater=none
        ```
    """

    name = "full_system"
    emits = "scores"
    trainable = True
    typer = "own"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {
            **SYSTEM_DEFAULTS,
            **{k: v for k, v in cfg.extra.items() if k in SYSTEM_DEFAULTS},
        }
        self.beta = float(self.dials["beta"])
        self.aligner_a = TypedPointerBaseline(self._pointer_config("a"))
        self.aligner_b = (
            TypedPointerBaseline(self._pointer_config("b")) if int(self.dials["ensemble"]) else None
        )
        self.rater = None
        if str(self.dials["rater"]) == "qlora":
            from retexo.baselines.llm import QLORA_DEFAULTS, ScriptRater

            rater_dials = {k: v for k, v in cfg.extra.items() if k in QLORA_DEFAULTS}
            self.rater = ScriptRater(
                rater_dials,
                device=cfg.device,
                annotate="inline" if int(self.dials["evidence"]) else "none",
            )
        self.llm_links: Dict[str, List[int]] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        if not int(self.dials["evidence"]):
            self.typer = "rule"

    # ---------- the pieces ----------

    def _pointer_config(self, which: str) -> BaselineConfig:
        extra = {k: v for k, v in self.cfg.extra.items() if k in POINTER_KEYS}
        extra["evidence"] = int(self.dials["evidence"])
        if which == "b":
            extra["dense_rate"] = float(self.dials["b_dense_rate"])
            extra["mlm_subst"] = int(self.dials["b_mlm"])
        seed = self.cfg.seed + (int(self.dials["b_seed_offset"]) if which == "b" else 0)
        return replace(
            self.cfg, seed=seed, extra=extra, out=Path(self.cfg.out) / f"aligner_{which}"
        )

    def gated_typer(self) -> GatedTyper:
        return GatedTyper(
            self.aligner_a.evidence_featurizer()
            if int(self.dials["gate"]) and int(self.dials["evidence"])
            else None
        )

    # ---------- training ----------

    def fit(
        self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
    ) -> FullSystem:
        if str(self.dials["load_a"]):
            self.aligner_a.load_into(Path(str(self.dials["load_a"])), log=log)
        else:
            self.aligner_a.fit(train, dev, log=log)
        if self.aligner_b is not None:
            if str(self.dials["load_b"]):
                self.aligner_b.load_into(Path(str(self.dials["load_b"])), log=log)
            else:
                self.aligner_b.fit(train, dev, log=log)
        if self.rater is not None:
            if str(self.dials["load_rater"]):
                self.rater.load_adapter(Path(str(self.dials["load_rater"])))
            else:
                self.rater.train(train, Path(self.cfg.out) / "rater", log=log)
        return self

    # ---------- the dev fold ----------

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """``beta`` (and a seed for ``theta``) by token accuracy on the dev fold,
        from the pieces' cached rows; the driver tunes ``theta`` again on the
        rows ``predict`` then builds with the chosen ``beta``."""
        from retexo.baselines.decoder import BaselineDecoder
        from retexo.baselines.record import RecordInterface

        if self.rater is None or not dev:
            return {"theta": float(self.dials["theta"])}
        pieces = [self.pieces(r) for r in dev]
        gold = [RecordInterface.links_of(r)[0] for r in dev]
        best = (-1.0, self.beta, float(self.dials["theta"]))
        for beta, theta in itertools.product(BETA_GRID, THETA_GRID):
            right = total = 0
            for record, piece, g in zip(dev, pieces, gold):
                rows = Ensemble.rows(
                    piece["rows_a"],
                    piece["rows_b"],
                    self.llm_links.get(record.id),
                    weight_a=float(self.dials["weight_a"]),
                    beta=beta,
                    top_k=int(self.dials["top_k"]),
                )
                links = BaselineDecoder.decode_default(rows, theta=theta, n_source=record.n_source)
                right += sum(int(a == b) for a, b in zip(links, g))
                total += len(g)
            acc = right / max(total, 1)
            if acc > best[0]:
                best = (acc, beta, theta)
        self.beta = best[1]
        if log:
            log(f"[full_system] dev: beta {best[1]} theta {best[2]} (token accuracy {best[0]:.4f})")
        return {"theta": best[2], "beta": best[1]}

    # ---------- inference ----------

    def pieces(self, record: Record) -> Dict[str, Any]:
        """A's and B's rows and the rater's links for one record, computed once."""
        if record.id not in self._cache:
            self._compute([record])
        return self._cache[record.id]

    def _compute(self, records: Sequence[Record]) -> None:
        todo = [r for r in records if r.id not in self._cache]
        if not todo:
            return
        preds_a = self.aligner_a.predict(todo)
        preds_b = self.aligner_b.predict(todo) if self.aligner_b is not None else [None] * len(todo)
        if self.rater is not None and self.rater.model is not None:
            from retexo.baselines.llm import ScriptFormat

            for record, text in zip(todo, self.rater.generate(todo)):
                self.llm_links[record.id] = ScriptFormat.parse_words(
                    text, record.source_tokens, record.reuse_tokens
                )[0]
        for record, pa, pb in zip(todo, preds_a, preds_b):
            self._cache[record.id] = {
                "rows_a": pa.scores,
                "rev_a": pa.rev_scores,
                "rows_b": pb.scores if pb is not None else None,
            }

    def predict(self, records: List[Record]) -> List[Prediction]:
        self._compute(records)
        out = []
        for record in records:
            pred = Prediction.empty(record.n_reuse)
            piece = self._cache[record.id]
            if piece["rows_a"] is None:
                out.append(pred)
                continue
            pred.scores = Ensemble.rows(
                piece["rows_a"],
                piece["rows_b"],
                self.llm_links.get(record.id),
                weight_a=float(self.dials["weight_a"]),
                beta=self.beta,
                top_k=int(self.dials["top_k"]),
            )
            pred.meta["llm_links"] = self.llm_links.get(record.id)
            out.append(pred)
        return out

    def _own_links_and_tags(
        self,
        aligner: TypedPointerBaseline,
        record: Record,
        rows: Rows,
        dials: Dict[str, float],
        typer: GatedTyper,
    ) -> Tuple[List[int], List[str]]:
        """What one aligner says on its own: its rows decoded at ``theta``, its
        typer's fine tags at those links, gated by the evidence."""
        from retexo.baselines.decoder import BaselineDecoder

        links = BaselineDecoder.decode_default(
            rows, theta=float(dials.get("theta", 0.45)), n_source=record.n_source
        )
        example = aligner.examples_of([record])[0]
        if aligner.recipe.typed:
            tags = aligner.model.predict_typed([example], [links])[0]
        else:
            from retexo.baselines.adapters import PredictionAdapter

            probe = Prediction(
                links=list(links), tags=[""] * record.n_reuse, frame=[0] * record.n_reuse
            )
            tags = PredictionAdapter.type_prediction(
                probe, record, "rule", self.featurizer, frame_rule="none"
            ).tags
            tags = [labels.TO_LINK_TAG.get(x, x) for x in tags]
        tags = [tag if s >= 0 else "INS" for tag, s in zip(tags, links)]
        return links, typer.gate(record, links, tags)

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """Decode the ensemble rows, name by the ensemble rule over the gated
        tags, frames from A's head, deletions, then the repairs."""
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.edit_typing.repair import Repairer

        pred = PredictionAdapter.decode_prediction(pred, self.decoder, dials, record)
        if pred.scores is None or self.aligner_a.model is None:
            return pred
        piece = self.pieces(record)
        typer = self.gated_typer()
        links_a, tags_a = self._own_links_and_tags(
            self.aligner_a, record, piece["rows_a"], dials, typer
        )
        links_b = tags_b = None
        if self.aligner_b is not None and piece["rows_b"] is not None:
            links_b, tags_b = self._own_links_and_tags(
                self.aligner_b, record, piece["rows_b"], dials, typer
            )
        fine = []
        for t, s in enumerate(pred.links):
            if s < 0:
                fine.append("INS")
                continue
            fine.append(
                Ensemble.tag(
                    t,
                    s,
                    record.source_tokens,
                    record.reuse_tokens,
                    links_a,
                    tags_a,
                    links_b,
                    tags_b,
                    typer.same_lemma(record, t, s),
                )
            )
        example = self.aligner_a.examples_of([record])[0]
        model_a = self.aligner_a.model
        if self.aligner_a.recipe.typed:
            frame = model_a.predict_frames([example], [pred.links])[0]
            frame_p = self.aligner_a.frame_probabilities(example)
        else:
            head = model_a.predict_frames([example])[0]
            frame = [int(f) if s < 0 else 0 for f, s in zip(head, pred.links)]
            frame_p = None
        links = list(pred.links)
        if int(self.dials["repairs"]):
            links, fine, frame = Repairer.apply(
                record.source_tokens,
                record.reuse_tokens,
                links,
                fine,
                frame,
                frame_p=frame_p,
                frame_threshold=float(self.dials["frame_threshold"]),
            )
        pred.links = [int(s) for s in links]
        pred.tags = [
            labels.canonical(tag)[0] or "SUBST" if s >= 0 else ""
            for tag, s in zip(fine, pred.links)
        ]
        pred.frame = [int(f) if s < 0 else 0 for f, s in zip(frame, pred.links)]
        pred.frame_p = frame_p
        used = {s for s in pred.links if s >= 0}
        pred.dels = [0 if s in used else 1 for s in range(record.n_source)]
        pred.meta["tags_a"] = tags_a
        pred.meta["links_a"] = links_a
        return pred

    def label_records(
        self, records: List[Record], dials: Optional[Dict[str, float]] = None
    ) -> List[Record]:
        """The records with the system's edges as their ``pred`` (note 33's
        distillation teacher)."""
        from retexo.baselines.record import RecordInterface

        dials = dials or {"theta": self.dials["theta"]}
        out = []
        for record, pred in zip(records, self.predict(records)):
            pred = self.postprocess(record, pred, dials)
            edges = RecordInterface.edges_from(pred.links, pred.tags, pred.frame, pred.extra)
            out.append(replace(record, pred={"edges": edges, "frame": pred.frame}))
        return out

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        path = Path(path)
        self.aligner_a.save(path / "aligner_a")
        if self.aligner_b is not None:
            self.aligner_b.save(path / "aligner_b")
        if self.rater is not None and self.rater.model is not None:
            self.rater.model.save_pretrained(str(path / "rater" / "adapter"))

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> FullSystem:
        path = Path(path)
        method = cls(cfg)
        method.aligner_a.load_into(path / "aligner_a")
        if method.aligner_b is not None and (path / "aligner_b" / "modules.pt").exists():
            method.aligner_b.load_into(path / "aligner_b")
        if method.rater is not None and (path / "rater" / "adapter").exists():
            method.rater.load_adapter(path / "rater" / "adapter")
        return method
