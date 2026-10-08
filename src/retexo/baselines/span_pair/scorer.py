# baselines/span_pair/scorer.py
"""The span-pair method row: the dials, training, the pruning grid and the per-record decoding."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry, labels
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Record, RecordInterface
from retexo.baselines.span_pair.candidates import SpanEnumerator
from retexo.baselines.span_pair.constants import (
    ADAPT,
    FRAME,
    INS,
    NONE,
    QUOTE,
    SPAN_DEFAULTS,
    WORD_TYPES,
    Span,
)
from retexo.baselines.span_pair.decoding import Expander, Segmenter, SpanChoice
from retexo.baselines.span_pair.head import SpanPairHead
from retexo.baselines.span_pair.runs import RunReader
from retexo.baselines.span_pair.substrate import Substrate
from retexo.baselines.typed_pointer import TypedPointerBaseline
from retexo.core.normalize import normalize

# =============================================================================
# SpanPairScorer
# =============================================================================


@BaselineRegistry.register
class SpanPairScorer(Baseline):
    """ "Seq2Edits (span-pair scorer)": runs first, words inside.

    ``cfg.extra`` (``SPAN_DEFAULTS``): ``L_max``, ``eps``, ``K_pairs``, ``swap``,
    ``gold_passes``, ``train_on``, ``theta`` (a fixed seed for the driver's
    dial), ``word_decoder`` (1 = the shared decoder on the word rows, the
    ablation), ``hidden``, ``lr_heads``, ``temperature``.

    Example:
        ```python
        method = SpanPairScorer(cfg).fit(train, dev, log=print)
        preds = [method.postprocess(r, p, {"theta": 0.3}) for r, p in zip(test, method.predict(test))]
        # python run_baseline.py --method span_pair --fold 4 --extra gold_passes=6
        ```
    """

    name = "span_pair"
    early_stopping_capable = True
    emits = "scores"
    trainable = True
    typer = "own"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {**SPAN_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in SPAN_DEFAULTS}}
        if cfg.train_on and "train_on" not in cfg.extra:
            self.dials["train_on"] = cfg.train_on
        self.L_max = int(self.dials["L_max"])
        self.substrate = Substrate(
            cfg.base_model,
            device=cfg.device,
            max_length=cfg.max_length,
            temperature=float(self.dials["temperature"]),
        )
        #: the frozen untyped pointer whose word grid prunes the pairs (note 15's model, ``load_base``);
        #: without one the substrate's cosine grid prunes, which is much weaker
        self.pruner: Optional[TypedPointerBaseline] = None
        self._pruning_rows: Dict[str, Any] = {}
        self.head: Optional[SpanPairHead] = None
        self._options: Dict[str, Dict[str, Any]] = {}
        #: candidate recall of the gold pairs, counted during the first training pass
        self.diagnostics: Optional[Dict[str, int]] = None

    # ---------- data ----------

    def training_records(self, records: Sequence[Record]) -> List[Record]:
        """Both orientations when ``swap`` says so; possible links dropped on ``train_on = sure``."""
        from dataclasses import replace

        out = []
        for record in records:
            if str(self.dials["train_on"]) == "sure":
                record = replace(record, links=[e for e in record.links if e.sure])
            out.append(record)
            if str(self.dials["swap"]) == "double":
                out.append(self.swapped(record))
        return out

    @staticmethod
    def swapped(record: Record) -> Record:
        """The record with the sides exchanged and its links inverted (``RecordCodec.swapped``)."""
        from retexo.baselines.record import RecordCodec

        return RecordCodec.swapped(record)

    # ---------- the pruning grid ----------

    def load_pruner(self, log=None) -> None:
        """The frozen untyped pointer of ``load_base``, built from its saved recipe."""
        import json

        path = Path(str(self.dials["load_base"]))
        if self.pruner is not None or not str(self.dials["load_base"]):
            return
        from dataclasses import replace

        saved = json.loads((path / "recipe.json").read_text())
        cfg = replace(
            self.cfg,
            base_model=saved.get("base_model", self.cfg.base_model),
            extra=dict(saved["recipe"]),
        )
        self.pruner = TypedPointerBaseline.load(path, cfg)
        if log:
            log(f"[span_pair] pruning grid from the frozen pointer at {path}")

    def prune_ahead(self, records: Sequence[Record]) -> None:
        """The frozen pointer's rows for many records at once (batched inside it)."""
        todo = [r for r in records if r.id not in self._pruning_rows]
        if self.pruner is None or not todo:
            return
        for record, pred in zip(todo, self.pruner.predict(todo)):
            self._pruning_rows[record.id] = pred.scores

    def pruning_grid(self, record: Record, h_r, h_s):
        """``[n_t, n_s]`` probabilities: the frozen pointer's rows when there is
        one, else the substrate's row-softmaxed cosine."""
        import numpy as np

        n_t, n_s = h_r.shape[0], h_s.shape[0]
        if self.pruner is None:
            return self.substrate.grid(h_r, h_s).numpy()
        if record.id not in self._pruning_rows:
            self._pruning_rows[record.id] = self.pruner.predict([record])[0].scores
        rows = self._pruning_rows[record.id] or []
        grid = np.zeros((n_t, n_s), dtype=np.float32)
        for t, row in enumerate(rows[:n_t]):
            for s_, p in row:
                if 0 <= s_ < n_s:
                    grid[t, s_] = float(p)
        return grid

    # ---------- scoring one record ----------

    def options_for(
        self, record: Record, h_r, h_s, *, gold_pairs: Sequence[Tuple[Span, Span]] = ()
    ):
        """Every candidate option of the record with its logits, grouped by reuse span.
        Returns ``(spans, per_span)`` where ``per_span[span] = (entries, logits)``:
        entries ``(source span or None, tag)`` in the logits' order."""
        import torch

        n_t, n_s = h_r.shape[0], h_s.shape[0]
        grid = self.pruning_grid(record, h_r, h_s)
        pairs = SpanEnumerator.candidate_pairs(
            grid,
            record.reuse_tokens[:n_t],
            record.source_tokens[:n_s],
            L_max=self.L_max,
            eps=float(self.dials["eps"]),
            K_pairs=int(self.dials["K_pairs"]),
            gold_pairs=gold_pairs,
        )
        reuse_spans = SpanEnumerator.enumerate_spans(n_t, self.L_max)
        source_spans = sorted({s for _, s in pairs})
        r_index = {sp: i for i, sp in enumerate(reuse_spans)}
        s_index = {sp: i for i, sp in enumerate(source_spans)}
        vec_r = self.head.span_vectors(h_r, reuse_spans)
        vec_s = self.head.span_vectors(h_s, source_spans) if source_spans else None
        null_logits = self.head.score_nulls(vec_r)  # [n_spans, 2]
        pair_logits = None
        if pairs:
            ri = torch.tensor([r_index[r] for r, _ in pairs], device=vec_r.device)
            si = torch.tensor([s_index[s] for _, s in pairs], device=vec_r.device)
            dlen = torch.tensor(
                [(r[1] - r[0]) - (s[1] - s[0]) for r, s in pairs], device=vec_r.device
            )
            pos = torch.tensor(
                [
                    ((r[0] + r[1]) / 2 / max(n_t, 1)) - ((s[0] + s[1]) / 2 / max(n_s, 1))
                    for r, s in pairs
                ],
                device=vec_r.device,
                dtype=vec_r.dtype,
            )
            pair_logits = self.head.score_pairs(vec_r[ri], vec_s[si], dlen, pos)  # [n_pairs, 2]
        by_span: Dict[Span, List[Tuple[int, Span]]] = {}
        for k, (r, s) in enumerate(pairs):
            by_span.setdefault(r, []).append((k, s))
        per_span: Dict[Span, Tuple[List[Tuple[Optional[Span], str]], Any]] = {}
        for span in reuse_spans:
            entries: List[Tuple[Optional[Span], str]] = [(None, INS), (None, FRAME), (None, NONE)]
            parts = [null_logits[r_index[span]]]
            for k, s in by_span.get(span, []):
                entries.extend([(s, QUOTE), (s, ADAPT)])
                parts.append(pair_logits[k])
            per_span[span] = (entries, torch.cat(parts))
        return grid, per_span

    # ---------- training ----------

    def validation_loss(self, records: Sequence[Record]) -> Optional[float]:
        """The run loss on ``records`` per term, as trained (no gradient; the diagnostics are left alone)."""
        import torch

        if self.head is None:
            return None
        saved, self.diagnostics = self.diagnostics, None
        total, count = 0.0, 0
        batch = max(1, self.cfg.batch_size)
        try:
            with torch.no_grad():
                for start in range(0, len(records), batch):
                    chunk = list(records[start : start + batch])
                    for record, (h_r, h_s) in zip(
                        chunk,
                        self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk]),
                    ):
                        loss, n = self.loss(record, h_r, h_s)
                        if loss is not None:
                            total += float(loss)
                            count += n
        finally:
            self.diagnostics = saved
        return total / count if count else None

    def loss(self, record: Record, h_r, h_s) -> Tuple[Any, int]:
        """One cross-entropy per gold run over its options, plus the word-type
        cross-entropy inside ADAPT runs; ``(loss, n_terms)``."""
        import torch

        runs = RunReader.chunked(RunReader.gold_runs(record), self.L_max)
        n_t, n_s = h_r.shape[0], h_s.shape[0]
        runs = [r for r in runs if r.end < n_t and (r.source is None or r.source[1] < n_s)]
        gold_pairs = [
            (r.span, r.source)
            for r in runs
            if r.source is not None
            and r.length <= self.L_max
            and r.source[1] - r.source[0] + 1 <= self.L_max + 2
        ]
        _, per_span = self.options_for(record, h_r, h_s, gold_pairs=gold_pairs)
        # the candidate recall of the gold pairs, before the gold was appended (the note's diagnostic)
        if self.diagnostics is not None:
            grid = self.pruning_grid(record, h_r, h_s)
            found = set(
                SpanEnumerator.candidate_pairs(
                    grid,
                    record.reuse_tokens[:n_t],
                    record.source_tokens[:n_s],
                    L_max=self.L_max,
                    eps=float(self.dials["eps"]),
                    K_pairs=int(self.dials["K_pairs"]),
                )
            )
            self.diagnostics["gold_pairs"] += len(gold_pairs)
            self.diagnostics["gold_pairs_in_candidates"] += sum(1 for g in gold_pairs if g in found)
            self.diagnostics["runs_too_long"] += sum(
                1 for r in runs if r.source is not None and (r.span, r.source) not in gold_pairs
            )  # source stretch beyond L_max + 2
        losses = []
        gold_spans = set()
        for run in runs:
            if run.length > self.L_max:
                continue
            entries, logits = per_span[run.span]
            target = (run.source, run.tag) if run.source is not None else (None, run.tag)
            if target not in entries:
                continue
            gold_spans.add(run.span)
            index = torch.tensor(entries.index(target), device=logits.device)
            losses.append(
                torch.nn.functional.cross_entropy(logits.unsqueeze(0), index.unsqueeze(0))
            )
        # every other reuse span is NONE: without this the segmentation would compare the
        # trained scores of gold-shaped spans against untrained ones (the first MTRef run, F1 0.66)
        none_terms = []
        for span, (entries, logits) in per_span.items():
            if span in gold_spans:
                continue
            index = torch.tensor(entries.index((None, NONE)), device=logits.device)
            none_terms.append(
                torch.nn.functional.cross_entropy(logits.unsqueeze(0), index.unsqueeze(0))
            )
        if none_terms:
            losses.append(torch.stack(none_terms).mean() * max(1, len(gold_spans)))
        links, tags, _, _ = RecordInterface.links_of(record)
        t_idx, s_idx, y = [], [], []
        for run in runs:
            if run.tag != ADAPT:
                continue
            for t in range(run.start, run.end + 1):
                s = links[t]
                if s >= 0 and tags[t] in WORD_TYPES:
                    t_idx.append(t)
                    s_idx.append(s)
                    y.append(WORD_TYPES.index(tags[t]))
        if t_idx:
            logits = self.head.word_types(h_r[t_idx], h_s[s_idx])
            losses.append(
                torch.nn.functional.cross_entropy(logits, torch.tensor(y, device=logits.device))
            )
        if not losses:
            return None, 0
        return torch.stack(losses).sum(), len(losses)

    def fit(
        self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
    ) -> SpanPairScorer:
        import torch
        from transformers import get_linear_schedule_with_warmup

        records = self.training_records(train)
        if not records:
            return self
        # ours' schedule when the run reads the shared training set (the schedule builds both orientations)
        from retexo.baselines.schedule import SharedSchedule

        schedule = SharedSchedule(self, train) if SharedSchedule.applies(self) else None
        synthetic = schedule.synthetic_epoch() if schedule is not None else []
        if schedule is not None:
            records = schedule.real_pass()
            if log:
                log(f"[span_pair] shared schedule {schedule.summary()}")
        self.load_pruner(log=log)
        self.prune_ahead(
            (schedule.real + schedule.negatives + schedule.synthetic)
            if schedule is not None
            else records
        )
        self.head = SpanPairHead(
            self.substrate.hidden_size,
            L_max=self.L_max,
            hidden=int(self.dials["hidden"]),
            device=self.cfg.device,
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": list(self.substrate.parameters()), "lr": self.cfg.learning_rate},
                {"params": self.head.parameters(), "lr": float(self.dials["lr_heads"])},
            ]
        )
        passes = max(1, int(self.dials["gold_passes"]))
        if self.cfg.smoke:
            passes = min(passes, 2)
        from retexo.baselines.early_stopping import EarlyStopping

        stopper = EarlyStopping.for_method(self, log=log)
        if stopper is not None:
            passes = stopper.max_epochs
        batch_size = max(1, self.cfg.batch_size)
        steps = (passes * max(1, -(-len(records) // batch_size))) + -(-len(synthetic) // batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, max(1, int(float(self.dials["warmup"]) * steps)), steps
        )
        rng = random.Random(self.cfg.seed)

        def run_pass(items: List[Record], label: str, on_batch=None) -> None:
            items = list(items)
            rng.shuffle(items)
            total, n_batches = 0.0, 0
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                encoded = self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])
                terms, count = [], 0
                for record, (h_r, h_s) in zip(chunk, encoded):
                    loss, n = self.loss(record, h_r, h_s)
                    if loss is not None:
                        terms.append(loss)
                        count += n
                if not terms:
                    continue
                loss = torch.stack(terms).sum() / max(count, 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += float(loss.detach())
                n_batches += 1
                if on_batch is not None:
                    on_batch(start + len(chunk), len(items))
            if log:
                log(
                    f"[span_pair] pass {label}: loss {total / max(n_batches, 1):.4f} ({len(items)} records)"
                )

        self.substrate.train()
        self.head.train()
        if synthetic:
            self.diagnostics = None
            monitor = stopper.monitor if stopper is not None else None
            run_pass(
                synthetic,
                "synthetic",
                monitor.progress("synthetic", int(self.cfg.extra.get("synthetic_evals", 4)))
                if monitor
                else None,
            )
            if monitor is not None:
                monitor.evaluate("synthetic", fraction=1.0)
                self.substrate.train()
                self.head.train()
        for pass_no in range(1, passes + 1):
            self.diagnostics = (
                {"gold_pairs": 0, "gold_pairs_in_candidates": 0, "runs_too_long": 0}
                if pass_no == 1
                else None
            )
            if schedule is not None and pass_no > 1:
                records = schedule.real_pass()  # a fresh synthetic draw every real pass
            run_pass(records, f"{pass_no}/{passes}")
            if pass_no == 1 and log and self.diagnostics and self.diagnostics["gold_pairs"]:
                d = self.diagnostics
                log(
                    f"[span_pair] candidate recall of gold pairs {d['gold_pairs_in_candidates'] / d['gold_pairs']:.3f} "
                    f"({d['gold_pairs']} pairs; {d['runs_too_long']} gold runs beyond L_max left out)"
                )
            self.last_diagnostics = self.diagnostics or getattr(self, "last_diagnostics", None)
            if stopper is not None:
                self.diagnostics = None
                keep_going = stopper.step(pass_no, self.modules())
                self.substrate.train()
                self.head.train()
                if not keep_going:
                    break
        if stopper is not None:
            stopper.restore(self.modules())
            stopper.release()
        self.diagnostics = None
        self.substrate.eval()
        self.head.eval()
        return self

    def modules(self) -> Dict[str, Any]:
        """The substrate encoder and the span-pair head."""
        return {"substrate": self.substrate, "head": self.head}

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """A fixed ``theta`` seeds the driver's dial; the driver's own dev tuning
        then runs on the word rows (pass ``--no-tune`` to keep the seed)."""
        return {"theta": float(self.dials["theta"])}

    # ---------- inference ----------

    def _score(self, records: Sequence[Record]) -> None:
        import torch

        todo = [
            r for r in records if r.id not in self._options and r.source_tokens and r.reuse_tokens
        ]
        if not todo or self.head is None:
            return
        self.prune_ahead(todo)
        self.substrate.eval()
        self.head.eval()
        with torch.no_grad():
            for start in range(0, len(todo), max(1, self.cfg.batch_size)):
                chunk = todo[start : start + max(1, self.cfg.batch_size)]
                for record, (h_r, h_s) in zip(
                    chunk, self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])
                ):
                    grid, per_span = self.options_for(record, h_r, h_s)
                    grid = grid if not hasattr(grid, "numpy") else grid.numpy()
                    options = []
                    for span, (entries, logits) in per_span.items():
                        log_p = torch.log_softmax(logits, dim=-1).cpu().tolist()
                        for (source, tag), lp in zip(entries, log_p):
                            options.append(
                                SpanChoice(
                                    span, source, tag, float(lp), float(torch.exp(torch.tensor(lp)))
                                )
                            )
                    self._options[record.id] = {
                        "options": options,
                        "grid": grid,
                        "h_r": h_r.cpu(),
                        "h_s": h_s.cpu(),
                        "n_t": h_r.shape[0],
                        "n_s": h_s.shape[0],
                    }

    def word_rows(self, record: Record, options: Sequence[SpanChoice]) -> Rows:
        """Word-level marginals for the dump and the ``word_decoder`` ablation:
        each reuse word's mass over source words, summed over the spans that
        cover it, position-wise inside each pair."""
        n = record.n_reuse
        mass: List[Dict[int, float]] = [dict() for _ in range(n)]
        for option in options:
            a, b = option.reuse
            if option.tag == NONE:
                continue
            if option.source is None:
                for t in range(a, b + 1):
                    mass[t][-1] = mass[t].get(-1, 0.0) + option.p
                continue
            c, d = option.source
            for k, t in enumerate(range(a, b + 1)):
                s = c + min(k, d - c)
                mass[t][s] = mass[t].get(s, 0.0) + option.p
        rows: Rows = []
        for t in range(n):
            total = sum(mass[t].values()) or 1.0
            row = sorted(((s, p / total) for s, p in mass[t].items()), key=lambda x: -x[1])
            if not any(s == -1 for s, _ in row):
                row.append((-1, 0.0))
            rows.append([(int(s), round(float(p), 6)) for s, p in row])
        return rows

    def candidate_recall(self, records: List[Record]) -> Dict[str, Any]:
        """The share of the gold span pairs of ``records`` that the pruning keeps as candidates (the ceiling the
        pruning puts on the scorer), counted as in training but on any records with links (the test fold)."""
        import torch

        found_n = gold_n = 0
        batch = max(1, self.cfg.batch_size)
        with torch.no_grad():
            for start in range(0, len(records), batch):
                chunk = [r for r in records[start : start + batch] if r.links]
                if not chunk:
                    continue
                for record, (h_r, h_s) in zip(
                    chunk, self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])
                ):
                    runs = RunReader.chunked(RunReader.gold_runs(record), self.L_max)
                    n_t, n_s = h_r.shape[0], h_s.shape[0]
                    gold = [
                        (r.span, r.source)
                        for r in runs
                        if r.end < n_t
                        and r.source is not None
                        and r.source[1] < n_s
                        and r.length <= self.L_max
                        and r.source[1] - r.source[0] + 1 <= self.L_max + 2
                    ]
                    grid = self.pruning_grid(record, h_r, h_s)
                    found = set(
                        SpanEnumerator.candidate_pairs(
                            grid,
                            record.reuse_tokens[:n_t],
                            record.source_tokens[:n_s],
                            L_max=self.L_max,
                            eps=float(self.dials["eps"]),
                            K_pairs=int(self.dials["K_pairs"]),
                        )
                    )
                    gold_n += len(gold)
                    found_n += sum(1 for g in gold if g in found)
        return {
            "gold_pairs": gold_n,
            "gold_pairs_in_candidates": found_n,
            "recall": round(found_n / gold_n, 5) if gold_n else None,
        }

    def predict(self, records: List[Record]) -> List[Prediction]:
        self._score(records)
        out = []
        for record in records:
            pred = Prediction.empty(record.n_reuse)
            stored = self._options.get(record.id)
            if stored is None:
                out.append(pred)
                continue
            pred.scores = self.word_rows(record, stored["options"])
            pred.meta["n_options"] = len(stored["options"])
            out.append(pred)
        return out

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """The segmentation, then the expansion; or the shared decoder on the
        word rows when ``word_decoder`` is on."""
        import torch

        stored = self._options.get(record.id)
        if stored is None or pred.scores is None:
            return pred
        theta = float(dials.get("theta", 0.0))
        if int(self.dials["word_decoder"]):
            from retexo.baselines.adapters import PredictionAdapter

            pred = PredictionAdapter.decode_prediction(pred, self.decoder, dials, record)
            pred.tags = [
                (
                    "COPY"
                    if normalize(record.reuse_tokens[t]) == normalize(record.source_tokens[s])
                    else self._type_of(stored, t, s)
                )
                if s >= 0
                else ""
                for t, s in enumerate(pred.links)
            ]
            return pred
        chosen = Segmenter.segment(stored["options"], stored["n_t"], theta=theta)
        with torch.no_grad():
            links, tags, frame, extra, rows = Expander.expand(
                chosen,
                stored["grid"],
                record.reuse_tokens[: stored["n_t"]],
                record.source_tokens[: stored["n_s"]],
                lambda t, s: self._type_of(stored, t, s),
            )
        n = record.n_reuse
        pred.links = links + [-1] * (n - len(links))
        pred.tags = [
            labels.canonical(tag)[0] or "" if s >= 0 else ""
            for tag, s in zip(tags + [""] * (n - len(tags)), pred.links)
        ]
        pred.frame = frame + [0] * (n - len(frame))
        pred.extra = extra
        pred.link_p = [
            float(dict(row).get(s, 0.0)) if s >= 0 else 0.0 for row, s in zip(rows, pred.links)
        ] + [0.0] * (n - len(rows))
        pred.meta["runs"] = [(c.reuse, c.source, c.tag, round(c.p, 4)) for c in chosen]
        return pred

    def _type_of(self, stored: Dict[str, Any], t: int, s: int) -> str:
        import torch

        with torch.no_grad():
            logits = self.head.word_types(
                stored["h_r"][t : t + 1].to(self.cfg.device),
                stored["h_s"][s : s + 1].to(self.cfg.device),
            )
        return WORD_TYPES[int(logits.argmax(dim=-1))]

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.head is None:
            return
        torch.save(
            {
                "model_name": self.substrate.model_name,
                "encoder": self.substrate.state_dict(),
                "head": self.head.state_dict(),
                "dials": self.dials,
            },
            path / "span_pair.pt",
        )

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> SpanPairScorer:
        import torch

        method = cls(cfg)
        state = torch.load(Path(path) / "span_pair.pt", map_location=cfg.device)
        method.load_pruner()
        method.substrate.load_state_dict(state["encoder"])
        method.head = SpanPairHead(
            method.substrate.hidden_size,
            L_max=method.L_max,
            hidden=int(method.dials["hidden"]),
            device=cfg.device,
        )
        method.head.load_state_dict(state["head"])
        method.substrate.eval()
        method.head.eval()
        return method
