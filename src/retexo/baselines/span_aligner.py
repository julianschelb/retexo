# retexo/baselines/span_aligner.py
"""Note 14: Nagata, Chousa and Nishino 2020's supervised span-prediction aligner.

Table 1's closest published *supervised* method: for every word of one
passage, ask a SQuAD v2.0 reading-comprehension question -- "where does this
word appear in the other passage, or nowhere?" The question is the whole
query passage with the query word delimited by two new tokens (``[Q]``,
``[/Q]``); the context is the other passage; two linear heads over the
context positions predict a start and an end, ``[CLS]`` stands for "no
answer." Both directions (reuse word asked against the source, source word
asked against the reuse) share one model; the shared decoder's bidirectional
average (``symmetrise_average``) is exactly the paper's own symmetrisation
(eq. 3) once each row already carries the best span's probability on every
word inside it, which is what ``rows="best"`` builds.

Two deliberate simplifications from the implementation note, both disclosed
here rather than silently taken: the best span is searched directly over
*word*-aligned subword ranges (the boundaries ``pair_encoding`` already
computes), not over every subword position with a separate word-inclusion
post-processing step -- the two coincide almost everywhere a real span
exists, and the note's own "longest token sequence strictly inside" rule
exists only to handle the rare disagreement; and training uses gold stage
only (every word of every training pair, both directions) -- the synthetic
pretraining stage (§Adaptation, a queries-of-queries corpus at roughly two
million questions per epoch) is not implemented, since both of the note's
own published-number checks already run with ``epochs_synthetic=0`` and it
was the dominant share of the note's three-GPU-hour-per-fold estimate.
``rows="marginal"`` (the Table 2 style variant) is likewise not implemented;
``rows="best"``, the faithful setting, is the only one this module builds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Edge, Record
from retexo.formulations.pair_encoding import PairEncoder, is_latin_bert, latin_bert_pieces

#: The two marker tokens delimiting the query word (Nagata's own ``¶``, split
#: into an open and a close marker so a repeated query word cannot be ambiguous).
Q_TOKEN, QEND_TOKEN = "[Q]", "[/Q]"

DEFAULT_MAX_WORDS = 3
DEFAULT_MAX_LENGTH = 320


# =============================================================================
# QueryResult
# =============================================================================


@dataclass
class QueryResult:
    """One query's answer: the best span's words (empty if null wins), its
    probability, and the null probability."""

    best_words: Tuple[int, ...]
    omega_best: float
    s_null: float


# =============================================================================
# SpanEncoder
# =============================================================================


class SpanEncoder:
    """Marks a query word with ``[Q]``/``[/Q]`` and encodes it against a context passage.

    Built lazily (no model download at construction) so unit tests can stub
    the backend. HuggingFace models go through ``pair_encoding``'s existing
    tokenizer path with the two markers registered as additional special
    tokens; Latin BERT's tensor2tensor vocabulary has no such mechanism, so
    its two marker ids are reserved past the end of the vocabulary, the way
    ``typed_pointer.py``'s ``enable_label_matching`` reserves its own.

    Example:
        ```python
        encoder = SpanEncoder("bert-base-multilingual-cased", device="cpu")
        batch, spans_b = encoder.encode_queries([(reuse_words, source_words, 0)])
        ```
    """

    def __init__(
        self, model_name: str, *, device: str = "cpu", max_length: int = DEFAULT_MAX_LENGTH
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.is_latin_bert = is_latin_bert(model_name)
        self._model = None
        self._pair_encoder: Optional[PairEncoder] = None
        self.q_id = 0
        self.qend_id = 0
        self.vocab_size = 0

    # ---------- lazy backend ----------

    @property
    def hidden_size(self) -> int:
        self._ensure_backend()
        return self._model.config.hidden_size

    def parameters(self):
        self._ensure_backend()
        return self._model.parameters()

    def train(self) -> None:
        self._ensure_backend()
        self._model.train()

    def eval(self) -> None:
        self._ensure_backend()
        self._model.eval()

    def _ensure_backend(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        self.torch = torch
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        if self.is_latin_bert:
            import huggingface_hub
            from locisimiles.tokenization.latin_bert import SubwordTextEncoder

            vocab = huggingface_hub.hf_hub_download(self.model_name, "vocab.txt")
            self._latin_encoder = SubwordTextEncoder.from_file(vocab)
            specials = {
                s: i
                for i, s in enumerate(self._latin_encoder._subtokens)
                if s in ("[CLS]", "[SEP]")
            }
            self._cls, self._sep = specials["[CLS]"], specials["[SEP]"]
            self.q_id, self.qend_id = (
                len(self._latin_encoder._subtokens),
                len(self._latin_encoder._subtokens) + 1,
            )
            self.vocab_size = self.qend_id + 1
            cls_id = self._cls
        else:
            self._pair_encoder = PairEncoder.build(self.model_name)
            added = self._pair_encoder.tokenizer.add_special_tokens(
                {"additional_special_tokens": [Q_TOKEN, QEND_TOKEN]}
            )
            self.vocab_size = len(self._pair_encoder.tokenizer)
            self.q_id, self.qend_id = self._pair_encoder.tokenizer.convert_tokens_to_ids(
                [Q_TOKEN, QEND_TOKEN]
            )
            cls_id = self._pair_encoder.tokenizer.cls_token_id
            if not added:
                return
        self._model.resize_token_embeddings(self.vocab_size)
        with torch.no_grad():  # the two new tokens start as [CLS]
            embeddings = self._model.get_input_embeddings().weight
            embeddings[self.q_id] = embeddings[cls_id]
            embeddings[self.qend_id] = embeddings[cls_id]

    # ---------- forward pass ----------

    def forward_hidden(self, batch: Dict):
        """The encoder's last hidden state for one already-built batch."""
        self._ensure_backend()
        return self._model(**batch).last_hidden_state

    # ---------- marking and batching ----------

    def encode_queries(self, items: Sequence[Tuple[Sequence[str], Sequence[str], int]]):
        """``(query_passage, context_passage, query_word_index)`` triples, heterogeneous across the batch.

        Returns ``(batch, spans_b)``: model inputs on ``self.device``, and
        per example the context passage's word spans (subword position
        ranges), unaffected by the marker insertion on the query side.
        """
        self._ensure_backend()
        if self.is_latin_bert:
            return self._encode_latin_bert(items)
        pairs = [(self._mark(list(query), i), list(context)) for query, context, i in items]
        batch, spans_b = self._pair_encoder.encode(pairs, self.max_length)
        return {k: v.to(self.device) for k, v in batch.items()}, spans_b

    @staticmethod
    def _mark(words: List[str], i: int) -> List[str]:
        return words[:i] + [Q_TOKEN, words[i], QEND_TOKEN] + words[i + 1 :]

    def _encode_latin_bert(self, items):
        torch = self.torch
        rows, types, spans_b = [], [], []
        for query, context, i in items:
            ids = [self._cls]
            for w in query[:i]:
                ids.extend(latin_bert_pieces(self._latin_encoder, w))
            ids.append(self.q_id)
            ids.extend(latin_bert_pieces(self._latin_encoder, query[i]))
            ids.append(self.qend_id)
            for w in query[i + 1 :]:
                ids.extend(latin_bert_pieces(self._latin_encoder, w))
            ids.append(self._sep)
            boundary = len(ids)
            row_spans = []
            for w in context:
                pieces = latin_bert_pieces(self._latin_encoder, w)
                if len(ids) + len(pieces) + 1 > self.max_length:
                    break
                row_spans.append((len(ids), len(ids) + len(pieces)))
                ids.extend(pieces)
            ids.append(self._sep)
            rows.append(ids[: self.max_length])
            types.append(([0] * boundary + [1] * (len(ids) - boundary))[: self.max_length])
            spans_b.append([s for s in row_spans if s[1] <= self.max_length])
        width = max(len(r) for r in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        token_types = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (row, row_types) in enumerate(zip(rows, types)):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention[i, : len(row)] = 1
            token_types[i, : len(row_types)] = torch.tensor(row_types)
        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention.to(self.device),
            "token_type_ids": token_types.to(self.device),
        }
        return batch, spans_b


# =============================================================================
# SpanHeads
# =============================================================================


class SpanHeads:
    """The SQuAD head: one linear layer, start and end logits from its two columns.

    Not a ``torch.nn.Module`` subclass on purpose (``formulations/token_classifier.py``'s
    house convention: a plain layer as an attribute, ``import torch`` lazily,
    so this module costs nothing at import time without torch installed).

    Example:
        ```python
        heads = SpanHeads(hidden_size=768)
        start_logits, end_logits = heads(hidden_states)
        ```
    """

    def __init__(self, hidden_size: int, device: str = "cpu"):
        import torch

        self.torch = torch
        self.layer = torch.nn.Linear(hidden_size, 2).to(device)

    def __call__(self, hidden):
        logits = self.layer(hidden)
        start, end = logits.split(1, dim=-1)
        return start.squeeze(-1), end.squeeze(-1)

    def parameters(self):
        return self.layer.parameters()

    def train(self) -> None:
        self.layer.train()

    def eval(self) -> None:
        self.layer.eval()


# =============================================================================
# span_scores: the best word-aligned span, and the null
# =============================================================================


def span_scores(
    start_logits, end_logits, spans_b: Sequence[Tuple[int, int]], max_words: int = DEFAULT_MAX_WORDS
) -> QueryResult:
    """The best span, searched over word-aligned subword ranges only (see the module
    docstring's first simplification), plus the null (``[CLS]``, position 0)."""
    import torch

    p_start = torch.softmax(start_logits, dim=-1)
    p_end = torch.softmax(end_logits, dim=-1)
    s_null = float(p_start[0] * p_end[0])
    if not spans_b:
        return QueryResult((), 0.0, s_null)
    best_words: Tuple[int, ...] = ()
    best_omega = -1.0
    for wi in range(len(spans_b)):
        for wj in range(wi, min(wi + max_words, len(spans_b))):
            k, end = spans_b[wi][0], spans_b[wj][1] - 1
            omega = float(p_start[k] * p_end[end])
            if omega > best_omega:
                best_omega, best_words = omega, tuple(range(wi, wj + 1))
    return QueryResult(best_words, best_omega, s_null)


# =============================================================================
# SpanAligner
# =============================================================================


@BaselineRegistry.register
class SpanAligner(Baseline):
    """The supervised span-prediction baseline of Table 1: trainable, both directions in one model.

    ``cfg.extra`` dials: ``max_answer_words`` (3), ``train_on`` (``all`` |
    ``sure``), ``encoder_lr`` (3e-5), ``heads_lr`` (1e-3), ``epochs`` (2),
    ``batch_size`` (12). Not implemented, disclosed in the module docstring:
    the synthetic pretraining stage (``epochs_synthetic`` is always 0) and
    ``rows="marginal"``.

    Example:
        ```python
        method = SpanAligner(BaselineConfig(device="cpu", extra={"model": "bert-base-multilingual-cased"}))
        method.fit(train, dev)
        pred = method.predict([record])[0]
        # python run_baseline.py --method span_aligner --set europarl_deen \\
        #     --base-model bert-base-multilingual-cased --extra epochs_synthetic=0 --decoder threshold
        ```
    """

    name = "span_aligner"
    early_stopping_capable = True
    emits = "scores"
    trainable = True
    typer = "rule"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        model_name = str(cfg.extra.get("model", cfg.base_model))
        self.encoder = SpanEncoder(
            model_name,
            device=cfg.device,
            max_length=int(cfg.extra.get("max_length", DEFAULT_MAX_LENGTH)),
        )
        self.heads: Optional[SpanHeads] = None
        self.max_answer_words = int(cfg.extra.get("max_answer_words", DEFAULT_MAX_WORDS))
        self.train_on = str(cfg.extra.get("train_on", cfg.train_on))
        self.encoder_lr = float(cfg.extra.get("encoder_lr", 3e-5))
        self.heads_lr = float(cfg.extra.get("heads_lr", 1e-3))
        self.epochs = int(cfg.extra.get("epochs", 2))
        self.batch_size = int(cfg.extra.get("batch_size", 12))

    # ---------- training data ----------

    @staticmethod
    def _targets(record: Record, *, reverse: bool, sure_only: bool) -> Dict[int, int]:
        """Word index (in the query side) -> its single gold target word (in the context side).

        Every word not in this map is a null question. Multi-word gold
        (SPLIT/MERGE) is reduced to its first edge -- disclosed in the module
        docstring's second simplification.
        """
        out: Dict[int, int] = {}
        for edge in record.links:
            if sure_only and not edge.sure:
                continue
            query, target = (edge.s, edge.r) if reverse else (edge.r, edge.s)
            out.setdefault(query, target)
        return out

    def queries_of(
        self, records: Sequence[Record]
    ) -> List[Tuple[List[str], List[str], int, Optional[Tuple[int, int]]]]:
        """One query per word of every record, both directions: ``(query_passage, context_passage, index, target)``."""
        sure_only = self.train_on == "sure"
        out = []
        for record in records:
            if not record.reuse_tokens or not record.source_tokens:
                continue
            fwd = self._targets(record, reverse=False, sure_only=sure_only)
            rev = self._targets(record, reverse=True, sure_only=sure_only)
            for t in range(record.n_reuse):
                target = (fwd[t], fwd[t]) if t in fwd else None
                out.append((record.reuse_tokens, record.source_tokens, t, target))
            for s in range(record.n_source):
                target = (rev[s], rev[s]) if s in rev else None
                out.append((record.source_tokens, record.reuse_tokens, s, target))
        return out

    # ---------- dev-fold dials ----------

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """A fixed ``cfg.extra["theta"]`` (the published checks' 0.4) seeds the driver's
        dial before its own dev-fold tuning runs; pass ``--no-tune`` to keep it fixed."""
        theta = self.cfg.extra.get("theta")
        return {"theta": float(theta)} if theta is not None else {}

    # ---------- training ----------

    def _query_loss(self, chunk):
        """Start plus end cross-entropy of one batch of questions ``(query, context, index, target)``."""
        import torch

        queries = [(q, c, i) for q, c, i, _ in chunk]
        batch, spans_b = self.encoder.encode_queries(queries)
        starts, ends = [], []
        for (_, _, _, target), spans in zip(chunk, spans_b):
            if target is None or target[0] >= len(spans) or target[1] >= len(spans):
                starts.append(0)
                ends.append(0)
            else:
                starts.append(spans[target[0]][0])
                ends.append(spans[target[1]][1] - 1)
        hidden = self.encoder.forward_hidden(batch)
        start_logits, end_logits = self.heads(hidden)
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(start_logits, torch.tensor(starts, device=self.cfg.device)) + loss_fn(
            end_logits, torch.tensor(ends, device=self.cfg.device)
        )

    def validation_loss(self, records: Sequence[Record]) -> Optional[float]:
        """The question loss on ``records`` (both directions, as trained; no gradient)."""
        import torch

        if self.heads is None:
            return None
        queries = self.queries_of(list(records))
        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(queries), self.batch_size):
                total += float(self._query_loss(queries[start : start + self.batch_size]))
                n += 1
        return total / n if n else None

    def fit(
        self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
    ) -> SpanAligner:
        import torch
        from transformers import get_linear_schedule_with_warmup

        examples = self.queries_of(train)
        if not examples:
            return self
        # ours' schedule when the run reads the shared training set: the questions already go both ways, so the
        # schedule hands over one orientation (the same count of oriented examples as ours)
        from retexo.baselines.schedule import SharedSchedule

        schedule = (
            SharedSchedule(self, train, both_directions_built_in=True)
            if SharedSchedule.applies(self)
            else None
        )
        synthetic = self.queries_of(schedule.synthetic_epoch()) if schedule is not None else []
        if schedule is not None:
            examples = self.queries_of(schedule.real_pass())
            if log:
                log(f"[span_aligner] shared schedule {schedule.summary()}")
        self.heads = SpanHeads(self.encoder.hidden_size, device=self.cfg.device)
        optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.heads.parameters(), "lr": self.heads_lr},
            ]
        )
        from retexo.baselines.early_stopping import EarlyStopping

        stopper = EarlyStopping.for_method(self, log=log)
        epochs = stopper.max_epochs if stopper is not None else self.epochs
        steps_per_epoch = max(1, -(-len(examples) // self.batch_size))
        total_steps = steps_per_epoch * epochs + -(-len(synthetic) // self.batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, max(1, total_steps // 10), total_steps
        )
        rng = random.Random(self.cfg.seed)

        def run_epoch(items, label: str, on_batch=None) -> None:
            items = list(items)
            rng.shuffle(items)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(items), self.batch_size):
                chunk = items[start : start + self.batch_size]
                loss = self._query_loss(chunk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.detach())
                n_batches += 1
                if on_batch is not None:
                    on_batch(start + len(chunk), len(items))
            if log:
                log(
                    f"[span_aligner] epoch {label}: loss {total_loss / max(n_batches, 1):.4f} ({len(items)} questions)"
                )

        self.encoder.train()
        self.heads.train()
        monitor = stopper.monitor if stopper is not None else None
        if synthetic:
            run_epoch(
                synthetic,
                "synthetic",
                monitor.progress("synthetic", int(self.cfg.extra.get("synthetic_evals", 4)))
                if monitor
                else None,
            )
            if monitor is not None:
                monitor.evaluate("synthetic", fraction=1.0)
        for epoch in range(epochs):
            if schedule is not None and epoch > 0:
                examples = self.queries_of(
                    schedule.real_pass()
                )  # a fresh synthetic draw every real pass
            run_epoch(examples, f"{epoch + 1}/{epochs}")
            if stopper is not None:
                keep_going = stopper.step(epoch + 1, self.modules())
                self.encoder.train()
                self.heads.train()
                if not keep_going:
                    break
        if stopper is not None:
            stopper.restore(self.modules())
            stopper.release()
        self.encoder.eval()
        self.heads.eval()
        return self

    def modules(self) -> Dict[str, Any]:
        """The fine-tuned encoder and the two span heads."""
        return {
            "encoder": self.encoder._model,
            "heads": self.heads.layer if self.heads is not None else None,
        }

    # ---------- inference ----------

    def _score_direction(
        self, query_words: Sequence[str], context_words: Sequence[str]
    ) -> List[QueryResult]:
        if not query_words or not context_words:
            return [QueryResult((), 0.0, 1.0) for _ in query_words]
        import torch

        items = [(query_words, context_words, i) for i in range(len(query_words))]
        out = []
        for start in range(0, len(items), max(self.batch_size, 1) * 4):
            chunk = items[start : start + max(self.batch_size, 1) * 4]
            batch, spans_b = self.encoder.encode_queries(chunk)
            with torch.no_grad():
                hidden = self.encoder.forward_hidden(batch)
                start_logits, end_logits = self.heads(hidden)
            for row in range(len(chunk)):
                out.append(
                    span_scores(
                        start_logits[row], end_logits[row], spans_b[row], self.max_answer_words
                    )
                )
        return out

    @staticmethod
    def _row(result: QueryResult) -> Rows:
        entries = [(w, result.omega_best) for w in result.best_words] + [(-1, result.s_null)]
        return sorted(entries, key=lambda x: -x[1])

    @staticmethod
    def _extra_edges(results: Sequence[QueryResult], *, reverse: bool) -> List[Edge]:
        """A multi-word best span becomes extra edges: SPLIT (one source, many reuse)
        when a reuse query's span covers several source words is not this case --
        that is the primary link plus extras sharing ``r``; the reverse direction's
        multi-word span shares ``s``."""
        out = []
        for index, result in enumerate(results):
            for word in result.best_words[1:]:
                out.append(
                    Edge(r=word, s=index, op="") if reverse else Edge(r=index, s=word, op="")
                )
        return out

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            pred = Prediction.empty(record.n_reuse)
            if not record.source_tokens or not record.reuse_tokens or self.heads is None:
                out.append(pred)
                continue
            fwd = self._score_direction(record.reuse_tokens, record.source_tokens)
            rev = self._score_direction(record.source_tokens, record.reuse_tokens)
            pred.scores = [self._row(r) for r in fwd]
            pred.rev_scores = [self._row(r) for r in rev]
            pred.extra = self._extra_edges(fwd, reverse=False) + self._extra_edges(
                rev, reverse=True
            )
            out.append(pred)
        return out

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        """The encoder's and heads' weights; the vocabulary and marker ids are
        rebuilt deterministically by ``load`` from the same ``model_name``."""
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder._ensure_backend()
        torch.save(
            {
                "model_name": self.encoder.model_name,
                "encoder_state": self.encoder._model.state_dict(),
                "heads_state": self.heads.layer.state_dict() if self.heads is not None else None,
            },
            path / "span_aligner.pt",
        )

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> SpanAligner:
        import torch

        checkpoint = torch.load(Path(path) / "span_aligner.pt", map_location=cfg.device)
        cfg.extra.setdefault("model", checkpoint["model_name"])
        method = cls(cfg)
        method.encoder._ensure_backend()
        method.encoder._model.load_state_dict(checkpoint["encoder_state"])
        method.heads = SpanHeads(method.encoder.hidden_size, device=cfg.device)
        if checkpoint["heads_state"] is not None:
            method.heads.layer.load_state_dict(checkpoint["heads_state"])
        return method
