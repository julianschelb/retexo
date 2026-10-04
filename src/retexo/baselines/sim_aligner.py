# retexo/baselines/sim_aligner.py
"""Note 12: the zero-supervision embedding aligner (SimAlign, awesome-align, Yousef).

Table 1's other non-neural-training competitor, "supervision: none": encode
each passage with a pretrained encoder, take one vector per word from layer
8 (mean over subwords), build the reuse-by-source similarity matrix, and
extract links from it. No fine-tuning, no fit. Three published recipes share
this shape and differ only in the similarity and the extraction rule --
SimAlign's ``(cos + 1) / 2`` and mutual-best Argmax, awesome-align's raw dot
product, row-softmax in both directions, and a threshold-intersection, and
Yousef's ``UGARIT/grc-alignment`` (XLM-R fine-tuned on Ancient Greek, Latin
and English through the awesome-align objectives) run through either recipe
-- so one class reports all three: :class:`Embedder` gets the vectors,
:class:`SimilarityMatrix` turns them into interface (I) rows, and the shared
decoder (``--decoder mutual`` for SimAlign's Argmax, ``--decoder intersect``
for awesome-align, the default bidirectional-average-then-Hungarian
otherwise) does the rest -- both already implement SimAlign's Argmax and
awesome-align's intersection over any interface (I) row, not just this
method's, so this module adds nothing to them.

Two encoders are reported side by side for Table 1 (``model=latin_bert``,
``model=ugarit``); ``laberta``, ``xlmr`` and ``mbert`` are appendix and
published-number-check settings, and any other HuggingFace repo id works
too. Latin BERT's vocabulary needs the tensor2tensor subword encoder
(``locisimiles.tokenization.latin_bert``, as ``aligner.py``'s
``ContextualAligner`` already uses); every other backbone goes through a
HuggingFace fast tokenizer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from retexo.aligners.agreement import AgreementDecoder
from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Record

#: cfg.extra["model"] short names, resolved to a HuggingFace repo id; any other
#: string is passed straight through.
MODEL_PRESETS = {
    "latin_bert": "ashleygong03/bamman-burns-latin-bert",
    "ugarit": "UGARIT/grc-alignment",
    "laberta": "bowphs/LaBerta",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}

#: The layer SimAlign (Figure 4), awesome-align (``--align_layer``) and Yousef all report as the peak.
LAYER = 8

#: Model names whose vocabulary needs the tensor2tensor encoder, not ``AutoTokenizer``.
LATIN_BERT_MARKERS = ("latin-bert", "latin_bert", "bamman")


# =============================================================================
# Embedder
# =============================================================================


class Embedder:
    """One contextual, layer-``layer`` vector per word, mean-pooled over subwords.

    Built lazily (no model download at construction) so unit tests can stub
    ``encode`` without a network. ``encode`` embeds one passage alone, the
    setting every published recipe uses; ``encode_pair`` embeds both passages
    in one joint forward pass (``[CLS] source [SEP] reuse [SEP]``), the
    ``joint=1`` appendix variant.

    Example:
        ```python
        embedder = Embedder("bert-base-multilingual-cased", device="cpu")
        vectors = embedder.encode(["ecce", "puella"])            # Tensor[2, hidden]
        source_vectors, reuse_vectors = embedder.encode_pair(source_words, reuse_words)
        ```
    """

    def __init__(self, model_name: str, *, layer: int = LAYER, device: str = "cpu", max_length: int = 256):
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.max_length = max_length
        from retexo.formulations.pair_encoding import is_latin_bert

        self.is_latin_bert = is_latin_bert(model_name)
        self._model = None
        self._tokenizer = None
        self._latin_encoder = None
        self._specials: Dict[str, int] = {}
        self._pair_encoder = None

    def _ensure_backend(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        self.torch = torch
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        if self.is_latin_bert:
            from locisimiles.tokenization.latin_bert import SubwordTextEncoder
            from retexo.formulations.pair_encoding import latin_bert_vocab

            self._latin_encoder = SubwordTextEncoder.from_file(latin_bert_vocab(self.model_name))
            self._specials = {s: i for i, s in enumerate(self._latin_encoder._subtokens)
                              if s in ("[CLS]", "[SEP]")}
        else:
            from transformers import AutoTokenizer

            try:
                # RoBERTa-style byte-level BPE (LaBerta, XLM-R) refuse pre-tokenized
                # input without this; inert on every other tokenizer.
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
            except (TypeError, ValueError):
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # ---------- one passage ----------

    def encode(self, words: Sequence[str], *, grad: bool = False):
        """One mean-pooled, layer-``self.layer`` vector per word of a single passage;
        ``grad=True`` keeps the graph for fine-tuning (``FineTunedSimAligner``)."""
        self._ensure_backend()
        if not words:
            return self.torch.zeros((0, self._model.config.hidden_size), device=self.device)
        return self._encode_latin_bert(words, grad) if self.is_latin_bert else self._encode_huggingface(words, grad)

    def _context(self, grad: bool):
        return self.torch.enable_grad() if grad else self.torch.no_grad()

    def _encode_latin_bert(self, words: Sequence[str], grad: bool = False):
        from retexo.core.normalize import normalize

        torch = self.torch
        ids = [self._specials["[CLS]"]]
        spans = []
        for word in words:
            from retexo.formulations.pair_encoding import latin_bert_pieces

            pieces = latin_bert_pieces(self._latin_encoder, word)
            spans.append((len(ids), len(ids) + len(pieces)))
            ids.extend(pieces)
        ids.append(self._specials["[SEP]"])
        ids = ids[: self.max_length]
        spans = [(a, b) for a, b in spans if b <= len(ids)]
        with self._context(grad):
            out = self._model(input_ids=torch.tensor([ids], device=self.device), output_hidden_states=True)
        hidden = out.hidden_states[self.layer][0]
        return torch.stack([hidden[a:b].mean(0) for a, b in spans]) if spans else hidden[:0]

    def _encode_huggingface(self, words: Sequence[str], grad: bool = False):
        torch = self.torch
        batch = self._tokenizer([list(words)], is_split_into_words=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        word_ids = batch.word_ids(0)
        with self._context(grad):
            out = self._model(**{k: v.to(self.device) for k, v in batch.items()}, output_hidden_states=True)
        hidden = out.hidden_states[self.layer][0]
        spans: Dict[int, List[int]] = {}
        for position, word in enumerate(word_ids):
            if word is not None:
                spans.setdefault(word, []).append(position)
        vectors = [hidden[positions].mean(0) for _, positions in sorted(spans.items())]
        return torch.stack(vectors) if vectors else hidden[:0]

    # ---------- both passages jointly ----------

    def encode_pair(self, source_words: Sequence[str], reuse_words: Sequence[str]):
        """Both passages from one joint forward pass: shared context, not the faithful setting."""
        self._ensure_backend()
        from retexo.formulations.pair_encoding import PairEncoder

        if self._pair_encoder is None:
            self._pair_encoder = PairEncoder.build(self.model_name)
        batch, reuse_spans = self._pair_encoder.encode([(list(source_words), list(reuse_words))], self.max_length)
        source_spans = self._pair_encoder.last_source_spans[0]
        torch = self.torch
        with torch.no_grad():
            out = self._model(**{k: v.to(self.device) for k, v in batch.items()}, output_hidden_states=True)
        hidden = out.hidden_states[self.layer][0]
        source_vectors = torch.stack([hidden[a:b].mean(0) for a, b in source_spans]) if source_spans else hidden[:0]
        reuse_vectors = torch.stack([hidden[a:b].mean(0) for a, b in reuse_spans[0]]) if reuse_spans[0] else hidden[:0]
        return source_vectors, reuse_vectors


# =============================================================================
# SimilarityMatrix
# =============================================================================


class SimilarityMatrix:
    """The reuse-by-source matrix of one pair, and every way of reading links from it.

    ``values[t, s]`` compares reuse word ``t`` and source word ``s`` -- a dot
    product (awesome-align) or a cosine (SimAlign) before ``rows``/``rev_rows``
    turn a side into interface (I) score rows: ``sim="dot"`` row-softmaxes
    each side (awesome-align's ``S_xy``/``S_yx``), ``sim="cos"`` renormalises
    ``(cos + 1) / 2`` per row (SimAlign). Both are one method, not two,
    because they differ only in which weights ``rows`` starts from.

    Example:
        ```python
        matrix = SimilarityMatrix.of(source_vectors, reuse_vectors, sim="dot")
        rows, rev_rows = matrix.rows(), matrix.rev_rows()
        pairs = matrix.itermax(n_max=2, alpha=0.9)          # SimAlign's Algorithm 1
        ```
    """

    def __init__(self, values: np.ndarray, sim: str = "dot", normalise: bool = True):
        if sim not in ("dot", "cos"):
            raise ValueError(f"unknown sim {sim!r}; expected 'dot' or 'cos'")
        self.values = values
        self.sim = sim
        #: With ``normalise=False`` a cosine row is left as ``(cos + 1) / 2`` per cell, so a
        #: null threshold theta is a cosine cut (cos >= 2 theta - 1) instead of a share of a
        #: row that sums to one: the null decision SimAlign lacks for non-parallel passages.
        self.normalise = normalise

    @classmethod
    def of(cls, source_vectors, reuse_vectors, *, sim: str = "dot", normalise: bool = True) -> "SimilarityMatrix":
        source = _as_numpy(source_vectors)
        reuse = _as_numpy(reuse_vectors)
        if sim == "cos":
            source = source / np.clip(np.linalg.norm(source, axis=-1, keepdims=True), 1e-9, None)
            reuse = reuse / np.clip(np.linalg.norm(reuse, axis=-1, keepdims=True), 1e-9, None)
        return cls(reuse @ source.T, sim=sim, normalise=normalise)

    @property
    def n_reuse(self) -> int:
        return self.values.shape[0]

    @property
    def n_source(self) -> int:
        return self.values.shape[1]

    # ---------- rows ----------

    def rows(self) -> Rows:
        """Interface (I): per reuse word, every source index, best first."""
        return self._rows(self.values)

    def rev_rows(self) -> Rows:
        """The transposed matrix's rows: per source word, every reuse index, best first."""
        return self._rows(self.values.T)

    def _rows(self, matrix: np.ndarray) -> Rows:
        if matrix.size == 0:
            return [[] for _ in range(matrix.shape[0])]
        if self.sim == "dot":
            shifted = matrix - matrix.max(axis=1, keepdims=True)
            weights = np.exp(shifted)
        else:
            weights = (matrix + 1.0) / 2.0
            if not self.normalise:
                return [sorted(((s, float(p)) for s, p in enumerate(row)), key=lambda x: -x[1]) for row in weights]
        totals = np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        probs = weights / totals
        return [sorted(((s, float(p)) for s, p in enumerate(row)), key=lambda x: -x[1]) for row in probs]

    # ---------- distortion and Itermax ----------

    def distorted(self, kappa: float) -> "SimilarityMatrix":
        """SimAlign's distortion (section 2.2): fade a match by how far it sits off the diagonal."""
        if kappa == 0.0 or self.values.size == 0:
            return self
        t = np.arange(self.n_reuse)[:, None] / max(self.n_reuse, 1)
        s = np.arange(self.n_source)[None, :] / max(self.n_source, 1)
        penalty = 1.0 - kappa * (s - t) ** 2
        return SimilarityMatrix(self.values * penalty, sim=self.sim, normalise=self.normalise)

    def itermax(self, n_max: int = 2, alpha: float = 0.9) -> Set[Tuple[int, int]]:
        """SimAlign's Algorithm 1: mutual-best, repeated, fading already-aligned rows/columns by ``alpha``."""
        aligned_t: Set[int] = set()
        aligned_s: Set[int] = set()
        pairs: Set[Tuple[int, int]] = set()
        if self.values.size == 0:
            return pairs
        for _ in range(n_max):
            row_aligned = np.array([t in aligned_t for t in range(self.n_reuse)])[:, None]
            col_aligned = np.array([s in aligned_s for s in range(self.n_source)])[None, :]
            mask = np.where(row_aligned & col_aligned, 0.0, np.where(row_aligned | col_aligned, alpha, 1.0))
            new_pairs = self._mutual_best(self.values * mask) - pairs
            if not new_pairs:
                break
            pairs |= new_pairs
            for t, s in new_pairs:
                aligned_t.add(t); aligned_s.add(s)
        return pairs

    @staticmethod
    def _mutual_best(matrix: np.ndarray) -> Set[Tuple[int, int]]:
        """Pairs that are each other's best; a zero row or column links nothing."""
        row_best = matrix.argmax(axis=1)
        col_best = matrix.argmax(axis=0)
        return {(t, int(s)) for t, s in enumerate(row_best) if matrix[t, s] > 0 and col_best[s] == t}


def _as_numpy(vectors) -> np.ndarray:
    return vectors.detach().cpu().numpy() if hasattr(vectors, "detach") else np.asarray(vectors)


# =============================================================================
# SimAligner
# =============================================================================


@BaselineRegistry.register
class SimAligner(Baseline):
    """The zero-supervision neural baseline of Table 1: no training, ``fit`` is a no-op.

    ``cfg.extra`` dials: ``model`` (a preset name -- ``latin_bert``,
    ``ugarit``, ``laberta``, ``xlmr``, ``mbert`` -- or any HuggingFace repo
    id; default ``latin_bert``), ``layer`` (8), ``sim`` (``dot``:
    awesome-align's row-softmax; ``cos``: SimAlign's ``(cos + 1) / 2``),
    ``joint`` (0: encode the two passages separately, the faithful setting
    every published tool uses; 1: one joint forward pass), ``kappa`` (0.0:
    SimAlign's positional distortion), ``null`` (``theta``: the shared
    decoder's threshold; ``entropy``: SimAlign's row/column entropy
    pre-filter, ``tau`` the 95th percentile of the dev fold's mutually-best
    edges' entropy), ``extract`` (``decoder``: the shared decoder, chosen by
    ``--decoder``; ``itermax``: SimAlign's own Algorithm 1, ``n_max`` 2 and
    ``alpha`` 0.9 unless overridden).

    Example:
        ```python
        method = SimAligner(BaselineConfig(device="cpu", extra={"model": "mbert", "sim": "cos"}))
        pred = method.predict([record])[0]
        # python run_baseline.py --method sim_aligner --fold 4 --extra model=latin_bert --decoder default
        # python run_baseline.py --method sim_aligner --set edinburgh --extra model=mbert,sim=dot,c=0.001 --decoder intersect
        ```
    """

    name = "sim_aligner"
    emits = "scores"
    trainable = False
    typer = "rule"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        model_name = str(cfg.extra.get("model", "latin_bert"))
        self.embedder = Embedder(MODEL_PRESETS.get(model_name, model_name),
                                 layer=int(cfg.extra.get("layer", LAYER)), device=cfg.device,
                                 max_length=cfg.max_length)
        self.sim = str(cfg.extra.get("sim", "dot"))
        self.normalise = bool(int(cfg.extra.get("norm", 1)))
        self.joint = bool(int(cfg.extra.get("joint", 0)))
        self.kappa = float(cfg.extra.get("kappa", 0.0))
        self.null = str(cfg.extra.get("null", "theta"))
        self.extract = str(cfg.extra.get("extract", "decoder"))
        self.entropy_tau: Optional[float] = None

    # ---------- dev-fold dials ----------

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """The entropy null's ``tau``: the 95th percentile of the dev fold's mutual-best edges' entropy."""
        if self.null != "entropy":
            return {}
        entropies: List[float] = []
        for record in dev:
            matrix = self._matrix(record)
            if not matrix.n_reuse or not matrix.n_source:
                continue
            kept = AgreementDecoder.mutual_argmax(matrix.rows(), matrix.rev_rows())
            entropies.extend(AgreementDecoder.entropy(row) for row in kept if row and row[0][0] >= 0)
        self.entropy_tau = float(np.percentile(entropies, 95)) if entropies else None
        if log:
            log(f"[sim_aligner] entropy tau tuned on dev: {self.entropy_tau}")
        return {}

    # ---------- inference ----------

    def _matrix(self, record: Record) -> SimilarityMatrix:
        if self.joint:
            source_vectors, reuse_vectors = self.embedder.encode_pair(record.source_tokens, record.reuse_tokens)
        else:
            source_vectors = self.embedder.encode(record.source_tokens)
            reuse_vectors = self.embedder.encode(record.reuse_tokens)
        matrix = SimilarityMatrix.of(source_vectors, reuse_vectors, sim=self.sim, normalise=self.normalise)
        return matrix.distorted(self.kappa)

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            pred = Prediction.empty(record.n_reuse)
            if not record.source_tokens or not record.reuse_tokens:
                out.append(pred)
                continue
            matrix = self._matrix(record)
            rows, rev_rows = matrix.rows(), matrix.rev_rows()
            if self.null == "entropy" and self.entropy_tau is not None:
                rows = AgreementDecoder.entropy_filter(rows, rev_rows, self.entropy_tau)
            pred.scores, pred.rev_scores = rows, rev_rows
            if self.extract == "itermax":
                n_max, alpha = int(self.cfg.extra.get("n_max", 2)), float(self.cfg.extra.get("alpha", 0.9))
                pred.meta["itermax_links"] = {t: s for t, s in matrix.itermax(n_max, alpha)}
            out.append(pred)
        return out

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """SimAlign's Itermax bypasses the shared decoder; every other setting uses it."""
        if "itermax_links" in pred.meta:
            from retexo.baselines.adapters import PredictionAdapter

            mapping = pred.meta.pop("itermax_links")
            pred.links = [mapping.get(t, -1) for t in range(record.n_reuse)]
            return PredictionAdapter.type_prediction(pred, record, self.typer, self.featurizer,
                                                     frame_rule=str(dials.get("frame_rule", "keyword")))
        return super().postprocess(record, pred, dials)


# =============================================================================
# The fine-tuned variant (awesome-align's supervised objective)
# =============================================================================


@BaselineRegistry.register
class FineTunedSimAligner(SimAligner):
    """The similarity aligner with its encoder fine-tuned on our links, then read like the frozen row.

    Awesome-align's supervised objective (Dou and Neubig 2021, section 3.3): for every annotated
    link (r, s) of a pair, the reuse word's similarity row, softmaxed over the source words, and
    the source word's column, softmaxed over the reuse words, should both put their mass on the
    other end of the link; the loss is the mean of the two cross-entropies over the pair's links.
    The similarity is the one the row extracts with (cosine, sharpened by ``temperature``, 0.1).
    Unlinked words and negatives carry no loss, as in awesome-align. Trained in ours' schedule on
    the shared training set, early stopped on the validation sample; ``predict`` is the frozen
    row's, on the fine-tuned encoder.

    Example:
        ```python
        # python run_baseline.py --method sim_aligner_ft --fold 4 --shared-data data/shared \
        #     --extra model=latin_bert,layer=12,sim=cos,norm=0,split_repair=1 --theta-grid 0.70:0.99:0.01
        ```
    """

    name = "sim_aligner_ft"
    trainable = True
    early_stopping_capable = True

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.temperature = float(cfg.extra.get("temperature", 0.1))
        self.pairs_per_step = int(cfg.extra.get("pairs_per_step", 16))
        self.max_epochs = int(cfg.extra.get("epochs", 3))            # without early stopping

    def modules(self) -> Dict[str, object]:
        return {"encoder": self.embedder._model}

    def validation_loss(self, records: Sequence[Record]) -> Optional[float]:
        """The alignment loss on ``records`` (no gradient), the mean over pairs with a linked word."""
        torch = self.embedder.torch
        losses = []
        with torch.no_grad():
            for record in records:
                loss = self.pair_loss(record)
                if loss is not None:
                    losses.append(float(loss))
        return sum(losses) / len(losses) if losses else None

    def pair_loss(self, record: Record):
        """The awesome-align supervised loss of one pair, or ``None`` when it has no link."""
        links = [(e.r, e.s) for e in record.links if e.r < record.n_reuse and e.s < record.n_source]
        if not links or not record.source_tokens or not record.reuse_tokens:
            return None
        torch = self.embedder.torch
        source = torch.nn.functional.normalize(self.embedder.encode(record.source_tokens, grad=True), dim=-1)
        reuse = torch.nn.functional.normalize(self.embedder.encode(record.reuse_tokens, grad=True), dim=-1)
        links = [(r, s) for r, s in links if r < reuse.shape[0] and s < source.shape[0]]    # truncated passages
        if not links:
            return None
        logits = reuse @ source.T / self.temperature
        rows = torch.tensor([r for r, _ in links], device=logits.device)
        cols = torch.tensor([s for _, s in links], device=logits.device)
        forward = torch.nn.functional.cross_entropy(logits[rows], cols)
        backward = torch.nn.functional.cross_entropy(logits.T[cols], rows)
        return (forward + backward) / 2

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
            ) -> "FineTunedSimAligner":
        import random

        from retexo.baselines.early_stopping import EarlyStopping
        from retexo.baselines.record import RecordCodec
        from retexo.baselines.schedule import SharedSchedule

        self.embedder._ensure_backend()
        torch = self.embedder.torch
        model = self.embedder._model
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate)
        schedule = SharedSchedule(self, train) if SharedSchedule.applies(self) else None
        stopper = EarlyStopping.for_method(self, log=log)
        epochs = stopper.max_epochs if stopper is not None else self.max_epochs
        rng = random.Random(self.cfg.seed)

        def run_epoch(records: List[Record], label: str, on_batch=None) -> None:
            records = list(records)
            rng.shuffle(records)
            model.train()
            total, n, pending = 0.0, 0, []
            for i, record in enumerate(records, 1):
                loss = self.pair_loss(record)
                if loss is not None:
                    pending.append(loss)
                if pending and (len(pending) >= self.pairs_per_step or i == len(records)):
                    step = torch.stack(pending).mean()
                    optimizer.zero_grad(); step.backward(); optimizer.step()
                    total += float(step.detach()); n += 1; pending = []
                    if on_batch is not None:
                        on_batch(i, len(records))
                        model.train()
            model.eval()
            if log:
                log(f"[sim_aligner_ft] epoch {label}: loss {total / max(n, 1):.4f} ({len(records)} pairs)")

        monitor = stopper.monitor if stopper is not None else None
        if schedule is not None:
            run_epoch(schedule.synthetic_epoch(), "synthetic", monitor.progress("synthetic", int(self.cfg.extra.get("synthetic_evals", 4))) if monitor else None)
            if monitor is not None:
                monitor.evaluate("synthetic", fraction=1.0)
            if log:
                log(f"[sim_aligner_ft] shared schedule {schedule.summary()}")
        real = list(train) + [RecordCodec.swapped(r) for r in train]
        for epoch in range(1, epochs + 1):
            run_epoch(schedule.real_pass() if schedule is not None else real, f"{epoch}/{epochs}")
            if stopper is not None and not stopper.step(epoch, self.modules()):
                break
        if stopper is not None:
            stopper.restore(self.modules())
            stopper.release()
        model.eval()
        return self

