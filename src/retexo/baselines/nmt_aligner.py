# retexo/baselines/nmt_aligner.py
"""Note 13: alignment read off a translation model, and an alignment layer on top.

The "pairs only" neural row: fine-tune an encoder-decoder (PhilTa for Latin,
mT5 for the checks) to rewrite the source passage into the reuse, on passage
pairs without word labels, and read the alignment off where the model attends
while writing each reuse word.

- **SHIFT-ATT** (Chen et al. 2020): the cross-attention at the decoder step
  whose *input* is the word to be aligned (row ``i + 1`` for reuse word ``i``,
  the naive reading takes row ``i``), heads averaged, at the layer the two
  directions agree on best (Chen's agreement criterion on the dev fold), under
  forced decoding of the real reuse.
- **The alignment layer** (Zenkel, Wuebker & DeNero 2020): one attention head
  on top of the frozen model (keys from the source input embeddings plus the
  encoder states, queries from the decoder states at ``i + 1``) with an explicit
  **null key**, optionally an unmasked self-attention over the decoder states
  (``full_context``), trained with the guided alignment loss on the model's own
  symmetrised SHIFT-ATT links, optionally with the contiguity loss.

Both directions (source to reuse, reuse to source); the shared decoder
symmetrises the rows. Emits interface (I) rows only; types come from the
shared typer.

    python run_baseline.py --method nmt_aligner --fold 4 --extra epochs=3
    python run_baseline.py --method nmt_aligner --set wpt_enfr --max-train 100000 --decoder gdf \\
        --extra backbone=google/mt5-base,reading=layer,epochs=1
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Record

#: The dials and the note's values.
NMT_DEFAULTS: Dict[str, Any] = {
    "backbone": "bowphs/PhilTa",
    "reading": "layer",
    "epochs": 3,
    "lr": 3e-4,
    "batch_size": 16,
    "layer_updates": 10000,
    "layer_lr": 1e-3,
    "lam": 0.0,
    "full_context": 0,
    "null": "key",
    "include_unlabeled": 1,
    "load_fwd": "",
    "load_bwd": "",
    "load_layer": "",
    "layer_fwd": -1,
    "layer_bwd": -1,
    "dev_pairs": 1000,
}


# =============================================================================
# Reading attention
# =============================================================================


class AttentionReader:
    """The two readings of a cross-attention tensor, and the subword pooling.

    Example:
        ```python
        rows = AttentionReader.read(att, tgt_word_ids, src_word_ids, n_t, n_s, shift=True)
        ```
    """

    @staticmethod
    def read(
        attention,
        tgt_word_ids: Sequence[Optional[int]],
        src_word_ids: Sequence[Optional[int]],
        n_t: int,
        n_s: int,
        *,
        shift: bool = True,
    ):
        """``attention`` is ``[heads, dec_len, src_len]`` for one pair (decoder
        position 0 the start token, position ``k`` the input ``y_{k-1}``);
        ``tgt_word_ids[k]`` names the reuse word of target piece ``k`` (labels,
        without the start token), ``src_word_ids[j]`` the source word of source
        piece ``j``. Returns ``[n_t, n_s]`` probabilities: rows of a word's pieces
        averaged, columns of a word's pieces summed, then normalised. SHIFT reads
        decoder row ``k + 1`` for target piece ``k``; naive reads row ``k``."""
        import numpy as np

        att = np.asarray(attention, dtype=np.float32).mean(axis=0)  # [dec_len, src_len]
        dec_len = att.shape[0]
        out = np.zeros((n_t, n_s), dtype=np.float32)
        counts = np.zeros(n_t, dtype=np.float32)
        for k, word in enumerate(tgt_word_ids):
            row = k + 1 if shift else k
            if word is None or word >= n_t or row >= dec_len:
                continue
            pooled = np.zeros(n_s, dtype=np.float32)
            for j, s_word in enumerate(src_word_ids):
                if s_word is not None and s_word < n_s and j < att.shape[1]:
                    pooled[s_word] += att[row, j]
            out[word] += pooled
            counts[word] += 1
        out = out / np.maximum(counts, 1)[:, None]
        sums = out.sum(axis=1, keepdims=True)
        return np.where(sums > 0, out / np.maximum(sums, 1e-9), out)

    @staticmethod
    def rows(matrix) -> Rows:
        """Interface (I) rows from a ``[n_t, n_s]`` probability matrix; the null
        gets what the row lacks to sum to one (the pieces the encoder truncated)."""
        out: Rows = []
        for row in matrix:
            entries = [(int(s), float(p)) for s, p in enumerate(row) if p > 0]
            null = max(0.0, 1.0 - sum(p for _, p in entries))
            entries.append((-1, null))
            out.append(sorted(entries, key=lambda x: -x[1]))
        return out


class LayerSelector:
    """Chen's agreement criterion: the pair of layers whose extracted
    alignments agree best between the two directions.

    Example:
        ```python
        l_fwd, l_bwd = LayerSelector.select(fwd_links_by_layer, bwd_links_by_layer)
        ```
    """

    @staticmethod
    def mutual_aer(fwd: Sequence[Sequence[int]], bwd: Sequence[Sequence[int]]) -> float:
        """AER of the forward links (reuse to source) against the backward links
        (source to reuse) read as the reference, over a list of pairs."""
        hits = n_f = n_b = 0
        for links_f, links_b in zip(fwd, bwd):
            back = {(s, t) for s, t in enumerate(links_b) if t >= 0}
            fore = {(s, t) for t, s in enumerate(links_f) if s >= 0}
            hits += len(fore & back)
            n_f += len(fore)
            n_b += len(back)
        return 1.0 - 2 * hits / max(n_f + n_b, 1)

    @classmethod
    def select(
        cls,
        fwd_by_layer: Sequence[Sequence[Sequence[int]]],
        bwd_by_layer: Sequence[Sequence[Sequence[int]]],
    ) -> Tuple[int, int]:
        best = (2.0, 0, 0)
        for lf, fwd in enumerate(fwd_by_layer):
            for lb, bwd in enumerate(bwd_by_layer):
                aer = cls.mutual_aer(fwd, bwd)
                if aer < best[0]:
                    best = (aer, lf, lb)
        return best[1], best[2]


# =============================================================================
# The translation model
# =============================================================================


class Seq2SeqWrapper:
    """A fine-tuned encoder-decoder and its forced-decoding attentions.

    Example:
        ```python
        model = Seq2SeqWrapper("google/mt5-base", device="cuda")
        model.fit(pairs, epochs=1, lr=3e-4, batch_size=16, log=print)
        matrices = model.attentions(pairs, layer=4)          # one [n_t, n_s] per pair
        ```
    """

    def __init__(self, model_name: str, *, device: str = "cpu", max_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    # ---------- Model access ----------

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    @property
    def n_layers(self) -> int:
        self._ensure()
        return int(
            self._model.config.num_decoder_layers
            if hasattr(self._model.config, "num_decoder_layers")
            else self._model.config.num_layers
        )

    @property
    def hidden_size(self) -> int:
        self._ensure()
        return int(self._model.config.d_model)

    def parameters(self):
        self._ensure()
        return self._model.parameters()

    def state_dict(self):
        self._ensure()
        return self._model.state_dict()

    def load_state_dict(self, state) -> None:
        self._ensure()
        self._model.load_state_dict(state)

    # ---------- encoding ----------

    def encode(self, pairs: Sequence[Tuple[Sequence[str], Sequence[str]]]):
        """Tokenised source and target of a batch, with word ids per piece."""
        self._ensure()
        tok = self._tokenizer
        src = tok(
            [list(s) for s, _ in pairs],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tgt = tok(
            [list(t) for _, t in pairs],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        src_words = [src.word_ids(i) for i in range(len(pairs))]
        tgt_words = [tgt.word_ids(i) for i in range(len(pairs))]
        labels = tgt["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        return src, labels, src_words, tgt_words

    # ---------- training ----------

    def fit(
        self,
        pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
        *,
        epochs: int = 1,
        lr: float = 3e-4,
        batch_size: int = 16,
        seed: int = 1,
        log=None,
        tag: str = "seq2seq",
        stopper=None,
    ) -> Seq2SeqWrapper:
        """Teacher-forced fine-tuning; with a ``stopper`` (``early_stopping.EarlyStopping`` on the
        validation pairs' loss) ``epochs`` is its maximum and the best epoch's weights are restored."""
        import torch
        from transformers import get_linear_schedule_with_warmup

        self._ensure()
        pairs = [p for p in pairs if p[0] and p[1]]
        if not pairs:
            return self
        if stopper is not None:
            epochs = stopper.max_epochs
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
        steps = epochs * max(1, -(-len(pairs) // batch_size))
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
        rng = random.Random(seed)
        order = list(pairs)
        self._model.train()
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            total, n = 0.0, 0
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                src, labels, _, _ = self.encode(chunk)
                out = self._model(
                    input_ids=src["input_ids"].to(self.device),
                    attention_mask=src["attention_mask"].to(self.device),
                    labels=labels.to(self.device),
                )
                optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += float(out.loss.detach())
                n += 1
                if log and n % 500 == 0:
                    log(f"[{tag}] epoch {epoch} step {n}: loss {total / n:.4f}")
            if log:
                log(
                    f"[{tag}] epoch {epoch}/{epochs}: loss {total / max(n, 1):.4f} over {len(order)} pairs"
                )
            if stopper is not None:
                keep_going = stopper.step(epoch, {"model": self._model})
                self._model.train()
                if not keep_going:
                    break
        if stopper is not None:
            stopper.restore({"model": self._model})
            stopper.release()
        self._model.eval()
        return self

    def loss_on(
        self, pairs: Sequence[Tuple[Sequence[str], Sequence[str]]], *, batch_size: int = 16
    ) -> float:
        """Mean teacher-forced loss over ``pairs`` (the early-stopping score of the translation stage)."""
        import torch

        self._ensure()
        pairs = [p for p in pairs if p[0] and p[1]]
        if not pairs:
            return 0.0
        self._model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                src, labels, _, _ = self.encode(pairs[start : start + batch_size])
                out = self._model(
                    input_ids=src["input_ids"].to(self.device),
                    attention_mask=src["attention_mask"].to(self.device),
                    labels=labels.to(self.device),
                )
                total += float(out.loss) * len(pairs[start : start + batch_size])
                n += len(pairs[start : start + batch_size])
        return total / max(n, 1)

    # ---------- forced decoding ----------

    def forward_batch(self, chunk, *, attentions: bool = False, hidden: bool = False):
        import torch

        src, labels, src_words, tgt_words = self.encode(chunk)
        labels_in = labels.clone()
        labels_in[labels_in == -100] = self._tokenizer.pad_token_id
        with torch.no_grad():
            out = self._model(
                input_ids=src["input_ids"].to(self.device),
                attention_mask=src["attention_mask"].to(self.device),
                labels=labels_in.to(self.device),
                output_attentions=attentions,
                output_hidden_states=hidden,
            )
        return out, src, labels, src_words, tgt_words

    def attentions_all_layers(
        self,
        pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
        *,
        batch_size: int = 16,
        shift: bool = True,
    ) -> List[List[Any]]:
        """Per layer, one ``[n_t, n_s]`` matrix per pair."""
        self._ensure()
        self._model.eval()
        per_layer: List[List[Any]] = [[] for _ in range(self.n_layers)]
        for start in range(0, len(pairs), batch_size):
            chunk = list(pairs[start : start + batch_size])
            out, src, labels, src_words, tgt_words = self.forward_batch(chunk, attentions=True)
            for layer, att in enumerate(out.cross_attentions):
                att = att.float().cpu().numpy()  # [B, heads, dec_len, src_len]
                for i, (s_words, t_words) in enumerate(zip(src_words, tgt_words)):
                    n_s, n_t = len(chunk[i][0]), len(chunk[i][1])
                    per_layer[layer].append(
                        AttentionReader.read(att[i], t_words, s_words, n_t, n_s, shift=shift)
                    )
        return per_layer

    def attentions(
        self, pairs, *, layer: int, batch_size: int = 16, shift: bool = True
    ) -> List[Any]:
        return self.attentions_all_layers(pairs, batch_size=batch_size, shift=shift)[layer]

    def states(self, chunk, *, layer: int):
        """For the alignment layer: per pair ``(keys [n_s, H], queries [n_t, H])``
        pooled to words -- keys = input embeddings + encoder states, queries = the
        decoder states of layer ``layer`` at position ``k + 1`` for target piece ``k``."""
        import torch

        out, src, labels, src_words, tgt_words = self.forward_batch(chunk, hidden=True)
        embeddings = self._model.get_input_embeddings()(src["input_ids"].to(self.device))
        enc = out.encoder_last_hidden_state + embeddings
        dec = out.decoder_hidden_states[min(layer + 1, len(out.decoder_hidden_states) - 1)]
        result = []
        for i, (s_words, t_words) in enumerate(zip(src_words, tgt_words)):
            n_s, n_t = len(chunk[i][0]), len(chunk[i][1])
            keys = torch.zeros((n_s, enc.shape[-1]), device=self.device)
            kc = torch.zeros(n_s, device=self.device)
            for j, w in enumerate(s_words):
                if w is not None and w < n_s:
                    keys[w] += enc[i, j]
                    kc[w] += 1
            queries = torch.zeros((n_t, dec.shape[-1]), device=self.device)
            qc = torch.zeros(n_t, device=self.device)
            for k, w in enumerate(t_words):
                if w is not None and w < n_t and k + 1 < dec.shape[1]:
                    queries[w] += dec[i, k + 1]
                    qc[w] += 1
            result.append((keys / kc.clamp(min=1)[:, None], queries / qc.clamp(min=1)[:, None]))
        return result


# =============================================================================
# The alignment layer
# =============================================================================


class AlignmentLayer:
    """One attention head with a null key over frozen states, and its losses.

    Example:
        ```python
        layer = AlignmentLayer(hidden=768, device="cpu", full_context=True)
        probs = layer(keys, queries)                    # [n_t, n_s + 1], the last column the null
        loss = layer.guided_loss(probs, links)          # links: -1 = null
        ```
    """

    def __init__(
        self, hidden: int, *, device: str = "cpu", full_context: bool = False, dropout: float = 0.1
    ):
        import torch

        self.hidden = hidden
        self.device = device
        self.full_context = full_context
        self.dropout = dropout
        # the frozen states are unnormalised (T5 hidden states run to norms in the hundreds);
        # without a normalisation the scores saturate the softmax and the layer never trains
        self.norm_k = torch.nn.LayerNorm(hidden).to(device)
        self.norm_q = torch.nn.LayerNorm(hidden).to(device)
        self.w_k = torch.nn.Linear(hidden, hidden, bias=False).to(device)
        self.w_q = torch.nn.Linear(hidden, hidden, bias=False).to(device)
        self.null_key = torch.nn.Parameter(torch.zeros(hidden, device=device).normal_(std=0.02))
        self.context = (
            torch.nn.MultiheadAttention(hidden, num_heads=4, batch_first=True).to(device)
            if full_context
            else None
        )
        self.training = False

    # ---------- Trainable surface ----------

    def modules(self):
        return [self.norm_k, self.norm_q, self.w_k, self.w_q] + (
            [self.context] if self.context is not None else []
        )

    def parameters(self):
        return [p for m in self.modules() for p in m.parameters()] + [self.null_key]

    # ---------- Modes and persistence ----------

    def train(self) -> None:
        self.training = True
        for m in self.modules():
            m.train()

    def eval(self) -> None:
        self.training = False
        for m in self.modules():
            m.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "w_k": self.w_k.state_dict(),
            "w_q": self.w_q.state_dict(),
            "null_key": self.null_key.detach().cpu(),
            "norm_k": self.norm_k.state_dict(),
            "norm_q": self.norm_q.state_dict(),
            "context": self.context.state_dict() if self.context is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        import torch

        self.w_k.load_state_dict(state["w_k"])
        self.w_q.load_state_dict(state["w_q"])
        if "norm_k" in state:
            self.norm_k.load_state_dict(state["norm_k"])
            self.norm_q.load_state_dict(state["norm_q"])
        with torch.no_grad():
            self.null_key.copy_(state["null_key"].to(self.device))
        if self.context is not None and state.get("context") is not None:
            self.context.load_state_dict(state["context"])

    # ---------- Forward pass ----------

    def logits(self, keys, queries, *, null_logit: Optional[float] = None):
        import torch

        q = self.norm_q(queries)
        if self.context is not None:
            q = q + self.context(
                q.unsqueeze(0), q.unsqueeze(0), q.unsqueeze(0), need_weights=False
            )[0].squeeze(0)
        k = torch.cat(
            [self.w_k(self.norm_k(keys)), self.w_k(self.null_key).unsqueeze(0)], dim=0
        )  # [n_s + 1, H]
        scores = (self.w_q(q) @ k.T) / (self.hidden**0.5)
        if null_logit is not None:
            scores = scores.clone()
            scores[:, -1] = null_logit
        if self.training and self.dropout > 0:
            scores = torch.nn.functional.dropout(scores, p=self.dropout, training=True)
        return scores

    def __call__(self, keys, queries, *, null_logit: Optional[float] = None):
        import torch

        return torch.softmax(self.logits(keys, queries, null_logit=null_logit), dim=-1)

    # ---------- Losses ----------

    @staticmethod
    def guided_loss(probs, links: Sequence[int]):
        """``-(1/m) sum_t log p(a_t | t)``, the null column for unaligned words."""
        import torch

        n_s = probs.shape[1] - 1
        targets = torch.tensor(
            [n_s if s is None or s < 0 else int(s) for s in links], device=probs.device
        )
        picked = probs[torch.arange(len(links), device=probs.device), targets]
        return -(torch.log(picked.clamp(min=1e-9))).mean()

    @staticmethod
    def contiguity_loss(probs):
        """Zenkel's 2 x 2 convolution over the source columns: ``-sum_t log max_s A_bar``."""
        import torch

        source = probs[:, :-1]
        if source.shape[0] < 2 or source.shape[1] < 2:
            return probs.sum() * 0.0
        pooled = torch.nn.functional.avg_pool2d(
            source.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=1
        ).squeeze()
        pooled = pooled.reshape(source.shape[0] - 1, -1)
        return -(torch.log(pooled.max(dim=1).values.clamp(min=1e-9))).sum() / max(
            source.shape[0], 1
        )


# =============================================================================
# The method row
# =============================================================================


@BaselineRegistry.register
class NMTAligner(Baseline):
    """ "Translation model + alignment layer": the pairs-only neural row.

    ``cfg.extra`` (``NMT_DEFAULTS``): ``backbone``, ``reading`` (``layer`` |
    ``shift_att`` | ``naive``), ``epochs``, ``lr``, ``batch_size``,
    ``layer_updates``, ``layer_lr``, ``lam`` (contiguity), ``full_context``,
    ``include_unlabeled``, ``load_fwd`` / ``load_bwd`` / ``load_layer``,
    ``layer_fwd`` / ``layer_bwd`` (fixed layers instead of the agreement grid).

    Example:
        ```python
        method = NMTAligner(cfg).fit(train, dev, log=print, unlabeled=test)
        preds = method.predict(test)              # scores and rev_scores per record
        # python run_baseline.py --method nmt_aligner --set wpt_enfr --max-train 100000 --decoder gdf
        ```
    """

    name = "nmt_aligner"
    early_stopping_capable = True
    emits = "scores"
    trainable = True
    typer = "rule"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {**NMT_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in NMT_DEFAULTS}}
        backbone = str(self.dials["backbone"])
        self.fwd = Seq2SeqWrapper(backbone, device=cfg.device, max_length=cfg.max_length)
        self.bwd = Seq2SeqWrapper(backbone, device=cfg.device, max_length=cfg.max_length)
        self.layer_fwd = int(self.dials["layer_fwd"])
        self.layer_bwd = int(self.dials["layer_bwd"])
        self.align_fwd: Optional[AlignmentLayer] = None
        self.align_bwd: Optional[AlignmentLayer] = None

    # ---------- data ----------

    @staticmethod
    def pairs_of(records: Sequence[Record]) -> List[Tuple[List[str], List[str]]]:
        return [
            (list(r.source_tokens), list(r.reuse_tokens))
            for r in records
            if r.source_tokens and r.reuse_tokens
        ]

    # ---------- training ----------

    def fit(
        self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
    ) -> NMTAligner:
        import torch

        # the trained-on pairs and the shared synthetic pairs as translation data (no label is read);
        # the test passages join only with ``include_unlabeled``
        records = (
            list(train)
            + list(self.shared_synthetic)
            + (list(unlabeled) if int(self.dials["include_unlabeled"]) else [])
        )
        from retexo.baselines.em_aligner import extra_bitext

        records += extra_bitext(self.cfg.extra.get("extra_bitext"), log=log, name="nmt_aligner")
        pairs = self.pairs_of(records)
        if not pairs:
            return self
        epochs = max(1, int(self.dials["epochs"]))
        if self.cfg.smoke:
            epochs = 1
        batch = int(self.dials["batch_size"])
        lr = float(self.dials["lr"])
        # early stopping of the translation stage on the validation pairs' loss (no alignment exists yet)
        from retexo.baselines.early_stopping import EarlyStopping, EarlyStoppingGroup

        valid = self.pairs_of(self.validation)
        stop_fwd = stop_bwd = None
        if valid:
            stop_fwd = EarlyStopping.for_method(
                self,
                log=log,
                score=lambda: self.fwd.loss_on(valid, batch_size=batch),
                higher_is_better=False,
                metric="validation translation loss, forward",
            )
            stop_bwd = EarlyStopping.for_method(
                self,
                log=log,
                score=lambda: self.bwd.loss_on([(t, s) for s, t in valid], batch_size=batch),
                higher_is_better=False,
                metric="validation translation loss, backward",
            )
            self.early_stopping = EarlyStoppingGroup({"forward": stop_fwd, "backward": stop_bwd})
        if str(self.dials["load_fwd"]):
            self.fwd.load_state_dict(
                torch.load(
                    Path(str(self.dials["load_fwd"])) / "seq2seq.pt", map_location=self.cfg.device
                )
            )
        else:
            self.fwd.fit(
                pairs,
                epochs=epochs,
                lr=lr,
                batch_size=batch,
                seed=self.cfg.seed,
                log=log,
                tag="nmt fwd",
                stopper=stop_fwd,
            )
        if str(self.dials["load_bwd"]):
            self.bwd.load_state_dict(
                torch.load(
                    Path(str(self.dials["load_bwd"])) / "seq2seq.pt", map_location=self.cfg.device
                )
            )
        else:
            self.bwd.fit(
                [(t, s) for s, t in pairs],
                epochs=epochs,
                lr=lr,
                batch_size=batch,
                seed=self.cfg.seed,
                log=log,
                tag="nmt bwd",
                stopper=stop_bwd,
            )
        dev_pairs = self.pairs_of(dev)[: int(self.dials["dev_pairs"])] or pairs[:200]
        if self.layer_fwd < 0 or self.layer_bwd < 0:
            self.select_layers(dev_pairs, log=log)
        if str(self.dials["reading"]) == "layer":
            self.train_alignment_layers(pairs, log=log)
        return self

    def select_layers(self, dev_pairs, *, log=None) -> None:
        """Chen's agreement grid over the two models' layers on the dev pairs."""
        fwd = self.fwd.attentions_all_layers(dev_pairs, batch_size=int(self.dials["batch_size"]))
        bwd = self.bwd.attentions_all_layers(
            [(t, s) for s, t in dev_pairs], batch_size=int(self.dials["batch_size"])
        )

        def argmax(mats):
            return [[int(row.argmax()) if row.sum() > 0 else -1 for row in m] for m in mats]

        self.layer_fwd, self.layer_bwd = LayerSelector.select(
            [argmax(m) for m in fwd], [argmax(m) for m in bwd]
        )
        if log:
            log(
                f"[nmt_aligner] layers by agreement: forward {self.layer_fwd}, backward {self.layer_bwd} "
                f"(of {len(fwd)}), mutual AER {LayerSelector.mutual_aer(argmax(fwd[self.layer_fwd]), argmax(bwd[self.layer_bwd])):.3f}"
            )

    def guided_targets(self, pairs, *, batch_size: int) -> Tuple[List[List[int]], List[List[int]]]:
        """The symmetrised SHIFT-ATT links of both models: grow-diag-final over
        the two argmax readings (Chen, Zenkel), no threshold; a word the
        symmetrisation leaves out is a null target."""
        from retexo.baselines.decoder import BaselineDecoder

        fwd_links, bwd_links = [], []
        for start in range(0, len(pairs), batch_size):
            chunk = list(pairs[start : start + batch_size])
            f = self.fwd.attentions(chunk, layer=self.layer_fwd, batch_size=batch_size)
            b = self.bwd.attentions(
                [(t, s) for s, t in chunk], layer=self.layer_bwd, batch_size=batch_size
            )
            for (s_tokens, _t_tokens), mf, mb in zip(chunk, f, b):
                rows, rev = AttentionReader.rows(mf), AttentionReader.rows(mb)
                links, _ = BaselineDecoder.decode_gdf(rows, rev, len(s_tokens))
                fwd_links.append(links)
                back = [-1] * len(s_tokens)
                for t, s in enumerate(links):
                    if s >= 0:
                        back[s] = t
                bwd_links.append(back)
        return fwd_links, bwd_links

    def train_alignment_layers(self, pairs, *, log=None) -> None:
        import torch

        batch = int(self.dials["batch_size"])
        updates = int(self.dials["layer_updates"])
        if self.cfg.smoke:
            updates = min(updates, 200)
        full = bool(int(self.dials["full_context"]))
        lam = float(self.dials["lam"])
        fwd_links, bwd_links = self.guided_targets(pairs, batch_size=batch)
        if log:
            log(
                f"[nmt_aligner] guided targets from {len(pairs)} pairs; training two alignment layers for {updates} updates"
            )
        for name, model, layer_index, links, oriented in (
            ("fwd", self.fwd, self.layer_fwd, fwd_links, pairs),
            ("bwd", self.bwd, self.layer_bwd, bwd_links, [(t, s) for s, t in pairs]),
        ):
            align = AlignmentLayer(model.hidden_size, device=self.cfg.device, full_context=full)
            optimizer = torch.optim.Adam(align.parameters(), lr=float(self.dials["layer_lr"]))
            rng = random.Random(self.cfg.seed)
            order = list(range(len(oriented)))
            align.train()
            done, total = 0, 0.0
            while done < updates:
                rng.shuffle(order)
                for start in range(0, len(order), batch):
                    idx = order[start : start + batch]
                    chunk = [oriented[i] for i in idx]
                    states = model.states(chunk, layer=layer_index)
                    losses = []
                    for i, (keys, queries) in zip(idx, states):
                        if keys.shape[0] == 0 or queries.shape[0] == 0:
                            continue
                        probs = align(keys.detach(), queries.detach())
                        loss = align.guided_loss(probs, links[i][: queries.shape[0]])
                        if lam > 0:
                            loss = loss + lam * align.contiguity_loss(probs)
                        losses.append(loss)
                    if not losses:
                        continue
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                    done += 1
                    if log and done % 500 == 0:
                        log(
                            f"[nmt_aligner] layer {name} update {done}/{updates}: guided loss {total / 500:.4f}"
                        )
                        total = 0.0
                    if done >= updates:
                        break
            align.eval()
            setattr(self, f"align_{name}", align)

    # ---------- inference ----------

    def rows_for(self, pairs, *, direction: str) -> List[Rows]:
        """Interface (I) rows of one direction from the configured reading."""
        model = self.fwd if direction == "fwd" else self.bwd
        layer_index = self.layer_fwd if direction == "fwd" else self.layer_bwd
        align = self.align_fwd if direction == "fwd" else self.align_bwd
        reading = str(self.dials["reading"])
        batch = int(self.dials["batch_size"])
        out: List[Rows] = []
        if reading == "layer" and align is not None:
            import torch

            for start in range(0, len(pairs), batch):
                chunk = list(pairs[start : start + batch])
                for (s_tokens, t_tokens), (keys, queries) in zip(
                    chunk, model.states(chunk, layer=layer_index)
                ):
                    if keys.shape[0] == 0 or queries.shape[0] == 0:
                        out.append([[(-1, 1.0)] for _ in t_tokens])
                        continue
                    with torch.no_grad():
                        probs = align(keys, queries).cpu().numpy()
                    rows: Rows = []
                    for t in range(len(t_tokens)):
                        if t < probs.shape[0]:
                            row = [(int(s), float(probs[t, s])) for s in range(len(s_tokens))] + [
                                (-1, float(probs[t, -1]))
                            ]
                        else:
                            row = [(-1, 1.0)]
                        rows.append(sorted(row, key=lambda x: -x[1]))
                    out.append(rows)
            return out
        shift = reading != "naive"
        for start in range(0, len(pairs), batch):
            chunk = list(pairs[start : start + batch])
            for matrix in model.attentions(
                chunk, layer=max(layer_index, 0), batch_size=batch, shift=shift
            ):
                out.append(AttentionReader.rows(matrix))
        return out

    def predict(self, records: List[Record]) -> List[Prediction]:
        preds = [Prediction.empty(r.n_reuse) for r in records]
        live = [(i, r) for i, r in enumerate(records) if r.source_tokens and r.reuse_tokens]
        if not live or self.fwd._model is None:
            return preds
        pairs = [(list(r.source_tokens), list(r.reuse_tokens)) for _, r in live]
        forward = self.rows_for(pairs, direction="fwd")
        backward = self.rows_for([(t, s) for s, t in pairs], direction="bwd")
        for (i, _), rows, rev in zip(live, forward, backward):
            preds[i].scores = rows
            preds[i].rev_scores = rev
        return preds

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        for name, model in (("fwd", self.fwd), ("bwd", self.bwd)):
            if model._model is not None:
                (path / name).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path / name / "seq2seq.pt")
        torch.save(
            {
                "layer_fwd": self.layer_fwd,
                "layer_bwd": self.layer_bwd,
                "align_fwd": self.align_fwd.state_dict() if self.align_fwd else None,
                "align_bwd": self.align_bwd.state_dict() if self.align_bwd else None,
                "full_context": bool(int(self.dials["full_context"])),
            },
            path / "layers.pt",
        )

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> NMTAligner:
        import torch

        path = Path(path)
        method = cls(cfg)
        method.fwd.load_state_dict(torch.load(path / "fwd" / "seq2seq.pt", map_location=cfg.device))
        method.bwd.load_state_dict(torch.load(path / "bwd" / "seq2seq.pt", map_location=cfg.device))
        state = torch.load(path / "layers.pt", map_location=cfg.device)
        method.layer_fwd, method.layer_bwd = int(state["layer_fwd"]), int(state["layer_bwd"])
        for name in ("fwd", "bwd"):
            if state.get(f"align_{name}") is not None:
                align = AlignmentLayer(
                    method.fwd.hidden_size,
                    device=cfg.device,
                    full_context=bool(state["full_context"]),
                )
                align.load_state_dict(state[f"align_{name}"])
                align.eval()
                setattr(method, f"align_{name}", align)
        return method
