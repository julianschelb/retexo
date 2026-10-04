# retexo/pretraining/stage0.py
"""Stage 0: continued pretraining of Latin BERT on reuse pairs, and the probe.

The pretraining definition (``Experiments/Pretraining Experiments - Definition``)
puts a stage 0 before the synthetic -> gold schedule every trainable row
follows: one encoder, three objectives summed, one epoch over the real-pairs
level (``retexo.pretraining.pool``). This module holds the objectives, the
trainer and the frozen geometry probe that gates the idea. Objective 1 (the
masked LM over the joint pair with a mask biased toward reused words) is
implemented; objectives 2 (contrastive lemma pairs) and 3 (pair
identification) keep their interfaces and raise until their turn.

    python run_stage0.py --objectives mlm --out runs/pretrain_mlm --smoke 200
    python run_stage0.py --probe runs/pretrain_mlm/latin-bert-reuse ashleygong03/bamman-burns-latin-bert

The checkpoint is saved as a directory the harness's ``--base-model`` accepts:
the encoder in Hugging Face format plus Latin BERT's ``vocab.txt`` and a
``latin_bert.marker``, so ``PairEncoder.build`` picks the tensor2tensor
subword encoder for it (``pair_encoding.is_latin_bert``).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from retexo.baselines.record import Record
from retexo.core.normalize import normalize
from retexo.pretraining.pool import PairPool

# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class Stage0Config:
    """The dials of the pretraining note, with its defaults.

    Attributes:
        base_model: The encoder to continue from.
        objectives: Which losses are summed (``mlm``, ``contrastive``, ``psi``).
        mask_rate: Share of subwords masked per pair.
        reuse_mask_share: Share of the masked positions drawn from words with a
            counterpart in the other passage (``rho`` in the note); 0 is uniform
            masking.
        contrastive_weight, psi_weight: ``lambda_c`` and ``lambda_p``.
        contrastive_temperature: InfoNCE temperature.
        layer: The layer the pointer reads and the probe measures.
        epochs, batch_size, learning_rate, warmup: The one-epoch schedule.
        max_length: Subwords per joint pair.
        device, seed: As everywhere.
        out: Where the checkpoint goes (``out / "latin-bert-reuse"``).
        log_every: Steps between loss lines.
    """

    base_model: str = "ashleygong03/bamman-burns-latin-bert"
    objectives: Tuple[str, ...] = ("mlm",)
    mask_rate: float = 0.15
    reuse_mask_share: float = 0.5
    contrastive_weight: float = 0.5
    psi_weight: float = 0.1
    contrastive_temperature: float = 0.05
    layer: int = 8
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup: float = 0.1
    max_length: int = 256
    device: str = "cuda"
    seed: int = 1
    out: Path = Path("runs/stage0")
    log_every: int = 25
    smoke: int = 0

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["out"] = str(self.out)
        return d


# =============================================================================
# Counterparts (what the mask is biased toward)
# =============================================================================


class Counterparts:
    """Which words of one passage have a counterpart in the other.

    "Identical or same-lemma" in the note; here identical after ``normalize``,
    or the same spelling key, or a shared stem of at least four characters
    with the same first letter -- the lemmatiser is too slow for 40,000
    sequences per epoch and the bias only has to point the mask at reused
    words, not decide their operation.

    Example:
        ```python
        Counterparts.mark(["arma", "virumque", "cano"], ["arma", "virum", "canit"])
        # -> [True, True, False]      (cano / canit share three characters, not four)
        ```
    """

    STEM = 4

    @classmethod
    def keys(cls, word: str) -> Tuple[str, str]:
        from retexo.edit_typing.repair import Repairer

        n = normalize(word)
        return n, (Repairer.spelling_key(word) if n else "")

    @classmethod
    def mark(cls, words: Sequence[str], others: Sequence[str]) -> List[bool]:
        other_forms, other_keys, other_stems = set(), set(), set()
        for w in others:
            n, k = cls.keys(w)
            if len(n) < 2:
                continue
            other_forms.add(n); other_keys.add(k)
            if len(n) >= cls.STEM:
                other_stems.add(n[: cls.STEM])
        out = []
        for w in words:
            n, k = cls.keys(w)
            hit = len(n) >= 2 and (n in other_forms or (k and k in other_keys) or (len(n) >= cls.STEM and n[: cls.STEM] in other_stems))
            out.append(bool(hit))
        return out


# =============================================================================
# Objective 1: the masked LM over the joint pair
# =============================================================================


class MaskedPairLM:
    """Masked LM over ``[CLS] source [SEP] reuse [SEP]``, the mask biased toward reused words.

    A masked word can be recovered from its counterpart across the ``[SEP]``;
    ``reuse_mask_share`` (rho) of the masked positions are drawn from subwords
    of words with a counterpart in the other passage, the rest uniformly over
    the passages' subwords; BERT's 80/10/10 replacement.

    Example:
        ```python
        mlm = MaskedPairLM(config, pair_encoder, vocab_size, mask_id)
        batch = mlm.batch([(source_words, reuse_words), ...], rng)     # input_ids, labels, ...
        loss = mlm.loss(model, batch)
        ```
    """

    def __init__(self, config: Stage0Config, pair_encoder, vocab_size: int, mask_id: int, special_ids: Sequence[int]):
        self.config = config
        self.encoder = pair_encoder
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.special_ids = set(int(i) for i in special_ids)

    def positions(self, ids: Sequence[int], source_spans, reuse_spans, source_hit: Sequence[bool],
                  reuse_hit: Sequence[bool], rng: random.Random) -> List[int]:
        """The subword positions to mask for one sequence."""
        candidates = [i for i, t in enumerate(ids) if int(t) not in self.special_ids]
        if not candidates:
            return []
        n_mask = max(1, int(round(self.config.mask_rate * len(candidates))))
        reused: List[int] = []
        for spans, hits in ((source_spans, source_hit), (reuse_spans, reuse_hit)):
            for (a, b), hit in zip(spans, hits):
                if hit:
                    reused.extend(range(a, b))
        reused = [i for i in reused if int(ids[i]) not in self.special_ids]
        n_biased = min(len(reused), int(round(self.config.reuse_mask_share * n_mask)))
        chosen = set(rng.sample(reused, n_biased)) if n_biased else set()
        rest = [i for i in candidates if i not in chosen]
        n_rest = min(len(rest), n_mask - len(chosen))
        chosen.update(rng.sample(rest, n_rest))
        return sorted(chosen)

    def batch(self, pairs: Sequence[Tuple[Sequence[str], Sequence[str]]], rng: random.Random):
        """Model inputs with ``labels`` (-100 off the mask) for a list of (source, reuse) word lists."""
        import torch

        encoded, reuse_spans = self.encoder.encode([(list(s), list(r)) for s, r in pairs], self.config.max_length)
        source_spans = self.encoder.last_source_spans
        input_ids = encoded["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)
        n_masked = 0
        n_biased_total = 0
        for row, (source, reuse) in enumerate(pairs):
            ids = input_ids[row].tolist()
            s_hit = Counterparts.mark(source, reuse)
            r_hit = Counterparts.mark(reuse, source)
            chosen = self.positions(ids, source_spans[row], reuse_spans[row], s_hit, r_hit, rng)
            for i in chosen:
                labels[row, i] = input_ids[row, i]
                draw = rng.random()
                if draw < 0.8:
                    input_ids[row, i] = self.mask_id
                elif draw < 0.9:
                    input_ids[row, i] = rng.randrange(5, self.vocab_size)
                n_masked += 1
            n_biased_total += sum(1 for i in chosen if any(a <= i < b for (a, b), h in zip(source_spans[row], s_hit) if h)
                                  or any(a <= i < b for (a, b), h in zip(reuse_spans[row], r_hit) if h))
        out = {k: v for k, v in encoded.items()}
        out["input_ids"] = input_ids
        out["labels"] = labels
        out["_n_masked"] = n_masked
        out["_n_biased"] = n_biased_total
        return out

    @staticmethod
    def loss(model, batch, device: str):
        import torch

        inputs = {k: v.to(device) for k, v in batch.items() if not k.startswith("_") and isinstance(v, torch.Tensor)}
        out = model(**inputs)
        return out.loss


# =============================================================================
# Objectives 2 and 3 (interfaces; their turn comes after objective 1's probe)
# =============================================================================


class ContrastiveLemmaPairs:
    """Objective 2: InfoNCE over resource pairs embedded as word-in-sentence vectors.

    A batch holds ``n`` resource pairs; each member is drawn in one of its
    sentences, every sentence is encoded alone (``[CLS] words [SEP]``), the
    lemma's subwords are mean-pooled at ``config.layer`` of the base encoder,
    and side A must be closer to its own side B than to the other same-POS
    lemmas' vectors of the batch (and back), at ``contrastive_temperature``;
    the pair's resource weight scales its term.

    Example:
        ```python
        objective = ContrastiveLemmaPairs(config, pair_encoder, resource_pairs, sentences)
        batch = objective.batch(rng)             # sentences, spans, pos, weights
        loss = objective.loss(model, batch, device)
        ```
    """

    def __init__(self, config: Stage0Config, pair_encoder, pairs, sentences, *, batch_pairs: int = 32):
        self.config = config
        self.encoder = pair_encoder
        self.pairs = list(pairs)
        self.sentences = sentences
        self.batch_pairs = batch_pairs

    def batch(self, rng: random.Random):
        """``batch_pairs`` resource pairs, one sentence per member; encoded as single passages."""
        chosen = rng.sample(self.pairs, min(self.batch_pairs, len(self.pairs)))
        items = []                      # (tokens, word index) for side A of pair 0, side B of pair 0, side A of pair 1, ...
        pos, weights = [], []
        for pair in chosen:
            a = rng.choice(self.sentences.contexts(pair.lemma_a))
            b = rng.choice(self.sentences.contexts(pair.lemma_b))
            items.append((list(a["tokens"]), int(a["index"]))); items.append((list(b["tokens"]), int(b["index"])))
            pos.append(pair.pos); weights.append(float(pair.weight))
        # a single passage through the pair encoder: the passage as the source side, an empty reuse side
        encoded, _ = self.encoder.encode([(tokens, []) for tokens, _ in items], self.config.max_length)
        spans = self.encoder.last_source_spans
        word_spans = []
        for (tokens, index), row_spans in zip(items, spans):
            word_spans.append(row_spans[index] if index < len(row_spans) else None)
        return {"encoded": encoded, "spans": word_spans, "pos": pos, "weights": weights}

    def loss(self, model, batch, device: str):
        import torch

        base = model.base_model if hasattr(model, "base_model") else model
        inputs = {k: v.to(device) for k, v in batch["encoded"].items() if isinstance(v, torch.Tensor)}
        states = base(**inputs, output_hidden_states=True).hidden_states
        hidden = states[min(self.config.layer, len(states) - 1)]        # a tiny test encoder has fewer layers
        vectors, keep = [], []
        for row, span in enumerate(batch["spans"]):
            if span is None:
                keep.append(False); vectors.append(hidden[row, 0]); continue
            a, b = span
            keep.append(True); vectors.append(hidden[row, a:b].mean(0))
        vectors = torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
        n = len(batch["pos"])
        side_a, side_b = vectors[0::2], vectors[1::2]                          # [n, H] each
        ok = torch.tensor([keep[2 * i] and keep[2 * i + 1] for i in range(n)], device=device)
        pos = batch["pos"]
        same_pos = torch.tensor([[pos[i] == pos[j] for j in range(n)] for i in range(n)], device=device)
        weights = torch.tensor(batch["weights"], device=device)
        logits = side_a @ side_b.T / self.config.contrastive_temperature     # [n, n]: A_i against every B_j
        logits = logits.masked_fill(~same_pos, float("-inf"))                 # negatives: same-POS lemmas only
        targets = torch.arange(n, device=device)
        loss_ab = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        loss_ba = torch.nn.functional.cross_entropy(logits.T.masked_fill(~same_pos.T, float("-inf")), targets, reduction="none")
        per_pair = 0.5 * (loss_ab + loss_ba) * weights
        per_pair = torch.where(ok, per_pair, torch.zeros_like(per_pair))
        return per_pair.sum() / max(int(ok.sum().item()), 1)


class PairIdentification:
    """Objective 3: a ``Linear(hidden, 2)`` head on the ``[CLS]`` vector of the joint pair --
    real reuse pair (label 1) or hard negative (label 0). Balanced by sampling one
    negative per real pair in every batch.

    Example:
        ```python
        objective = PairIdentification(config, pair_encoder, hidden_size, negatives)
        batch = objective.batch(real_pairs, rng)
        loss = objective.loss(model, batch, device)
        ```
    """

    def __init__(self, config: Stage0Config, pair_encoder, hidden_size: int, negatives: Sequence[Tuple[List[str], List[str]]]):
        import torch

        self.config = config
        self.encoder = pair_encoder
        self.negatives = list(negatives)
        self.head = torch.nn.Linear(hidden_size, 2).to(config.device)

    def parameters(self):
        return self.head.parameters()

    def batch(self, real: Sequence[Tuple[Sequence[str], Sequence[str]]], rng: random.Random):
        negs = rng.sample(self.negatives, min(len(real), len(self.negatives))) if self.negatives else []
        pairs = [(list(s), list(r)) for s, r in real] + [(list(s), list(r)) for s, r in negs]
        labels = [1] * len(real) + [0] * len(negs)
        encoded, _ = self.encoder.encode(pairs, self.config.max_length)
        return {"encoded": encoded, "labels": labels}

    def loss(self, model, batch, device: str):
        import torch

        base = model.base_model if hasattr(model, "base_model") else model
        inputs = {k: v.to(device) for k, v in batch["encoded"].items() if isinstance(v, torch.Tensor)}
        cls = base(**inputs).last_hidden_state[:, 0]
        logits = self.head(cls)
        return torch.nn.functional.cross_entropy(logits, torch.tensor(batch["labels"], device=device))


# =============================================================================
# Trainer
# =============================================================================


class Stage0Trainer:
    """One epoch of the summed objectives on one encoder; writes one checkpoint.

    Example:
        ```python
        trainer = Stage0Trainer(Stage0Config(objectives=("mlm",), out=Path("runs/pretrain_mlm")))
        trainer.fit(pool, log=print)
        path = trainer.save()          # runs/pretrain_mlm/latin-bert-reuse
        # then: run_baseline.py --method typed_pointer --base-model runs/pretrain_mlm/latin-bert-reuse
        ```
    """

    CHECKPOINT_NAME = "latin-bert-reuse"

    def __init__(self, config: Optional[Stage0Config] = None):
        self.config = config or Stage0Config()
        self.model = None
        self.pair_encoder = None
        self.history: List[Dict[str, float]] = []

    # ---------- backend ----------

    def _load(self):
        import torch
        from transformers import AutoModelForMaskedLM

        from retexo.formulations.pair_encoding import PairEncoder

        torch.manual_seed(self.config.seed)
        self.model = AutoModelForMaskedLM.from_pretrained(self.config.base_model).to(self.config.device)
        self.pair_encoder = PairEncoder.build(self.config.base_model)
        specials = [getattr(self.pair_encoder, name) for name in ("PAD", "CLS", "SEP") if hasattr(self.pair_encoder, name)]
        mask_id = self._mask_id()
        vocab = int(self.model.config.vocab_size)
        self.mlm = MaskedPairLM(self.config, self.pair_encoder, vocab, mask_id, specials + [mask_id])

    def _mask_id(self) -> int:
        enc = getattr(self.pair_encoder, "encoder", None)
        if enc is not None and hasattr(enc, "_subtoken_to_id"):
            return int(enc._subtoken_to_id.get("[MASK]", 4))
        tok = getattr(self.pair_encoder, "tokenizer", None)
        if tok is not None and getattr(tok, "mask_token_id", None) is not None:
            return int(tok.mask_token_id)
        return 4

    # ---------- training ----------

    def fit(self, pool: PairPool, resource_pairs=None, *, sentences=None, negatives=None, psi_pool: Optional[PairPool] = None,
            log=None) -> "Stage0Trainer":
        """One epoch: per step one MLM batch (if ``mlm`` is on), one contrastive batch (if ``contrastive``),
        one PSI batch (if ``psi``); ``L = L_MLM + lambda_c L_c + lambda_p L_p``; every loss logged on its own.
        ``psi_pool`` gives the PSI objective its own real pairs (Data Scale: the masked LM reads the all-overlap
        pool, the other objectives keep the classifier's); without it both read ``pool``."""
        import torch

        if self.model is None:
            self._load()
        cfg = self.config
        rng = random.Random(cfg.seed)
        use_mlm = "mlm" in cfg.objectives
        use_c = "contrastive" in cfg.objectives
        use_p = "psi" in cfg.objectives
        self.contrastive = ContrastiveLemmaPairs(cfg, self.pair_encoder, resource_pairs, sentences,
                                                 batch_pairs=cfg.batch_size) if use_c else None
        self.psi = PairIdentification(cfg, self.pair_encoder, int(self.model.config.hidden_size), negatives or []) if use_p else None
        items = list(pool.both_orientations())
        if cfg.smoke:
            items = items[: cfg.smoke]
        rng.shuffle(items)
        psi_items = items
        if use_p and psi_pool is not None:
            psi_items = list(psi_pool.both_orientations())[: cfg.smoke or None]
            rng.shuffle(psi_items)
        steps_per_epoch = max(1, math.ceil(len(items) / cfg.batch_size))
        if not use_mlm and use_c:                      # contrastive alone: one epoch over the resource pairs
            steps_per_epoch = max(1, math.ceil(len(self.contrastive.pairs) / cfg.batch_size))
            if cfg.smoke:
                steps_per_epoch = min(steps_per_epoch, max(1, cfg.smoke // cfg.batch_size))
        total_steps = steps_per_epoch * cfg.epochs
        warmup = int(cfg.warmup * total_steps)
        params = list(self.model.parameters()) + (list(self.psi.parameters()) if self.psi else [])
        optimiser = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.01)

        def lr_at(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)
        if log:
            log(f"[stage0] objectives {list(cfg.objectives)}; {len(items)} sequences ({len(pool)} pairs, both orientations)"
                f"{f', {len(self.contrastive.pairs)} resource pairs' if use_c else ''}{f', {len(self.psi.negatives)} negatives' if use_p else ''}; "
                f"{total_steps} steps of {cfg.batch_size}, lr {cfg.learning_rate}, warm-up {warmup}, rho {cfg.reuse_mask_share}, "
                f"lambda_c {cfg.contrastive_weight}, lambda_p {cfg.psi_weight}")
        self.model.train()
        step = 0
        started = time.time()
        quarter = max(1, total_steps // 4)
        for epoch in range(cfg.epochs):
            rng.shuffle(items)
            running = {"mlm": 0.0, "contrastive": 0.0, "psi": 0.0}
            running_n, masked, biased = 0, 0, 0
            for k in range(steps_per_epoch):
                chunk = items[(k * cfg.batch_size) % max(len(items), 1):][: cfg.batch_size] if items else []
                loss = None
                if use_mlm and chunk:
                    batch = self.mlm.batch([(s, r) for s, r, _ in chunk], rng)
                    part = self.mlm.loss(self.model, batch, cfg.device)
                    running["mlm"] += float(part.item()); masked += batch["_n_masked"]; biased += batch["_n_biased"]
                    loss = part
                if use_c:
                    part = cfg.contrastive_weight * self.contrastive.loss(self.model, self.contrastive.batch(rng), cfg.device)
                    running["contrastive"] += float(part.item()) / cfg.contrastive_weight
                    loss = part if loss is None else loss + part
                psi_chunk = chunk if psi_items is items else (
                    psi_items[(k * cfg.batch_size) % max(len(psi_items), 1):][: cfg.batch_size] if psi_items else [])
                if use_p and psi_chunk:
                    part = cfg.psi_weight * self.psi.loss(self.model, self.psi.batch([(s, r) for s, r, _ in psi_chunk], rng), cfg.device)
                    running["psi"] += float(part.item()) / cfg.psi_weight
                    loss = part if loss is None else loss + part
                if loss is None:
                    continue
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step(); scheduler.step()
                step += 1
                running_n += 1
                if log and (step % cfg.log_every == 0 or step == total_steps):
                    means = {k2: v / max(running_n, 1) for k2, v in running.items()}
                    entry = {"step": step}
                    if use_mlm:
                        entry["mlm"] = means["mlm"]
                        log(f"[stage0] mlm step {step}/{total_steps}: loss {means['mlm']:.4f} (masked {masked}, biased share "
                            f"{biased / max(masked, 1):.2f}, {(time.time() - started) / 60:.1f} min)")
                    if use_c:
                        entry["contrastive"] = means["contrastive"]
                        log(f"[stage0] contrastive step {step}/{total_steps}: loss {means['contrastive']:.4f} (InfoNCE, log(batch) = {math.log(cfg.batch_size):.2f})")
                    if use_p:
                        entry["psi"] = means["psi"]
                        log(f"[stage0] psi step {step}/{total_steps}: loss {means['psi']:.4f}")
                    self.history.append(entry)
                    running = {"mlm": 0.0, "contrastive": 0.0, "psi": 0.0}; running_n, masked, biased = 0, 0, 0
                if step % quarter == 0 and step < total_steps:
                    self.save(self.config.out / f"checkpoint-{step}")
        return self

    # ---------- persistence ----------

    def save(self, path: Optional[Path] = None) -> Path:
        """The encoder (``AutoModel`` format, MLM head dropped) plus what the harness needs
        to recognise a Latin BERT checkpoint in a local directory."""
        path = Path(path) if path is not None else self.config.out / self.CHECKPOINT_NAME
        path.mkdir(parents=True, exist_ok=True)
        encoder = self.model.base_model if hasattr(self.model, "base_model") else self.model
        encoder.save_pretrained(path)
        self.model.save_pretrained(path / "with_mlm_head") if path.name == self.CHECKPOINT_NAME else None
        from retexo.formulations.pair_encoding import is_latin_bert, latin_bert_vocab

        if is_latin_bert(self.config.base_model):
            shutil.copy(latin_bert_vocab(self.config.base_model), path / "vocab.txt")
            (path / "latin_bert.marker").write_text(self.config.base_model + "\n")
        else:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(self.config.base_model).save_pretrained(path)
        (path / "stage0.json").write_text(json.dumps({"config": self.config.as_dict(), "history": self.history}, indent=1))
        return path


# =============================================================================
# Probe
# =============================================================================


@dataclass
class ProbeResult:
    """What the frozen probe reports for one checkpoint.

    Attributes:
        model: The checkpoint measured.
        auc_lexical_vs_nonlink: Gold lexical links (SYN / POS / NE-SUB / SUBST) against random non-link cells.
        auc_lexical_vs_competitors: The same links against the other source words of the same reuse word.
        auc_subst_vs_nonlink: The SUBST subset alone against random non-link cells.
        spread: Bootstrap standard deviation of ``auc_lexical_vs_nonlink``.
        cos_copy, cos_morph, cos_lexical, cos_nonlink: Mean cosines per class.
        n: Pairs measured per class.
    """

    model: str
    auc_lexical_vs_nonlink: float
    auc_lexical_vs_competitors: float
    auc_subst_vs_nonlink: float
    spread: float
    cos_copy: float
    cos_morph: float
    cos_lexical: float
    cos_nonlink: float
    n: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class GeometryProbe:
    """E39 replicated on a checkpoint, frozen: do gold-linked substitution pairs
    separate from non-link cells, with nothing trained?

    The gate of the pretraining note: stage 0 goes on only if the AUC rises
    by more than its bootstrap spread and the COPY cosine does not fall.

    Example:
        ```python
        probe = GeometryProbe(layer=8, device="cuda")
        before = probe.run("ashleygong03/bamman-burns-latin-bert", records)
        after = probe.run("runs/pretrain_mlm/latin-bert-reuse", records)
        ```
    """

    LEXICAL = ("SYN", "POS", "NE-SUB", "SUBST")

    def __init__(self, *, layer: int = 8, device: str = "cpu", max_length: int = 256, nonlinks_per_record: int = 20,
                 bootstrap: int = 1000, seed: int = 1):
        self.layer = layer
        self.device = device
        self.max_length = max_length
        self.nonlinks_per_record = nonlinks_per_record
        self.bootstrap = bootstrap
        self.seed = seed

    def _vectors(self, model_name: str, records: Sequence[Record], log=None):
        """Per record: (source word vectors, reuse word vectors) from one joint encoding at ``layer``."""
        import torch
        from transformers import AutoModel

        from retexo.formulations.pair_encoding import PairEncoder

        encoder = PairEncoder.build(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(records), 16):
                chunk = records[i:i + 16]
                batch, reuse_spans = encoder.encode([(list(r.source_tokens), list(r.reuse_tokens)) for r in chunk], self.max_length)
                source_spans = encoder.last_source_spans
                hidden = model(**{k: v.to(self.device) for k, v in batch.items()}, output_hidden_states=True).hidden_states[self.layer]
                for row in range(len(chunk)):
                    h = hidden[row].float().cpu()
                    s_vec = {j: h[a:b].mean(0) for j, (a, b) in enumerate(source_spans[row])}
                    r_vec = {j: h[a:b].mean(0) for j, (a, b) in enumerate(reuse_spans[row])}
                    out.append((s_vec, r_vec))
                if log and (i // 16) % 20 == 0:
                    log(f"[probe] {min(i + 16, len(records))}/{len(records)} records encoded")
        return out

    @staticmethod
    def _cos(a, b) -> float:
        import torch

        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    @staticmethod
    def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
        import numpy as np

        if not pos or not neg:
            return float("nan")
        from scipy.stats import rankdata

        p, n = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
        ranks = rankdata(np.concatenate([p, n]))              # ties share their average rank
        r_pos = ranks[: len(p)].sum()
        return float((r_pos - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))

    def run(self, model_name: str, records: Sequence[Record], *, log=None) -> ProbeResult:
        import numpy as np

        rng = random.Random(self.seed)
        vectors = self._vectors(model_name, records, log=log)
        cos: Dict[str, List[float]] = {"COPY": [], "MORPH": [], "LEXICAL": [], "SUBST": [], "NONLINK": [], "COMPETITOR": []}
        alpha = lambda w: any(c.isalpha() for c in w)  # noqa: E731
        for record, (s_vec, r_vec) in zip(records, vectors):
            linked = {(e.r, e.s) for e in record.links}
            for e in record.links:
                if e.r not in r_vec or e.s not in s_vec:
                    continue
                c = self._cos(r_vec[e.r], s_vec[e.s])
                if e.op == "COPY":
                    cos["COPY"].append(c)
                elif e.op == "MORPH":
                    cos["MORPH"].append(c)
                elif e.op in self.LEXICAL:
                    cos["LEXICAL"].append(c)
                    if e.op == "SUBST":
                        cos["SUBST"].append(c)
                    # the competitors: every other alphabetic source word for this reuse word
                    others = [s for s in s_vec if s != e.s and alpha(record.source_tokens[s])]
                    for s in rng.sample(others, min(len(others), 10)):
                        cos["COMPETITOR"].append(self._cos(r_vec[e.r], s_vec[s]))
            # random non-link cells between alphabetic words
            cells = [(t, s) for t in r_vec for s in s_vec
                     if (t, s) not in linked and alpha(record.reuse_tokens[t]) and alpha(record.source_tokens[s])]
            for t, s in rng.sample(cells, min(len(cells), self.nonlinks_per_record)):
                cos["NONLINK"].append(self._cos(r_vec[t], s_vec[s]))
        auc = self._auc(cos["LEXICAL"], cos["NONLINK"])
        boots = []
        np_rng = np.random.default_rng(self.seed)
        pos, neg = np.asarray(cos["LEXICAL"]), np.asarray(cos["NONLINK"])
        for _ in range(self.bootstrap):
            if len(pos) and len(neg):
                boots.append(self._auc(list(np_rng.choice(pos, len(pos))), list(np_rng.choice(neg, len(neg)))))
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
        return ProbeResult(
            model=model_name, auc_lexical_vs_nonlink=auc,
            auc_lexical_vs_competitors=self._auc(cos["LEXICAL"], cos["COMPETITOR"]),
            auc_subst_vs_nonlink=self._auc(cos["SUBST"], cos["NONLINK"]),
            spread=float(np.std(boots)) if boots else float("nan"),
            cos_copy=mean(cos["COPY"]), cos_morph=mean(cos["MORPH"]), cos_lexical=mean(cos["LEXICAL"]), cos_nonlink=mean(cos["NONLINK"]),
            n={k: len(v) for k, v in cos.items()},
        )
