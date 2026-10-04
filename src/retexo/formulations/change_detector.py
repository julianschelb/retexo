# retexo/formulations/change_detector.py
"""
E0: does a reuse token differ from its source token, yes or no.

The simplest thing the tagging architecture can be asked, and deliberately so.
With one-to-one operations the answer is computable by string comparison, which
makes this a **plumbing check with a known ceiling** rather than a test of
whether the model understands Latin: if it cannot reach a score that ``!=``
reaches, the pair encoding, the word-to-subword alignment or the label mapping
is broken, and every later result would inherit the fault.

Only one head, and no pointer. The pointer is not in the loop here, so its
128-position cap cannot contaminate the measurement.

The one variable is where a word's representation comes from. 59% of Latin
words split into more than one subword under XLM-R, and the tagging formulation
labels the **first** piece — so the model judges *uirumque* from ``▁u``, and
*primarius* from ``▁primari`` while its source counterpart ``▁primus`` is one
unsplit token. ``pooling="mean"`` averages the word's pieces instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from retexo.formulations.pair_encoding import PairEncoder

#: E24: fine tags a hand-labelled SUBST may resolve to, and the sentinel used
#: as the typer target for such a link.
LEXICAL_TAGS = frozenset({"SYN", "HYPER", "HYPO", "ANT", "SYN-DIST", "NE-SUB", "POS",
                          "SUBST"})
GROUP_TARGET = -2

# =============================================================================
# Config and examples
# =============================================================================


@dataclass(frozen=True)
class ChangeDetectorConfig:
    """Knobs for the binary change detector."""

    base_model: str = "FacebookAI/xlm-roberta-base"

    #: ``first`` reads the word's first subword, as the tagging formulation
    #: does today; ``mean`` averages every subword of the word.
    pooling: str = "first"

    max_length: int = 256
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    device: str = "cuda"
    seed: int = 42

    #: Add a second binary head over the *source* side. ``DEL`` has no reuse
    #: position to attach to, so without this it cannot be expressed at all
    #: (E1). Deriving it instead would need the pointer, which this
    #: formulation deliberately does not have.
    source_head: bool = False

    #: Operation vocabulary. Empty means the binary task (changed / not, as in
    #: E0 and E1); a tuple makes the operation head n-way and labels come from
    #: ``ChangeExample.operations`` instead of ``.labels``.
    operations: tuple = ()

    #: Per-class loss weights, keyed by operation. A rare class costs almost
    #: nothing to ignore -- SUBST is 3.7% of tokens, so never predicting it is
    #: close to loss-optimal -- and this is the cheapest way to say otherwise.
    class_weights: tuple = ()

    #: Added to a class's logit at inference only. GECToR's dial: it trades
    #: precision for recall without retraining, and can be tuned on a
    #: validation split rather than baked into the weights.
    logit_bias: tuple = ()

    #: Focal-loss gamma. 0 disables it. Down-weights examples the model already
    #: gets right, which here means the flood of easy INS tokens.
    focal_gamma: float = 0.0

    #: Pointer head: for each reuse word, *which* source word it came from, or
    #: none at all. The tagging head cannot express correspondence -- it labels
    #: a reuse word without reference to any source word -- which is why SUBST
    #: and INS are confusable at all: 59 of the 70 missed SUBSTs in E5 went to
    #: INS, not to COPY. A model that has to name the source word can only call
    #: something INS by declining to point anywhere.
    pointer: bool = False

    #: Loss weight on "came from nowhere". Only 14.5% of reuse words align, so
    #: pointing at the null every time is nearly loss-optimal in exactly the way
    #: never predicting SUBST was. 0.2 is roughly the inverse of that rate,
    #: which is where a balanced objective sits -- taken from the measurement
    #: rather than swept, since E5 spent six arms establishing that this family
    #: of correction helps monotonically over the range that matters.
    null_pointer_weight: float = 0.2
    #: How the pointer turns two contextual vectors into a score.
    #:
    #: ``"dot"``     f(h_j).g(h_i)/sqrt(d) over the full hidden width -- the
    #:               Vinyals pointer this project has used throughout.
    #: ``"colbert"`` ColBERT's late interaction: project both sides to
    #:               ``pointer_dim``, L2-normalise, and take the cosine, scaled
    #:               by a learned temperature. ColBERT's MaxSim is a per-query
    #:               argmax over document tokens, which is what this pointer
    #:               already computes -- so the architectures differ only in
    #:               the geometry of the comparison, and this makes that
    #:               difference testable rather than assumed.
    #:
    #: The learned null is kept in both: ColBERT has none, because in retrieval
    #: every query token must match something, whereas 84% of reuse tokens here
    #: align to nothing at all.
    pointer_style: str = "dot"
    pointer_dim: int = 128
    #: Sharpness of the softmax over candidates under ``"colbert"``. Cosine is
    #: bounded in [-1, 1], so without this the distribution is nearly flat.
    #: ColBERT itself has no temperature -- it sums raw MaxSim scores and never
    #: takes a softmax -- so this is an addition the task requires, and is
    #: stated rather than buried in the constructor.
    pointer_temperature: float = 20.0
    #: One scalar at 2e-5 moved 0.0002 in 200 steps: a constant pretending to be
    #: learned. Its own group, several orders up, lets it actually be fitted.
    temperature_lr: float = 1e-2

    #: Split each batch across every visible GPU. The encoder is small enough
    #: that this is worth it only at larger batch sizes, but both A40s are
    #: otherwise idle. Outputs are gathered back onto device 0, so the pooling
    #: below is unaffected.
    data_parallel: bool = False

    #: E24 -- the link typer. For every reuse word the pointer links to a
    #: source word, name the *kind* of change among the fine operations
    #: (NOP, MORPH, SYN, HYPER, HYPO, ANT, SYN-DIST, NE-SUB, POS, SPLIT,
    #: MERGE). Empty disables the head. The input is the two contextual
    #: vectors, their difference and product, and -- when ``feature_dim`` is
    #: set -- the symbolic evidence vector for the pair, which is what E2's
    #: encoder-only typer lacked when it could not tell a hypernym from a
    #: synonym.
    fine_operations: tuple = ()
    feature_dim: int = 0
    #: Ablation switch: keep the head but zero the evidence, so the typer sees
    #: only what E2's did.
    use_link_features: bool = True
    typer_hidden: int = 256
    fine_class_weights: tuple = ()
    #: The two new heads start from random weights; at the encoder's 2e-5 a
    #: fresh MLP barely moves in the few hundred steps a run has (the smoke
    #: run's typer predicted NOP for everything). Their own group, as the
    #: pointer temperature already has.
    typer_lr: float = 1e-3
    #: Frame tokens are ~5% of reuse words; the positive class is up-weighted
    #: so the head does not learn to say "no frame" everywhere.
    frame_positive_weight: float = 3.0
    #: A hand-labelled SUBST is a lexical change whose relation the labeller
    #: did not name. Rather than mask it, it supervises the *group*: the loss
    #: is the negative log of the total probability on the lexical classes.
    #: Without this, eight gold passes that name only NOP and MORPH teach the
    #: typer to forget every lexical class it learned on synthetic data --
    #: which is exactly what the medium smoke run showed (SUBST 0.000).
    group_loss_weight: float = 1.0
    #: E25: build a TypedPointer instead of this class when reloading a saved
    #: model. Set by run_e25; read by run_e5.load_model.
    typed_pointer: bool = False
    #: Failure-mode dry run, chain 2 row 6: word-level input channels ("lemma", "lemma,pos",
    #: "lemma,pos,morph") summed onto every piece's word embedding; "" = none.
    channels: str = ""
    #: learning rate of the channel tables; 0 = the fresh heads' rate (typer_lr), else its own group
    channel_lr: float = 0.0
    #: Failure-mode dry run, row 9 (the unified test): a second view on the same encoder -- a span
    #: head (start / end over the source words, spans of up to ``span_max_len`` words, its own null)
    #: whose per-word marginal is averaged with the pointer's at ``span_weight`` before the decoder.
    span_view: bool = False
    span_weight: float = 0.5
    span_max_len: int = 3
    #: Failure-mode dry run, chain 15 (2026-09-19, the unified test, second attempt): the stretch tower --
    #: the span view and the stretch head read their own copy of the encoder's last ``stretch_tower``
    #: layers (0 = the shared vectors, row 9's setting), so the two views keep separate representations
    #: inside one model; the stretch head labels every reuse word VERBATIM / ALLUSION / FRAME / NOMATCH and,
    #: at ``null_fuse`` != 0, its log p(NOMATCH) is added to the pointer's null logit at that (learned)
    #: weight: the per-stretch null prior, learned jointly instead of a global theta.
    stretch_tower: int = 0
    stretch_head: bool = False
    null_fuse: float = 0.0
    stretch_weight: float = 1.0
    #: Chain 16 (2026-09-19, the slot convolution): a small 2-D convolution over each pair's matrix of
    #: first-pass link probabilities (plus an anchor channel: same form or lemma), whose output is added
    #: to every cell before the null competition -- Stengel-Eskin et al. 2019's alignment-matrix
    #: convolution, i.e. the annotator's slot convention ("everything around it matches") as a learned,
    #: differentiable score term. ``slot_conv`` = hidden channels (0 = off), ``slot_kernel`` the window.
    slot_conv: int = 0
    slot_kernel: int = 3
    #: chain 18: the convolution reads the anchor channel and the mask only (no first-pass probabilities), so
    #: the bump follows copies and inflections around a cell and not the model's own confidence
    slot_anchor_only: bool = False
    #: E24 -- a binary head over reuse words: does this word belong to an
    #: attribution formula (FRAME)? Supervised by the constructed frames and by
    #: the 153 hand-labelled spans in the training folds.
    frame_head: bool = False


@dataclass(frozen=True)
class ChangeExample:
    """One (source, reuse) pair with a binary label per reuse word.

    ``operations`` records which operation produced each reuse word, so recall
    can be decomposed by operation type at scoring time. It is never shown to
    the model.
    """

    source_tokens: List[str]
    target_tokens: List[str]
    labels: List[int]
    operations: List[str]
    n_operations: int
    #: Per *source* word, 1 where the reuse dropped it. Only used when the
    #: config asks for a source head; ``None`` means the question was not put.
    source_labels: Optional[List[int]] = None
    source_operations: Optional[List[str]] = None
    #: Per *reuse* word, the index of the source word it came from, or -1 for a
    #: word the citing author supplied. This is the pointer's supervision, and
    #: the generator knows it exactly: it spliced the fragment in.
    alignments: Optional[List[int]] = None
    #: E24. Per reuse word, the fine operation of its link ("INS" for none,
    #: "?" for a link whose kind is unknown and must not be trained on), a
    #: 0/1 frame flag, and the symbolic evidence vector of its gold link.
    fine_operations: Optional[List[str]] = None
    frame_labels: Optional[List[int]] = None
    link_features: Optional[List[Optional[List[float]]]] = None
    #: E25. The evidence for *every* (reuse word, source word) pair, as a
    #: float16 array of shape [n_reuse, n_source, feature_dim]. Filled by
    #: ``e24.featurize_pairs`` and read by the typed pointer, which needs it
    #: inside the alignment decision rather than after it.
    pair_features: Optional[object] = None


# =============================================================================
# Model
# =============================================================================


class ChangeDetector:
    """A binary token classifier over the reuse side of a pair."""

    def __init__(self, config: Optional[ChangeDetectorConfig] = None) -> None:
        import torch
        from transformers import AutoModel

        self.config = config or ChangeDetectorConfig()
        if self.config.pooling not in ("first", "mean"):
            raise ValueError(f"unknown pooling: {self.config.pooling!r}")

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

        self._pair_encoder = PairEncoder.build(self.config.base_model)
        # ModernBERT torch.compiles its MLP by default, and DataParallel
        # replicates modules by FX-tracing them -- FX cannot trace a
        # dynamo-optimised function, so every ModernBERT run died at the first
        # forward pass. Disabling the compiled reference path costs a little
        # speed and makes the backbone usable.
        encoder_kwargs = {}
        if "modernbert" in self.config.base_model.lower():
            encoder_kwargs["reference_compile"] = False
        self._encoder = AutoModel.from_pretrained(self.config.base_model,
                                                  **encoder_kwargs)
        hidden = self._encoder.config.hidden_size
        self._classes = list(self.config.operations) or None
        self._index = {c: i for i, c in enumerate(self._classes or [])}
        self._head = torch.nn.Linear(hidden, len(self._classes) if self._classes else 2)
        self._source_head = (
            torch.nn.Linear(hidden, 2) if self.config.source_head else None
        )
        # Two projections and a learned null, after Vinyals' pointer networks:
        # score(j, i) = f(h_j).g(h_i)/sqrt(d), with the null competing in the
        # same softmax so that "from nowhere" is a choice among the candidates
        # rather than a threshold on top of them. Scaling by sqrt(d) keeps the
        # dot products in a range where the softmax has usable gradients, which
        # at hidden 768 it otherwise does not.
        colbert = self.config.pointer_style == "colbert"
        width = self.config.pointer_dim if colbert else hidden
        self._pointer_source = (
            torch.nn.Linear(hidden, width) if self.config.pointer else None
        )
        self._pointer_target = (
            torch.nn.Linear(hidden, width) if self.config.pointer else None
        )
        self._pointer_null = (
            torch.nn.Parameter(torch.zeros(width).normal_(std=0.02))
            if self.config.pointer else None
        )
        # Cosine is bounded in [-1, 1], so a fixed 1/sqrt(d) would leave the
        # softmax almost flat; ColBERT-style scoring gets a learned temperature
        # instead, initialised where the logits have usable gradients.
        self._pointer_temperature = (
            torch.nn.Parameter(torch.tensor(float(self.config.pointer_temperature)))
            if self.config.pointer and colbert else None
        )
        self._pointer_scale = 1.0 if colbert else hidden ** 0.5
        if self._pointer_null is not None:
            self._pointer_null.data = self._pointer_null.data.to(self.config.device)
        if self._pointer_temperature is not None:
            self._pointer_temperature.data = \
                self._pointer_temperature.data.to(self.config.device)
        # E24: the link typer names the kind of change on a link the pointer
        # has already placed. It reads both words' contextual vectors, their
        # elementwise difference and product, and the symbolic evidence for
        # the pair. The pointer is left exactly as it was: locating is solved,
        # and this head is not allowed to disturb it beyond sharing the encoder.
        self._fine = list(self.config.fine_operations) or None
        self._fine_index = {c: i for i, c in enumerate(self._fine or [])}
        #: The classes a hand-labelled SUBST may resolve to.
        self._lexical_index = [i for i, c in enumerate(self._fine or [])
                               if c in LEXICAL_TAGS]
        self._typer = None
        self._typer_evidence = None
        if self._fine:
            self._typer = torch.nn.Sequential(
                torch.nn.Linear(4 * hidden, self.config.typer_hidden),
                torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(self.config.typer_hidden, len(self._fine)))
            if self.config.use_link_features and self.config.feature_dim:
                # A direct linear path from the evidence to the logits. Fed
                # through the MLP beside 3,072 contextual dimensions, 23 flags
                # were drowned: the first full run's typer scored *below* the
                # lemma rule whose output it could see. Here "same lemma ->
                # MORPH" is one weight.
                self._typer_evidence = torch.nn.Linear(self.config.feature_dim,
                                                       len(self._fine))
        self._frame_head = (torch.nn.Linear(hidden, 2)
                            if self.config.frame_head else None)
        self._channels = None
        if getattr(self.config, "channels", ""):
            from retexo.formulations.word_channels import WordChannels

            self._channels = WordChannels(tuple(k.strip() for k in self.config.channels.split(",") if k.strip()), hidden)
        for module in (self._encoder, self._head, self._source_head,
                       self._pointer_source, self._pointer_target,
                       self._typer, self._typer_evidence, self._frame_head, self._channels):
            if module is not None:
                module.to(self.config.device)

        self.gpus = 1
        if self.config.data_parallel and self.config.device.startswith("cuda"):
            available = torch.cuda.device_count()
            if available > 1:
                self._encoder = torch.nn.DataParallel(self._encoder)
                self.gpus = available

        self.losses: List[float] = []
        #: Reuse words lost to truncation. Silent data loss would make the
        #: ceiling unreachable for reasons that have nothing to do with the
        #: model, so it is counted rather than ignored.
        self.truncated_words = 0

    # ---------- Encoding ----------

    def _run_encoder(self, batch):
        """The backbone's forward; with word channels the piece embeddings are built here."""
        if self._channels is None or "channel_ids" not in batch:
            return self._encoder(**{k: v for k, v in batch.items() if k != "channel_ids"})
        encoder = getattr(self._encoder, "module", self._encoder)
        embeddings = encoder.embeddings
        table = getattr(embeddings, "word_embeddings", None) or getattr(embeddings, "tok_embeddings")
        embeds = table(batch["input_ids"]) + self._channels(batch["channel_ids"])
        rest = {k: v for k, v in batch.items() if k not in ("input_ids", "channel_ids")}
        return self._encoder(inputs_embeds=embeds, **rest)

    def _encode(self, examples: Sequence[ChangeExample]):
        import torch

        pairs = [(list(e.source_tokens), list(e.target_tokens)) for e in examples]
        batch, spans = self._pair_encoder.encode(pairs, self.config.max_length)
        self._last_label_positions = getattr(self._pair_encoder, "last_label_positions", None)   # E35
        if self._channels is not None:
            batch["channel_ids"] = self._channels.ids(
                tuple(batch["input_ids"].shape), spans, getattr(self._pair_encoder, "last_source_spans", ()), pairs)

        rows, starts, ends, targets, align = [], [], [], [], []
        fine_targets, frame_targets, features = [], [], []
        feature_dim = self.config.feature_dim
        for row, example in enumerate(examples):
            if len(spans[row]) < len(example.target_tokens):
                self.truncated_words += len(example.target_tokens) - len(spans[row])
            for word, (start, end) in enumerate(spans[row]):
                if word >= len(example.labels):
                    break
                rows.append(row)
                starts.append(start)
                ends.append(max(end, start + 1))
                if self._classes:
                    tag = example.operations[word] if word < len(example.operations) else "INS"
                    targets.append(self._index.get(tag, -100))
                else:
                    targets.append(example.labels[word])
                # Source word this reuse word came from, -1 for none. Unlabelled
                # (-100) where the example does not carry alignments at all, so
                # gold pairs without them contribute no pointer loss.
                links = example.alignments
                link = links[word] if links and word < len(links) else -100
                align.append(link)
                # E24: fine tag of the link, frame flag, and evidence vector.
                # -100 wherever the example does not say, so a pair labelled
                # only coarsely trains the coarse heads and nothing else.
                fine = example.fine_operations
                tag = fine[word] if fine and word < len(fine) else None
                if self._fine and link is not None and link >= 0:
                    # "?" is a lexical change of unknown kind: the group target
                    fine_targets.append(GROUP_TARGET if tag == "?"
                                        else self._fine_index.get(tag, -100))
                else:
                    fine_targets.append(-100)
                frames = example.frame_labels
                frame_targets.append(int(frames[word])
                                     if frames and word < len(frames) else -100)
                phi = example.link_features
                vec = phi[word] if phi and word < len(phi) else None
                features.append(list(vec) if vec is not None and len(vec) == feature_dim
                                else [0.0] * feature_dim)
        source = None
        # The source side is encoded for the deletion head, and also for the
        # pointer, which needs every source word as a candidate whether or not
        # it carries a deletion label.
        if self.config.source_head or self.config.pointer:
            src_spans = self._pair_encoder.last_source_spans
            s_rows, s_starts, s_ends, s_targets, s_words = [], [], [], [], []
            for row, example in enumerate(examples):
                labels = example.source_labels or []
                limit = (len(example.source_tokens) if self.config.pointer
                         else len(labels))
                for word, (start, end) in enumerate(src_spans[row]):
                    if word >= limit:
                        break
                    s_rows.append(row); s_starts.append(start)
                    s_ends.append(max(end, start + 1)); s_words.append(word)
                    # -100 where the pointer widened the set past the labels,
                    # so the deletion head is unaffected by the extra words.
                    s_targets.append(labels[word] if word < len(labels) else -100)
            source = (torch.tensor(s_rows), torch.tensor(s_starts),
                      torch.tensor(s_ends), torch.tensor(s_targets),
                      torch.tensor(s_words))
        extra = (torch.tensor(fine_targets), torch.tensor(frame_targets),
                 torch.tensor(features, dtype=torch.float32)
                 if features else torch.zeros((0, feature_dim)))
        return (batch, torch.tensor(rows), torch.tensor(starts),
                torch.tensor(ends), torch.tensor(targets), source,
                torch.tensor(align), extra)

    def _word_vectors(self, hidden, rows, starts, ends):
        """One vector per labelled word, pooled as the config asks."""
        import torch

        if self.config.pooling == "first":
            return hidden[rows, starts]
        # Mean over each word's subwords. Spans are short, so a gather over a
        # padded index matrix is cheaper than a Python loop per word.
        widths = (ends - starts)
        longest = int(widths.max().item()) if len(widths) else 1
        offsets = torch.arange(longest, device=hidden.device).unsqueeze(0)
        index = starts.unsqueeze(1).to(hidden.device) + offsets
        mask = offsets < widths.unsqueeze(1).to(hidden.device)
        index = index.clamp(max=hidden.shape[1] - 1)
        gathered = hidden[rows.unsqueeze(1).to(hidden.device), index]
        gathered = gathered * mask.unsqueeze(-1)
        return gathered.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _logits(self, examples: Sequence[ChangeExample]):
        import torch

        batch, rows, starts, ends, targets, source, align, extra = self._encode(examples)
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        hidden = self._run_encoder(batch).last_hidden_state
        vectors = self._word_vectors(hidden, rows, starts, ends)
        logits = self._head(vectors)
        source_out = None
        s_vectors = None
        if source is not None and source[0].numel():
            s_rows, s_starts, s_ends, s_targets, s_words = source
            s_vectors = self._word_vectors(hidden, s_rows, s_starts, s_ends)
            if self._source_head is not None:
                source_out = (self._source_head(s_vectors),
                              s_targets.to(self.config.device))
        pointer_out = None
        if self.config.pointer and s_vectors is not None and rows.numel():
            pointer_out = self._pointer_scores(
                vectors, rows, s_vectors, source[0], source[4], align)
        # E24 heads. The typer is trained on the *gold* link -- the source
        # word the example says the reuse word came from -- so that typing is
        # learned separately from locating and a pointer error never teaches
        # the typer a wrong relation.
        typer_out, frame_out = None, None
        fine_t, frame_t, phi = extra
        if self._typer is not None and s_vectors is not None and rows.numel():
            typer_out = self._typer_logits(vectors, rows, s_vectors, source[0],
                                           source[4], align, fine_t, phi)
        if self._frame_head is not None and rows.numel():
            frame_out = (self._frame_head(vectors), frame_t.to(self.config.device))
        return (logits, targets.to(self.config.device), source_out, pointer_out,
                (typer_out, frame_out))

    def _typer_forward(self, h_t, h_s, phi):
        """Logits for one link: both vectors, how they differ, plus evidence."""
        import torch

        logits = self._typer(torch.cat([h_t, h_s, (h_t - h_s).abs(), h_t * h_s], dim=-1))
        if self._typer_evidence is not None:
            logits = logits + self._typer_evidence(phi.to(h_t.device))
        return logits

    @staticmethod
    def evidence_veto(logits, phi, index: Dict[str, int]):
        """Rule out fine tags the evidence makes impossible, with -inf logits.

        NOP if and only if the forms are identical after normalization; NE-SUB
        only between two names; SPLIT / MERGE only with an enclitic on exactly
        one side whose stem matches the other. Feature columns follow
        ``link_features.FEATURE_NAMES``: 0 same_form, 12 both_names,
        14 enclitic_src, 15 enclitic_tgt, 16 enclitic_stem_match.
        """
        import torch

        masked = logits.clone()
        neg = float("-inf")
        same = phi[:, 0] > 0.5
        if "NOP" in index:
            masked[~same, index["NOP"]] = neg
            others = torch.tensor([i for c, i in index.items() if c != "NOP"],
                                  device=logits.device, dtype=torch.long)
            rows_same = same.nonzero(as_tuple=True)[0]
            if len(others) and len(rows_same):
                masked[rows_same.unsqueeze(1), others.unsqueeze(0)] = neg
        if "NE-SUB" in index:
            masked[phi[:, 12] < 0.5, index["NE-SUB"]] = neg
        if "MORPH" in index:
            # an inflection shares its stem: same lemma, or at least half the
            # characters in common (Threicius/Thracius, neque/nec pass; the
            # oculos/lumina the head once called MORPH does not)
            masked[(phi[:, 1] < 0.5) & (phi[:, 18] < 0.5), index["MORPH"]] = neg
        enclitic_ok = (phi[:, 16] > 0.5) & ((phi[:, 14] > 0.5) != (phi[:, 15] > 0.5))
        for tag in ("SPLIT", "MERGE"):
            if tag in index:
                masked[~enclitic_ok, index[tag]] = neg
        return masked

    def _evidence_veto(self, logits, phi):
        """Definitions the evidence settles are not left to the classifier.

        A link is NOP if and only if the two forms are identical after
        normalization; NE-SUB needs two names; SPLIT and MERGE need an
        enclitic on exactly one side whose stem matches the other. The second
        full run's typer called *aetate -> aevo* a NOP and *ipsis -> bestias*
        a name substitution, both impossible by definition, with the deciding
        flag sitting in its input. Feature indices follow ``FEATURE_NAMES``:
        0 same_form, 12 both_names, 14/15 enclitic_src/tgt, 16 stem match.
        """
        if not self.config.use_link_features or phi.shape[-1] < 17:
            return logits
        return self.evidence_veto(logits, phi.to(logits.device), self._fine_index)

    def _typer_logits(self, vectors, rows, s_vectors, s_rows, s_words, align,
                      fine_t, phi):
        """Logits over the fine operations for every reuse word with a known link."""
        import torch

        flat = {}
        for k, (r, w) in enumerate(zip(s_rows.tolist(), s_words.tolist())):
            flat[(r, w)] = k
        sel_t, sel_s = [], []
        for k, (r, a, f) in enumerate(zip(rows.tolist(), align.tolist(), fine_t.tolist())):
            if a >= 0 and f != -100 and (r, a) in flat:
                sel_t.append(k); sel_s.append(flat[(r, a)])
        if not sel_t:
            return None
        device = vectors.device
        h_t = vectors[torch.tensor(sel_t, device=device)]
        h_s = s_vectors[torch.tensor(sel_s, device=device)]
        evidence = phi[torch.tensor(sel_t)]
        logits = self._typer_forward(h_t, h_s, evidence)
        return logits, fine_t[torch.tensor(sel_t)].to(device)


    def _pointer_scores(self, vectors, rows, s_vectors, s_rows, s_words, align):
        """Scores over [null, every source word of the same pair].

        Candidates differ per pair, so they are gathered into a padded matrix
        and the positions belonging to other pairs are masked to -inf before the
        softmax -- otherwise a reuse word could point into a different pair's
        source, which is not merely wrong but trivially separable.
        """
        import torch

        device = self.config.device
        rows = rows.to(device); s_rows = s_rows.to(device)
        s_words = s_words.to(device); align = align.to(device)

        # every pair of the chunk, whether or not both sides survived truncation: a source row
        # with no target row (or the reverse) must still index inside the tables
        n_rows = int(max(rows.max().item(), s_rows.max().item())) + 1 if rows.numel() and s_rows.numel() else 0
        counts = torch.bincount(s_rows, minlength=n_rows) if n_rows else s_rows.new_zeros(0)
        width = int(counts.max().item()) if counts.numel() else 0
        if width == 0:
            return None

        # column of each source word within its own pair (a running index per row, whatever
        # the order of ``s_rows``), and the flat index back into s_vectors
        offsets = torch.cumsum(counts, 0) - counts
        order = torch.argsort(s_rows, stable=True)
        columns_sorted = torch.arange(len(s_rows), device=device) - offsets[s_rows[order]]
        columns = torch.empty_like(columns_sorted)
        columns[order] = columns_sorted
        gather = torch.zeros((n_rows, width), dtype=torch.long, device=device)
        valid = torch.zeros((n_rows, width), dtype=torch.bool, device=device)
        gather[s_rows, columns] = torch.arange(len(s_rows), device=device)
        valid[s_rows, columns] = True

        projected = self._pointer_source(s_vectors)               # [S, H]
        queries = self._pointer_target(vectors)                   # [T, H]
        null_vector = self._pointer_null
        if self.config.pointer_style == "colbert":
            # Late interaction: unit vectors, so the dot product is a cosine and
            # the temperature -- not 1/sqrt(d) -- sets how sharp the softmax
            # over candidates is.
            #
            # The null is deliberately NOT normalised. Normalising it caps its
            # score at the temperature, so it could only rotate and never grow,
            # and against ~26 candidates in a softmax it would lose almost
            # always. Leaving its magnitude free is what lets the model learn
            # *how strongly* to decline -- which is the whole difficulty here,
            # since 84% of reuse tokens align to nothing.
            projected = torch.nn.functional.normalize(projected, dim=-1)
            queries = torch.nn.functional.normalize(queries, dim=-1)
            scale = 1.0 / self._pointer_temperature.clamp(min=1e-3)
        else:
            scale = self._pointer_scale
        candidates = projected[gather[rows]]                      # [T, width, H]
        scores = torch.einsum("th,twh->tw", queries, candidates) / scale
        scores = scores.masked_fill(~valid[rows], float("-inf"))
        null = (queries @ null_vector) / scale
        scores = torch.cat([null.unsqueeze(1), scores], dim=1)    # column 0 = null

        # gold column: 0 for null, else the source word's own column + 1
        word_to_column = torch.zeros((n_rows, width), dtype=torch.long, device=device)
        word_to_column[s_rows, s_words.clamp(max=width - 1)] = columns
        present = torch.zeros((n_rows, width), dtype=torch.bool, device=device)
        present[s_rows, s_words.clamp(max=width - 1)] = True
        safe = align.clamp(min=0, max=width - 1)
        found = present[rows, safe] & (align >= 0)
        gold = torch.where(found, word_to_column[rows, safe] + 1,
                           torch.zeros_like(align))
        # A word whose source was truncated away has no column to point at, so
        # it is dropped rather than relabelled as null -- calling it null would
        # teach the model that truncation means the author invented the word.
        gold = torch.where((align >= 0) & ~found, torch.full_like(align, -100), gold)
        gold = torch.where(align == -100, torch.full_like(align, -100), gold)
        return scores, gold

    # ---------- Training ----------

    def _loss_functions(self) -> Dict[str, object]:
        """The training losses and their class weights (one place for ``fit`` and ``evaluation_loss``)."""
        import torch

        weights = None
        if self.config.class_weights and self._classes:
            table = dict(self.config.class_weights)
            weights = torch.tensor(
                [float(table.get(c, 1.0)) for c in self._classes],
                device=self.config.device)
        fine_weights = None
        if self._fine and self.config.fine_class_weights:
            table = dict(self.config.fine_class_weights)
            fine_weights = torch.tensor([float(table.get(c, 1.0)) for c in self._fine],
                                        device=self.config.device)
        return {
            "weights": weights, "fine_weights": fine_weights,
            "loss": torch.nn.CrossEntropyLoss(ignore_index=-100, weight=weights),
            # The source head is binary (deleted / not), so it must not inherit a
            # weight vector sized for the operation classes.
            "source": torch.nn.CrossEntropyLoss(ignore_index=-100),
            "frame": torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                weight=torch.tensor([1.0, float(self.config.frame_positive_weight)], device=self.config.device)),
        }

    def _chunk_loss(self, chunk: Sequence[ChangeExample], fns: Dict[str, object]):
        """The training loss of one batch, or ``None`` when it has no target."""
        logits, targets, source_out, pointer_out, extra = self._logits(chunk)
        if targets.numel() == 0:
            return None
        loss = (self._focal(logits, targets, fns["weights"])
                if self.config.focal_gamma else fns["loss"](logits, targets))
        if source_out is not None:
            # Both sides weigh equally: a missed deletion is as wrong
            # as a missed substitution.
            loss = loss + fns["source"](*source_out)
        if pointer_out is not None:
            loss = loss + self._pointer_loss(*pointer_out)
        typer_out, frame_out = extra
        if typer_out is not None:
            loss = loss + self._typer_loss(*typer_out, fns["fine_weights"])
        if frame_out is not None and bool((frame_out[1] != -100).any()):
            loss = loss + fns["frame"](*frame_out)
        return loss

    def evaluation_loss(self, examples: Sequence[ChangeExample]) -> Optional[float]:
        """The training loss on ``examples`` without a gradient (the validation loss), the mean over batches."""
        import torch

        fns = self._loss_functions()
        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                loss = self._chunk_loss(list(examples[start:start + self.config.batch_size]), fns)
                if loss is not None:
                    total += float(loss.item()); n += 1
        return total / n if n else None

    def fit(self, examples: Sequence[ChangeExample], *, log=None,
            on_batch: Optional[Callable[[int, int], None]] = None) -> "ChangeDetector":
        """Train on ``examples``; ``on_batch(done, total)`` is called after every optimizer step (a training
        monitor's evaluations inside the epoch)."""
        import torch

        parameters = list(self._encoder.parameters()) + list(self._head.parameters())
        if self._source_head is not None:
            parameters += list(self._source_head.parameters())
        if self._channels is not None:
            parameters += list(self._channels.parameters())
        if self.config.pointer:
            parameters += list(self._pointer_source.parameters())
            parameters += list(self._pointer_target.parameters())
            parameters += [self._pointer_null]
        groups = [{"params": parameters, "lr": self.config.learning_rate}]
        fresh = []
        if self._typer is not None:
            fresh += list(self._typer.parameters())
        if self._typer_evidence is not None:
            fresh += list(self._typer_evidence.parameters())
        if self._frame_head is not None:
            fresh += list(self._frame_head.parameters())
        if fresh:
            groups.append({"params": fresh, "lr": self.config.typer_lr})
        if self.config.pointer and self._pointer_temperature is not None:
            groups.append({"params": [self._pointer_temperature],
                           "lr": self.config.temperature_lr})
        optimizer = torch.optim.AdamW(groups, lr=self.config.learning_rate)
        fns = self._loss_functions()
        order = list(examples)
        rng = random.Random(self.config.seed)

        self._encoder.train(); self._head.train()
        if self._source_head is not None:
            self._source_head.train()
        if self._typer is not None:
            self._typer.train()
        if self._frame_head is not None:
            self._frame_head.train()
        if self.config.pointer:
            self._pointer_source.train(); self._pointer_target.train()
        for epoch in range(self.config.epochs):
            rng.shuffle(order)
            epoch_losses = []
            for start in range(0, len(order), self.config.batch_size):
                chunk = order[start : start + self.config.batch_size]
                if not chunk:
                    continue
                loss = self._chunk_loss(chunk, fns)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                if on_batch is not None:
                    on_batch(start + len(chunk), len(order))
            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            self.losses.append(mean_loss)
            if log:
                log(f"    epoch {epoch + 1}/{self.config.epochs}  loss {mean_loss:.4f}")
        return self

    # ---------- Inference ----------

    def _typer_loss(self, logits, targets, weights):
        """Cross-entropy on links whose kind is known; a group term on the rest.

        A hand-labelled SUBST says only "changed, lexically". Its loss is the
        negative log of the probability mass on the lexical classes together,
        so real data teaches the boundary between NOP / MORPH and the lexical
        group while the constructed data, where every link is typed, decides
        the relation inside the group.
        """
        import torch
        import torch.nn.functional as F

        exact = targets >= 0
        group = targets == GROUP_TARGET
        loss = logits.sum() * 0.0
        if bool(exact.any()):
            loss = loss + F.cross_entropy(logits[exact], targets[exact], weight=weights)
        if bool(group.any()) and self._lexical_index:
            log_p = F.log_softmax(logits[group], dim=-1)
            idx = torch.tensor(self._lexical_index, device=logits.device)
            in_group = torch.logsumexp(log_p[:, idx], dim=-1)
            loss = loss + self.config.group_loss_weight * (-in_group).mean()
        return loss

    def _pointer_loss(self, scores, gold):
        """Cross-entropy over the candidates, with the null down-weighted.

        Only 14.5% of reuse words align, so a pointer trained on a plain
        objective learns to point nowhere -- the same corner E5's unweighted
        SUBST fell into. The weight applies to the *gold* being null rather than
        to the prediction, so it re-prices the examples the model would
        otherwise be right to ignore.
        """
        import torch
        import torch.nn.functional as F

        losses = F.cross_entropy(scores, gold, ignore_index=-100, reduction="none")
        labelled = gold != -100
        if not bool(labelled.any()):
            return scores.sum() * 0.0
        weights = torch.where(gold == 0,
                              torch.full_like(losses, self.config.null_pointer_weight),
                              torch.ones_like(losses))
        weights = weights * labelled.float()
        return (losses * weights).sum() / weights.sum().clamp(min=1e-6)

    def _focal(self, logits, targets, weights):
        """Focal loss: down-weight what the model already gets right.

        With 86% of tokens INS, plain cross-entropy is dominated by examples
        that are already correct. Focal loss scales each by ``(1 - p)^gamma``,
        so the rare classes keep influencing the gradient.
        """
        import torch
        import torch.nn.functional as F

        log_probability = F.log_softmax(logits, dim=-1)
        valid = targets != -100
        if not valid.any():
            return logits.sum() * 0.0
        logits, targets = log_probability[valid], targets[valid]
        chosen = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - chosen.exp()) ** self.config.focal_gamma) * chosen
        if weights is not None:
            loss = loss * weights[targets]
        return loss.mean()

    def evaluate_loss(self, examples: Sequence[ChangeExample]) -> float:
        """Mean loss on held-out examples, for the early-stopping criterion."""
        import torch

        self._encoder.eval(); self._head.eval()
        if self._source_head is not None:
            self._source_head.eval()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        losses = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                logits, targets, source_out, _, _ = self._logits(chunk)
                if targets.numel() == 0:
                    continue
                loss = (self._focal(logits, targets, None)    # unweighted, like the plain branch's loss_fn
                        if self.config.focal_gamma else loss_fn(logits, targets))
                if source_out is not None:
                    loss = loss + loss_fn(*source_out)
                losses.append(float(loss.item()))
        self._encoder.train(); self._head.train()
        if self._source_head is not None:
            self._source_head.train()
        return sum(losses) / max(len(losses), 1)

    def predict_operations(self, examples: Sequence[ChangeExample]) -> List[List[str]]:
        """Per example, the predicted operation tag per reuse word."""
        if not self._classes:
            raise ValueError("configure `operations` to predict tags")
        import torch

        self._encoder.eval(); self._head.eval()
        out: List[List[str]] = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length)
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    tags = ["INS"] * len(example.target_tokens)
                    usable = [(w, a, b) for w, (a, b) in enumerate(spans[row])
                              if w < len(example.target_tokens)]
                    if usable:
                        st = torch.tensor([a for _, a, _ in usable])
                        en = torch.tensor([max(b, a + 1) for _, a, b in usable])
                        rw = torch.full_like(st, row)
                        vectors = self._word_vectors(hidden, rw, st, en)
                        logits = self._head(vectors)
                        if self.config.logit_bias and self._classes:
                            table = dict(self.config.logit_bias)
                            bias = torch.tensor(
                                [float(table.get(c, 0.0)) for c in self._classes],
                                device=logits.device)
                            logits = logits + bias
                        chosen = logits.argmax(dim=-1).tolist()
                        for (word, _, _), guess in zip(usable, chosen):
                            tags[word] = self._classes[guess]
                    out.append(tags)
        self._encoder.train(); self._head.train()
        return out

    def predict_alignment(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """Per example, the source word each reuse word points at, -1 for none.

        Scored a pair at a time: the candidate set is that pair's own source
        words, so there is nothing to pad and nothing to mask. Words lost to
        truncation come back as -1, which is also what the caller reads as INS.
        """
        import torch

        if not self.config.pointer:
            raise ValueError("predict_alignment needs pointer=True")
        self._encoder.eval(); self._pointer_source.eval(); self._pointer_target.eval()
        out: List[List[int]] = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length)
                source_spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    links = [-1] * len(example.target_tokens)
                    words = [(w, a, b) for w, (a, b) in enumerate(spans[row])
                             if w < len(example.target_tokens)]
                    sources = [(w, a, b) for w, (a, b) in enumerate(source_spans[row])
                               if w < len(example.source_tokens)]
                    if words and sources:
                        def vectors_for(items):
                            st = torch.tensor([a for _, a, _ in items])
                            en = torch.tensor([max(b, a + 1) for _, a, b in items])
                            return self._word_vectors(
                                hidden, torch.full_like(st, row), st, en)

                        queries = self._pointer_target(vectors_for(words))
                        keys = self._pointer_source(vectors_for(sources))
                        # Must mirror _pointer_scores exactly. A model trained
                        # on cosines and scored on raw dot products puts the
                        # null on a scale it never saw, which destroys the
                        # decline decision while leaving the candidate ranking
                        # roughly intact -- so it looks like a model that
                        # aligns well and cannot say "nothing", not like a bug.
                        scale = self._pointer_scale
                        if self.config.pointer_style == "colbert":
                            queries = torch.nn.functional.normalize(queries, dim=-1)
                            keys = torch.nn.functional.normalize(keys, dim=-1)
                            scale = 1.0 / self._pointer_temperature.clamp(min=1e-3)
                        scores = queries @ keys.T / scale
                        null = (queries @ self._pointer_null) / scale
                        scores = torch.cat([null.unsqueeze(1), scores], dim=1)
                        chosen = scores.argmax(dim=-1).tolist()
                        for (word, _, _), guess in zip(words, chosen):
                            links[word] = -1 if guess == 0 else sources[guess - 1][0]
                    out.append(links)
        self._encoder.train(); self._pointer_source.train(); self._pointer_target.train()
        return out

    def predict_alignment_scores(self, examples: Sequence[ChangeExample]):
        """Per reuse word, the probability of every candidate it could point at.

        Returns, for each example, a list over reuse words of
        ``[(source_index, probability), ...]`` sorted best first, with
        ``source_index == -1`` standing for the null. The argmax of this is what
        ``predict_alignment`` returns; the rest is what the model nearly chose,
        which is the part worth showing a reader.
        """
        import torch

        if not self.config.pointer:
            raise ValueError("predict_alignment_scores needs pointer=True")
        self._encoder.eval(); self._pointer_source.eval(); self._pointer_target.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length)
                source_spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    per_word = [[] for _ in example.target_tokens]
                    words = [(w, a, b) for w, (a, b) in enumerate(spans[row])
                             if w < len(example.target_tokens)]
                    sources = [(w, a, b) for w, (a, b) in enumerate(source_spans[row])
                               if w < len(example.source_tokens)]
                    if words and sources:
                        def vectors_for(items):
                            st = torch.tensor([a for _, a, _ in items])
                            en = torch.tensor([max(b, a + 1) for _, a, b in items])
                            return self._word_vectors(
                                hidden, torch.full_like(st, row), st, en)

                        queries = self._pointer_target(vectors_for(words))
                        keys = self._pointer_source(vectors_for(sources))
                        # Must mirror _pointer_scores exactly. A model trained
                        # on cosines and scored on raw dot products puts the
                        # null on a scale it never saw, which destroys the
                        # decline decision while leaving the candidate ranking
                        # roughly intact -- so it looks like a model that
                        # aligns well and cannot say "nothing", not like a bug.
                        scale = self._pointer_scale
                        if self.config.pointer_style == "colbert":
                            queries = torch.nn.functional.normalize(queries, dim=-1)
                            keys = torch.nn.functional.normalize(keys, dim=-1)
                            scale = 1.0 / self._pointer_temperature.clamp(min=1e-3)
                        scores = queries @ keys.T / scale
                        null = (queries @ self._pointer_null) / scale
                        scores = torch.cat([null.unsqueeze(1), scores], dim=1)
                        probability = torch.softmax(scores, dim=-1)
                        candidates = [-1] + [w for w, _, _ in sources]
                        for (word, _, _), row_p in zip(words, probability):
                            pairs = sorted(zip(candidates, row_p.tolist()),
                                           key=lambda x: -x[1])
                            per_word[word] = pairs
                    out.append(per_word)
        self._encoder.train(); self._pointer_source.train(); self._pointer_target.train()
        return out

    def predict_typed(self, examples: Sequence[ChangeExample], alignments,
                      featurizer=None) -> List[List[str]]:
        """E24: name the operation on every predicted link.

        ``alignments`` is whatever the inference stack decided -- Hungarian,
        identity bonus, null scale -- so the typer names links as they will
        be reported, not as the raw pointer would have placed them. Words
        with no link come back as ``"INS"``. ``featurizer`` supplies the
        symbolic evidence for each (source word, reuse word); without one the
        evidence is zero, which is the ablation.
        """
        import torch

        if self._typer is None:
            raise ValueError("configure `fine_operations` to predict typed links")
        self._encoder.eval(); self._typer.eval()
        out: List[List[str]] = []
        dim = self.config.feature_dim
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length)
                source_spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    links = alignments[start + row]
                    tags = ["INS"] * len(example.target_tokens)
                    t_span = {w: (a, b) for w, (a, b) in enumerate(spans[row])
                              if w < len(example.target_tokens)}
                    s_span = {w: (a, b) for w, (a, b) in enumerate(source_spans[row])
                              if w < len(example.source_tokens)}
                    chosen = [(t, s) for t, s in enumerate(links)
                              if s is not None and s >= 0 and t in t_span and s in s_span]
                    if not chosen:
                        out.append(tags); continue
                    def vectors_for(items):
                        st = torch.tensor([a for a, _ in items])
                        en = torch.tensor([max(b, a + 1) for a, b in items])
                        return self._word_vectors(hidden, torch.full_like(st, row), st, en)
                    h_t = vectors_for([t_span[t] for t, _ in chosen])
                    h_s = vectors_for([s_span[s] for _, s in chosen])
                    rows_phi = []
                    for t, s in chosen:
                        if featurizer is not None and dim:
                            rows_phi.append(featurizer(
                                example.source_tokens[s], example.target_tokens[t], s, t,
                                len(example.source_tokens), len(example.target_tokens)))
                        else:
                            rows_phi.append([0.0] * dim)
                    phi = torch.tensor(rows_phi, dtype=torch.float32)
                    if phi.shape[1] != dim:
                        phi = torch.zeros((len(chosen), dim))
                    logits = self._evidence_veto(self._typer_forward(h_t, h_s, phi), phi)
                    for (t, _), k in zip(chosen, logits.argmax(dim=-1).tolist()):
                        tags[t] = self._fine[k]
                    out.append(tags)
        self._encoder.train(); self._typer.train()
        return out

    def refine_heads(self, examples: Sequence[ChangeExample], *, epochs: int = 6,
                     lr: float = 1e-3, batch_size: int = 512, log=None,
                     frame_examples: Optional[Sequence[ChangeExample]] = None) -> None:
        """E24: train the typer and frame head alone, on cached encodings.

        The two heads ride along with the encoder for a few hundred steps
        during the main fit, which is not enough for a head that starts from
        random weights to converge. Encoding every example once with the
        encoder frozen and then fitting the heads for several epochs over the
        cached vectors costs seconds per epoch and lets them actually fit --
        without touching the pointer, which is what the shared encoder would
        otherwise pay for.
        """
        import torch

        if self._typer is None and self._frame_head is None:
            return
        device = self.config.device
        link_rows, frame_rows = [], []
        # the frame head may be refined on a different (gold-heavier) set than
        # the typer: constructed frames repeat 117 templates, and a head that
        # memorises them misses the formulas the held-out authors actually use
        frame_set = {id(e) for e in (frame_examples if frame_examples is not None else examples)}
        todo = list(examples)
        if frame_examples is not None:
            seen = {id(e) for e in todo}
            todo += [e for e in frame_examples if id(e) not in seen]
        typer_set = {id(e) for e in examples}
        self._encoder.eval()
        with torch.no_grad():
            for start in range(0, len(todo), self.config.batch_size):
                chunk = list(todo[start : start + self.config.batch_size])
                if not chunk:
                    continue
                # a subclass may return more (the typed pointer adds the word index)
                batch, rows, starts, ends, _, source, align, extra = self._encode(chunk)[:8]
                batch = {k: v.to(device) for k, v in batch.items()}
                hidden = self._run_encoder(batch).last_hidden_state
                vectors = self._word_vectors(hidden, rows, starts, ends)
                fine_t, frame_t, phi = extra
                in_frame_set = torch.tensor([id(chunk[r]) in frame_set for r in rows.tolist()])
                in_typer_set = torch.tensor([id(chunk[r]) in typer_set for r in rows.tolist()])
                if self._frame_head is not None:
                    keep = (frame_t != -100) & in_frame_set
                    if bool(keep.any()):
                        frame_rows.append((vectors[keep.to(device)].half().cpu(),
                                           frame_t[keep]))
                fine_t = torch.where(in_typer_set, fine_t, torch.full_like(fine_t, -100))
                if self._typer is not None and source is not None and source[0].numel():
                    s_rows, s_starts, s_ends, _, s_words = source
                    s_vectors = self._word_vectors(hidden, s_rows, s_starts, s_ends)
                    flat = {(r, w): k for k, (r, w) in
                            enumerate(zip(s_rows.tolist(), s_words.tolist()))}
                    sel_t, sel_s = [], []
                    for k, (r, a, f) in enumerate(zip(rows.tolist(), align.tolist(),
                                                      fine_t.tolist())):
                        if a >= 0 and f != -100 and (r, a) in flat:
                            sel_t.append(k); sel_s.append(flat[(r, a)])
                    if sel_t:
                        it = torch.tensor(sel_t, device=device)
                        link_rows.append((vectors[it].half().cpu(),
                                          s_vectors[torch.tensor(sel_s, device=device)].half().cpu(),
                                          phi[torch.tensor(sel_t)], fine_t[torch.tensor(sel_t)]))
        self._encoder.train()

        fine_weights = None
        if self._fine and self.config.fine_class_weights:
            table = dict(self.config.fine_class_weights)
            fine_weights = torch.tensor([float(table.get(c, 1.0)) for c in self._fine],
                                        device=device)
        frame_loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(self.config.frame_positive_weight)], device=device))

        if link_rows and self._typer is not None:
            h_t = torch.cat([r[0] for r in link_rows]); h_s = torch.cat([r[1] for r in link_rows])
            phi = torch.cat([r[2] for r in link_rows]); y = torch.cat([r[3] for r in link_rows])
            params = list(self._typer.parameters()) + (
                list(self._typer_evidence.parameters()) if self._typer_evidence is not None else [])
            opt = torch.optim.Adam(params, lr=lr)
            n = len(y)
            for epoch in range(epochs):
                order = torch.randperm(n)
                total = 0.0
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    logits = self._typer_forward(h_t[idx].float().to(device),
                                                 h_s[idx].float().to(device), phi[idx])
                    loss = self._typer_loss(logits, y[idx].to(device), fine_weights)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += float(loss.item()) * len(idx)
                if log:
                    log(f"    typer refine {epoch + 1}/{epochs}  loss {total / n:.4f}  ({n:,} links)")
        if frame_rows and self._frame_head is not None:
            h = torch.cat([r[0] for r in frame_rows]); y = torch.cat([r[1] for r in frame_rows])
            opt = torch.optim.Adam(self._frame_head.parameters(), lr=lr)
            n = len(y)
            for epoch in range(epochs):
                order = torch.randperm(n)
                total = 0.0
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    loss = frame_loss_fn(self._frame_head(h[idx].float().to(device)),
                                         y[idx].to(device))
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += float(loss.item()) * len(idx)
                if log:
                    log(f"    frame refine {epoch + 1}/{epochs}  loss {total / n:.4f}  ({n:,} words)")

    def predict_frames(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """E24: per reuse word, 1 if it belongs to an attribution formula."""
        import torch

        if self._frame_head is None:
            return [[0] * len(e.target_tokens) for e in examples]
        self._encoder.eval(); self._frame_head.eval()
        out: List[List[int]] = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length)
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    flags = [0] * len(example.target_tokens)
                    usable = [(w, a, b) for w, (a, b) in enumerate(spans[row])
                              if w < len(example.target_tokens)]
                    if usable:
                        st = torch.tensor([a for _, a, _ in usable])
                        en = torch.tensor([max(b, a + 1) for _, a, b in usable])
                        vec = self._word_vectors(hidden, torch.full_like(st, row), st, en)
                        chosen = self._frame_head(vec).argmax(dim=-1).tolist()
                        for (w, _, _), c in zip(usable, chosen):
                            flags[w] = int(c)
                    out.append(flags)
        self._encoder.train(); self._frame_head.train()
        return out

    def operations_from_alignment(self, examples, alignments=None) -> List[List[str]]:
        """Read the operations off the pointer instead of off the tagging head.

        Pointing nowhere is INS; pointing at a word with the same normalized
        form is COPY; pointing at a different word is SUBST. The distinction the
        tagging head could not draw -- 59 of 70 missed SUBSTs went to INS -- is
        here a consequence of where the pointer landed rather than a class the
        model has to name.
        """
        from retexo.core.normalize import normalize

        if alignments is None:
            alignments = self.predict_alignment(examples)
        out: List[List[str]] = []
        for example, links in zip(examples, alignments):
            tags = []
            for word, token in enumerate(example.target_tokens):
                source = links[word] if word < len(links) else -1
                if source < 0 or source >= len(example.source_tokens):
                    tags.append("INS")
                elif normalize(token) == normalize(example.source_tokens[source]):
                    tags.append("COPY")
                else:
                    tags.append("SUBST")
            out.append(tags)
        return out

    def predict(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """Per example, one 0/1 prediction per reuse word.

        Words lost to truncation are predicted 0, so the returned list always
        matches the example's own length and scoring never silently shortens.
        """
        import torch

        self._encoder.eval(); self._head.eval()
        out: List[List[int]] = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length,
                )
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    predictions = [0] * len(example.labels)
                    usable = [(w, s, e) for w, (s, e) in enumerate(spans[row])
                              if w < len(example.labels)]
                    if usable:
                        words = torch.tensor([w for w, _, _ in usable])
                        starts = torch.tensor([s for _, s, _ in usable])
                        ends = torch.tensor([max(e, s + 1) for _, s, e in usable])
                        rows = torch.full_like(starts, row)
                        vectors = self._word_vectors(hidden, rows, starts, ends)
                        chosen = self._head(vectors).argmax(dim=-1).tolist()
                        for word, prediction in zip(words.tolist(), chosen):
                            predictions[word] = int(prediction)
                    out.append(predictions)
        return out


    def predict_source(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """Per example, one 0/1 prediction per *source* word (1 = deleted).

        Only meaningful with ``source_head=True``; otherwise every word is
        predicted 0, which is the honest answer for a model that was never
        asked the question.
        """
        import torch

        if self._source_head is None:
            return [[0] * len(e.source_tokens) for e in examples]

        self._encoder.eval(); self._head.eval(); self._source_head.eval()
        out: List[List[int]] = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, _ = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length,
                )
                spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    predictions = [0] * len(example.source_tokens)
                    usable = [(w, a, b) for w, (a, b) in enumerate(spans[row])
                              if w < len(example.source_tokens)]
                    if usable:
                        starts = torch.tensor([a for _, a, _ in usable])
                        ends = torch.tensor([max(b, a + 1) for _, a, b in usable])
                        rows = torch.full_like(starts, row)
                        vectors = self._word_vectors(hidden, rows, starts, ends)
                        chosen = self._source_head(vectors).argmax(dim=-1).tolist()
                        for (word, _, _), guess in zip(usable, chosen):
                            predictions[word] = int(guess)
                    out.append(predictions)
        return out


#: Backward-compatible module-level alias.
evidence_veto = ChangeDetector.evidence_veto

__all__ = ["ChangeDetector", "ChangeDetectorConfig", "ChangeExample"]
