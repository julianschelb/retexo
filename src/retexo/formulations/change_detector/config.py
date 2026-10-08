# formulations/change_detector/config.py
"""Configuration for the binary change detector and every optional head it can grow."""

from __future__ import annotations

from dataclasses import dataclass

# =============================================================================
# ChangeDetectorConfig
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
