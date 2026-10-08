# formulations/change_detector/detector.py
"""The change detector model, composed from the encoding, loss, training and prediction mixins."""

from __future__ import annotations

import random
from typing import List, Optional

from retexo.formulations.change_detector.config import ChangeDetectorConfig
from retexo.formulations.change_detector.constants import LEXICAL_TAGS
from retexo.formulations.change_detector.encoding import _EncodingMixin
from retexo.formulations.change_detector.losses import _LossMixin
from retexo.formulations.change_detector.prediction import _PredictionMixin
from retexo.formulations.change_detector.training import _TrainingMixin
from retexo.formulations.pair_encoding import PairEncoder

# =============================================================================
# Model
# =============================================================================


class ChangeDetector(_EncodingMixin, _LossMixin, _TrainingMixin, _PredictionMixin):
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
        self._encoder = AutoModel.from_pretrained(self.config.base_model, **encoder_kwargs)
        hidden = self._encoder.config.hidden_size
        self._classes = list(self.config.operations) or None
        self._index = {c: i for i, c in enumerate(self._classes or [])}
        self._head = torch.nn.Linear(hidden, len(self._classes) if self._classes else 2)
        self._source_head = torch.nn.Linear(hidden, 2) if self.config.source_head else None
        # Two projections and a learned null, after Vinyals' pointer networks:
        # score(j, i) = f(h_j).g(h_i)/sqrt(d), with the null competing in the
        # same softmax so that "from nowhere" is a choice among the candidates
        # rather than a threshold on top of them. Scaling by sqrt(d) keeps the
        # dot products in a range where the softmax has usable gradients, which
        # at hidden 768 it otherwise does not.
        colbert = self.config.pointer_style == "colbert"
        width = self.config.pointer_dim if colbert else hidden
        self._pointer_source = torch.nn.Linear(hidden, width) if self.config.pointer else None
        self._pointer_target = torch.nn.Linear(hidden, width) if self.config.pointer else None
        self._pointer_null = (
            torch.nn.Parameter(torch.zeros(width).normal_(std=0.02))
            if self.config.pointer
            else None
        )
        # Cosine is bounded in [-1, 1], so a fixed 1/sqrt(d) would leave the
        # softmax almost flat; ColBERT-style scoring gets a learned temperature
        # instead, initialised where the logits have usable gradients.
        self._pointer_temperature = (
            torch.nn.Parameter(torch.tensor(float(self.config.pointer_temperature)))
            if self.config.pointer and colbert
            else None
        )
        self._pointer_scale = 1.0 if colbert else hidden**0.5
        if self._pointer_null is not None:
            self._pointer_null.data = self._pointer_null.data.to(self.config.device)
        if self._pointer_temperature is not None:
            self._pointer_temperature.data = self._pointer_temperature.data.to(self.config.device)
        # E24: the link typer names the kind of change on a link the pointer
        # has already placed. It reads both words' contextual vectors, their
        # elementwise difference and product, and the symbolic evidence for
        # the pair. The pointer is left exactly as it was: locating is solved,
        # and this head is not allowed to disturb it beyond sharing the encoder.
        self._fine = list(self.config.fine_operations) or None
        self._fine_index = {c: i for i, c in enumerate(self._fine or [])}
        #: The classes a hand-labelled SUBST may resolve to.
        self._lexical_index = [i for i, c in enumerate(self._fine or []) if c in LEXICAL_TAGS]
        self._typer = None
        self._typer_evidence = None
        if self._fine:
            self._typer = torch.nn.Sequential(
                torch.nn.Linear(4 * hidden, self.config.typer_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.config.typer_hidden, len(self._fine)),
            )
            if self.config.use_link_features and self.config.feature_dim:
                # A direct linear path from the evidence to the logits. Fed
                # through the MLP beside 3,072 contextual dimensions, 23 flags
                # were drowned: the first full run's typer scored *below* the
                # lemma rule whose output it could see. Here "same lemma ->
                # MORPH" is one weight.
                self._typer_evidence = torch.nn.Linear(self.config.feature_dim, len(self._fine))
        self._frame_head = torch.nn.Linear(hidden, 2) if self.config.frame_head else None
        self._channels = None
        if getattr(self.config, "channels", ""):
            from retexo.formulations.word_channels import WordChannels

            self._channels = WordChannels(
                tuple(k.strip() for k in self.config.channels.split(",") if k.strip()), hidden
            )
        for module in (
            self._encoder,
            self._head,
            self._source_head,
            self._pointer_source,
            self._pointer_target,
            self._typer,
            self._typer_evidence,
            self._frame_head,
            self._channels,
        ):
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


#: Backward-compatible module-level alias.
evidence_veto = ChangeDetector.evidence_veto
