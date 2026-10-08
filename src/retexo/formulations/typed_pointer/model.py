# formulations/typed_pointer/model.py
"""The ``TypedPointer`` class, assembled from the concern mixins."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from retexo.formulations.change_detector import GROUP_TARGET, ChangeDetector
from retexo.formulations.typed_pointer.aux_losses import AuxiliaryLossMixin
from retexo.formulations.typed_pointer.cells import (
    CELLS_FROM,
    NULL_FRAME,
    NULL_INS,
    STRETCH_MODES,
)
from retexo.formulations.typed_pointer.encoding import EncodingMixin
from retexo.formulations.typed_pointer.grid import CellGridMixin
from retexo.formulations.typed_pointer.losses import LossesMixin
from retexo.formulations.typed_pointer.prediction import PredictionMixin
from retexo.formulations.typed_pointer.training import TrainingMixin


# =============================================================================
# The model
# =============================================================================
class TypedPointer(
    EncodingMixin,
    CellGridMixin,
    AuxiliaryLossMixin,
    LossesMixin,
    PredictionMixin,
    TrainingMixin,
    ChangeDetector,
):
    """One softmax per reuse word over (source word, type) cells and two nulls.

    Built on the parent's encoder, pair encoding and word pooling. The parent's
    pointer projections become the *locate* term, its typer MLP the *name* term
    and its evidence layer the *evidence* term; a second learned null is added
    for FRAME. The parent's operation head and source head are left unused.
    """

    # ---------- pure helpers, testable without a model ----------

    @staticmethod
    def cell_index(column: int, k: int, n_types: int) -> int:
        return CELLS_FROM + column * n_types + k

    @classmethod
    def allowed_cells(
        cls, column: Optional[int], fine: int, frame: int, n_types: int, lexical: Sequence[int]
    ) -> List[int]:
        """Which flattened cells a gold label permits.

        ``column`` is the source column (None for a gold null); ``fine`` is the
        type index, ``GROUP_TARGET`` for "lexical, kind unknown", -100 for
        unknown; ``frame`` is 1 / 0 / -100.
        """
        if column is None:
            if frame == 1:
                return [NULL_FRAME]
            if frame == 0:
                return [NULL_INS]
            return [NULL_INS, NULL_FRAME]
        if fine == GROUP_TARGET:
            return [cls.cell_index(column, k, n_types) for k in lexical]
        if fine == -100:
            return [cls.cell_index(column, k, n_types) for k in range(n_types)]
        return [cls.cell_index(column, fine, n_types)]

    # ---------- construction ----------

    def __init__(self, config=None) -> None:
        import torch

        super().__init__(config)
        if not self.config.pointer or not self._fine:
            raise ValueError("TypedPointer needs pointer=True and fine_operations")
        if self.config.pointer_style != "dot":
            raise ValueError("TypedPointer is written for pointer_style='dot'")
        width = self._pointer_null.shape[0]
        self._frame_null = torch.nn.Parameter(
            torch.zeros(width).normal_(std=0.02).to(self.config.device)
        )
        self.K = len(self._fine)
        #: v6, factorized: p(s, k | t) = p(s | t) . p(k | t, s). The location
        #: softmax gets its own scalar evidence term so the dictionary still
        #: helps *find* the link; the type softmax lives at the chosen source
        #: and its confidence can no longer move the link. Five variants of the
        #: single (s, k) softmax showed the two cannot share one scale.
        self.factorized = True
        #: E26 route B: train under the resources' restriction (see _restrictions)
        self.restrict_training = False
        #: E27 route D: weight of the per-pair link-density prior (0 = off)
        self.density_weight = 0.0
        #: E31 (closed, not used): the hierarchical group-before-operation typer
        #: has been removed; ``group_loss_beta`` stays for the lexical-group loss.
        self.group_loss_beta = 1.0
        self.fine_confidence = 0.6
        self._loc_evidence = (
            torch.nn.Linear(self.config.feature_dim, 1).to(self.config.device)
            if self.config.use_link_features and self.config.feature_dim
            else None
        )
        #: v7: location gets its own pair scorer. The joint form's alignment
        #: (0.947) came partly from the name MLP acting as a richer locator
        #: than the dot product; factorizing (v6) took that away from location
        #: and it scored like a plain pointer again (0.915). Same input as the
        #: name term, separate parameters, one scalar out -- so location and
        #: naming have their own scales and their own capacity.
        hidden = width
        #: E34: a three-way pair head (no match / cit. / cf.) on the first token's state,
        #: trained jointly when ``pair_head_weight`` > 0 and an example carries ``pair_label``
        self._pair_head = torch.nn.Linear(hidden, 3).to(self.config.device)
        self.pair_head_weight = 0.0
        #: E36: per-token state embeddings for iterative refinement (see retexo.refinement.refine)
        self._state_reuse = torch.nn.Embedding(4, hidden).to(self.config.device)
        self._state_source = torch.nn.Embedding(3, hidden).to(self.config.device)
        self._state_link = torch.nn.Linear(hidden, hidden, bias=False).to(self.config.device)
        torch.nn.init.zeros_(self._state_reuse.weight)
        torch.nn.init.zeros_(self._state_source.weight)
        torch.nn.init.zeros_(self._state_link.weight)  # pass one == the model as it is
        #: E37: Sinkhorn (train-through-the-stack) -- the one-to-one constraint in the loss
        self.sinkhorn_weight = 0.0
        self.sinkhorn_iters = 10
        self.sinkhorn_decode = False
        self.sinkhorn_temperature = (
            1.0  # decoding: divide log-probabilities by this before balancing
        )
        self._sk_bin = torch.nn.Embedding(1, 1).to(
            self.config.device
        )  # the source-side dustbin score
        torch.nn.init.zeros_(self._sk_bin.weight)
        self.sinkhorn_src_head = False  # E37b: each source word its own "deleted" score
        #: E35: the operations as label tokens; the typer matches pairs against label states
        self.label_matching = False
        self.mc_dropout = False  # E30 step 0: keep the location MLP's dropout on at prediction
        self.zero_shot_index = None  # a fine index whose examples the type loss never sees
        d_match = 128
        self._match_pair = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden, self.config.typer_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.config.typer_hidden, d_match),
        ).to(self.config.device)
        self._match_label = torch.nn.Sequential(
            torch.nn.Linear(hidden, self.config.typer_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.typer_hidden, d_match),
        ).to(self.config.device)
        self._match_scale = float(d_match) ** 0.5
        self._sk_src = torch.nn.Linear(hidden, 1).to(self.config.device)
        torch.nn.init.zeros_(self._sk_src.weight)
        torch.nn.init.zeros_(self._sk_src.bias)
        #: E30: self-training on agreed links -- weight, agreement threshold, scope
        self.self_train_weight = 0.0
        self.self_train_threshold = 0.15
        self.self_train_scope = "open"  # open (words the gold leaves unlinked) | all
        self.refine_mode = "none"  # none | random | chain | gold  (training states)
        self.refine_rollin = 0.0  # probability of a model roll-in state instead
        self._refine_rng = random.Random(self.config.seed + 36)
        self._loc_mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden, self.config.typer_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.config.typer_hidden, 1),
        ).to(self.config.device)
        #: chunk of reuse words scored at once; the name term materialises a
        #: [words x candidates x 4*hidden] block, so this bounds memory
        self.score_chunk = 256
        # row 9: the span view -- a second location scorer on the same word vectors, span-shaped
        self._span_q = self._span_start = self._span_end = None
        self._span_null = None
        self._last_span = None
        self._span_marginals = {}
        if getattr(self.config, "span_view", False):
            self._span_q = torch.nn.Linear(hidden, hidden).to(self.config.device)
            self._span_start = torch.nn.Linear(hidden, hidden).to(self.config.device)
            self._span_end = torch.nn.Linear(hidden, hidden).to(self.config.device)
            self._span_null = torch.nn.Parameter(torch.zeros(hidden, device=self.config.device))
        # chain 15: the stretch tower, the stretch head and the null fusion
        self._tower = None
        self._tower_k = 0
        self._last_tower_hidden = None
        self._stretch_head = None
        self._null_fuse = None
        self._last_stretch = None
        k = int(getattr(self.config, "stretch_tower", 0) or 0)
        if k > 0:
            import copy

            base = getattr(self._encoder, "module", self._encoder)
            layers = base.encoder.layer
            self._tower_k = min(k, len(layers))
            self._tower = torch.nn.ModuleList(
                [copy.deepcopy(layer) for layer in layers[-self._tower_k :]]
            ).to(self.config.device)
        if getattr(self.config, "stretch_head", False):
            self._stretch_head = torch.nn.Linear(hidden, len(STRETCH_MODES)).to(self.config.device)
            if float(getattr(self.config, "null_fuse", 0.0)) != 0.0:
                self._null_fuse = torch.nn.Parameter(
                    torch.tensor(float(self.config.null_fuse), device=self.config.device)
                )
        # chain 16: the slot convolution over the per-pair score matrix
        self._slot_conv = None
        c = int(getattr(self.config, "slot_conv", 0) or 0)
        if c > 0:
            ksz = max(3, int(getattr(self.config, "slot_kernel", 3)) | 1)
            self._slot_conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, c, ksz, padding=ksz // 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(c, 1, ksz, padding=ksz // 2),
            ).to(self.config.device)
            torch.nn.init.zeros_(self._slot_conv[2].weight)  # starts as the plain pointer
            torch.nn.init.zeros_(self._slot_conv[2].bias)
        #: loss weight on a gold FRAME null. 3% of reuse words: at the INS
        #: weight (0.2) it never learned to beat the INS null; at 1.75 it did
        self.frame_null_weight = 1.75
        #: weight of the auxiliary naming loss at the gold cell (v5); 0 disables
        self.name_loss_weight = 1.0
