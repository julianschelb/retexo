# retexo/formulations/typed_pointer.py
"""
E25: a typed pointer -- one softmax over every (source word, type) cell.

E24 answered "which source word?" with a pointer and "what kind of change?" with
a second head that ran afterwards, plus a frame head, an operation head and a
source head, each with its own loss. The evidence about a word pair -- shared
lemma, a wordnet entry, two proper names -- reached only the second head, so it
could never help the pointer *find* the link, even though E23 measured that it
can (alignment 0.941 -> 0.964 when lemma similarity re-ranks the shortlist).

Here the two questions are one decision. For reuse word t the head scores every
cell (s, k) -- source word s, type k -- plus two nulls, and takes one softmax:

    score(t, s, k) = f(h_t).g(h_s)/sqrt(d)                     locate
                   + u_k . MLP[h_t, h_s, |h_t-h_s|, h_t*h_s]   name
                   + v_k . phi(t, s)                          evidence
    score(t, o, INS)   = f(h_t).n_INS
    score(t, o, FRAME) = f(h_t).n_FRAME

The loss is one cross-entropy per reuse word over a *set* of allowed cells, so
a label that names only part of the answer -- a hand-labelled SUBST that says
"lexical, kind unknown" -- supervises exactly what it knows.

Inference marginalises: p(s | t) = sum_k p(s, k | t), and hands those to the
same assignment stack E24 used, so the two are compared on identical terms.
The type is the argmax at the chosen cell, under the same evidence vetoes.

Everything that was a head in E24 and is a definition -- COPY/SUBST/INS from the
cell, DEL from what no cell consumed, REORDER/QUOTE/ADAPT/DISPERSE from the
alignment -- is derived, not learned.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

from retexo.formulations.change_detector import (
    GROUP_TARGET,
    ChangeDetector,
    ChangeExample,
)

#: chain 15: the stretch head's modes (the paper's modes of reuse) and the one the null fusion reads
STRETCH_MODES = ("VERBATIM", "ALLUSION", "FRAME", "NOMATCH")
STRETCH_NOMATCH = STRETCH_MODES.index("NOMATCH")

# cell layout of the flattened logits, per reuse word:
#   0            (null, INS)
#   1            (null, FRAME)
#   2 + c*K + k  (source column c, type k)
NULL_INS, NULL_FRAME, CELLS_FROM = 0, 1, 2


# =============================================================================
# The model
# =============================================================================


class TypedPointer(ChangeDetector):
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

    # ---------- encoding: the parent's, plus the target word index ----------

    def _run_encoder(self, batch):
        """The backbone's forward; with a stretch tower, the encoder's hidden states before its last
        ``stretch_tower`` layers are run through the tower's own copy of those layers as well, and kept
        for the span view and the stretch head (``_last_tower_hidden``)."""
        if self._tower is None or self._channels is not None:
            self._last_tower_hidden = None
            return super()._run_encoder(batch)
        kwargs = {k: v for k, v in batch.items() if k != "channel_ids"}
        out = self._encoder(**kwargs, output_hidden_states=True)
        h = out.hidden_states[-1 - self._tower_k]
        base = getattr(self._encoder, "module", self._encoder)
        mask = kwargs.get("attention_mask")
        ext = base.get_extended_attention_mask(mask, mask.shape) if mask is not None else None
        self._tower.train(self._encoder.training)
        for layer in self._tower:
            h = layer(h, attention_mask=ext)[0]
        self._last_tower_hidden = h
        return out

    def _encode(self, examples: Sequence[ChangeExample]):
        import torch

        out = super()._encode(examples)
        rows = out[1]
        seen: Dict[int, int] = {}
        t_words = []
        for r in rows.tolist():
            t_words.append(seen.get(r, 0))
            seen[r] = seen.get(r, 0) + 1
        return out + (torch.tensor(t_words),)

    # ---------- the cell grid ----------

    def _cells(self, hidden, rows, starts, ends, t_words, source, examples):
        """Flattened logits [T, 2 + width*K] and the bookkeeping to read them."""
        import torch

        vectors = self._word_vectors(hidden, rows, starts, ends)
        s_rows, s_starts, s_ends, _, s_words = source
        s_vectors = self._word_vectors(hidden, s_rows, s_starts, s_ends)
        vectors, s_vectors = self._apply_states(
            vectors, s_vectors, rows, t_words, s_rows, s_words, examples
        )
        if self.label_matching:  # E35: the [LBL] states, [B, K, H]
            pos = torch.tensor(self._last_label_positions, device=hidden.device)
            self._last_label_states = hidden[
                torch.arange(hidden.shape[0], device=hidden.device)[:, None], pos
            ]
        tower = None
        if self._last_tower_hidden is not None:
            tower = (
                self._word_vectors(self._last_tower_hidden, rows, starts, ends),
                self._word_vectors(self._last_tower_hidden, s_rows, s_starts, s_ends),
            )
        return self._cells_from_vectors(
            vectors, s_vectors, rows, t_words, s_rows, s_words, examples, tower=tower
        )

    def _apply_states(self, vectors, s_vectors, rows, t_words, s_rows, s_words, examples):
        """E36: add each token's state embedding; a linked reuse word also receives
        a projection of its linked source word's vector. Examples without a
        ``refine_state`` are left untouched (pass one)."""
        import torch

        states = [getattr(ex, "refine_state", None) for ex in examples]
        if all(st is None for st in states):
            return vectors, s_vectors
        device = vectors.device
        # source side first, so the linked-source vectors are the un-stated ones
        s_idx = [(r, w) for r, w in zip(s_rows.tolist(), s_words.tolist())]
        s_state = torch.tensor(
            [
                states[r].source[w] if states[r] is not None and w < len(states[r].source) else 0
                for r, w in s_idx
            ],
            device=device,
        )
        s_pos = {rw: i for i, rw in enumerate(s_idx)}
        t_idx = [(r, w) for r, w in zip(rows.tolist(), t_words.tolist())]
        t_state = torch.tensor(
            [
                states[r].reuse[w] if states[r] is not None and w < len(states[r].reuse) else 0
                for r, w in t_idx
            ],
            device=device,
        )
        link_src = torch.zeros_like(vectors)
        for i, (r, w) in enumerate(t_idx):
            st = states[r]
            if st is not None and w < len(st.link) and st.link[w] >= 0:
                j = s_pos.get((r, st.link[w]))
                if j is not None:
                    link_src[i] = s_vectors[j]
        vectors = vectors + self._state_reuse(t_state) + self._state_link(link_src)
        s_vectors = s_vectors + self._state_source(s_state)
        return vectors, s_vectors

    def _cells_from_vectors(
        self, vectors, s_vectors, rows, t_words, s_rows, s_words, examples, tower=None
    ):
        """The grid from already-pooled word vectors (also used on cached ones)."""
        import torch

        device = self.config.device
        vectors = vectors.to(device)
        s_vectors = s_vectors.to(device)
        rows_d = rows.to(device)
        s_rows_d = s_rows.to(device)
        s_words_d = s_words.to(device)
        # E37b: the per-source dustbin scores, kept for the Sinkhorn loss and decoding
        self._last_src_bins = (
            s_rows.tolist(),
            s_words.tolist(),
            self._sk_src(s_vectors).squeeze(-1),
        )

        # a row can hold source words but no reuse words (the reuse text was
        # truncated away behind a long source): size the grid by both sides
        n_rows = int(max(rows_d.max().item(), s_rows_d.max().item())) + 1
        counts = torch.bincount(s_rows_d, minlength=n_rows)
        width = int(counts.max().item()) if counts.numel() else 0
        if width == 0:
            return None
        offsets = torch.cumsum(counts, 0) - counts
        columns = torch.arange(len(s_rows_d), device=device) - offsets[s_rows_d]
        gather = torch.zeros((n_rows, width), dtype=torch.long, device=device)
        valid = torch.zeros((n_rows, width), dtype=torch.bool, device=device)
        gather[s_rows_d, columns] = torch.arange(len(s_rows_d), device=device)
        valid[s_rows_d, columns] = True
        # source word index sitting in each (row, column), for the evidence lookup
        word_at = torch.full((n_rows, width), -1, dtype=torch.long, device=device)
        word_at[s_rows_d, columns] = s_words_d

        # ---- locate: [T, width]
        keys = self._pointer_source(s_vectors)
        queries = self._pointer_target(vectors)
        cand = keys[gather[rows_d]]  # [T, width, H]
        locate = torch.einsum("th,twh->tw", queries, cand) / self._pointer_scale
        null_ins = (queries @ self._pointer_null) / self._pointer_scale
        null_frame = (queries @ self._frame_null) / self._pointer_scale

        # ---- row 9: the span view on the same vectors; chain 15: on the stretch tower's vectors
        t_vec = vectors if tower is None else tower[0].to(device)
        t_svec = s_vectors if tower is None else tower[1].to(device)
        if self._span_q is not None:
            self._last_span = self._span_scores(t_vec, t_svec[gather[rows_d]], valid[rows_d])
        # ---- chain 15: the stretch head, and its NOMATCH belief fused into the pointer's null
        if self._stretch_head is not None:
            self._last_stretch = self._stretch_head(t_vec)  # [T, 4]
            if self._null_fuse is not None:
                null_ins = (
                    null_ins
                    + self._null_fuse
                    * torch.log_softmax(self._last_stretch, dim=-1)[:, STRETCH_NOMATCH]
                )

        # ---- evidence: phi[t, c] from each example's pair matrix
        F = self.config.feature_dim
        phi = torch.zeros((len(rows_d), width, F), dtype=torch.float32)
        rows_l, t_l, word_l = rows_d.tolist(), t_words.tolist(), word_at.tolist()
        for i, (r, t) in enumerate(zip(rows_l, t_l)):
            pf = getattr(examples[r], "pair_features", None)
            if pf is None or t >= pf.shape[0]:
                continue
            cols = word_l[r]
            for c, s in enumerate(cols):
                if s >= 0 and s < pf.shape[1]:
                    phi[i, c] = torch.from_numpy(pf[t, s].astype("float32"))
        phi = phi.to(device)

        # ---- name + evidence: [T, width, K], chunked over reuse words
        T = len(rows_d)
        cells = torch.empty((T, width, self.K), device=device)
        h_s_all = s_vectors[gather[rows_d]]  # [T, width, H]
        loc_extra = torch.zeros((T, width), device=device)
        for a in range(0, T, self.score_chunk):
            b = min(a + self.score_chunk, T)
            h_t = vectors[a:b].unsqueeze(1).expand(-1, width, -1)  # [t, width, H]
            h_s = h_s_all[a:b]
            pair = torch.cat([h_t, h_s, (h_t - h_s).abs(), h_t * h_s], dim=-1)
            if self.label_matching:  # E35: match against the label states
                q = self._match_pair(pair)  # [c, width, d]
                lab = self._match_label(self._last_label_states.to(device))[
                    rows_d[a:b]
                ]  # [c, K, d]
                name = torch.einsum("cwd,ckd->cwk", q, lab) / self._match_scale
            else:
                name = self._typer(pair)
            loc_extra[a:b] = self._loc_mlp(pair).squeeze(-1)
            block = locate[a:b].unsqueeze(-1) + name
            if self._typer_evidence is not None and self.config.use_link_features:
                block = block + self._typer_evidence(phi[a:b])
            cells[a:b] = block
        cells = cells.masked_fill(~valid[rows_d].unsqueeze(-1), float("-inf"))
        flat = torch.cat(
            [null_ins.unsqueeze(1), null_frame.unsqueeze(1), cells.reshape(T, width * self.K)],
            dim=1,
        )
        # factorized view: location logits [T, 2 + width] and name logits [T, width, K]
        loc = locate + loc_extra
        if self._loc_evidence is not None:
            loc = loc + self._loc_evidence(phi).squeeze(-1)
        # ---- chain 16: the slot convolution -- every cell reads its neighbourhood in the pair's matrix
        if self._slot_conv is not None:
            bump = self._slot_bump(loc, null_ins, valid, rows_d, t_words, phi)
            loc = loc + bump
            cells = cells + bump.unsqueeze(-1)
            flat = torch.cat(
                [null_ins.unsqueeze(1), null_frame.unsqueeze(1), cells.reshape(T, width * self.K)],
                dim=1,
            )
        loc = loc.masked_fill(~valid[rows_d], float("-inf"))
        loc_flat = torch.cat([null_ins.unsqueeze(1), null_frame.unsqueeze(1), loc], dim=1)
        name = cells - locate.unsqueeze(-1)  # the name + evidence terms alone
        # v7: the frame decision is per reuse word, from the parent's frame head
        frame_logits = self._frame_head(vectors) if self._frame_head is not None else None
        self._last_factorized = (loc_flat, name, frame_logits)
        return flat, valid[rows_d], word_at[rows_d], phi

    # ---------- training ----------

    def _sample_states(self, examples):
        """E36: one training state per example -- from the gold (random / chain / gold)
        or, with probability ``refine_rollin``, from the model's own first pass."""
        from retexo.refinement.refine import State

        rollin_wanted = [
            ex
            for ex in examples
            if self.refine_rollin > 0 and self._refine_rng.random() < self.refine_rollin
        ]
        rolled = {}
        if rollin_wanted:
            for ex in rollin_wanted:
                object.__setattr__(ex, "refine_state", None)
            links_all = self.predict_links(rollin_wanted)
            frames_all = self.predict_frames(rollin_wanted, links_all)
            for ex, links, frames in zip(rollin_wanted, links_all, frames_all):
                rolled[id(ex)] = State.from_script(links, frames, len(ex.source_tokens))
            self._encoder.train()
            self._typer.train()
            self._loc_mlp.train()
        for ex in examples:
            if id(ex) in rolled:
                st = rolled[id(ex)]
            else:
                pf = getattr(ex, "pair_features", None)
                tiers = None
                if pf is not None and self.refine_mode == "chain":
                    from retexo.edit_typing.downstream import ScriptFeaturizer

                    tiers = ScriptFeaturizer.tier_grid(pf)
                st = State.sample_for_training(ex, self.refine_mode, self._refine_rng, tiers)
            object.__setattr__(ex, "refine_state", st)

    def _self_training_loss(self, examples, rows, t_words, word_at):
        """E30 (awesome-align's self-training objective): links both directions agree
        on become targets. Forward p(s|t) from the current factorized grid; reverse
        p(t|s) from a second, swapped pass (its own encoder call; gradients flow
        into both). Agreed = both above the threshold. Scope ``open``: only reuse
        words the gold leaves unlinked (a labelled link is already trained by the
        CE); ``all``: every reuse word. Never on a negative pair.
        loss = -mean_{(t,s) in A} [log p(s|t) + log p(t|s)] / 2."""
        import torch

        from retexo.aligners.agreement import PairSwap

        eligible = [
            i
            for i, ex in enumerate(examples)
            if getattr(ex, "negative_kind", None) is None and ex.alignments is not None
        ]
        if not eligible:
            return None
        loc_flat, _, _ = self._last_factorized  # forward, with grad
        p_fwd = torch.softmax(loc_flat, dim=1)  # [T, 2 + width]
        # the reverse pass on the swapped examples (featurized copies cached on the example)
        rev = []
        for i in eligible:
            ex = examples[i]
            r = getattr(ex, "swapped_copy", None)
            if r is None:
                r = PairSwap.example(ex)
                object.__setattr__(ex, "swapped_copy", r)
            rev.append(r)
        batch, r_rows, r_starts, r_ends, _, r_source, _, _, r_t_words = self._encode(rev)
        if r_source is None or not r_source[0].numel() or not r_rows.numel():
            return None
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        hidden = self._run_encoder(batch).last_hidden_state
        keep_fwd = self._last_factorized
        got = self._cells(hidden, r_rows, r_starts, r_ends, r_t_words, r_source, rev)
        if got is None:
            self._last_factorized = keep_fwd
            return None
        _, _, r_word_at, _ = got
        r_loc, _, _ = self._last_factorized
        self._last_factorized = keep_fwd
        p_rev = torch.softmax(r_loc, dim=1)  # rows = source words of the pair
        # index the reverse rows: (example index in `rev`, source word) -> row
        r_index = {(r, w): k for k, (r, w) in enumerate(zip(r_rows.tolist(), r_t_words.tolist()))}
        r_word_l = r_word_at.tolist()
        word_l = word_at.tolist()
        pos = {i: k for k, i in enumerate(eligible)}
        terms = []
        for k, (r, t) in enumerate(zip(rows.tolist(), t_words.tolist())):
            if r not in pos:
                continue
            ex = examples[r]
            gold = ex.alignments[t] if t < len(ex.alignments) else -100
            if gold == -100 or (self.self_train_scope == "open" and gold >= 0):
                continue
            for c, s in enumerate(word_l[k]):
                if s < 0:
                    continue
                pf = p_fwd[k, CELLS_FROM + c]
                if float(pf) <= self.self_train_threshold:
                    continue
                rk = r_index.get((pos[r], s))
                if rk is None:
                    continue
                try:
                    rc = r_word_l[rk].index(t)
                except ValueError:
                    continue
                pr = p_rev[rk, CELLS_FROM + rc]
                if float(pr) > self.self_train_threshold:
                    terms.append(
                        -(torch.log(pf.clamp(min=1e-9)) + torch.log(pr.clamp(min=1e-9))) / 2
                    )
        self._last_agreed = len(terms)
        if not terms:
            return None
        return torch.stack(terms).mean()

    def _sinkhorn_loss(self, loc_flat, word_at, align, restrict=None):
        """E37: per example, -log of the Sinkhorn-balanced assignment at the gold
        cell -- linked words at their source, unlinked words in the dustbin column
        (null weight), unconsumed source words in the dustbin row (null weight)."""
        import torch

        from retexo.aligners.sinkhorn import SinkhornBalancer

        rows = getattr(self, "_sk_rows", None)
        if rows is None:
            return None
        device = loc_flat.device
        rows_l, word_l, align_l = rows.tolist(), word_at.tolist(), align.tolist()
        by_ex = {}
        for i, (r, a) in enumerate(zip(rows_l, align_l)):
            if a != -100:
                by_ex.setdefault(r, []).append(i)
        total, weight_sum = loc_flat.new_zeros(()), 0.0
        nw = float(self.config.null_pointer_weight)
        col_bin = self._sk_bin.weight[0, 0]
        sb_rows, sb_words, sb_vals = getattr(self, "_last_src_bins", (None, None, None))
        sb_index = (
            {(r, w): k for k, (r, w) in enumerate(zip(sb_rows, sb_words))}
            if sb_rows is not None
            else {}
        )
        for r, idx in by_ex.items():
            cols = [c for c, w in enumerate(word_l[idx[0]]) if w >= 0]
            if not cols or len(idx) == 0:
                continue
            L = loc_flat[idx][:, CELLS_FROM:][:, cols]  # [T_e, S_e]
            if restrict is not None:
                L = L.masked_fill(restrict.to(device)[idx][:, CELLS_FROM:][:, cols], float("-inf"))
            row_bins = torch.logsumexp(loc_flat[idx][:, :CELLS_FROM], dim=1)  # [T_e]
            S_e = len(cols)
            src_of_col = [word_l[idx[0]][c] for c in cols]
            if self.sinkhorn_src_head and sb_index:
                ks = [sb_index.get((r, sw)) for sw in src_of_col]
                col_bins = torch.stack([sb_vals[k] if k is not None else col_bin for k in ks])
            else:
                col_bins = col_bin
            logP = SinkhornBalancer.log_sinkhorn_torch(
                L, row_bins, col_bins, iters=self.sinkhorn_iters
            )
            consumed = set()
            for k, i in enumerate(idx):
                a = align_l[i]
                if a >= 0 and a in src_of_col:
                    c = src_of_col.index(a)
                    if torch.isfinite(logP[k, c]):
                        total = total - logP[k, c]
                        weight_sum += 1.0
                        consumed.add(a)
                elif a < 0:
                    total = total - nw * logP[k, S_e]
                    weight_sum += nw
            for c, sw in enumerate(src_of_col):
                if sw not in consumed:
                    total = total - nw * logP[len(idx), c]
                    weight_sum += nw
        if weight_sum == 0:
            return None
        return total / weight_sum

    def predict_links(self, examples):
        from retexo.aligners.assignment import AssignmentPolicy

        scores = self.predict_alignment_scores(examples)
        if self.sinkhorn_decode:
            import numpy as np

            from retexo.aligners.sinkhorn import SinkhornBalancer

            col_bin = float(self._sk_bin.weight[0, 0])
            out = []
            for sc, ex in zip(scores, examples):
                bins = getattr(ex, "sk_src_bins", None) if self.sinkhorn_src_head else None
                cb = (
                    np.array([bins.get(s_, col_bin) for s_ in range(len(ex.source_tokens))])
                    if bins
                    else col_bin
                )
                out.append(
                    SinkhornBalancer.balanced_scores(
                        sc,
                        len(ex.source_tokens),
                        col_bins=cb,
                        iters=self.sinkhorn_iters,
                        temperature=self.sinkhorn_temperature,
                    )
                )
            scores = out
        return AssignmentPolicy.links_hungarian(scores)

    def predict_iterative(self, examples, passes: int = 2, until_stable: bool = False):
        """E36 inference: pass one undecided, later passes with the previous script
        as state. Returns per pass (links, frames), every pass complete for every pair.

        With ``until_stable`` a pair is frozen once a pass reproduces the previous
        script exactly (the model is deterministic in eval mode, so every later pass would too),
        and the loop ends when all pairs are frozen or at the cap ``passes``.
        ``self.settled_at[i]`` is the pass at which pair i stopped changing, 0 if it
        was still changing at the cap."""
        from retexo.refinement.refine import State

        examples = list(examples)
        for ex in examples:
            object.__setattr__(ex, "refine_state", None)
        history, active, prev = [], list(range(len(examples))), None
        settled = [0] * len(examples)
        for k in range(1, passes + 1):
            if not active:
                break
            sub = [examples[i] for i in active]
            links_a = self.predict_links(sub)
            frames_a = self.predict_frames(sub, links_a)
            if prev is None:
                links, frames = list(links_a), list(frames_a)
            else:
                links, frames = list(prev[0]), list(prev[1])
                for i, link, f in zip(active, links_a, frames_a):
                    links[i], frames[i] = link, f
            history.append((links, frames))
            if until_stable and prev is not None:
                still = []
                for i in active:
                    same = list(links[i]) == list(prev[0][i]) and list(frames[i]) == list(
                        prev[1][i]
                    )
                    if same:
                        settled[i] = k
                    else:
                        still.append(i)
                active = still
            prev = (links, frames)
            for ex, link, f in zip(examples, links, frames):
                object.__setattr__(
                    ex, "refine_state", State.from_script(link, f, len(ex.source_tokens))
                )
        for ex in examples:
            object.__setattr__(ex, "refine_state", None)
        self.settled_at = settled
        return history

    def _typed_loss(self, examples: Sequence[ChangeExample]):
        if self.refine_mode != "none":
            self._sample_states(examples)
        batch, rows, starts, ends, _, source, align, extra, t_words = self._encode(examples)
        if source is None or not source[0].numel() or not rows.numel():
            return None
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        hidden = self._run_encoder(batch).last_hidden_state
        got = self._cells(hidden, rows, starts, ends, t_words, source, examples)
        if got is None:
            return None
        flat, valid, word_at, _ = got
        fine_t, frame_t, _ = extra
        stretch_t = (
            self._stretch_targets(fine_t, frame_t, align)
            if self._stretch_head is not None
            else None
        )
        if self.zero_shot_index is not None:  # E35 arm Z
            fine_t = fine_t.masked_fill(fine_t == self.zero_shot_index, -100)
        restrict, type_mask = self._restrictions(examples, rows, t_words, word_at)
        if type_mask is not None:
            fine_t = fine_t.masked_fill(type_mask, -100)
        self._sk_rows = rows
        loss = self._loss_from_cells(flat, word_at, align, fine_t, frame_t, restrict=restrict)
        if self._span_q is not None and loss is not None:
            span_loss = self._span_loss(word_at, align)
            if span_loss is not None:
                loss = loss + span_loss
        if stretch_t is not None and loss is not None and self._last_stretch is not None:
            import torch

            if bool((stretch_t >= 0).any()):
                loss = loss + float(
                    getattr(self.config, "stretch_weight", 1.0)
                ) * torch.nn.functional.cross_entropy(
                    self._last_stretch, stretch_t.to(self._last_stretch.device), ignore_index=-100
                )
        if self.density_weight > 0 and self.factorized and loss is not None:
            loss = loss + self.density_weight * self._density_loss(examples, rows)
        if self.self_train_weight > 0 and loss is not None:
            so = self._self_training_loss(examples, rows, t_words, word_at)
            if so is not None:
                loss = loss + self.self_train_weight * so
        if self.pair_head_weight > 0 and loss is not None:
            labels = [getattr(ex, "pair_label", None) for ex in examples]
            keep = [i for i, label in enumerate(labels) if label is not None]
            if keep:
                import torch

                pooled = hidden[keep, 0]  # [n, H]
                target = torch.tensor([labels[i] for i in keep], device=pooled.device)
                loss = loss + self.pair_head_weight * torch.nn.functional.cross_entropy(
                    self._pair_head(pooled), target
                )
        return loss

    def enable_label_matching(self, glosses, zero_shot=None):
        """E35: prefix the operations' glosses (per fine operation, in order; empty
        lists for the no-gloss arm) and type by matching. ``zero_shot`` names a
        fine operation whose examples the type loss must never see."""
        import torch

        enc = self._pair_encoder
        if not hasattr(enc, "LBL"):
            raise RuntimeError("label matching needs the Latin BERT subword pair encoder")
        if len(glosses) != len(self._fine):
            raise ValueError(f"{len(glosses)} glosses for {len(self._fine)} operations")
        enc.label_prefix = [list(g) for g in glosses]
        base = self._encoder.module if hasattr(self._encoder, "module") else self._encoder
        base.resize_token_embeddings(enc.vocab_size_with_labels)
        with torch.no_grad():  # the new token starts as [CLS]
            emb = base.get_input_embeddings().weight
            emb[enc.LBL] = emb[enc.CLS]
        self.label_matching = True
        if zero_shot is not None:
            if zero_shot not in self._fine_index:
                raise KeyError(zero_shot)
            self.zero_shot_index = self._fine_index[zero_shot]

    def predict_pair(self, examples: Sequence[ChangeExample]):
        """E34: per example, [p(no match), p(cit.), p(cf.)] from the pair head."""
        import torch

        self._encoder.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch = self._encode(chunk)[0]
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch).last_hidden_state
                out.extend(torch.softmax(self._pair_head(hidden[:, 0]), dim=-1).cpu().tolist())
        self._encoder.train()
        return out

    def _density_loss(self, examples, rows):
        """E27 route D: the expected share of linked reuse words per pair,
        pulled toward the reference type's gold density (``density_target``
        on the example; examples without one contribute nothing)."""
        import torch

        loc_flat, _, _ = self._last_factorized
        p = torch.softmax(loc_flat, dim=1)
        p_link = 1.0 - (p[:, NULL_INS] + p[:, NULL_FRAME])
        rows_d = rows.to(p_link.device)
        terms = []
        for r, ex in enumerate(examples):
            target = getattr(ex, "density_target", None)
            if target is None:
                continue
            mask = rows_d == r
            if bool(mask.any()):
                terms.append((p_link[mask].mean() - float(target)) ** 2)
        return torch.stack(terms).mean() if terms else loc_flat.new_zeros(())

    def _restrictions(self, examples, rows, t_words, word_at):
        """E26 (route B): what the resources have already settled for this batch.

        Per row, the location columns the resources *close* (an attested word
        may link only to its attested candidates, or to nothing), and the rows
        whose type the lookup already named (no type loss there: the head trains
        on the residual). Both come from attributes ``run_e26`` sets on the
        examples -- ``allowed_sources`` (per reuse word: list of source indices,
        or None for an open word) and ``type_open`` (per reuse word: bool).
        """
        import torch

        if not self.restrict_training:
            return None, None
        T, width = word_at.shape
        restrict = torch.zeros((T, CELLS_FROM + width), dtype=torch.bool)
        type_mask = torch.zeros(T, dtype=torch.bool)
        word_l = word_at.tolist()
        any_restriction = False
        for i, (r, t) in enumerate(zip(rows.tolist(), t_words.tolist())):
            ex = examples[r]
            allowed = getattr(ex, "allowed_sources", None)
            if allowed is not None and t < len(allowed) and allowed[t] is not None:
                ok = set(allowed[t])
                for c, s in enumerate(word_l[i]):
                    if s >= 0 and s not in ok:
                        restrict[i, CELLS_FROM + c] = True
                any_restriction = True
            open_ = getattr(ex, "type_open", None)
            if open_ is not None and t < len(open_) and not open_[t]:
                type_mask[i] = True
        return (restrict if any_restriction else None), type_mask

    def _loss_from_cells(self, flat, word_at, align, fine_t, frame_t, restrict=None):
        if self.factorized:
            return self._loss_factorized(word_at, align, fine_t, frame_t, restrict=restrict)
        return self._loss_joint(flat, word_at, align, fine_t, frame_t)

    def _loss_factorized(self, word_at, align, fine_t, frame_t, restrict=None):
        """-log p(s* | t) - log p(k* | t, s*): two softmaxes, one sum."""
        import torch

        loc_flat, name, frame_logits = self._last_factorized
        device = loc_flat.device
        T, n_loc = loc_flat.shape
        word_l = word_at.tolist()
        table = dict(self.config.fine_class_weights) if self.config.fine_class_weights else {}
        # ---- location: allowed columns of [INS, FRAME, s_0 .. s_w]
        allowed = torch.zeros((T, n_loc), dtype=torch.bool)
        weight = torch.zeros(T)
        rows_i, cols_i, tk = [], [], []
        for i, (a, f, _fr) in enumerate(zip(align.tolist(), fine_t.tolist(), frame_t.tolist())):
            if a == -100:
                continue
            if a < 0:
                # "no source" -- either null; whether it is a frame is the
                # frame head's question, asked below on the same rows
                allowed[i, NULL_INS] = allowed[i, NULL_FRAME] = True
                weight[i] = self.config.null_pointer_weight
                continue
            try:
                col = word_l[i].index(a)
            except ValueError:
                continue
            allowed[i, CELLS_FROM + col] = True
            # E48: a declined substitution costs the same as a declined copy, though
            # substitutions are 7% of the links and almost all of the misses. link_weights
            # (fine operation -> weight) re-prices the location loss per kind of link.
            weight[i] = 1.0
            lw = getattr(self, "link_weights", None)
            if lw and f != -100 and 0 <= f < len(self.config.fine_operations):
                weight[i] = float(lw.get(self.config.fine_operations[f], 1.0))
            if f != -100:
                rows_i.append(i)
                cols_i.append(col)
                tk.append(f)
        if restrict is not None:
            # the resources closed these columns: the softmax runs over what is
            # left ({candidates, nulls}); a gold column they closed drops the row
            allowed = allowed & ~restrict
        keep = allowed.any(dim=1)
        loss = loc_flat.new_zeros(())  # not sum()*0: the grid holds -inf, and -inf*0 prints NaN
        if bool(keep.any()):
            k = keep.to(device)
            logits = loc_flat[k]
            if restrict is not None:
                logits = logits.masked_fill(restrict.to(device)[k], float("-inf"))
            ok = allowed.to(device)[k]
            w = weight.to(device)[k]
            log_all = torch.logsumexp(logits, dim=1)
            log_ok = torch.logsumexp(logits.masked_fill(~ok, float("-inf")), dim=1)
            loss = loss + ((-(log_ok - log_all)) * w).sum() / w.sum().clamp(min=1e-6)
        if self.sinkhorn_weight > 0:
            sk = self._sinkhorn_loss(loc_flat, word_at, align, restrict)
            if sk is not None:
                loss = loss + self.sinkhorn_weight * sk
        # ---- frame: per reuse word, where the label says
        if frame_logits is not None:
            ft = frame_t.to(device)
            has = ft != -100
            if bool(has.any()):
                w2 = torch.tensor([1.0, float(self.config.frame_positive_weight)], device=device)
                loss = loss + torch.nn.functional.cross_entropy(
                    frame_logits[has], ft[has], weight=w2
                )
        # ---- type, at the gold source only (E24's protection, kept)
        if rows_i:
            ri = torch.tensor(rows_i, device=device)
            ci = torch.tensor(cols_i, device=device)
            at_gold = name[ri, ci]  # [n, K]
            tk_t = torch.tensor(tk, device=device)
            exact = tk_t >= 0
            group = tk_t == GROUP_TARGET
            fine_w = torch.tensor([float(table.get(c, 1.0)) for c in self._fine], device=device)
            if bool(exact.any()):
                loss = loss + torch.nn.functional.cross_entropy(
                    at_gold[exact], tk_t[exact], weight=fine_w
                )
            if bool(group.any()) and self._lexical_index:
                # a hand-labelled SUBST: the lexical cells -- under the hierarchy
                # this is the sense + residual groups, unchanged
                lp = torch.log_softmax(at_gold[group], dim=-1)
                idx = torch.tensor(self._lexical_index, device=device)
                loss = (
                    loss
                    + self.config.group_loss_weight * (-torch.logsumexp(lp[:, idx], dim=-1)).mean()
                )
        return loss

    def _loss_joint(self, flat, word_at, align, fine_t, frame_t):
        import torch

        device = flat.device
        T, n_cells = flat.shape
        width = (n_cells - CELLS_FROM) // self.K

        # column of the gold source word within its row, or None
        allowed = torch.zeros((T, n_cells), dtype=torch.bool)
        weight = torch.zeros(T)
        table = dict(self.config.fine_class_weights) if self.config.fine_class_weights else {}
        word_l = word_at.tolist()
        for i, (a, f, fr) in enumerate(zip(align.tolist(), fine_t.tolist(), frame_t.tolist())):
            if a == -100:
                continue  # no supervision at all
            if a < 0:
                cells_ok = self.allowed_cells(None, f, fr, self.K, self._lexical_index)
                # INS nulls are 84% of reuse words and are down-weighted as in
                # every pointer since E5. A FRAME null is ~3% of words: at the
                # same 0.2 the frame null never learned to beat the INS null
                # (v1: FRAME F1 0.000). It gets the frame head's positive weight.
                weight[i] = self.frame_null_weight if fr == 1 else self.config.null_pointer_weight
            else:
                try:
                    col = word_l[i].index(a)
                except ValueError:
                    continue  # source truncated away
                cells_ok = self.allowed_cells(col, f, fr, self.K, self._lexical_index)
                weight[i] = float(table.get(self._fine[f], 1.0)) if 0 <= f < self.K else 1.0
            allowed[i, cells_ok] = True
        keep = allowed.any(dim=1)
        if not bool(keep.any()):
            return flat.sum() * 0.0
        allowed = allowed.to(device)[keep.to(device)]
        weight = weight.to(device)[keep.to(device)]
        logits = flat[keep.to(device)]
        log_all = torch.logsumexp(logits, dim=1)
        log_ok = torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), dim=1)
        losses = -(log_ok - log_all)
        loss = (losses * weight).sum() / weight.sum().clamp(min=1e-6)

        # v5: an auxiliary naming loss at the gold cell -- "given the right
        # source word, which type?" -- a softmax over K at that source only.
        # Every attempt to give the name terms extra training *after* the joint
        # fit (v2, v3c, v4) cost alignment, because they sit inside the same
        # softmax as location. Trained *together* with the joint loss from the
        # start, nothing is ever recalibrated afterwards.
        if self.name_loss_weight:
            rows_i, cols_i, tk = [], [], []
            for i, (a, f) in enumerate(zip(align.tolist(), fine_t.tolist())):
                if a < 0 or f == -100:
                    continue
                try:
                    col = word_l[i].index(a)
                except ValueError:
                    continue
                rows_i.append(i)
                cols_i.append(col)
                tk.append(f)
            if rows_i:
                ri = torch.tensor(rows_i, device=device)
                ci = torch.tensor(cols_i, device=device)
                grid = flat[:, CELLS_FROM:].reshape(T, width, self.K)
                at_gold = grid[ri, ci]  # [n, K]
                tk_t = torch.tensor(tk, device=device)
                exact = tk_t >= 0
                group = tk_t == GROUP_TARGET
                name = at_gold.sum() * 0.0
                if bool(exact.any()):
                    name = name + torch.nn.functional.cross_entropy(at_gold[exact], tk_t[exact])
                if bool(group.any()) and self._lexical_index:
                    lp = torch.log_softmax(at_gold[group], dim=-1)
                    idx = torch.tensor(self._lexical_index, device=device)
                    name = name + (-torch.logsumexp(lp[:, idx], dim=-1)).mean()
                loss = loss + self.name_loss_weight * name
        return loss

    def _slot_bump(self, loc, null_ins, valid, rows_d, t_words, phi):
        """The convolution's addition per cell, [T, width]: the first-pass probabilities of every pair laid
        out as a [reuse x source] matrix (rows in reuse order), an anchor channel (same form or same lemma
        from the evidence vector) and the validity mask, convolved, read back at each reuse word's row."""
        import torch

        device = loc.device
        T, width = loc.shape
        masked = loc.masked_fill(~valid[rows_d], float("-inf"))
        prob = torch.softmax(torch.cat([null_ins.unsqueeze(1), masked], dim=1), dim=1)[
            :, 1:
        ]  # [T, width]
        prob = prob.masked_fill(~valid[rows_d], 0.0)
        anchor = (
            torch.clamp(phi[:, :, 0] + phi[:, :, 1], 0.0, 1.0)
            if phi.shape[-1] >= 2
            else torch.zeros_like(prob)
        )
        n_rows = int(rows_d.max().item()) + 1
        t_idx = t_words.to(device)
        max_t = int(t_idx.max().item()) + 1
        grid = torch.zeros((n_rows, 3, max_t, width), device=device)
        grid[rows_d, 0, t_idx] = (
            prob if not getattr(self.config, "slot_anchor_only", False) else torch.zeros_like(prob)
        )
        grid[rows_d, 1, t_idx] = anchor.masked_fill(~valid[rows_d], 0.0)
        grid[rows_d, 2, t_idx] = valid[rows_d].float()
        out = self._slot_conv(grid)[:, 0]  # [n_rows, max_t, width]
        bump = out[rows_d, t_idx]  # [T, width]
        return bump.masked_fill(~valid[rows_d], 0.0)

    def _extra_fresh_parameters(self):
        """Parameters a subclass adds to the heads' learning-rate group: the span view's, the stretch head's, the slot convolution's."""
        out = []
        if self._slot_conv is not None:
            out += list(self._slot_conv.parameters())
        if self._span_q is not None:
            out += (
                list(self._span_q.parameters())
                + list(self._span_start.parameters())
                + list(self._span_end.parameters())
                + [self._span_null]
            )
        if self._stretch_head is not None:
            out += list(self._stretch_head.parameters())
            if self._null_fuse is not None:
                out.append(self._null_fuse)
        return out

    def _stretch_targets(self, fine_t, frame_t, align):
        """The mode of every reuse word from the targets the example carries: FRAME where the frame flag
        is set, NOMATCH for an unlinked word, VERBATIM for a NOP / MORPH link, ALLUSION for a lexical or
        cardinality link (the group target included); -100 where the example does not say."""
        import torch

        names = self._fine or []
        out = []
        for f, fr, a in zip(fine_t.tolist(), frame_t.tolist(), align.tolist()):
            if fr == 1:
                out.append(STRETCH_MODES.index("FRAME"))
            elif a == -100:
                out.append(-100)
            elif a < 0:
                out.append(STRETCH_NOMATCH)
            elif f == GROUP_TARGET:
                out.append(STRETCH_MODES.index("ALLUSION"))
            elif 0 <= f < len(names):
                out.append(
                    STRETCH_MODES.index("VERBATIM")
                    if names[f] in ("NOP", "MORPH")
                    else STRETCH_MODES.index("ALLUSION")
                )
            else:
                out.append(-100)
        return torch.tensor(out, dtype=torch.long)

    # ---------- row 9: the span view ----------

    def _span_scores(self, vectors, h_s_all, valid_rows):
        """Span logits per reuse word: ``[T, 1 + L * width]`` -- the null first, then for every
        length 1..L the spans starting at each source column (start + end scores); invalid
        spans at -inf. Also the per-column marginal probability ``[T, width]`` and ``p(null) [T]``."""
        import torch

        scale = self._pointer_scale
        q = self._span_q(vectors)  # [T, H]
        start = torch.einsum("th,twh->tw", q, self._span_start(h_s_all)) / scale  # [T, width]
        end = torch.einsum("th,twh->tw", q, self._span_end(h_s_all)) / scale
        null = (q @ self._span_null) / scale  # [T]
        T, width = start.shape
        L = max(1, int(self.config.span_max_len))
        spans = torch.full((T, L, width), float("-inf"), device=start.device)
        for length in range(L):
            if length >= width:
                break
            ok = valid_rows[:, : width - length] & valid_rows[:, length:]
            spans[:, length, : width - length] = (
                start[:, : width - length] + end[:, length:]
            ).masked_fill(~ok, float("-inf"))
        logits = torch.cat([null.unsqueeze(1), spans.reshape(T, L * width)], dim=1)
        prob = torch.softmax(logits, dim=1)
        p_null = prob[:, 0]
        p_spans = prob[:, 1:].reshape(T, L, width)
        marginal = torch.zeros((T, width), device=start.device)
        for length in range(L):
            for offset in range(length + 1):  # a span of length+1 words covers a..a+length
                if width - length > 0:
                    marginal[:, offset : offset + width - length] += p_spans[
                        :, length, : width - length
                    ]
        # the covered mass of a multi-word span lands on every word it covers, so the row is renormalised
        # with the null to a distribution over columns + null (what the decoder's theta reads)
        total = p_null + marginal.sum(dim=1)
        return logits, marginal / total.unsqueeze(1), p_null / total

    def _span_loss(self, word_at, align):
        """-log p(the gold word's single-word span | t), the null for an unlinked word."""
        import torch

        logits, _, _ = self._last_span
        T, _ = logits.shape
        targets = torch.full((T,), -100, dtype=torch.long)
        weight = torch.ones(T)
        word_l = word_at.tolist()
        for i, a in enumerate(align.tolist()):
            if a == -100:
                continue
            if a < 0:
                targets[i] = 0
                weight[i] = self.config.null_pointer_weight
                continue
            try:
                targets[i] = 1 + word_l[i].index(a)  # length-1 span at that column
            except ValueError:
                continue
        keep = targets >= 0
        if not bool(keep.any()):
            return None
        losses = torch.nn.functional.cross_entropy(
            logits[keep.to(logits.device)], targets[keep].to(logits.device), reduction="none"
        )
        return (losses * weight[keep].to(logits.device)).sum() / weight[keep].sum().to(
            logits.device
        )

    def evaluation_loss(self, examples: Sequence[ChangeExample]) -> Optional[float]:
        """The cell loss on ``examples`` without a gradient (the validation loss), the mean over batches."""
        import torch

        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                loss = self._typed_loss(list(examples[start : start + self.config.batch_size]))
                if loss is not None:
                    total += float(loss.item())
                    n += 1
        return total / n if n else None

    def fit(
        self,
        examples: Sequence[ChangeExample],
        *,
        log=None,
        on_batch: Optional[Callable[[int, int], None]] = None,
    ) -> TypedPointer:
        """Train on ``examples``; ``on_batch(done, total)`` is called after every optimizer step."""
        import torch

        encoder_params = list(self._encoder.parameters())
        encoder_params += list(self._pointer_source.parameters())
        encoder_params += list(self._pointer_target.parameters())
        encoder_params += [self._pointer_null, self._frame_null]
        if self._tower is not None:
            encoder_params += list(self._tower.parameters())
        fresh = list(self._typer.parameters())
        if self._typer_evidence is not None:
            fresh += list(self._typer_evidence.parameters())
        if self._loc_evidence is not None:
            fresh += list(self._loc_evidence.parameters())
        fresh += list(self._loc_mlp.parameters())
        fresh += list(self._pair_head.parameters())
        fresh += list(self._sk_bin.parameters())
        fresh += list(self._sk_src.parameters())
        fresh += list(self._match_pair.parameters()) + list(self._match_label.parameters())
        fresh += (
            list(self._state_reuse.parameters())
            + list(self._state_source.parameters())
            + list(self._state_link.parameters())
        )
        if self._frame_head is not None:
            fresh += list(self._frame_head.parameters())
        fresh += list(self._extra_fresh_parameters())
        groups = [
            {"params": encoder_params, "lr": self.config.learning_rate},
            {"params": fresh, "lr": self.config.typer_lr},
        ]
        if self._channels is not None:
            if self.config.channel_lr > 0:
                groups.append(
                    {"params": list(self._channels.parameters()), "lr": self.config.channel_lr}
                )
            else:
                fresh += list(self._channels.parameters())
        optimizer = torch.optim.AdamW(groups)
        order = list(examples)
        rng = random.Random(self.config.seed)
        self._encoder.train()
        self._typer.train()
        self._loc_mlp.train()
        for epoch in range(self.config.epochs):
            rng.shuffle(order)
            total, n = 0.0, 0
            for start in range(0, len(order), self.config.batch_size):
                chunk = order[start : start + self.config.batch_size]
                if not chunk:
                    continue
                loss = self._typed_loss(chunk)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                n += 1
                if on_batch is not None:
                    on_batch(start + len(chunk), len(order))
            self.losses.append(total / max(n, 1))
            if log:
                log(f"    epoch {epoch + 1}/{self.config.epochs}  loss {total / max(n, 1):.4f}")
        return self

    def refine_joint(
        self,
        examples: Sequence[ChangeExample],
        *,
        epochs: int = 4,
        lr: float = 5e-4,
        log=None,
        freeze_locate: bool = True,
    ) -> None:
        """More epochs of the *same* cell loss, with the encoder frozen.

        The parent's refinement retrains the name and evidence terms as a
        standalone classifier. That is right when they feed a separate head
        and wrong here, where they sit inside one softmax with the locate term
        and the nulls: rescaling them freely destroyed the alignment (v2:
        0.943 -> 0.884). So the extra epochs use the joint loss over cached
        word vectors -- every non-encoder parameter trains, calibrated against
        the others, at a fraction of a full epoch's cost.
        """
        import torch

        cache = []
        self._encoder.eval()
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, rows, starts, ends, _, source, align, extra, t_words = self._encode(chunk)
                if source is None or not source[0].numel() or not rows.numel():
                    continue
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch).last_hidden_state
                vectors = self._word_vectors(hidden, rows, starts, ends).half().cpu()
                s_rows, s_starts, s_ends, _, s_words = source
                s_vectors = self._word_vectors(hidden, s_rows, s_starts, s_ends).half().cpu()
                fine_t, frame_t, _ = extra
                cache.append(
                    (
                        vectors,
                        s_vectors,
                        rows,
                        t_words,
                        s_rows,
                        s_words,
                        chunk,
                        align,
                        fine_t,
                        frame_t,
                    )
                )
        self._encoder.train()
        # v3c refined everything and lost the alignment again (0.947 -> 0.890):
        # the refinement set is synthetic-heavy, and re-exposing the *locate*
        # projections to it is the extra synthetic epoch E7 showed costs gold
        # score. The locate term is frozen by default; the name terms and the
        # nulls train against it, with the joint loss, and stay calibrated.
        params = [self._pointer_null, self._frame_null] + list(self._typer.parameters())
        if self._typer_evidence is not None:
            params += list(self._typer_evidence.parameters())
        if not freeze_locate:
            params += list(self._pointer_source.parameters()) + list(
                self._pointer_target.parameters()
            )
        opt = torch.optim.Adam(params, lr=lr)
        rng = random.Random(self.config.seed + 1)
        for epoch in range(epochs):
            rng.shuffle(cache)
            total, n = 0.0, 0
            for (
                vectors,
                s_vectors,
                rows,
                t_words,
                s_rows,
                s_words,
                chunk,
                align,
                fine_t,
                frame_t,
            ) in cache:
                got = self._cells_from_vectors(
                    vectors.float(), s_vectors.float(), rows, t_words, s_rows, s_words, chunk
                )
                if got is None:
                    continue
                flat, _, word_at, _ = got
                loss = self._loss_from_cells(flat, word_at, align, fine_t, frame_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
            if log:
                log(
                    f"    joint refine {epoch + 1}/{epochs}  loss {total / max(n, 1):.4f}"
                    f"  ({len(cache)} batches)"
                )

    # ---------- inference ----------

    def predict_cells(self, examples: Sequence[ChangeExample]):
        """Per example, per reuse word: (candidate source words, [n_cand, K] logits,
        [2] null logits), with the evidence vetoes applied to the cells."""
        import torch

        self._encoder.eval()
        self._typer.eval()
        self._loc_mlp.train() if self.mc_dropout else self._loc_mlp.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, rows, starts, ends, _, source, align, extra, t_words = self._encode(chunk)
                per_example = [[None] * len(e.target_tokens) for e in chunk]
                if source is not None and source[0].numel() and rows.numel():
                    batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                    hidden = self._run_encoder(batch_on).last_hidden_state
                    got = self._cells(hidden, rows, starts, ends, t_words, source, chunk)
                    if self.sinkhorn_src_head and getattr(self, "_last_src_bins", None) is not None:
                        sb_rows, sb_words, sb_vals = self._last_src_bins
                        vals = sb_vals.detach().cpu().tolist()
                        per_bins = {}
                        for r_, w_, v_ in zip(sb_rows, sb_words, vals):
                            per_bins.setdefault(r_, {})[w_] = v_
                        for r_, ex_ in enumerate(chunk):
                            object.__setattr__(ex_, "sk_src_bins", per_bins.get(r_, {}))
                    if got is not None:
                        flat, valid, word_at, phi = got
                        T, n_cells = flat.shape
                        width = (n_cells - CELLS_FROM) // self.K
                        frame_src = None
                        if self.factorized:
                            loc_flat, name, frame_logits = self._last_factorized
                            cells = name  # [T, width, K], type logits
                            nulls_src = loc_flat[:, :CELLS_FROM]
                            loc_src = loc_flat[:, CELLS_FROM:]
                            frame_src = frame_logits
                        else:
                            cells = flat[:, CELLS_FROM:].reshape(T, width, self.K)
                            nulls_src = flat[:, :CELLS_FROM]
                            loc_src = None
                        if self.config.use_link_features:
                            vetoed = self.evidence_veto(
                                cells.reshape(T * width, self.K),
                                phi.reshape(T * width, -1),
                                self._fine_index,
                            )
                            cells = vetoed.reshape(T, width, self.K)
                        cells = cells.cpu()
                        nulls = nulls_src.cpu()
                        loc_c = loc_src.cpu() if loc_src is not None else None
                        fr_c = frame_src.cpu() if frame_src is not None else None
                        valid_l, word_l = valid.cpu(), word_at.cpu()
                        span_marg = span_null = None
                        if self._span_q is not None and self._last_span is not None:
                            _, span_marg, span_null = self._last_span
                            span_marg, span_null = span_marg.cpu(), span_null.cpu()
                        for i, (r, t) in enumerate(zip(rows.tolist(), t_words.tolist())):
                            cols = [c for c in range(width) if bool(valid_l[i, c])]
                            if span_marg is not None:
                                self._span_marginals[(start + r, t)] = (
                                    float(span_null[i]),
                                    span_marg[i, cols].tolist(),
                                )
                            per_example[r][t] = (
                                [int(word_l[i, c]) for c in cols],
                                cells[i, cols],  # [n_cand, K]
                                nulls[i],  # [2]
                                loc_c[i, cols] if loc_c is not None else None,  # [n_cand]
                                fr_c[i] if fr_c is not None else None,
                            )  # [2]
                out.extend(per_example)
        self._encoder.train()
        self._typer.train()
        self._loc_mlp.train()
        return out

    def predict_alignment_scores(self, examples: Sequence[ChangeExample]):
        """The parent's interface: per reuse word, [(source, p)...] best first,
        with source -1 for the null -- marginalised over types."""
        import torch

        out = []
        self._span_marginals = {}
        for r, (_ex, words) in enumerate(zip(examples, self.predict_cells(examples))):
            per_word = []
            for t, w in enumerate(words):
                if w is None:
                    per_word.append([])
                    continue
                cands, cells, nulls, loc, _ = w
                if loc is not None:  # factorized: p(s | t) directly
                    p = torch.softmax(torch.cat([nulls, loc]), dim=0)
                    p_null = float(p[NULL_INS] + p[NULL_FRAME])
                    p_cells = p[CELLS_FROM:]
                else:
                    logits = torch.cat([nulls, cells.reshape(-1)])
                    p = torch.softmax(logits, dim=0)
                    p_null = float(p[NULL_INS] + p[NULL_FRAME])
                    p_cells = p[CELLS_FROM:].reshape(len(cands), self.K).sum(dim=1)
                span = self._span_marginals.get((r, t))
                if span is not None and len(span[1]) == len(cands):  # row 9: the two views averaged
                    w_s = float(self.config.span_weight)
                    p_null = (1 - w_s) * p_null + w_s * span[0]
                    p_cells = (1 - w_s) * p_cells + w_s * torch.tensor(span[1], dtype=p_cells.dtype)
                pairs = [(-1, p_null)] + [(s, float(v)) for s, v in zip(cands, p_cells)]
                per_word.append(sorted(pairs, key=lambda x: -x[1]))
            out.append(per_word)
        return out

    def predict_typed(
        self, examples: Sequence[ChangeExample], alignments, featurizer=None, detail: bool = False
    ):
        """The type at each chosen cell. ``featurizer`` is accepted for interface
        parity and unused: the evidence is already inside the cells.

        ``detail`` is kept for interface parity with earlier versions of this
        method (the hierarchical decode it once carried is closed, E31); no
        caller passes ``detail=True`` today, and every entry of ``details`` is
        ``None``."""

        out, details = [], []
        for ex, words, links in zip(examples, self.predict_cells(examples), alignments):
            tags = ["INS"] * len(ex.target_tokens)
            info = [None] * len(ex.target_tokens)
            for t, w in enumerate(words):
                s = links[t] if t < len(links) else -1
                if w is None or s is None or s < 0:
                    continue
                cands, cells, _, _, _ = w
                if s not in cands:
                    continue
                logits = cells[cands.index(s)]
                tags[t] = self._fine[int(logits.argmax())]
            out.append(tags)
            details.append(info)
        return (out, details) if detail else out

    def predict_frames(self, examples: Sequence[ChangeExample], alignments=None) -> List[List[int]]:
        """A word the assignment left unlinked is a frame word if the FRAME null
        beats the INS null."""
        out = []
        cells = self.predict_cells(examples)
        for i, (ex, words) in enumerate(zip(examples, cells)):
            flags = [0] * len(ex.target_tokens)
            links = alignments[i] if alignments is not None else None
            for t, w in enumerate(words):
                if w is None:
                    continue
                if links is not None and t < len(links) and links[t] is not None and links[t] >= 0:
                    continue
                if w[4] is not None:  # v7: the frame head decides
                    flags[t] = int(w[4][1] > w[4][0])
                else:
                    nulls = w[2]
                    flags[t] = int(nulls[NULL_FRAME] > nulls[NULL_INS])
            out.append(flags)
        return out

    def predict_source(self, examples: Sequence[ChangeExample], alignments=None) -> List[List[int]]:
        """DEL is what no link consumed. Without alignments, everything is kept."""
        out = []
        for i, ex in enumerate(examples):
            if alignments is None:
                out.append([0] * len(ex.source_tokens))
                continue
            used = {s for s in alignments[i] if s is not None and s >= 0}
            out.append([0 if s in used else 1 for s in range(len(ex.source_tokens))])
        return out


#: Backward-compatible module-level aliases.
cell_index = TypedPointer.cell_index
allowed_cells = TypedPointer.allowed_cells
