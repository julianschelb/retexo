# formulations/typed_pointer/grid.py
"""The (source word, type) cell grid and its auxiliary views: locate, name, evidence."""

from __future__ import annotations

from retexo.formulations.change_detector import GROUP_TARGET
from retexo.formulations.typed_pointer.cells import STRETCH_MODES, STRETCH_NOMATCH


# =============================================================================
# The cell grid
# =============================================================================
class CellGridMixin:
    """The flattened (source column, type) cell logits and their auxiliary views."""

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

    # ---------- chain 16: the slot convolution ----------

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

    # ---------- chain 15: the stretch head's targets ----------

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
