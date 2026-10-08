# formulations/typed_pointer/aux_losses.py
"""The experiment-flagged auxiliary losses: E26, E27, E30 and E37."""

from __future__ import annotations

from retexo.formulations.typed_pointer.cells import CELLS_FROM, NULL_FRAME, NULL_INS


# =============================================================================
# The auxiliary losses
# =============================================================================
class AuxiliaryLossMixin:
    """The E26-E37 auxiliary objectives, each switched off by default."""

    # ---------- E26: resource restrictions ----------

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

    # ---------- E27: the link-density prior ----------

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

    # ---------- E30: self-training on agreed links ----------

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

    # ---------- E37: the Sinkhorn assignment loss ----------

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
