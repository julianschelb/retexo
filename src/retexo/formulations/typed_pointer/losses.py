# formulations/typed_pointer/losses.py
"""The cell losses: location, type, frame and span, factorized or joint."""

from __future__ import annotations

from typing import Optional, Sequence

from retexo.formulations.change_detector import GROUP_TARGET, ChangeExample
from retexo.formulations.typed_pointer.cells import CELLS_FROM, NULL_FRAME, NULL_INS


# =============================================================================
# The losses
# =============================================================================
class LossesMixin:
    """The location, type, frame and span losses over the cell grid."""

    # ---------- the cell losses ----------

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

    # ---------- row 9: the span view ----------

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

    # ---------- validation ----------

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
