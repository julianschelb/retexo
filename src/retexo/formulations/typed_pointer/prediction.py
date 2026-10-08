# formulations/typed_pointer/prediction.py
"""Inference: cells, links, frames and types read off the cell grid."""

from __future__ import annotations

from typing import List, Sequence

from retexo.formulations.change_detector import ChangeExample
from retexo.formulations.typed_pointer.cells import CELLS_FROM, NULL_FRAME, NULL_INS


# =============================================================================
# Prediction
# =============================================================================
class PredictionMixin:
    """Prediction over the cell grid: links, types, frames and sources."""

    # ---------- E35: label matching ----------

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

    # ---------- E34: the pair head ----------

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

    # ---------- links and iterative refinement ----------

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
