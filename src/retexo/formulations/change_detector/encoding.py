# formulations/change_detector/encoding.py
"""Encoding and forward pass of the change detector: pooling, heads and scores."""

from __future__ import annotations

from typing import Dict, Sequence

from retexo.formulations.change_detector.constants import GROUP_TARGET
from retexo.formulations.change_detector.example import ChangeExample

# =============================================================================
# _EncodingMixin
# =============================================================================


class _EncodingMixin:
    """Encoding and forward-pass methods of the change detector."""

    # ---------- Encoding ----------

    def _run_encoder(self, batch):
        """The backbone's forward; with word channels the piece embeddings are built here."""
        if self._channels is None or "channel_ids" not in batch:
            return self._encoder(**{k: v for k, v in batch.items() if k != "channel_ids"})
        encoder = getattr(self._encoder, "module", self._encoder)
        embeddings = encoder.embeddings
        table = getattr(embeddings, "word_embeddings", None) or embeddings.tok_embeddings
        embeds = table(batch["input_ids"]) + self._channels(batch["channel_ids"])
        rest = {k: v for k, v in batch.items() if k not in ("input_ids", "channel_ids")}
        return self._encoder(inputs_embeds=embeds, **rest)

    def _encode(self, examples: Sequence[ChangeExample]):
        import torch

        pairs = [(list(e.source_tokens), list(e.target_tokens)) for e in examples]
        batch, spans = self._pair_encoder.encode(pairs, self.config.max_length)
        self._last_label_positions = getattr(
            self._pair_encoder, "last_label_positions", None
        )  # E35
        if self._channels is not None:
            batch["channel_ids"] = self._channels.ids(
                tuple(batch["input_ids"].shape),
                spans,
                getattr(self._pair_encoder, "last_source_spans", ()),
                pairs,
            )

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
                    fine_targets.append(
                        GROUP_TARGET if tag == "?" else self._fine_index.get(tag, -100)
                    )
                else:
                    fine_targets.append(-100)
                frames = example.frame_labels
                frame_targets.append(int(frames[word]) if frames and word < len(frames) else -100)
                phi = example.link_features
                vec = phi[word] if phi and word < len(phi) else None
                features.append(
                    list(vec)
                    if vec is not None and len(vec) == feature_dim
                    else [0.0] * feature_dim
                )
        source = None
        # The source side is encoded for the deletion head, and also for the
        # pointer, which needs every source word as a candidate whether or not
        # it carries a deletion label.
        if self.config.source_head or self.config.pointer:
            src_spans = self._pair_encoder.last_source_spans
            s_rows, s_starts, s_ends, s_targets, s_words = [], [], [], [], []
            for row, example in enumerate(examples):
                labels = example.source_labels or []
                limit = len(example.source_tokens) if self.config.pointer else len(labels)
                for word, (start, end) in enumerate(src_spans[row]):
                    if word >= limit:
                        break
                    s_rows.append(row)
                    s_starts.append(start)
                    s_ends.append(max(end, start + 1))
                    s_words.append(word)
                    # -100 where the pointer widened the set past the labels,
                    # so the deletion head is unaffected by the extra words.
                    s_targets.append(labels[word] if word < len(labels) else -100)
            source = (
                torch.tensor(s_rows),
                torch.tensor(s_starts),
                torch.tensor(s_ends),
                torch.tensor(s_targets),
                torch.tensor(s_words),
            )
        extra = (
            torch.tensor(fine_targets),
            torch.tensor(frame_targets),
            torch.tensor(features, dtype=torch.float32)
            if features
            else torch.zeros((0, feature_dim)),
        )
        return (
            batch,
            torch.tensor(rows),
            torch.tensor(starts),
            torch.tensor(ends),
            torch.tensor(targets),
            source,
            torch.tensor(align),
            extra,
        )

    def _word_vectors(self, hidden, rows, starts, ends):
        """One vector per labelled word, pooled as the config asks."""
        import torch

        if self.config.pooling == "first":
            return hidden[rows, starts]
        # Mean over each word's subwords. Spans are short, so a gather over a
        # padded index matrix is cheaper than a Python loop per word.
        widths = ends - starts
        longest = int(widths.max().item()) if len(widths) else 1
        offsets = torch.arange(longest, device=hidden.device).unsqueeze(0)
        index = starts.unsqueeze(1).to(hidden.device) + offsets
        mask = offsets < widths.unsqueeze(1).to(hidden.device)
        index = index.clamp(max=hidden.shape[1] - 1)
        gathered = hidden[rows.unsqueeze(1).to(hidden.device), index]
        gathered = gathered * mask.unsqueeze(-1)
        return gathered.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    # ---------- Forward pass ----------

    def _logits(self, examples: Sequence[ChangeExample]):
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
                source_out = (self._source_head(s_vectors), s_targets.to(self.config.device))
        pointer_out = None
        if self.config.pointer and s_vectors is not None and rows.numel():
            pointer_out = self._pointer_scores(
                vectors, rows, s_vectors, source[0], source[4], align
            )
        # E24 heads. The typer is trained on the *gold* link -- the source
        # word the example says the reuse word came from -- so that typing is
        # learned separately from locating and a pointer error never teaches
        # the typer a wrong relation.
        typer_out, frame_out = None, None
        fine_t, frame_t, phi = extra
        if self._typer is not None and s_vectors is not None and rows.numel():
            typer_out = self._typer_logits(
                vectors, rows, s_vectors, source[0], source[4], align, fine_t, phi
            )
        if self._frame_head is not None and rows.numel():
            frame_out = (self._frame_head(vectors), frame_t.to(self.config.device))
        return (
            logits,
            targets.to(self.config.device),
            source_out,
            pointer_out,
            (typer_out, frame_out),
        )

    # ---------- Link typer ----------

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
            others = torch.tensor(
                [i for c, i in index.items() if c != "NOP"], device=logits.device, dtype=torch.long
            )
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

    def _typer_logits(self, vectors, rows, s_vectors, s_rows, s_words, align, fine_t, phi):
        """Logits over the fine operations for every reuse word with a known link."""
        import torch

        flat = {}
        for k, (r, w) in enumerate(zip(s_rows.tolist(), s_words.tolist())):
            flat[(r, w)] = k
        sel_t, sel_s = [], []
        for k, (r, a, f) in enumerate(zip(rows.tolist(), align.tolist(), fine_t.tolist())):
            if a >= 0 and f != -100 and (r, a) in flat:
                sel_t.append(k)
                sel_s.append(flat[(r, a)])
        if not sel_t:
            return None
        device = vectors.device
        h_t = vectors[torch.tensor(sel_t, device=device)]
        h_s = s_vectors[torch.tensor(sel_s, device=device)]
        evidence = phi[torch.tensor(sel_t)]
        logits = self._typer_forward(h_t, h_s, evidence)
        return logits, fine_t[torch.tensor(sel_t)].to(device)

    # ---------- Pointer scoring ----------

    def _pointer_scores(self, vectors, rows, s_vectors, s_rows, s_words, align):
        """Scores over [null, every source word of the same pair].

        Candidates differ per pair, so they are gathered into a padded matrix
        and the positions belonging to other pairs are masked to -inf before the
        softmax -- otherwise a reuse word could point into a different pair's
        source, which is not merely wrong but trivially separable.
        """
        import torch

        device = self.config.device
        rows = rows.to(device)
        s_rows = s_rows.to(device)
        s_words = s_words.to(device)
        align = align.to(device)

        # every pair of the chunk, whether or not both sides survived truncation: a source row
        # with no target row (or the reverse) must still index inside the tables
        n_rows = (
            int(max(rows.max().item(), s_rows.max().item())) + 1
            if rows.numel() and s_rows.numel()
            else 0
        )
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

        projected = self._pointer_source(s_vectors)  # [S, H]
        queries = self._pointer_target(vectors)  # [T, H]
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
        candidates = projected[gather[rows]]  # [T, width, H]
        scores = torch.einsum("th,twh->tw", queries, candidates) / scale
        scores = scores.masked_fill(~valid[rows], float("-inf"))
        null = (queries @ null_vector) / scale
        scores = torch.cat([null.unsqueeze(1), scores], dim=1)  # column 0 = null

        # gold column: 0 for null, else the source word's own column + 1
        word_to_column = torch.zeros((n_rows, width), dtype=torch.long, device=device)
        word_to_column[s_rows, s_words.clamp(max=width - 1)] = columns
        present = torch.zeros((n_rows, width), dtype=torch.bool, device=device)
        present[s_rows, s_words.clamp(max=width - 1)] = True
        safe = align.clamp(min=0, max=width - 1)
        found = present[rows, safe] & (align >= 0)
        gold = torch.where(found, word_to_column[rows, safe] + 1, torch.zeros_like(align))
        # A word whose source was truncated away has no column to point at, so
        # it is dropped rather than relabelled as null -- calling it null would
        # teach the model that truncation means the author invented the word.
        gold = torch.where((align >= 0) & ~found, torch.full_like(align, -100), gold)
        gold = torch.where(align == -100, torch.full_like(align, -100), gold)
        return scores, gold
