# formulations/change_detector/prediction.py
"""The inference API of the change detector: one prediction per head, per reuse word."""

from __future__ import annotations

from typing import List, Sequence

from retexo.formulations.change_detector.example import ChangeExample

# =============================================================================
# _PredictionMixin
# =============================================================================


class _PredictionMixin:
    """The prediction methods of the change detector."""

    # ---------- Operation tags ----------

    def predict_operations(self, examples: Sequence[ChangeExample]) -> List[List[str]]:
        """Per example, the predicted operation tag per reuse word."""
        if not self._classes:
            raise ValueError("configure `operations` to predict tags")
        import torch

        self._encoder.eval()
        self._head.eval()
        out: List[List[str]] = []
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
                    tags = ["INS"] * len(example.target_tokens)
                    usable = [
                        (w, a, b)
                        for w, (a, b) in enumerate(spans[row])
                        if w < len(example.target_tokens)
                    ]
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
                                device=logits.device,
                            )
                            logits = logits + bias
                        chosen = logits.argmax(dim=-1).tolist()
                        for (word, _, _), guess in zip(usable, chosen):
                            tags[word] = self._classes[guess]
                    out.append(tags)
        self._encoder.train()
        self._head.train()
        return out

    # ---------- Pointer alignment ----------

    def predict_alignment(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """Per example, the source word each reuse word points at, -1 for none.

        Scored a pair at a time: the candidate set is that pair's own source
        words, so there is nothing to pad and nothing to mask. Words lost to
        truncation come back as -1, which is also what the caller reads as INS.
        """
        import torch

        if not self.config.pointer:
            raise ValueError("predict_alignment needs pointer=True")
        self._encoder.eval()
        self._pointer_source.eval()
        self._pointer_target.eval()
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
                source_spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    links = [-1] * len(example.target_tokens)
                    words = [
                        (w, a, b)
                        for w, (a, b) in enumerate(spans[row])
                        if w < len(example.target_tokens)
                    ]
                    sources = [
                        (w, a, b)
                        for w, (a, b) in enumerate(source_spans[row])
                        if w < len(example.source_tokens)
                    ]
                    if words and sources:

                        def vectors_for(items, hidden=hidden, row=row):
                            st = torch.tensor([a for _, a, _ in items])
                            en = torch.tensor([max(b, a + 1) for _, a, b in items])
                            return self._word_vectors(hidden, torch.full_like(st, row), st, en)

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
        self._encoder.train()
        self._pointer_source.train()
        self._pointer_target.train()
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
        self._encoder.eval()
        self._pointer_source.eval()
        self._pointer_target.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length,
                )
                source_spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    per_word = [[] for _ in example.target_tokens]
                    words = [
                        (w, a, b)
                        for w, (a, b) in enumerate(spans[row])
                        if w < len(example.target_tokens)
                    ]
                    sources = [
                        (w, a, b)
                        for w, (a, b) in enumerate(source_spans[row])
                        if w < len(example.source_tokens)
                    ]
                    if words and sources:

                        def vectors_for(items, hidden=hidden, row=row):
                            st = torch.tensor([a for _, a, _ in items])
                            en = torch.tensor([max(b, a + 1) for _, a, b in items])
                            return self._word_vectors(hidden, torch.full_like(st, row), st, en)

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
                            pairs = sorted(zip(candidates, row_p.tolist()), key=lambda x: -x[1])
                            per_word[word] = pairs
                    out.append(per_word)
        self._encoder.train()
        self._pointer_source.train()
        self._pointer_target.train()
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

    # ---------- Link typing ----------

    def predict_typed(
        self, examples: Sequence[ChangeExample], alignments, featurizer=None
    ) -> List[List[str]]:
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
        self._encoder.eval()
        self._typer.eval()
        out: List[List[str]] = []
        dim = self.config.feature_dim
        with torch.no_grad():
            for start in range(0, len(examples), self.config.batch_size):
                chunk = list(examples[start : start + self.config.batch_size])
                if not chunk:
                    continue
                batch, spans = self._pair_encoder.encode(
                    [(list(e.source_tokens), list(e.target_tokens)) for e in chunk],
                    self.config.max_length,
                )
                source_spans = self._pair_encoder.last_source_spans
                batch_on = {k: v.to(self.config.device) for k, v in batch.items()}
                hidden = self._run_encoder(batch_on).last_hidden_state
                for row, example in enumerate(chunk):
                    links = alignments[start + row]
                    tags = ["INS"] * len(example.target_tokens)
                    t_span = {
                        w: (a, b)
                        for w, (a, b) in enumerate(spans[row])
                        if w < len(example.target_tokens)
                    }
                    s_span = {
                        w: (a, b)
                        for w, (a, b) in enumerate(source_spans[row])
                        if w < len(example.source_tokens)
                    }
                    chosen = [
                        (t, s)
                        for t, s in enumerate(links)
                        if s is not None and s >= 0 and t in t_span and s in s_span
                    ]
                    if not chosen:
                        out.append(tags)
                        continue

                    def vectors_for(items, hidden=hidden, row=row):
                        st = torch.tensor([a for a, _ in items])
                        en = torch.tensor([max(b, a + 1) for a, b in items])
                        return self._word_vectors(hidden, torch.full_like(st, row), st, en)

                    h_t = vectors_for([t_span[t] for t, _ in chosen])
                    h_s = vectors_for([s_span[s] for _, s in chosen])
                    rows_phi = []
                    for t, s in chosen:
                        if featurizer is not None and dim:
                            rows_phi.append(
                                featurizer(
                                    example.source_tokens[s],
                                    example.target_tokens[t],
                                    s,
                                    t,
                                    len(example.source_tokens),
                                    len(example.target_tokens),
                                )
                            )
                        else:
                            rows_phi.append([0.0] * dim)
                    phi = torch.tensor(rows_phi, dtype=torch.float32)
                    if phi.shape[1] != dim:
                        phi = torch.zeros((len(chosen), dim))
                    logits = self._evidence_veto(self._typer_forward(h_t, h_s, phi), phi)
                    for (t, _), k in zip(chosen, logits.argmax(dim=-1).tolist()):
                        tags[t] = self._fine[k]
                    out.append(tags)
        self._encoder.train()
        self._typer.train()
        return out

    # ---------- Frames ----------

    def predict_frames(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """E24: per reuse word, 1 if it belongs to an attribution formula."""
        import torch

        if self._frame_head is None:
            return [[0] * len(e.target_tokens) for e in examples]
        self._encoder.eval()
        self._frame_head.eval()
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
                    flags = [0] * len(example.target_tokens)
                    usable = [
                        (w, a, b)
                        for w, (a, b) in enumerate(spans[row])
                        if w < len(example.target_tokens)
                    ]
                    if usable:
                        st = torch.tensor([a for _, a, _ in usable])
                        en = torch.tensor([max(b, a + 1) for _, a, b in usable])
                        vec = self._word_vectors(hidden, torch.full_like(st, row), st, en)
                        chosen = self._frame_head(vec).argmax(dim=-1).tolist()
                        for (w, _, _), c in zip(usable, chosen):
                            flags[w] = int(c)
                    out.append(flags)
        self._encoder.train()
        self._frame_head.train()
        return out

    # ---------- Change detection ----------

    def predict(self, examples: Sequence[ChangeExample]) -> List[List[int]]:
        """Per example, one 0/1 prediction per reuse word.

        Words lost to truncation are predicted 0, so the returned list always
        matches the example's own length and scoring never silently shortens.
        """
        import torch

        self._encoder.eval()
        self._head.eval()
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
                    usable = [
                        (w, s, e) for w, (s, e) in enumerate(spans[row]) if w < len(example.labels)
                    ]
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

        self._encoder.eval()
        self._head.eval()
        self._source_head.eval()
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
                    usable = [
                        (w, a, b)
                        for w, (a, b) in enumerate(spans[row])
                        if w < len(example.source_tokens)
                    ]
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
