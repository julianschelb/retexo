# formulations/typed_pointer/encoding.py
"""The encoder pass with its stretch tower, the per-word bookkeeping, and the E36 states."""

from __future__ import annotations

from typing import Dict, Sequence

from retexo.formulations.change_detector import ChangeExample


# =============================================================================
# Encoding
# =============================================================================
class EncodingMixin:
    """The parent's encoder run (plus the stretch tower) and the E36 state embeddings."""

    # ---------- the parent's encoder, plus the target word index ----------

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

    # ---------- E36: the refinement states ----------

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
