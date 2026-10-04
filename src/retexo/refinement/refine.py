# retexo/refinement/refine.py
"""
E36: iterative refinement -- the script fed back as a per-token state.

Every reuse word today decides on its own; the only coupling is the Hungarian
afterwards. A second pass that *sees* the first pass's script can tell an
isolated coincidental link from a real single-word reuse, and can decide an
open word with the verbatim skeleton in view. The state is one small
embedding per token, added to the pooled word vectors before the heads:

    reuse word   0 undecided | 1 linked (+ the linked source word's vector) | 2 declined | 3 frame
    source word  0 undecided | 1 consumed | 2 free

Pass one runs with everything undecided, which is the model as it is.

Training states are built from the gold script -- randomly corrupted (the
base), or as an easy-to-hard chain (the same-form skeleton, then the lemma
links with MORPH, then everything) -- and, with probability rho, from the
model's own first pass (roll-in), the gold being the expert in every case.
Every constructor below is a classmethod on :class:`State`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

UNDECIDED, LINKED, DECLINED, FRAME = 0, 1, 2, 3
S_UNDECIDED, S_CONSUMED, S_FREE = 0, 1, 2


@dataclass
class State:
    """The refiner's per-token view of a first-pass script.

    Attributes:
        reuse: Per reuse word, one of ``UNDECIDED``/``LINKED``/``DECLINED``/``FRAME``.
        link: Per reuse word, the source index when ``LINKED``, else -1.
        source: Per source word, one of ``S_UNDECIDED``/``S_CONSUMED``/``S_FREE``.
    """

    reuse: List[int]
    link: List[int]
    source: List[int]

    @classmethod
    def undecided(cls, n_reuse: int, n_source: int) -> "State":
        """Pass one: nothing decided yet."""
        return cls([UNDECIDED] * n_reuse, [-1] * n_reuse, [S_UNDECIDED] * n_source)

    @classmethod
    def from_script(cls, links: Sequence[int], frames: Optional[Sequence[int]], n_source: int) -> "State":
        """A decided state from links (and frame flags) -- a model's script or the gold."""
        reuse, link = [], []
        for t, s in enumerate(links):
            if s is not None and s >= 0:
                reuse.append(LINKED); link.append(int(s))
            elif frames is not None and t < len(frames) and frames[t]:
                reuse.append(FRAME); link.append(-1)
            else:
                reuse.append(DECLINED); link.append(-1)
        used = {s for s in link if s >= 0}
        source = [S_CONSUMED if s in used else S_FREE for s in range(n_source)]
        return cls(reuse, link, source)

    # ---------- training states from the gold ----------

    @classmethod
    def corrupted(
        cls, gold_links: Sequence[int], gold_frames: Optional[Sequence[int]], n_source: int,
        rng: random.Random, *, flip: float = 0.2, add: float = 0.2,
    ) -> "State":
        """The gold script with a share of its links declined and as many false links
        added among the unlinked words (to a free source word), frames kept."""
        links = list(gold_links)
        linked = [t for t, s in enumerate(links) if s >= 0]
        for t in linked:
            if rng.random() < flip:
                links[t] = -1
        n_add = int(round(add * len(linked)))
        free = [s for s in range(n_source) if s not in set(links)]
        open_words = [t for t, s in enumerate(links) if s < 0 and not (gold_frames and gold_frames[t])]
        rng.shuffle(open_words)
        for t in open_words[:n_add]:
            if not free:
                break
            links[t] = free.pop(rng.randrange(len(free)))
        return cls.from_script(links, gold_frames, n_source)

    @classmethod
    def chain(
        cls, gold_links: Sequence[int], gold_frames: Optional[Sequence[int]], n_source: int,
        tiers, stage: int,
    ) -> "State":
        """Easy-to-hard: stage 1 keeps only same-form links with a unique candidate
        (the verbatim skeleton), stage 2 adds every link the resources attest (form or
        lemma), stage 3 is the full gold. ``tiers`` is the [n_reuse, n_source] tier
        grid from retexo.edit_typing.downstream.ScriptFeaturizer.tier_grid; None means every word is a tier-0 word.
        Words not yet decided are UNDECIDED (not declined): the model must still
        decide them."""
        import numpy as np

        n_t = len(gold_links)
        if stage >= 3 or tiers is None:
            if tiers is None and stage < 3:
                return cls.undecided(n_t, n_source)
            return cls.from_script(gold_links, gold_frames, n_source)
        tiers = np.asarray(tiers)
        best = tiers.max(axis=1) if tiers.size else np.zeros(n_t, dtype=int)
        count = (tiers == best[:, None]).sum(axis=1) if tiers.size else np.zeros(n_t, dtype=int)
        reuse, link = [UNDECIDED] * n_t, [-1] * n_t
        for t, s in enumerate(gold_links):
            if s < 0:
                continue
            keep = (best[t] == 4 and count[t] == 1) if stage == 1 else (best[t] >= 3)
            if keep:
                reuse[t] = LINKED; link[t] = int(s)
        used = {s for s in link if s >= 0}
        source = [S_CONSUMED if s in used else S_UNDECIDED for s in range(n_source)]
        return cls(reuse, link, source)

    @classmethod
    def sample_for_training(cls, example, mode: str, rng: random.Random, tiers=None) -> "State":
        """One state for one training example: ``random`` corruption of the gold, or a
        stage of the ``chain`` drawn uniformly (0 = undecided, 1, 2, 3 = gold)."""
        n_t, n_s = len(example.target_tokens), len(example.source_tokens)
        gold = list(example.alignments) if example.alignments else [-1] * n_t
        frames = list(example.frame_labels) if example.frame_labels else None
        if mode == "random":
            if rng.random() < 0.25:
                return cls.undecided(n_t, n_s)
            return cls.corrupted(gold, frames, n_s, rng)
        if mode == "chain":
            stage = rng.randrange(0, 4)
            if stage == 0:
                return cls.undecided(n_t, n_s)
            return cls.chain(gold, frames, n_s, tiers, stage)
        if mode == "gold":
            return cls.from_script(gold, frames, n_s)
        return cls.undecided(n_t, n_s)

    # ---------- diagnostics ----------

    def n_decided(self) -> int:
        return sum(1 for r in self.reuse if r != UNDECIDED)

    @staticmethod
    def isolated_false_links(links: Sequence[int], gold: Sequence[int]) -> Tuple[int, int]:
        """(false links in ``links``, of which isolated between two unlinked words)."""
        false = iso = 0
        n = len(links)
        for t, s in enumerate(links):
            if s >= 0 and gold[t] < 0:
                false += 1
                l_off = t == 0 or links[t - 1] < 0
                r_off = t + 1 >= n or links[t + 1] < 0
                iso += int(l_off and r_off)
        return false, iso
