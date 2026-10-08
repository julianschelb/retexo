# baselines/span_pair/constants.py
"""The span tags, the word types, the dials and the shared span alias."""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Span tags: two for matched pairs, two nulls per reuse span, and NONE ("this
#: span is not a run"), the target of every reuse span that is not a gold run so
#: that the scores of all spans are comparable in the segmentation; NONE is never chosen.
QUOTE, ADAPT, INS, FRAME, NONE = "QUOTE", "ADAPT", "INS", "FRAME", "NONE"
PAIR_TAGS = (QUOTE, ADAPT)
NULL_TAGS = (INS, FRAME, NONE)

#: Word-level types inside an ADAPT span.
WORD_TYPES = ("COPY", "MORPH", "SUBST")

#: The dials and their values from the note.
SPAN_DEFAULTS: Dict[str, Any] = {
    "L_max": 6,
    "eps": 0.01,
    "K_pairs": 2000,
    "swap": "double",
    "gold_passes": 6,
    "train_on": "all",
    "word_decoder": 0,
    "theta": 0.0,
    "hidden": 256,
    "warmup": 0.1,
    "lr_heads": 1e-3,
    "temperature": 0.1,
    "load_base": "",
}

#: A span is inclusive, ``(first, last)``.
Span = Tuple[int, int]

ENCLITICS = ("que", "ue", "ne")
