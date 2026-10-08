# formulations/change_detector/example.py
"""One labelled (source, reuse) pair, as the change detector and its heads consume it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# =============================================================================
# ChangeExample
# =============================================================================


@dataclass(frozen=True)
class ChangeExample:
    """One (source, reuse) pair with a binary label per reuse word.

    ``operations`` records which operation produced each reuse word, so recall
    can be decomposed by operation type at scoring time. It is never shown to
    the model.
    """

    source_tokens: List[str]
    target_tokens: List[str]
    labels: List[int]
    operations: List[str]
    n_operations: int
    #: Per *source* word, 1 where the reuse dropped it. Only used when the
    #: config asks for a source head; ``None`` means the question was not put.
    source_labels: Optional[List[int]] = None
    source_operations: Optional[List[str]] = None
    #: Per *reuse* word, the index of the source word it came from, or -1 for a
    #: word the citing author supplied. This is the pointer's supervision, and
    #: the generator knows it exactly: it spliced the fragment in.
    alignments: Optional[List[int]] = None
    #: E24. Per reuse word, the fine operation of its link ("INS" for none,
    #: "?" for a link whose kind is unknown and must not be trained on), a
    #: 0/1 frame flag, and the symbolic evidence vector of its gold link.
    fine_operations: Optional[List[str]] = None
    frame_labels: Optional[List[int]] = None
    link_features: Optional[List[Optional[List[float]]]] = None
    #: E25. The evidence for *every* (reuse word, source word) pair, as a
    #: float16 array of shape [n_reuse, n_source, feature_dim]. Filled by
    #: ``e24.featurize_pairs`` and read by the typed pointer, which needs it
    #: inside the alignment decision rather than after it.
    pair_features: Optional[object] = None
