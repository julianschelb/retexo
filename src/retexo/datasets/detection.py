# retexo/datasets/detection.py
"""
The distance lives inside the window.

``FRAME`` is nearly free by design -- describing the citing author's prose is
one act, not evidence of distance -- so a full-passage normalized cost dilutes
both classes with shared near-zero context and destroys detection (measured:
AUC 0.807 under the old trimmed labels collapsed to 0.588 under teacher labels
with naive normalization, and recovered to 0.804 with the definition here).

The detection score is therefore the in-window cost density: total cost minus
the FRAME acts, divided by the larger of the two window widths. Lower is more
related.
"""

from __future__ import annotations

from typing import Optional, Tuple

from retexo.core.script import EditScript


class DetectionScore:
    """The in-window cost density of a script: total cost minus FRAME acts,
    divided by the larger window width. Lower is more related.

    Example:
        ```python
        score = DetectionScore.window_density(example.script, example.fragment_span)
        ```
    """

    @staticmethod
    def window_widths(script: EditScript, fragment_span: Optional[Tuple[int, int]] = None):
        """Source and target window widths, read off the script's FRAME ops."""
        consumed = sum(
            len(op.source_indices)
            for op in script.operations
            if op.tag == "FRAME" and not op.target_indices
        )
        written = sum(
            len(op.target_indices)
            for op in script.operations
            if op.tag == "FRAME" and op.target_indices
        )
        source_window = max(len(script.source_tokens) - consumed, 1)
        if fragment_span is not None:
            target_window = max(fragment_span[1] - fragment_span[0], 1)
        else:
            target_window = max(len(script.target_tokens) - written, 1)
        return source_window, target_window

    @classmethod
    def window_density(
        cls, script: EditScript, fragment_span: Optional[Tuple[int, int]] = None, costs=None
    ) -> float:
        """In-window cost per token of the larger window. Lower = more related."""
        frame_ops = sum(1 for op in script.operations if op.tag == "FRAME")
        frame_cost = (
            script._registry()[  # noqa: SLF001 - cost of one FRAME act
                "FRAME"
            ].default_cost
            if frame_ops
            else 0.0
        )
        in_window = script.cost(costs) - frame_cost * frame_ops
        source_window, target_window = cls.window_widths(script, fragment_span)
        return in_window / max(source_window, target_window)
