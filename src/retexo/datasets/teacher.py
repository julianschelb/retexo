# retexo/datasets/teacher.py
"""
Relabelling real pairs: the structural half of the teacher.

The old path *trimmed* a pair to its located window and planned that, so the
citing author's prose either vanished from the example or, inside the window,
was charged as per-token insertion. The teacher keeps both passages whole,
plans only the located window with the oracle, and describes everything outside
it as ``FRAME`` -- one operation per contiguous run, on either side: framing
prose in the reuse is *written* by a FRAME, and source context not carried over
is *consumed* by one. Both replay as structure rather than being priced as
residual, which is the "describe, don't charge" argument of D-40 made concrete.

The examples this produces match inference conditions: a model sees full
passages, not pre-trimmed ones, and learns to name the framing itself.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from retexo.datasets.localize import Localized
from retexo.operations import EditOperation
from retexo.core.script import EditScript


class RelabellingTeacher:
    """Plans only the located window with the oracle and frames everything
    else, so a real pair keeps both passages whole.

    Args:
        aligner: Optional :class:`~retexo.aligners.aligner.ContextualAligner`
            converting resource-silent DEL+INS residue into arrows.
        aligner_threshold: Mutual-best-match floor for a conversion.

    Example:
        ```python
        teacher = RelabellingTeacher()
        script, span = teacher.relabel_pair(source, target, oracle)
        ```
    """

    def __init__(self, *, aligner=None, aligner_threshold: float = 0.70):
        self.aligner = aligner
        self.aligner_threshold = aligner_threshold

    def relabel_pair(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        oracle,
    ) -> Tuple[EditScript, Tuple[int, int]]:
        """A full-passage script: oracle inside the window, FRAME outside it.

        Returns the script and the fragment's span inside the target.
        """
        span = Localized.find(source_tokens, target_tokens)
        inner = oracle.plan_tokens(span.source, span.target)
        if self.aligner is not None:
            inner = self._add_aligner_arrows(inner, span)

        operations: List[EditOperation] = []
        self._frame_consuming(operations, 0, span.source_offset, source_tokens)
        self._frame_consuming(
            operations,
            span.source_offset + len(span.source),
            len(source_tokens),
            source_tokens,
        )
        self._frame_writing(operations, 0, span.target_offset, target_tokens)

        for op in inner.operations:
            operations.append(EditOperation(
                op.tag,
                tuple(i + span.source_offset for i in op.source_indices),
                tuple(i + span.target_offset for i in op.target_indices),
                op.source_tokens,
                op.target_tokens,
            ))

        fragment_end = span.target_offset + len(span.target)
        self._frame_writing(operations, fragment_end, len(target_tokens), target_tokens)

        script = EditScript(
            list(source_tokens), list(target_tokens), operations, inner.registry
        )
        return script, (span.target_offset, fragment_end)

    @staticmethod
    def _frame_writing(operations, begin: int, end: int, tokens) -> None:
        """Context in the reuse: one FRAME writing the span."""
        if end > begin:
            operations.append(EditOperation(
                "FRAME", (), tuple(range(begin, end)), (),
                tuple(tokens[begin:end]),
            ))

    @staticmethod
    def _frame_consuming(operations, begin: int, end: int, tokens) -> None:
        """Context in the source: one FRAME consuming the span, writing nothing."""
        if end > begin:
            operations.append(EditOperation(
                "FRAME", tuple(range(begin, end)), (),
                tuple(tokens[begin:end]), (),
            ))

    def _add_aligner_arrows(self, inner, span):
        """Convert resource-silent DEL+INS residue into arrows the aligner trusts.

        Only mutual-best matches at or above :attr:`aligner_threshold` convert,
        and each conversion replaces one deletion and one insertion with a
        single ``SYN-DIST`` operation carrying the aligner's evidence -- the
        same tag the static vectors use, since both assert a distributional
        relation; the provenance is recorded in the run configuration rather
        than the tag.
        """
        deleted = {i for op in inner.operations if op.tag == "DEL"
                   for i in op.source_indices}
        inserted = {j for op in inner.operations if op.tag == "INS"
                    for j in op.target_indices}
        if not deleted or not inserted:
            return inner

        src = sorted(deleted)
        tgt = sorted(inserted)
        matches = self.aligner.mutual_best(
            [span.source[i] for i in src], [span.target[j] for j in tgt],
            threshold=self.aligner_threshold,
        )
        if not matches:
            return inner

        convert = {}
        for a, b, _score in matches:
            convert[src[a]] = tgt[b]
        converted_targets = set(convert.values())

        operations = []
        for op in inner.operations:
            if op.tag == "DEL" and op.source_indices[0] in convert:
                i = op.source_indices[0]
                j = convert[i]
                operations.append(EditOperation(
                    "SYN-DIST", (i,), (j,), (span.source[i],), (span.target[j],)
                ))
            elif op.tag == "INS" and op.target_indices[0] in converted_targets:
                continue
            else:
                operations.append(op)
        return EditScript(
            list(inner.source_tokens), list(inner.target_tokens), operations,
            inner.registry,
        )
