# retexo/formulations/encoding.py
"""
Turning examples into model inputs, one function per formulation.

The formulations differ in how a script is represented as a prediction target,
and this is where that difference lives. Keeping the encodings together makes
it visible that they are three views of the same supervision rather than three
datasets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from retexo.core.script import EditScript

#: Written for reuse positions that no source token produced.
NULL_SOURCE = -1

#: Tags no model predicts directly. Both are read off the alignment instead: a
#: reuse position with no source is an insertion, a source position with no
#: reuse is a deletion. Deriving them keeps the two statements consistent by
#: construction, and denies a model the option of accounting for a pair it
#: cannot explain by insertion and deletion alone.
DERIVED_TAGS = ("INS", "DEL")


class ScriptEncoder:
    """Turns scripts into model inputs, one method per formulation.

    Example:
        ```python
        source, target = ScriptEncoder.to_seq2seq(script)
        labels = ScriptEncoder.to_token_labels(script)
        ```
    """

    # ---------- sequence to sequence ----------

    @staticmethod
    def to_seq2seq(script: EditScript, *, atomic_tags: bool = True) -> Tuple[str, str]:
        """Render a script as an (input, target) string pair.

        The input marks source positions so the decoder can refer to them; the
        target is the serialized script. With ``atomic_tags`` the operation
        names are written as single bracketed tokens, which a tokenizer can be
        taught as one symbol each rather than spelling out.
        """
        marked = " ".join(f"{i}:{tok}" for i, tok in enumerate(script.source_tokens))
        text = " ".join(script.target_tokens)
        model_input = f"source: {marked} reuse: {text}"

        parts: List[str] = []
        for op in script.operations:
            tag = f"<{op.tag}>" if atomic_tags else op.tag
            src = ",".join(str(i) for i in op.source_indices) or "-"
            tgt = ",".join(str(i) for i in op.target_indices) or "-"
            tokens = " ".join(op.target_tokens)
            parts.append(f"{tag} {src}>{tgt} {tokens}".strip())
        return model_input, " | ".join(parts)

    @staticmethod
    def operation_tokens(tags: Sequence[str]) -> List[str]:
        """Vocabulary entries to add so each operation name is a single token."""
        return [f"<{tag}>" for tag in tags]

    # ---------- token classification ----------

    @staticmethod
    def to_token_labels(script: EditScript) -> Dict[str, List]:
        """Per reuse token, which operation produced it and from which source token.

        Deletions carry no reuse position and so contribute no label; they are
        recovered at decode time from the source positions nothing points at.

        Returns:
            ``op_labels`` — one tag per reuse position;
            ``source_indices`` — the source position each reuse position came
            from, or ``NULL_SOURCE`` for insertions.
        """
        n = len(script.target_tokens)
        op_labels: List[Optional[str]] = [None] * n
        source_indices: List[int] = [NULL_SOURCE] * n

        for op in script.operations:
            if not op.target_indices:
                continue
            for position, target_index in enumerate(op.target_indices):
                if not 0 <= target_index < n:
                    continue
                op_labels[target_index] = op.tag
                if op.source_indices:
                    source = op.source_indices[min(position, len(op.source_indices) - 1)]
                    source_indices[target_index] = source

        return {
            "tokens": list(script.target_tokens),
            "op_labels": [label or "INS" for label in op_labels],
            "source_indices": source_indices,
        }

    @staticmethod
    def label_vocabulary(tags: Sequence[str]) -> Dict[str, int]:
        """A stable tag-to-index mapping for the operation head."""
        return {tag: i for i, tag in enumerate(sorted(tags))}


#: Backward-compatible module-level aliases.
to_seq2seq = ScriptEncoder.to_seq2seq
operation_tokens = ScriptEncoder.operation_tokens
to_token_labels = ScriptEncoder.to_token_labels
label_vocabulary = ScriptEncoder.label_vocabulary
