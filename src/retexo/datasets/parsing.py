# retexo/datasets/parsing.py
"""
Reading a script back from generated text.

Model output is untrusted: it may name an operation that does not exist, point
outside the passage, or not parse at all. Every failure returns ``None`` rather
than raising, so that unusable output stays a measurable rate instead of an
exception, and is reported separately from output that parses but is wrong.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from retexo.operations import EditOperation, OperationRegistry
from retexo.core.script import EditScript


class ScriptParser:
    """Parses a serialized script back from generated text, or fails safe.

    Example:
        ```python
        script = ScriptParser.parse("<NOP> 0>0 uox | <DEL> 1>-", source, target)
        ```
    """

    PATTERN = re.compile(
        r"<?(?P<tag>[A-Z][A-Z\-]*)>?\s+(?P<src>[\d,]+|-)\s*>\s*(?P<tgt>[\d,]+|-)\s*(?P<toks>[^|]*)"
    )

    @staticmethod
    def _indices(field: str) -> tuple:
        if field.strip() == "-":
            return ()
        return tuple(int(x) for x in field.split(",") if x.strip().isdigit())

    @classmethod
    def parse(
        cls,
        text: str,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        registry: Optional[OperationRegistry] = None,
    ) -> Optional[EditScript]:
        """Parse serialized operations, or return ``None`` if the text is unusable."""
        registry = registry or OperationRegistry.default()
        operations: List[EditOperation] = []

        for chunk in text.split("|"):
            match = cls.PATTERN.search(chunk)
            if match is None:
                continue
            tag = match.group("tag")
            if tag not in registry:
                return None
            source_indices = cls._indices(match.group("src"))
            target_indices = cls._indices(match.group("tgt"))
            if any(i >= len(source_tokens) for i in source_indices):
                return None
            tokens = tuple(match.group("toks").split())
            operations.append(
                EditOperation(
                    tag=tag,
                    source_indices=source_indices,
                    target_indices=target_indices,
                    source_tokens=tuple(source_tokens[i] for i in source_indices),
                    target_tokens=tokens or tuple(),
                )
            )

        if not operations:
            return None
        return EditScript(list(source_tokens), list(target_tokens), operations, registry)
