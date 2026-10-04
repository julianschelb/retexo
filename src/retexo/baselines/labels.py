# retexo/baselines/labels.py
"""Every label space of the paper and every collapse between them, once.

The definition (section 3) fixes one nested inventory. The finest level on an
edge is V3 (COPY, MORPH, SYN, POS, NE-SUB, SUBST, SPLIT, MERGE, plus the
derived REORDER, FRAME, QUOTE, DISPERSE and the residual INS, DEL); the type
groups, the modes, V1 and V0 are collapse functions of it. The project had
these conversions in five places with five spellings (``e4.coarse``,
``run_e25.to_four``, ``e32.to_coarse``, ``e28.FINE_LEXICAL``,
``decode.LINK_TAGS``); this module replaces them with one name per level.

Two vocabularies meet here. The record spells the copy ``COPY``; the decoder
(``decode.LINK_TAGS``) spells it ``NOP`` and lists neither ``COPY`` nor
``SUBST``. ``canonical`` maps the old names in (NOP to COPY, SYN-DIST to SYN,
the demoted relations to SUBST with the relation kept as ``detail``), and
``TO_LINK_TAG`` maps COPY back to NOP on the way out to ``decode_script``.
The structural markers (REORDER, QUOTE, DISPERSE) are token flags the decoder
derives from the geometry of the links; a token's label at any level is the
collapse of its edge operation, never a marker.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

V0 = ("KEEP", "REPLACE", "INS", "DEL")
V1 = ("COPY", "MORPH", "SUBST", "INS", "DEL")
MODES = ("VERBATIM", "ALLUSION", "FRAME", "NOMATCH")
GROUPS = ("form", "lexical-semantic", "cardinality", "structural", "residual")
V3 = ("COPY", "MORPH", "SYN", "POS", "NE-SUB", "SUBST", "SPLIT", "MERGE",
      "REORDER", "FRAME", "QUOTE", "DISPERSE", "INS", "DEL")
#: What an edge may carry.
EDGE_OPS = ("COPY", "MORPH", "SYN", "POS", "NE-SUB", "SUBST", "SPLIT", "MERGE")
LEXICAL = ("SYN", "POS", "NE-SUB", "SUBST")
LEVELS = ("V0", "V1", "mode", "group", "V3")

#: Old spellings and demoted relations; ``None`` means the tag is dropped.
ALIASES: Dict[str, Optional[str]] = {
    "NOP": "COPY", "SYN-DIST": "SYN", "HYPER": "SUBST", "HYPO": "SUBST",
    "ANT": "SUBST", "CO-HYPO": "SUBST", "ADAPT": None,
    "FORM": "MORPH", "SENSE": "SUBST", "LINK": "SUBST",
}
#: Relations kept as ``detail`` when demoted (SYN-DIST folds into SYN, the others into SUBST).
DEMOTED = ("HYPER", "HYPO", "ANT", "CO-HYPO", "SYN-DIST")
#: V3 to ``decode.LINK_TAGS`` on the way into ``decode_script``.
TO_LINK_TAG = {"COPY": "NOP"}
#: The closed ``detail`` list (definition section 3.1b): one lexical value at most, then MORPH features.
LEXICAL_DETAILS = {
    "SUBST": ("HYPER", "HYPO", "CO-HYPO", "ANT", "META", "NUM", "PRON", "FUNC", "CONSTR"),
    "SYN": ("SYN-DIST",),
}
MORPH_FEATURES = ("CASE", "NUMBER", "GENDER", "TENSE", "MOOD", "PERSON", "VOICE", "DEGREE")
#: Operations whose ``detail`` may carry MORPH features (the form changed as well as the lemma).
INFLECTABLE = ("MORPH",) + LEXICAL

#: The paper's vocabulary on disk (the Operation Vocabulary note, 2026-09-22): eight exclusive
#: labels, MORPH spelled INFLECT, and the lexical relations SYN, POS, NE-SUB as SUBST details.
#: The code keeps its own spelling (``EDGE_OPS``); ``to_paper`` and ``from_paper`` are the
#: only crossing, used by the record codec for files whose provenance names a ``format``.
PAPER_OPS = ("INS", "DEL", "COPY", "INFLECT", "SUBST", "SPLIT", "MERGE", "FRAME")
PAPER_EDGE_OPS = ("COPY", "INFLECT", "SUBST", "SPLIT", "MERGE")
#: SUBST details in the paper's closed list; the first three are code labels of their own.
PAPER_SUBST_DETAILS = ("SYN", "POS", "NE-SUB", "SYN-DIST") + LEXICAL_DETAILS["SUBST"]
#: The paper format's version, written into ``provenance.format`` by the gold freeze.
PAPER_FORMAT = "2026-09-24"

_GROUP = {
    "COPY": "form", "MORPH": "form",
    "SYN": "lexical-semantic", "POS": "lexical-semantic", "NE-SUB": "lexical-semantic",
    "SUBST": "lexical-semantic",
    "SPLIT": "cardinality", "MERGE": "cardinality",
    "REORDER": "structural", "FRAME": "structural", "QUOTE": "structural", "DISPERSE": "structural",
    "INS": "residual", "DEL": "residual",
}
_V1 = {
    "COPY": "COPY", "MORPH": "MORPH",
    "SYN": "SUBST", "POS": "SUBST", "NE-SUB": "SUBST", "SUBST": "SUBST",
    "SPLIT": "SUBST", "MERGE": "SUBST",
    "REORDER": "COPY", "QUOTE": "COPY", "DISPERSE": "SUBST",
    "FRAME": "INS", "INS": "INS", "DEL": "DEL",
}
_V0 = {
    "COPY": "KEEP", "MORPH": "REPLACE", "SYN": "REPLACE", "POS": "REPLACE", "NE-SUB": "REPLACE",
    "SUBST": "REPLACE", "SPLIT": "REPLACE", "MERGE": "REPLACE",
    "REORDER": "KEEP", "QUOTE": "KEEP", "DISPERSE": "REPLACE",
    "FRAME": "INS", "INS": "INS", "DEL": "DEL",
}
_MODE_OF_GROUP = {
    "form": "VERBATIM", "lexical-semantic": "ALLUSION", "cardinality": "ALLUSION",
    "residual": "NOMATCH",
}


class Labels:
    """Every label space of the paper and every collapse between them.

    Example:
        ```python
        op, detail = Labels.canonical("HYPER")     # ("SUBST", "HYPER")
        Labels.collapse("mode", op)                 # "ALLUSION"
        ```
    """

    #: The vocabularies themselves, one class attribute per level.
    V0 = V0
    V1 = V1
    MODES = MODES
    GROUPS = GROUPS
    V3 = V3
    EDGE_OPS = EDGE_OPS
    LEXICAL = LEXICAL
    LEVELS = LEVELS
    ALIASES = ALIASES
    DEMOTED = DEMOTED
    TO_LINK_TAG = TO_LINK_TAG
    LEXICAL_DETAILS = LEXICAL_DETAILS
    MORPH_FEATURES = MORPH_FEATURES
    INFLECTABLE = INFLECTABLE
    PAPER_OPS = PAPER_OPS
    PAPER_EDGE_OPS = PAPER_EDGE_OPS
    PAPER_SUBST_DETAILS = PAPER_SUBST_DETAILS
    PAPER_FORMAT = PAPER_FORMAT

    @staticmethod
    def canonical(tag: Optional[str]) -> Tuple[Optional[str], str]:
        """An old or fine tag to ``(op, detail)`` in the record's vocabulary.

        ``canonical("NOP") == ("COPY", "")``, ``canonical("HYPER") == ("SUBST", "HYPER")``;
        a dropped tag (ADAPT) gives ``(None, "")``; ``""`` and ``None`` pass through as ``(None, "")``.
        """
        if not tag:
            return None, ""
        tag = str(tag)
        if tag in ALIASES:
            detail = tag if tag in DEMOTED else ""
            return ALIASES[tag], detail
        return tag, ""

    @staticmethod
    def parse_detail(op: str, detail: Optional[str]) -> Tuple[str, Tuple[str, ...]]:
        """Split a ``detail`` string into ``(lexical, morph_features)`` and check it
        against the closed list for ``op`` (section 3.1b): at most one lexical
        value, then MORPH features in the fixed order; MERGE carries ``+<index>``.

        ``parse_detail("SUBST", "HYPER+CASE") == ("HYPER", ("CASE",))``;
        ``parse_detail("MORPH", "") == ("", ())``. Raises ``ValueError`` otherwise.
        """
        if not detail:
            return "", ()
        if op == "MERGE":
            if not (detail.startswith("+") and detail[1:].isdigit()):
                raise ValueError(f"MERGE detail must be '+<index>', got {detail!r}")
            return detail, ()
        parts = [p.strip().upper() for p in str(detail).split("+") if p.strip()]
        lexical = [p for p in parts if p in LEXICAL_DETAILS.get(op, ())]
        morph = [p for p in parts if p in MORPH_FEATURES]
        if len(lexical) > 1:
            raise ValueError(f"{op}: at most one lexical detail, got {lexical}")
        unknown = [p for p in parts if p not in lexical and p not in morph]
        if unknown or (morph and op not in INFLECTABLE) or len(morph) != len(set(morph)):
            raise ValueError(f"{op}: detail {detail!r} is not in the closed list of section 3.1b")
        ordered = tuple(f for f in MORPH_FEATURES if f in morph)
        return (lexical[0] if lexical else ""), ordered

    @staticmethod
    def join_detail(lexical: str, morph: Sequence[str]) -> str:
        """The inverse of ``parse_detail``: ``join_detail("HYPER", ("CASE",)) == "HYPER+CASE"``."""
        return "+".join([lexical] * bool(lexical) + [f for f in MORPH_FEATURES if f in morph])

    @staticmethod
    def normalise_detail(op: str, detail: Optional[str]) -> str:
        """``detail`` in the fixed order of ``parse_detail`` (MOOD+TENSE becomes TENSE+MOOD);
        a detail outside the closed list is returned unchanged rather than raising."""
        if not detail:
            return ""
        try:
            lexical, morph = Labels.parse_detail(op, detail)
        except ValueError:
            return str(detail)
        return lexical if op == "MERGE" else Labels.join_detail(lexical, morph)

    @staticmethod
    def to_paper(op: str, detail: Optional[str] = "") -> Tuple[str, str]:
        """A code edge to the paper's spelling: MORPH becomes INFLECT; SYN, POS and NE-SUB
        become SUBST with the relation first in ``detail`` (``("SYN", "CASE")`` to
        ``("SUBST", "SYN+CASE")``); SYN's own SYN-DIST detail replaces the SYN relation."""
        detail = Labels.normalise_detail(op, detail)
        if op == "MORPH":
            return "INFLECT", detail
        if op in ("SYN", "POS", "NE-SUB"):
            parts = [p for p in detail.split("+") if p]
            if op == "SYN" and parts and parts[0] == "SYN-DIST":
                return "SUBST", "+".join(parts)
            return "SUBST", "+".join([op] + parts)
        return op, detail

    @staticmethod
    def from_paper(op: str, detail: Optional[str] = "") -> Tuple[str, str]:
        """The inverse of ``to_paper``; code spellings (MORPH, SYN ...) pass through, so a
        reader can call it on files of either format."""
        detail = str(detail or "")
        if op == "INFLECT":
            return "MORPH", Labels.normalise_detail("MORPH", detail)
        if op == "SUBST" and detail:
            parts = [p.strip().upper() for p in detail.split("+") if p.strip()]
            head = parts[0]
            if head in ("SYN", "POS", "NE-SUB"):
                code = head
                rest = parts[1:]
                return code, Labels.normalise_detail(code, "+".join(rest))
            if head == "SYN-DIST":
                return "SYN", Labels.normalise_detail("SYN", "+".join(parts))
        return op, Labels.normalise_detail(op, detail)

    @staticmethod
    def _known(op: str) -> str:
        if op not in _GROUP:
            raise ValueError(f"unknown operation {op!r}; expected one of {V3}")
        return op

    @classmethod
    def to_v0(cls, op: str) -> str:
        return _V0[cls._known(op)]

    @classmethod
    def to_v1(cls, op: str) -> str:
        return _V1[cls._known(op)]

    @classmethod
    def to_group(cls, op: str) -> str:
        return _GROUP[cls._known(op)]

    @classmethod
    def to_mode(cls, op: str, in_frame: bool = False) -> str:
        """The mode of a token: FRAME wins over everything, then the group decides."""
        if in_frame or op == "FRAME":
            return "FRAME"
        group = cls.to_group(op)
        if group == "structural":
            return "VERBATIM" if op in ("REORDER", "QUOTE") else "ALLUSION"
        return _MODE_OF_GROUP[group]

    @classmethod
    def to_v3(cls, op: str) -> str:
        return cls._known(op)

    @classmethod
    def collapse(cls, level: str, op: str, in_frame: bool = False) -> str:
        """``op`` at ``level`` (``V0`` | ``V1`` | ``mode`` | ``group`` | ``V3``)."""
        if level == "V0":
            return "INS" if in_frame and op == "INS" else cls.to_v0(op)
        if level == "V1":
            return cls.to_v1(op)
        if level == "mode":
            return cls.to_mode(op, in_frame)
        if level == "group":
            return "structural" if in_frame and op == "INS" else cls.to_group(op)
        if level == "V3":
            return "FRAME" if in_frame and op == "INS" else cls.to_v3(op)
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")

    @staticmethod
    def classes(level: str) -> Tuple[str, ...]:
        """The class list of a level, in reporting order."""
        return {"V0": V0, "V1": V1, "mode": MODES, "group": GROUPS, "V3": V3}[level]

    @classmethod
    def token_labels(cls, links: Sequence[int], tags: Sequence[str], frame: Sequence[int],
                     level: str) -> List[str]:
        """One label per reuse word at ``level`` from the interface (II) triple.

        Unlinked words are INS (FRAME where the frame mask says so, at levels
        that have it: V3, mode, group); a linked word's label is the collapse
        of its tag, which must be set (``""`` on a linked word is the scorer's
        error).
        """
        out = []
        for t, s in enumerate(links):
            in_frame = bool(frame[t]) if t < len(frame) else False
            if s is None or s < 0:
                out.append(cls.collapse(level, "INS", in_frame))
            else:
                tag = tags[t] if t < len(tags) else ""
                op, _ = cls.canonical(tag)
                if not op:
                    raise ValueError(f"linked reuse word {t} has no tag")
                out.append(cls.collapse(level, op, False))
        return out

    @classmethod
    def tag_labels(cls, tags: Sequence[str], frame: Sequence[int], level: str) -> List[str]:
        """One label per reuse word at ``level`` from the tags alone (a tagger's
        output: no links). A word tagged ``FRAME`` or flagged by the frame mask is
        FRAME at the levels that have it; an empty or ``INS`` tag is INS; any other
        tag collapses to ``level``."""
        #: a mode tagger's vocabulary read at the other levels (approximate; that row reports mode F1 only)
        mode_as_op = {"VERBATIM": "COPY", "ALLUSION": "SUBST", "FRAME": "FRAME", "NOMATCH": "INS"}
        out = []
        for t, tag in enumerate(tags):
            in_frame = bool(frame[t]) if t < len(frame) else False
            if tag in MODES:
                if level == "mode":
                    out.append(tag)
                    continue
                tag = mode_as_op[tag]
            op, _ = cls.canonical(tag)
            if op in (None, "", "INS") or op == "FRAME":
                out.append(cls.collapse(level, "INS", in_frame or op == "FRAME"))
            else:
                out.append(cls.collapse(level, op, False))
        return out

    @staticmethod
    def source_labels(links: Sequence[int], n_source: int) -> List[str]:
        """Per source word: KEEP where a reuse word points at it, DEL otherwise."""
        used = {s for s in links if s is not None and s >= 0}
        return ["KEEP" if s in used else "DEL" for s in range(n_source)]


#: Backward-compatible module-level aliases; ``labels.canonical(...)`` etc. still work.
canonical = Labels.canonical
to_v0 = Labels.to_v0
to_v1 = Labels.to_v1
to_group = Labels.to_group
to_mode = Labels.to_mode
to_v3 = Labels.to_v3
collapse = Labels.collapse
classes = Labels.classes
token_labels = Labels.token_labels
source_labels = Labels.source_labels
parse_detail = Labels.parse_detail
tag_labels = Labels.tag_labels
join_detail = Labels.join_detail
