# retexo/aligners/decode.py
"""
From a typed alignment to the paper's object: a replayable, costed EditScript.

The model emits three things per reuse word -- which source word it came from
(the pointer), what kind of change the link records (the typer), and whether
the word belongs to an attribution formula (the frame head) -- plus a deletion
flag per source word. Everything above the token is *derived* here, by rules
that need no learning because they are definitions:

    REORDER   a link outside a longest increasing subsequence of source
              positions, taken in reuse order -- the oracle's own rule
    QUOTE     at least four consecutive reuse words copied unchanged from
              consecutive source positions, in order
    ADAPT     a region of reused material whose substitutions cluster: at
              least two typed substitutions inside a run of links with no gap
              wider than one word
    DISPERSE  the reused material straddles a clause boundary on the reuse
              side -- punctuation that closes a clause falls between two links
    SPLIT     a link whose source carries an enclitic the reuse does not
    MERGE     the reverse
    DEL       a source word no link consumes
    FRAME     a maximal run of frame-flagged reuse words, priced once

The result can be handed to :class:`retexo.core.scriba.Scriba` to replay, which
is the paper's definition of a correct script, and to the cost model, which is
its definition of a distance.

    script = ScriptDecoder.decode_script(source, reuse, alignment, fine_tags, frame_mask)
"""

from __future__ import annotations

import bisect
import re
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

from retexo.core.normalize import normalize
from retexo.operations import EditOperation, OperationRegistry
from retexo.core.script import EditScript


class ScriptDecoder:
    """Derives the full typed, replayable script from a pointer's per-word
    outputs, by rules that need no learning because they are definitions.

    Example:
        ```python
        script = ScriptDecoder.decode_script(source_tokens, target_tokens, alignment,
                                             fine_tags, frame_mask)
        view = ScriptDecoder.per_token_view(script)
        ```
    """

    #: Fine tags the typer may emit for a link. Anything else is treated as a
    #: plain substitution.
    LINK_TAGS: ClassVar[Tuple[str, ...]] = ("NOP", "MORPH", "SYN", "HYPER", "HYPO", "ANT", "SYN-DIST",
                                            "NE-SUB", "POS", "SPLIT", "MERGE",
                                            "FORM", "SENSE")        # E31 group-level back-offs

    #: Tags that count as "the word was reworked" when looking for ADAPT regions.
    SUBSTITUTION_TAGS: ClassVar[frozenset] = frozenset(t for t in LINK_TAGS if t not in ("NOP",))

    #: Punctuation that closes a clause, on the reuse side.
    CLAUSE_END = re.compile(r"[.;:?!]$")

    QUOTE_MIN = 4

    # ---------- structural derivations ----------

    @staticmethod
    def lis_positions(values: Sequence[int]) -> Set[int]:
        """Positions of one longest strictly increasing subsequence."""
        if not values:
            return set()
        tails: List[int] = []
        tail_positions: List[int] = []
        previous = [-1] * len(values)
        for position, value in enumerate(values):
            slot = bisect.bisect_left(tails, value)
            if slot == len(tails):
                tails.append(value); tail_positions.append(position)
            else:
                tails[slot] = value; tail_positions[slot] = position
            previous[position] = tail_positions[slot - 1] if slot > 0 else -1
        keep: Set[int] = set()
        position = tail_positions[-1]
        while position != -1:
            keep.add(position)
            position = previous[position]
        return keep

    @classmethod
    def reordered_targets(cls, alignment: Sequence[int]) -> Set[int]:
        """Reuse positions whose link crosses another's, by the LIS rule.

        The same derivation the EditPlan oracle uses on its own alignments, so
        a REORDER predicted here and a REORDER in the gold are the same object.
        """
        linked = [(t, s) for t, s in enumerate(alignment) if s >= 0]
        if len(linked) < 2:
            return set()
        keep = cls.lis_positions([s for _, s in linked])
        return {t for k, (t, _) in enumerate(linked) if k not in keep}

    @classmethod
    def quote_spans(cls, alignment: Sequence[int], tags: Sequence[str],
                    minimum: int = QUOTE_MIN) -> List[Tuple[int, int]]:
        """Maximal runs of >= ``minimum`` in-order verbatim links, as (start, end)."""
        spans, start = [], None
        for t in range(len(alignment) + 1):
            ok = (t < len(alignment) and alignment[t] >= 0 and tags[t] == "NOP"
                  and (start is None or alignment[t] == alignment[t - 1] + 1))
            if ok and start is None:
                start = t
            elif not ok:
                if start is not None and t - start >= minimum:
                    spans.append((start, t - 1))
                start = t if (t < len(alignment) and alignment[t] >= 0
                              and tags[t] == "NOP") else None
        return spans

    @classmethod
    def adapt_spans(cls, alignment: Sequence[int], tags: Sequence[str],
                    max_gap: int = 1, min_subst: int = 2) -> List[Tuple[int, int]]:
        """Regions of reuse where substitutions concentrate."""
        spans, start, last = [], None, None
        for t, s in enumerate(alignment):
            if s < 0:
                continue
            if start is None or t - last > max_gap + 1:
                if start is not None:
                    spans.append((start, last))
                start = t
            last = t
        if start is not None:
            spans.append((start, last))
        return [(a, b) for a, b in spans
                if sum(1 for t in range(a, b + 1)
                       if alignment[t] >= 0 and tags[t] in cls.SUBSTITUTION_TAGS) >= min_subst]

    @classmethod
    def dispersed(cls, target_tokens: Sequence[str], alignment: Sequence[int]) -> bool:
        """Whether the reused material straddles a clause boundary on the reuse side."""
        linked = [t for t, s in enumerate(alignment) if s >= 0]
        if len(linked) < 2:
            return False
        first, last = linked[0], linked[-1]
        return any(cls.CLAUSE_END.search(target_tokens[t]) for t in range(first, last))

    @classmethod
    def clean_frame_mask(cls, frame_mask: Sequence[int], min_run: int = 2,
                         fill_gap: int = 1) -> List[int]:
        """A frame is a formula, not a word: close one-word gaps, drop one-word runs.

        The hand-labelled frames average six words and none is a single word, so
        a lone flagged word is noise and a lone unflagged word inside a run is a
        boundary slip.
        """
        mask = [int(bool(f)) for f in frame_mask]
        n = len(mask)
        # fill short gaps between flagged words
        t = 0
        while t < n:
            if mask[t] == 0 and 0 < t and t + fill_gap < n and mask[t - 1]:
                run_end = t
                while run_end < n and mask[run_end] == 0 and run_end - t < fill_gap:
                    run_end += 1
                if run_end < n and mask[run_end] and run_end - t <= fill_gap:
                    for k in range(t, run_end):
                        mask[k] = 1
                    t = run_end
                    continue
            t += 1
        # drop runs shorter than the minimum
        for a, b in cls.frame_spans(mask):
            if b - a + 1 < min_run:
                for k in range(a, b + 1):
                    mask[k] = 0
        return mask

    @staticmethod
    def frame_spans(frame_mask: Sequence[int]) -> List[Tuple[int, int]]:
        spans, start = [], None
        for t in range(len(frame_mask) + 1):
            on = t < len(frame_mask) and frame_mask[t]
            if on and start is None:
                start = t
            elif not on and start is not None:
                spans.append((start, t - 1)); start = None
        return spans

    # ---------- decoding ----------

    @classmethod
    def decode_script(cls, source_tokens: Sequence[str], target_tokens: Sequence[str],
                      alignment: Sequence[int], fine_tags: Optional[Sequence[str]] = None,
                      frame_mask: Optional[Sequence[int]] = None,
                      source_deleted: Optional[Sequence[int]] = None,
                      registry: Optional[OperationRegistry] = None) -> EditScript:
        """Assemble the full typed script from the model's per-word outputs.

        ``fine_tags`` may be missing for a link, in which case the tag is read off
        the strings: identical after normalization is NOP, otherwise SUBST is not
        in the inventory and the link is recorded as ``SYN-DIST``-free plain
        substitution under the closest available tag, ``MORPH`` if the lemma is
        unknown -- no: a link with no evidence is emitted as ``SUBST`` so that a
        reader can see the typer declined. The registry does not price ``SUBST``,
        and the cost model treats it as a synonym-priced change.
        """
        registry = registry or OperationRegistry.default()
        source, target = list(source_tokens), list(target_tokens)
        n_t = len(target)
        alignment = [alignment[t] if t < len(alignment) else -1 for t in range(n_t)]
        alignment = [s if (s is not None and 0 <= s < len(source)) else -1 for s in alignment]
        tags: List[str] = []
        for t in range(n_t):
            s = alignment[t]
            if s < 0:
                tags.append("INS"); continue
            tag = fine_tags[t] if fine_tags is not None and t < len(fine_tags) and fine_tags[t] else None
            if normalize(source[s]) == normalize(target[t]):
                tag = "NOP"
            elif tag not in cls.LINK_TAGS:
                tag = "SUBST"
            tags.append(tag)
        # A frame is the citing author's own words by definition, so a predicted
        # frame never covers a word that has a source: the link wins.
        frame_mask = [int(bool(frame_mask[t])) if frame_mask is not None and t < len(frame_mask)
                      else 0 for t in range(n_t)]
        frame_mask = [f if alignment[t] < 0 else 0 for t, f in enumerate(frame_mask)]
        frame_mask = cls.clean_frame_mask(frame_mask)
        frames = cls.frame_spans(frame_mask)
        frame_at = {a: (a, b) for a, b in frames}
        # QUOTE is a *writing* operation in the inventory -- "copy a contiguous
        # span verbatim as one act" -- so it stands in for the copies it covers
        # rather than sitting on top of them, which is what the verifier checks.
        quotes = cls.quote_spans(alignment, tags)
        quote_at = {a: (a, b) for a, b in quotes}

        operations: List[EditOperation] = []
        # ---- reuse side, in order: frames and quotes as one act each, links,
        #      insertions
        t = 0
        while t < n_t:
            if t in frame_at:
                a, b = frame_at[t]
                operations.append(EditOperation(
                    "FRAME", (), tuple(range(a, b + 1)), (), tuple(target[a:b + 1]),
                    "attribution"))
                t = b + 1
                continue
            if t in quote_at:
                a, b = quote_at[t]
                operations.append(EditOperation(
                    "QUOTE", tuple(alignment[k] for k in range(a, b + 1)),
                    tuple(range(a, b + 1)),
                    tuple(source[alignment[k]] for k in range(a, b + 1)),
                    tuple(target[a:b + 1]), "verbatim run"))
                t = b + 1
                continue
            s = alignment[t]
            if s < 0:
                operations.append(EditOperation("INS", (), (t,), (), (target[t],)))
            else:
                operations.append(EditOperation(tags[t], (s,), (t,), (source[s],),
                                                (target[t],)))
            t += 1
        # ---- source side: what no link consumed
        used = {s for s in alignment if s >= 0}
        for s, token in enumerate(source):
            deleted = source_deleted[s] if source_deleted is not None and s < len(source_deleted) else 1
            if s not in used and deleted:
                operations.append(EditOperation("DEL", (s,), (), (token,), ()))
        # ---- structure: markers derived from the alignment
        # Markers: they occupy nothing and exist to be counted, so they carry
        # source indices only, as the builder emits them.
        for t in sorted(cls.reordered_targets(alignment)):
            s = alignment[t]
            operations.append(EditOperation("REORDER", (s,), (), (source[s],), (),
                                            "crossing"))
        for a, b in cls.adapt_spans(alignment, tags):
            idx = tuple(alignment[t] for t in range(a, b + 1) if alignment[t] >= 0)
            operations.append(EditOperation(
                "ADAPT", idx, (), tuple(source[s] for s in idx), (),
                f"substitutions cluster in reuse {a}-{b}"))
        if cls.dispersed(target, alignment):
            idx = tuple(alignment[t] for t in range(n_t) if alignment[t] >= 0)
            operations.append(EditOperation(
                "DISPERSE", idx, (), tuple(source[s] for s in idx), (),
                "across a clause boundary"))
        return EditScript(source, target, operations, registry)

    # ---------- views used by the scorer ----------

    @classmethod
    def per_token_view(cls, script: EditScript) -> Dict[str, List]:
        """Per-word labels a scorer can compare against the hand labels."""
        n_t = len(script.target_tokens)
        tags = ["INS"] * n_t
        frame = [0] * n_t
        reorder = [0] * n_t
        quote = [0] * n_t
        link = [-1] * n_t
        for op in script.operations:
            if op.tag == "FRAME":
                for t in op.target_indices:
                    frame[t] = 1
            elif op.tag == "QUOTE":
                # one act standing in for its copies: expand it back to per-word
                for t, s in zip(op.target_indices, op.source_indices):
                    quote[t] = 1; tags[t] = "NOP"; link[t] = s
            elif op.tag in ("REORDER", "ADAPT", "DISPERSE", "DEL"):
                continue
            elif op.target_indices:
                t = op.target_indices[0]
                tags[t] = op.tag
                if op.source_indices:
                    link[t] = op.source_indices[0]
        for t in cls.reordered_targets(link):
            reorder[t] = 1
        return {"tags": tags, "frame": frame, "reorder": reorder, "quote": quote,
                "link": link}


#: Kept for callers that want the constant without naming the class.
LINK_TAGS = ScriptDecoder.LINK_TAGS
QUOTE_MIN = ScriptDecoder.QUOTE_MIN
