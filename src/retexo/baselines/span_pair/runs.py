# baselines/span_pair/runs.py
"""The record's gold read as maximal runs, and runs cut to a training length."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from retexo.baselines.record import Record, RecordInterface
from retexo.baselines.span_pair.constants import ADAPT, FRAME, INS, QUOTE, Span

# =============================================================================
# Run
# =============================================================================


@dataclass(frozen=True)
class Run:
    """A maximal stretch of the reuse with one span tag.

    Example:
        ```python
        Run(start=5, end=6, tag="QUOTE", source=(5, 6))   # words 5..6 copy source 5..6
        Run(start=3, end=4, tag="INS", source=None)
        ```
    """

    start: int
    end: int
    tag: str
    source: Optional[Span] = None

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def span(self) -> Span:
        return (self.start, self.end)


# =============================================================================
# RunReader
# =============================================================================


class RunReader:
    """The record's gold as runs, and back.

    Example:
        ```python
        runs = RunReader.gold_runs(record)
        ```
    """

    @staticmethod
    def chunked(runs: Sequence[Run], max_length: int) -> List[Run]:
        """Runs longer than ``max_length`` cut into consecutive pieces of at most
        that length (the source stretch cut at the same offsets), so a long
        quotation still teaches the scorer its parts."""
        out: List[Run] = []
        for run in runs:
            if run.length <= max_length:
                out.append(run)
                continue
            for start in range(run.start, run.end + 1, max_length):
                end = min(start + max_length - 1, run.end)
                source = None
                if run.source is not None:
                    offset = start - run.start
                    source = (run.source[0] + offset, run.source[0] + offset + (end - start))
                out.append(Run(start, end, run.tag, source))
        return out

    @staticmethod
    def gold_runs(record: Record) -> List[Run]:
        """Maximal runs of consecutive reuse positions whose links land on
        consecutive source positions in order; QUOTE when every edge is COPY,
        ADAPT otherwise; INS or FRAME (by the frame flag) for sourceless runs."""
        links, tags, frame, _ = RecordInterface.links_of(record)
        n = len(links)
        runs: List[Run] = []
        t = 0
        while t < n:
            if links[t] < 0:
                kind = FRAME if frame[t] else INS
                u = t
                while u + 1 < n and links[u + 1] < 0 and (FRAME if frame[u + 1] else INS) == kind:
                    u += 1
                runs.append(Run(t, u, kind, None))
            else:
                u = t
                while u + 1 < n and links[u + 1] == links[u] + 1:
                    u += 1
                all_copy = all(tags[k] == "COPY" for k in range(t, u + 1))
                runs.append(Run(t, u, QUOTE if all_copy else ADAPT, (links[t], links[u])))
            t = u + 1
        return runs
