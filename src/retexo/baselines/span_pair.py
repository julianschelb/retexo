# retexo/baselines/span_pair.py
"""Note 21: the span-pair scorer, Seq2Edits' idea for our setting.

Instead of a label per word, the model decides at the level of *runs*: "this
stretch of the reuse comes from that stretch of the source, by this kind of
change", then names the words inside. Seq2Edits (Stahlberg & Kumar 2020) emits
(tag, span end, replacement) triples autoregressively; here the replacement is
a *reference* to a source span, so the cheap and faithful form is a scorer
over candidate span pairs:

1. a **substrate** gives one vector per word of the jointly encoded pair (the
   row's own encoder, trained jointly) and a word grid for pruning;
2. **candidate pairs**: reuse spans of length 1 to ``L_max`` against source spans
   of length 1 to ``L_max + 2`` whose rectangle holds a grid cell above ``eps``
   or an equal normalised form, capped at ``K_pairs`` per record; every reuse
   span also pairs with the two nulls INS and FRAME;
3. a **span-pair head** scores each pair for the span tags QUOTE and ADAPT and
   each reuse span for the two nulls, one softmax per reuse span over
   ``{INS, FRAME} u {(source span, tag)}``, the pointer's cell softmax lifted
   to spans; a **word-type head** names the word pairs inside ADAPT spans;
4. **targets** are the record's maximal monotone runs (QUOTE when every edge is
   COPY, ADAPT otherwise, INS and FRAME for sourceless runs; a crossing or a
   gap starts a new run);
5. **decoding** is a segmentation dynamic programme over reuse positions (every
   position covered by exactly one chosen span, scores are log probabilities,
   null spans of length one always available) with a greedy one-to-one repair
   on the source side, then the expansion of each chosen pair to word links
   (position-wise on equal lengths, the Hungarian inside the rectangle
   otherwise, an enclitic SPLIT/MERGE on a length mismatch of one).

ADAPT is an internal segmentation label: it never reaches an edge. The class
overrides ``postprocess`` with its own segmentation; the word rows it emits
serve the dump and the ``word_decoder=1`` ablation.

    python run_baseline.py --method span_pair --fold 4 --smoke 20 --device cpu --extra gold_passes=1
    python run_baseline.py --method span_pair --set multimwa_mtref --base-model bert-base-cased \\
        --extra gold_passes=6,swap=double,L_max=6
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry, labels
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Edge, Record, RecordInterface
from retexo.baselines.typed_pointer import TypedPointerBaseline
from retexo.core.normalize import normalize

#: Span tags: two for matched pairs, two nulls per reuse span, and NONE ("this
#: span is not a run"), the target of every reuse span that is not a gold run so
#: that the scores of all spans are comparable in the segmentation; NONE is never chosen.
QUOTE, ADAPT, INS, FRAME, NONE = "QUOTE", "ADAPT", "INS", "FRAME", "NONE"
PAIR_TAGS = (QUOTE, ADAPT)
NULL_TAGS = (INS, FRAME, NONE)

#: Word-level types inside an ADAPT span.
WORD_TYPES = ("COPY", "MORPH", "SUBST")

#: The dials and their values from the note.
SPAN_DEFAULTS: Dict[str, Any] = {"L_max": 6, "eps": 0.01, "K_pairs": 2000, "swap": "double", "gold_passes": 6,
                                 "train_on": "all", "word_decoder": 0, "theta": 0.0, "hidden": 256, "warmup": 0.1,
                                 "lr_heads": 1e-3, "temperature": 0.1, "load_base": ""}

#: A span is inclusive, ``(first, last)``.
Span = Tuple[int, int]

ENCLITICS = ("que", "ue", "ne")


# =============================================================================
# Runs: the record read as spans
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
                out.append(run); continue
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


# =============================================================================
# Candidate spans and pairs
# =============================================================================


class SpanEnumerator:
    """Spans and the pruned pairs of one record.

    Example:
        ```python
        spans = SpanEnumerator.enumerate_spans(5, 3)             # 12 spans
        pairs = SpanEnumerator.candidate_pairs(grid, reuse, source, L_max=6)
        ```
    """

    @staticmethod
    def enumerate_spans(n: int, max_length: int) -> List[Span]:
        return [(a, b) for a in range(n) for b in range(a, min(n, a + max_length))]

    @classmethod
    def candidate_pairs(cls, grid, reuse_tokens: Sequence[str], source_tokens: Sequence[str], *, L_max: int = 6,
                        eps: float = 0.01, K_pairs: int = 2000, gold_pairs: Sequence[Tuple[Span, Span]] = ()
                        ) -> List[Tuple[Span, Span]]:
        """Two tiers. First, for every cell above ``eps`` and every equal-form
        cell, the 1x1 pair and every equal-length diagonal pair through it (the
        shape of almost every gold run), kept whole. Then the unequal-length
        rectangles (a difference of at most two) holding a cell above ``eps``,
        ranked by their best cell and the smaller area, up to ``K_pairs`` in
        all; the gold pairs always included (training)."""
        import numpy as np

        grid = np.asarray(grid, dtype=np.float32)
        T, S = grid.shape
        if T == 0 or S == 0:
            return list(gold_pairs)
        forms_r = [normalize(w) for w in reuse_tokens]
        forms_s = [normalize(w) for w in source_tokens]
        anchors = {(t, u) for t in range(T) for u in range(S)
                   if grid[t, u] > eps or (forms_r[t] and forms_r[t] == forms_s[u])}
        kept: List[Tuple[Span, Span]] = []
        seen = set()
        for t, u in sorted(anchors, key=lambda c: -grid[c]):
            for length in range(1, L_max + 1):
                for offset in range(length):
                    a, c = t - offset, u - offset
                    b, d = a + length - 1, c + length - 1
                    if a < 0 or c < 0 or b >= T or d >= S:
                        continue
                    pair = ((a, b), (c, d))
                    if pair not in seen:
                        seen.add(pair); kept.append(pair)
        above = (grid > eps).astype(np.int32)
        scored: List[Tuple[float, int, Span, Span]] = []
        for a, b in cls.enumerate_spans(T, L_max):
            col_any = np.concatenate([[0], np.cumsum(above[a:b + 1].any(axis=0).astype(np.int32))])
            col_max = grid[a:b + 1].max(axis=0)
            length_r = b - a + 1
            for length_s in range(max(1, length_r - 2), min(L_max + 2, S, length_r + 2) + 1):
                if length_s == length_r:
                    continue
                starts = np.arange(0, S - length_s + 1)
                keep = (col_any[starts + length_s] - col_any[starts]) > 0
                peak = np.max(np.stack([col_max[k:k + len(starts)] for k in range(length_s)]), axis=0)
                for c in starts[keep]:
                    pair = ((a, b), (int(c), int(c) + length_s - 1))
                    if pair not in seen:
                        scored.append((float(peak[c]), -(length_r * length_s), pair[0], pair[1]))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        for _, _, r, s_ in scored:
            if len(kept) >= K_pairs:
                break
            kept.append((r, s_)); seen.add((r, s_))
        for pair in gold_pairs:
            if pair not in seen:
                kept.append(pair); seen.add(pair)
        return kept


# =============================================================================
# The heads
# =============================================================================


class SpanPairHead:
    """The span representation, the pair scorer and the null scorer.

    A span is ``[h_start ; h_end ; mean(h) ; phi_len]``; a pair
    ``[r ; s ; |r - s| ; r * s ; phi_dlen ; phi_pos]`` scored for QUOTE and ADAPT;
    a reuse span alone scored for INS and FRAME.

    Example:
        ```python
        head = SpanPairHead(hidden_size=768, L_max=6, device="cpu")
        pair_logits = head.score_pairs(vec_r, vec_s, dlen, pos)     # [n, 2]
        ```
    """

    def __init__(self, hidden_size: int, *, L_max: int, hidden: int = 256, device: str = "cpu", embed: int = 16):
        import torch

        self.L_max = L_max
        self.embed = embed
        span_dim = 3 * hidden_size + embed
        self.len_embedding = torch.nn.Embedding(L_max + 4, embed).to(device)
        self.dlen_embedding = torch.nn.Embedding(2 * (L_max + 3) + 1, embed).to(device)
        self.pair_mlp = torch.nn.Sequential(torch.nn.Linear(4 * span_dim + embed + 1, hidden), torch.nn.ReLU(),
                                            torch.nn.Linear(hidden, len(PAIR_TAGS))).to(device)
        self.null_mlp = torch.nn.Sequential(torch.nn.Linear(span_dim, hidden), torch.nn.ReLU(),
                                            torch.nn.Linear(hidden, len(NULL_TAGS))).to(device)
        self.type_mlp = torch.nn.Sequential(torch.nn.Linear(4 * hidden_size, hidden), torch.nn.ReLU(),
                                            torch.nn.Linear(hidden, len(WORD_TYPES))).to(device)
        self.device = device

    def modules(self):
        return [self.len_embedding, self.dlen_embedding, self.pair_mlp, self.null_mlp, self.type_mlp]

    def parameters(self):
        return [p for m in self.modules() for p in m.parameters()]

    def train(self) -> None:
        for m in self.modules():
            m.train()

    def eval(self) -> None:
        for m in self.modules():
            m.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {f"{i}": m.state_dict() for i, m in enumerate(self.modules())}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for i, m in enumerate(self.modules()):
            m.load_state_dict(state[f"{i}"])

    def span_vectors(self, words, spans: Sequence[Span]):
        """``[h_start ; h_end ; mean ; phi_len]`` for each span over ``words`` [n, H]."""
        import torch

        if not spans:
            return words.new_zeros((0, 3 * words.shape[1] + self.embed))
        starts = torch.tensor([a for a, _ in spans], device=words.device)
        ends = torch.tensor([b for _, b in spans], device=words.device)
        means = torch.stack([words[a:b + 1].mean(dim=0) for a, b in spans])
        lengths = torch.tensor([min(b - a + 1, self.L_max + 3) for a, b in spans], device=words.device)
        return torch.cat([words[starts], words[ends], means, self.len_embedding(lengths)], dim=-1)

    def score_pairs(self, vec_r, vec_s, dlen, pos):
        """``[n, 2]`` logits (QUOTE, ADAPT) for ``n`` pairs of span vectors."""
        import torch

        dlen_idx = (dlen.clamp(-(self.L_max + 3), self.L_max + 3) + self.L_max + 3)
        feats = torch.cat([vec_r, vec_s, (vec_r - vec_s).abs(), vec_r * vec_s, self.dlen_embedding(dlen_idx),
                           pos.unsqueeze(-1)], dim=-1)
        return self.pair_mlp(feats)

    def score_nulls(self, vec_r):
        return self.null_mlp(vec_r)

    def word_types(self, h_t, h_s):
        """``[n, 3]`` logits over COPY, MORPH, SUBST for ``n`` word pairs."""
        import torch

        return self.type_mlp(torch.cat([h_t, h_s, (h_t - h_s).abs(), h_t * h_s], dim=-1))


# =============================================================================
# The substrate: word vectors and a pruning grid
# =============================================================================


class Substrate:
    """The joint encoder: one vector per word of both sides, and a word grid
    (row-softmaxed cosine at ``temperature``) that prunes the candidate pairs.

    Example:
        ```python
        sub = Substrate("bert-base-cased", device="cpu")
        (h_r, h_s), grid = sub.encode_one(reuse, source)
        ```
    """

    def __init__(self, model_name: str, *, device: str = "cpu", max_length: int = 256, temperature: float = 0.1):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self._model = None
        self._pair_encoder = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        from retexo.formulations.pair_encoding import PairEncoder

        kwargs = {"reference_compile": False} if "modernbert" in self.model_name.lower() else {}
        self._model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device)
        self._pair_encoder = PairEncoder.build(self.model_name)

    @property
    def hidden_size(self) -> int:
        self._ensure()
        return int(self._model.config.hidden_size)

    def parameters(self):
        self._ensure()
        return self._model.parameters()

    def train(self) -> None:
        self._ensure(); self._model.train()

    def eval(self) -> None:
        self._ensure(); self._model.eval()

    def state_dict(self):
        self._ensure()
        return self._model.state_dict()

    def load_state_dict(self, state) -> None:
        self._ensure()
        self._model.load_state_dict(state)

    @staticmethod
    def _pool(hidden_row, spans: Sequence[Span]):
        import torch

        return torch.stack([hidden_row[a:max(b, a + 1)].mean(dim=0) for a, b in spans]) if spans else hidden_row[:0]

    def encode(self, pairs: Sequence[Tuple[Sequence[str], Sequence[str]]]):
        """Per pair ``(reuse_words [n_t, H], source_words [n_s, H])``; words the
        encoder truncated away are absent (callers cut to the shorter length)."""
        self._ensure()
        batch, reuse_spans = self._pair_encoder.encode([(list(s), list(r)) for s, r in pairs], self.max_length)
        source_spans = list(self._pair_encoder.last_source_spans)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self._model(**batch).last_hidden_state
        out = []
        for i in range(len(pairs)):
            out.append((self._pool(hidden[i], reuse_spans[i]), self._pool(hidden[i], source_spans[i])))
        return out

    def grid(self, h_r, h_s):
        """Row-softmaxed cosine similarity, a probability per (reuse, source) cell."""
        import torch

        if h_r.shape[0] == 0 or h_s.shape[0] == 0:
            return torch.zeros((h_r.shape[0], h_s.shape[0]))
        r = torch.nn.functional.normalize(h_r.detach(), dim=-1)
        s = torch.nn.functional.normalize(h_s.detach(), dim=-1)
        return torch.softmax((r @ s.T) / self.temperature, dim=-1).cpu()


# =============================================================================
# Segmentation and expansion
# =============================================================================


@dataclass
class SpanChoice:
    """One scored option of a reuse span: a source span with a tag, or a null."""

    reuse: Span
    source: Optional[Span]
    tag: str
    log_p: float
    p: float


class Segmenter:
    """The span-level dynamic programme and the source-side repair.

    Example:
        ```python
        chosen = Segmenter.segment(options, n_reuse, theta=0.3)
        ```
    """

    @staticmethod
    def segment(options: Sequence[SpanChoice], n_reuse: int, *, theta: float = 0.0) -> List[SpanChoice]:
        """The partition of ``[0, n)`` into reuse spans maximising the summed log
        probability of each span's best option (a pair only when its probability
        is at least ``theta``, else its best null); a length-one INS span is
        always available; then two chosen pairs whose source spans overlap
        resolve to the higher-scoring one, the other falling to its null."""
        import math

        best_pair: Dict[Span, SpanChoice] = {}
        best_null: Dict[Span, SpanChoice] = {}
        for option in options:
            if option.tag == NONE:
                continue                                    # "not a run" is never a choice
            table = best_null if option.source is None else best_pair
            if option.reuse not in table or option.log_p > table[option.reuse].log_p:
                table[option.reuse] = option
        choice: Dict[Span, SpanChoice] = {}
        for span in set(best_pair) | set(best_null):
            pair, null = best_pair.get(span), best_null.get(span)
            if pair is not None and pair.p >= theta and (null is None or pair.log_p >= null.log_p):
                choice[span] = pair
            elif null is not None:
                choice[span] = null
            elif pair is not None and pair.p >= theta:
                choice[span] = pair
        floor = math.log(1e-6)
        score = [float("-inf")] * (n_reuse + 1)
        back: List[Optional[Span]] = [None] * (n_reuse + 1)
        score[0] = 0.0
        for end in range(1, n_reuse + 1):
            for span, option in choice.items():
                a, b = span
                if b + 1 != end:
                    continue
                cand = score[a] + option.log_p
                if cand > score[end]:
                    score[end], back[end] = cand, span
            if back[end] is None:                          # a length-one null is always available
                score[end], back[end] = score[end - 1] + floor, (end - 1, end - 1)
                choice.setdefault((end - 1, end - 1), SpanChoice((end - 1, end - 1), None, INS, floor, 0.0))
        chosen: List[SpanChoice] = []
        end = n_reuse
        while end > 0:
            span = back[end]
            chosen.append(choice[span])
            end = span[0]
        chosen.reverse()
        return Segmenter.repair(chosen, best_null)

    @staticmethod
    def repair(chosen: List[SpanChoice], best_null: Dict[Span, SpanChoice]) -> List[SpanChoice]:
        """Greedy one-to-one on the source side: of two pairs whose source spans
        overlap, the lower-scoring one becomes its null."""
        out = list(chosen)
        order = sorted(range(len(out)), key=lambda i: -out[i].log_p)
        used: List[Span] = []
        for i in order:
            option = out[i]
            if option.source is None:
                continue
            c, d = option.source
            if any(c <= d2 and c2 <= d for c2, d2 in used):
                null = best_null.get(option.reuse)
                out[i] = null if null is not None else SpanChoice(option.reuse, None, INS, float("-inf"), 0.0)
            else:
                used.append((c, d))
        return out


class Expander:
    """Chosen span pairs to word links, tags, extra edges and interface (I) rows.

    Example:
        ```python
        links, tags, frame, extra, rows = Expander.expand(chosen, grid, reuse, source, type_logits)
        ```
    """

    @staticmethod
    def _enclitic_split(word: str) -> bool:
        key = normalize(word)
        return any(key.endswith(e) and len(key) - len(e) >= 3 for e in ENCLITICS)

    @classmethod
    def word_pairs(cls, choice: SpanChoice, grid) -> Tuple[List[Tuple[int, int]], List[Edge]]:
        """The word links of one pair: position-wise on equal lengths, the
        Hungarian inside the rectangle otherwise; returns ``(links, extra)``."""
        a, b = choice.reuse
        c, d = choice.source
        n_r, n_s = b - a + 1, d - c + 1
        if n_r == n_s:
            return [(a + k, c + k) for k in range(n_r)], []
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        block = np.asarray(grid)[a:b + 1, c:d + 1]
        rows_i, cols_i = linear_sum_assignment(-block)
        return [(a + int(i), c + int(j)) for i, j in zip(rows_i, cols_i)], []

    @classmethod
    def expand(cls, chosen: Sequence[SpanChoice], grid, reuse_tokens: Sequence[str], source_tokens: Sequence[str],
               type_of) -> Tuple[List[int], List[str], List[int], List[Edge], Rows]:
        """``type_of(t, s) -> str`` names a word pair inside an ADAPT span."""
        n = len(reuse_tokens)
        links, tags, frame = [-1] * n, [""] * n, [0] * n
        extra: List[Edge] = []
        rows: Rows = [[(-1, 1.0)] for _ in range(n)]
        for choice in chosen:
            a, b = choice.reuse
            if choice.source is None:
                for t in range(a, b + 1):
                    frame[t] = int(choice.tag == FRAME)
                    rows[t] = [(-1, round(max(choice.p, 1e-6), 6))]
                continue
            pairs, more = cls.word_pairs(choice, grid)
            for t, s in pairs:
                links[t] = s
                tags[t] = "COPY" if choice.tag == QUOTE else type_of(t, s)
                rows[t] = [(s, round(choice.p, 6)), (-1, round(1.0 - choice.p, 6))]
            c, d = choice.source
            n_r, n_s = b - a + 1, d - c + 1
            linked_t = {t for t, _ in pairs}
            linked_s = {s for _, s in pairs}
            if n_r == n_s + 1:                              # two reuse words render one source word: SPLIT
                lone = next((t for t in range(a, b + 1) if t not in linked_t), None)
                if lone is not None:
                    neighbour = lone - 1 if lone - 1 >= a and links[lone - 1] >= 0 else lone + 1 if lone + 1 <= b else None
                    if neighbour is not None and links[neighbour] >= 0 and cls._enclitic_split(source_tokens[links[neighbour]]):
                        links[lone] = links[neighbour]; tags[lone] = "SPLIT"; tags[neighbour] = "SPLIT"
                        rows[lone] = [(links[lone], round(choice.p, 6)), (-1, round(1.0 - choice.p, 6))]
            elif n_s == n_r + 1:                            # one reuse word renders two source words: MERGE
                lone_s = next((s for s in range(c, d + 1) if s not in linked_s), None)
                if lone_s is not None:
                    t_near = next((t for t in range(a, b + 1) if links[t] in (lone_s - 1, lone_s + 1)), None)
                    if t_near is not None and cls._enclitic_split(reuse_tokens[t_near]):
                        tags[t_near] = "MERGE"
                        extra.append(Edge(t_near, lone_s, "MERGE"))
        return links, tags, frame, extra, rows


# =============================================================================
# The method row
# =============================================================================


@BaselineRegistry.register
class SpanPairScorer(Baseline):
    """"Seq2Edits (span-pair scorer)": runs first, words inside.

    ``cfg.extra`` (``SPAN_DEFAULTS``): ``L_max``, ``eps``, ``K_pairs``, ``swap``,
    ``gold_passes``, ``train_on``, ``theta`` (a fixed seed for the driver's
    dial), ``word_decoder`` (1 = the shared decoder on the word rows, the
    ablation), ``hidden``, ``lr_heads``, ``temperature``.

    Example:
        ```python
        method = SpanPairScorer(cfg).fit(train, dev, log=print)
        preds = [method.postprocess(r, p, {"theta": 0.3}) for r, p in zip(test, method.predict(test))]
        # python run_baseline.py --method span_pair --fold 4 --extra gold_passes=6
        ```
    """

    name = "span_pair"
    early_stopping_capable = True
    emits = "scores"
    trainable = True
    typer = "own"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {**SPAN_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in SPAN_DEFAULTS}}
        if cfg.train_on and "train_on" not in cfg.extra:
            self.dials["train_on"] = cfg.train_on
        self.L_max = int(self.dials["L_max"])
        self.substrate = Substrate(cfg.base_model, device=cfg.device, max_length=cfg.max_length,
                                   temperature=float(self.dials["temperature"]))
        #: the frozen untyped pointer whose word grid prunes the pairs (note 15's model, ``load_base``);
        #: without one the substrate's cosine grid prunes, which is much weaker
        self.pruner: Optional[TypedPointerBaseline] = None
        self._pruning_rows: Dict[str, Any] = {}
        self.head: Optional[SpanPairHead] = None
        self._options: Dict[str, Dict[str, Any]] = {}
        #: candidate recall of the gold pairs, counted during the first training pass
        self.diagnostics: Optional[Dict[str, int]] = None

    # ---------- data ----------

    def training_records(self, records: Sequence[Record]) -> List[Record]:
        """Both orientations when ``swap`` says so; possible links dropped on ``train_on = sure``."""
        from dataclasses import replace

        out = []
        for record in records:
            if str(self.dials["train_on"]) == "sure":
                record = replace(record, links=[e for e in record.links if e.sure])
            out.append(record)
            if str(self.dials["swap"]) == "double":
                out.append(self.swapped(record))
        return out

    @staticmethod
    def swapped(record: Record) -> Record:
        """The record with the sides exchanged and its links inverted (``RecordCodec.swapped``)."""
        from retexo.baselines.record import RecordCodec

        return RecordCodec.swapped(record)

    # ---------- the pruning grid ----------

    def load_pruner(self, log=None) -> None:
        """The frozen untyped pointer of ``load_base``, built from its saved recipe."""
        import json

        path = Path(str(self.dials["load_base"]))
        if self.pruner is not None or not str(self.dials["load_base"]):
            return
        from dataclasses import replace

        saved = json.loads((path / "recipe.json").read_text())
        cfg = replace(self.cfg, base_model=saved.get("base_model", self.cfg.base_model), extra=dict(saved["recipe"]))
        self.pruner = TypedPointerBaseline.load(path, cfg)
        if log:
            log(f"[span_pair] pruning grid from the frozen pointer at {path}")

    def prune_ahead(self, records: Sequence[Record]) -> None:
        """The frozen pointer's rows for many records at once (batched inside it)."""
        todo = [r for r in records if r.id not in self._pruning_rows]
        if self.pruner is None or not todo:
            return
        for record, pred in zip(todo, self.pruner.predict(todo)):
            self._pruning_rows[record.id] = pred.scores

    def pruning_grid(self, record: Record, h_r, h_s):
        """``[n_t, n_s]`` probabilities: the frozen pointer's rows when there is
        one, else the substrate's row-softmaxed cosine."""
        import numpy as np

        n_t, n_s = h_r.shape[0], h_s.shape[0]
        if self.pruner is None:
            return self.substrate.grid(h_r, h_s).numpy()
        if record.id not in self._pruning_rows:
            self._pruning_rows[record.id] = self.pruner.predict([record])[0].scores
        rows = self._pruning_rows[record.id] or []
        grid = np.zeros((n_t, n_s), dtype=np.float32)
        for t, row in enumerate(rows[:n_t]):
            for s_, p in row:
                if 0 <= s_ < n_s:
                    grid[t, s_] = float(p)
        return grid

    # ---------- scoring one record ----------

    def options_for(self, record: Record, h_r, h_s, *, gold_pairs: Sequence[Tuple[Span, Span]] = ()):
        """Every candidate option of the record with its logits, grouped by reuse span.
        Returns ``(spans, per_span)`` where ``per_span[span] = (entries, logits)``:
        entries ``(source span or None, tag)`` in the logits' order."""
        import torch

        n_t, n_s = h_r.shape[0], h_s.shape[0]
        grid = self.pruning_grid(record, h_r, h_s)
        pairs = SpanEnumerator.candidate_pairs(grid, record.reuse_tokens[:n_t], record.source_tokens[:n_s],
                                               L_max=self.L_max, eps=float(self.dials["eps"]),
                                               K_pairs=int(self.dials["K_pairs"]), gold_pairs=gold_pairs)
        reuse_spans = SpanEnumerator.enumerate_spans(n_t, self.L_max)
        source_spans = sorted({s for _, s in pairs})
        r_index = {sp: i for i, sp in enumerate(reuse_spans)}
        s_index = {sp: i for i, sp in enumerate(source_spans)}
        vec_r = self.head.span_vectors(h_r, reuse_spans)
        vec_s = self.head.span_vectors(h_s, source_spans) if source_spans else None
        null_logits = self.head.score_nulls(vec_r)                                   # [n_spans, 2]
        pair_logits = None
        if pairs:
            ri = torch.tensor([r_index[r] for r, _ in pairs], device=vec_r.device)
            si = torch.tensor([s_index[s] for _, s in pairs], device=vec_r.device)
            dlen = torch.tensor([(r[1] - r[0]) - (s[1] - s[0]) for r, s in pairs], device=vec_r.device)
            pos = torch.tensor([((r[0] + r[1]) / 2 / max(n_t, 1)) - ((s[0] + s[1]) / 2 / max(n_s, 1)) for r, s in pairs],
                               device=vec_r.device, dtype=vec_r.dtype)
            pair_logits = self.head.score_pairs(vec_r[ri], vec_s[si], dlen, pos)    # [n_pairs, 2]
        by_span: Dict[Span, List[Tuple[int, Span]]] = {}
        for k, (r, s) in enumerate(pairs):
            by_span.setdefault(r, []).append((k, s))
        per_span: Dict[Span, Tuple[List[Tuple[Optional[Span], str]], Any]] = {}
        for span in reuse_spans:
            entries: List[Tuple[Optional[Span], str]] = [(None, INS), (None, FRAME), (None, NONE)]
            parts = [null_logits[r_index[span]]]
            for k, s in by_span.get(span, []):
                entries.extend([(s, QUOTE), (s, ADAPT)])
                parts.append(pair_logits[k])
            per_span[span] = (entries, torch.cat(parts))
        return grid, per_span

    # ---------- training ----------

    def validation_loss(self, records: Sequence[Record]) -> Optional[float]:
        """The run loss on ``records`` per term, as trained (no gradient; the diagnostics are left alone)."""
        import torch

        if self.head is None:
            return None
        saved, self.diagnostics = self.diagnostics, None
        total, count = 0.0, 0
        batch = max(1, self.cfg.batch_size)
        try:
            with torch.no_grad():
                for start in range(0, len(records), batch):
                    chunk = list(records[start:start + batch])
                    for record, (h_r, h_s) in zip(chunk, self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])):
                        loss, n = self.loss(record, h_r, h_s)
                        if loss is not None:
                            total += float(loss); count += n
        finally:
            self.diagnostics = saved
        return total / count if count else None

    def loss(self, record: Record, h_r, h_s) -> Tuple[Any, int]:
        """One cross-entropy per gold run over its options, plus the word-type
        cross-entropy inside ADAPT runs; ``(loss, n_terms)``."""
        import torch

        runs = RunReader.chunked(RunReader.gold_runs(record), self.L_max)
        n_t, n_s = h_r.shape[0], h_s.shape[0]
        runs = [r for r in runs if r.end < n_t and (r.source is None or r.source[1] < n_s)]
        gold_pairs = [(r.span, r.source) for r in runs
                      if r.source is not None and r.length <= self.L_max and r.source[1] - r.source[0] + 1 <= self.L_max + 2]
        _, per_span = self.options_for(record, h_r, h_s, gold_pairs=gold_pairs)
        # the candidate recall of the gold pairs, before the gold was appended (the note's diagnostic)
        if self.diagnostics is not None:
            grid = self.pruning_grid(record, h_r, h_s)
            found = set(SpanEnumerator.candidate_pairs(grid, record.reuse_tokens[:n_t], record.source_tokens[:n_s],
                                                       L_max=self.L_max, eps=float(self.dials["eps"]),
                                                       K_pairs=int(self.dials["K_pairs"])))
            self.diagnostics["gold_pairs"] += len(gold_pairs)
            self.diagnostics["gold_pairs_in_candidates"] += sum(1 for g in gold_pairs if g in found)
            self.diagnostics["runs_too_long"] += sum(1 for r in runs if r.source is not None and (r.span, r.source) not in gold_pairs)   # source stretch beyond L_max + 2
        losses = []
        gold_spans = set()
        for run in runs:
            if run.length > self.L_max:
                continue
            entries, logits = per_span[run.span]
            target = (run.source, run.tag) if run.source is not None else (None, run.tag)
            if target not in entries:
                continue
            gold_spans.add(run.span)
            index = torch.tensor(entries.index(target), device=logits.device)
            losses.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0), index.unsqueeze(0)))
        # every other reuse span is NONE: without this the segmentation would compare the
        # trained scores of gold-shaped spans against untrained ones (the first MTRef run, F1 0.66)
        none_terms = []
        for span, (entries, logits) in per_span.items():
            if span in gold_spans:
                continue
            index = torch.tensor(entries.index((None, NONE)), device=logits.device)
            none_terms.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0), index.unsqueeze(0)))
        if none_terms:
            losses.append(torch.stack(none_terms).mean() * max(1, len(gold_spans)))
        links, tags, _, _ = RecordInterface.links_of(record)
        t_idx, s_idx, y = [], [], []
        for run in runs:
            if run.tag != ADAPT:
                continue
            for t in range(run.start, run.end + 1):
                s = links[t]
                if s >= 0 and tags[t] in WORD_TYPES:
                    t_idx.append(t); s_idx.append(s); y.append(WORD_TYPES.index(tags[t]))
        if t_idx:
            logits = self.head.word_types(h_r[t_idx], h_s[s_idx])
            losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor(y, device=logits.device)))
        if not losses:
            return None, 0
        return torch.stack(losses).sum(), len(losses)

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
           ) -> "SpanPairScorer":
        import torch
        from transformers import get_linear_schedule_with_warmup

        records = self.training_records(train)
        if not records:
            return self
        # ours' schedule when the run reads the shared training set (the schedule builds both orientations)
        from retexo.baselines.schedule import SharedSchedule

        schedule = SharedSchedule(self, train) if SharedSchedule.applies(self) else None
        synthetic = schedule.synthetic_epoch() if schedule is not None else []
        if schedule is not None:
            records = schedule.real_pass()
            if log:
                log(f"[span_pair] shared schedule {schedule.summary()}")
        self.load_pruner(log=log)
        self.prune_ahead((schedule.real + schedule.negatives + schedule.synthetic) if schedule is not None else records)
        self.head = SpanPairHead(self.substrate.hidden_size, L_max=self.L_max, hidden=int(self.dials["hidden"]),
                                 device=self.cfg.device)
        optimizer = torch.optim.AdamW([{"params": list(self.substrate.parameters()), "lr": self.cfg.learning_rate},
                                       {"params": self.head.parameters(), "lr": float(self.dials["lr_heads"])}])
        passes = max(1, int(self.dials["gold_passes"]))
        if self.cfg.smoke:
            passes = min(passes, 2)
        from retexo.baselines.early_stopping import EarlyStopping

        stopper = EarlyStopping.for_method(self, log=log)
        if stopper is not None:
            passes = stopper.max_epochs
        batch_size = max(1, self.cfg.batch_size)
        steps = (passes * max(1, -(-len(records) // batch_size))) + -(-len(synthetic) // batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(float(self.dials["warmup"]) * steps)), steps)
        rng = random.Random(self.cfg.seed)

        def run_pass(items: List[Record], label: str, on_batch=None) -> None:
            items = list(items)
            rng.shuffle(items)
            total, n_batches = 0.0, 0
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                encoded = self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])
                terms, count = [], 0
                for record, (h_r, h_s) in zip(chunk, encoded):
                    loss, n = self.loss(record, h_r, h_s)
                    if loss is not None:
                        terms.append(loss); count += n
                if not terms:
                    continue
                loss = torch.stack(terms).sum() / max(count, 1)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
                total += float(loss.detach()); n_batches += 1
                if on_batch is not None:
                    on_batch(start + len(chunk), len(items))
            if log:
                log(f"[span_pair] pass {label}: loss {total / max(n_batches, 1):.4f} ({len(items)} records)")

        self.substrate.train(); self.head.train()
        if synthetic:
            self.diagnostics = None
            monitor = stopper.monitor if stopper is not None else None
            run_pass(synthetic, "synthetic", monitor.progress("synthetic", int(self.cfg.extra.get("synthetic_evals", 4))) if monitor else None)
            if monitor is not None:
                monitor.evaluate("synthetic", fraction=1.0)
                self.substrate.train(); self.head.train()
        for pass_no in range(1, passes + 1):
            self.diagnostics = {"gold_pairs": 0, "gold_pairs_in_candidates": 0, "runs_too_long": 0} if pass_no == 1 else None
            if schedule is not None and pass_no > 1:
                records = schedule.real_pass()                      # a fresh synthetic draw every real pass
            run_pass(records, f"{pass_no}/{passes}")
            if pass_no == 1 and log and self.diagnostics and self.diagnostics["gold_pairs"]:
                d = self.diagnostics
                log(f"[span_pair] candidate recall of gold pairs {d['gold_pairs_in_candidates'] / d['gold_pairs']:.3f} "
                    f"({d['gold_pairs']} pairs; {d['runs_too_long']} gold runs beyond L_max left out)")
            self.last_diagnostics = self.diagnostics or getattr(self, "last_diagnostics", None)
            if stopper is not None:
                self.diagnostics = None
                keep_going = stopper.step(pass_no, self.modules())
                self.substrate.train(); self.head.train()
                if not keep_going:
                    break
        if stopper is not None:
            stopper.restore(self.modules())
            stopper.release()
        self.diagnostics = None
        self.substrate.eval(); self.head.eval()
        return self

    def modules(self) -> Dict[str, Any]:
        """The substrate encoder and the span-pair head."""
        return {"substrate": self.substrate, "head": self.head}

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """A fixed ``theta`` seeds the driver's dial; the driver's own dev tuning
        then runs on the word rows (pass ``--no-tune`` to keep the seed)."""
        return {"theta": float(self.dials["theta"])}

    # ---------- inference ----------

    def _score(self, records: Sequence[Record]) -> None:
        import torch

        todo = [r for r in records if r.id not in self._options and r.source_tokens and r.reuse_tokens]
        if not todo or self.head is None:
            return
        self.prune_ahead(todo)
        self.substrate.eval(); self.head.eval()
        with torch.no_grad():
            for start in range(0, len(todo), max(1, self.cfg.batch_size)):
                chunk = todo[start:start + max(1, self.cfg.batch_size)]
                for record, (h_r, h_s) in zip(chunk, self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])):
                    grid, per_span = self.options_for(record, h_r, h_s)
                    grid = grid if not hasattr(grid, "numpy") else grid.numpy()
                    options = []
                    for span, (entries, logits) in per_span.items():
                        log_p = torch.log_softmax(logits, dim=-1).cpu().tolist()
                        for (source, tag), lp in zip(entries, log_p):
                            options.append(SpanChoice(span, source, tag, float(lp), float(torch.exp(torch.tensor(lp)))))
                    self._options[record.id] = {"options": options, "grid": grid, "h_r": h_r.cpu(), "h_s": h_s.cpu(),
                                                "n_t": h_r.shape[0], "n_s": h_s.shape[0]}

    def word_rows(self, record: Record, options: Sequence[SpanChoice]) -> Rows:
        """Word-level marginals for the dump and the ``word_decoder`` ablation:
        each reuse word's mass over source words, summed over the spans that
        cover it, position-wise inside each pair."""
        n = record.n_reuse
        mass: List[Dict[int, float]] = [dict() for _ in range(n)]
        for option in options:
            a, b = option.reuse
            if option.tag == NONE:
                continue
            if option.source is None:
                for t in range(a, b + 1):
                    mass[t][-1] = mass[t].get(-1, 0.0) + option.p
                continue
            c, d = option.source
            for k, t in enumerate(range(a, b + 1)):
                s = c + min(k, d - c)
                mass[t][s] = mass[t].get(s, 0.0) + option.p
        rows: Rows = []
        for t in range(n):
            total = sum(mass[t].values()) or 1.0
            row = sorted(((s, p / total) for s, p in mass[t].items()), key=lambda x: -x[1])
            if not any(s == -1 for s, _ in row):
                row.append((-1, 0.0))
            rows.append([(int(s), round(float(p), 6)) for s, p in row])
        return rows

    def candidate_recall(self, records: List[Record]) -> Dict[str, Any]:
        """The share of the gold span pairs of ``records`` that the pruning keeps as candidates (the ceiling the
        pruning puts on the scorer), counted as in training but on any records with links (the test fold)."""
        import torch

        found_n = gold_n = 0
        batch = max(1, self.cfg.batch_size)
        with torch.no_grad():
            for start in range(0, len(records), batch):
                chunk = [r for r in records[start:start + batch] if r.links]
                if not chunk:
                    continue
                for record, (h_r, h_s) in zip(chunk, self.substrate.encode([(r.source_tokens, r.reuse_tokens) for r in chunk])):
                    runs = RunReader.chunked(RunReader.gold_runs(record), self.L_max)
                    n_t, n_s = h_r.shape[0], h_s.shape[0]
                    gold = [(r.span, r.source) for r in runs if r.end < n_t and r.source is not None and r.source[1] < n_s
                            and r.length <= self.L_max and r.source[1] - r.source[0] + 1 <= self.L_max + 2]
                    grid = self.pruning_grid(record, h_r, h_s)
                    found = set(SpanEnumerator.candidate_pairs(grid, record.reuse_tokens[:n_t], record.source_tokens[:n_s],
                                                               L_max=self.L_max, eps=float(self.dials["eps"]),
                                                               K_pairs=int(self.dials["K_pairs"])))
                    gold_n += len(gold)
                    found_n += sum(1 for g in gold if g in found)
        return {"gold_pairs": gold_n, "gold_pairs_in_candidates": found_n,
                "recall": round(found_n / gold_n, 5) if gold_n else None}

    def predict(self, records: List[Record]) -> List[Prediction]:
        self._score(records)
        out = []
        for record in records:
            pred = Prediction.empty(record.n_reuse)
            stored = self._options.get(record.id)
            if stored is None:
                out.append(pred)
                continue
            pred.scores = self.word_rows(record, stored["options"])
            pred.meta["n_options"] = len(stored["options"])
            out.append(pred)
        return out

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """The segmentation, then the expansion; or the shared decoder on the
        word rows when ``word_decoder`` is on."""
        import torch

        stored = self._options.get(record.id)
        if stored is None or pred.scores is None:
            return pred
        theta = float(dials.get("theta", 0.0))
        if int(self.dials["word_decoder"]):
            from retexo.baselines.adapters import PredictionAdapter

            pred = PredictionAdapter.decode_prediction(pred, self.decoder, dials, record)
            pred.tags = [("COPY" if normalize(record.reuse_tokens[t]) == normalize(record.source_tokens[s]) else
                          self._type_of(stored, t, s)) if s >= 0 else "" for t, s in enumerate(pred.links)]
            return pred
        chosen = Segmenter.segment(stored["options"], stored["n_t"], theta=theta)
        with torch.no_grad():
            links, tags, frame, extra, rows = Expander.expand(
                chosen, stored["grid"], record.reuse_tokens[:stored["n_t"]], record.source_tokens[:stored["n_s"]],
                lambda t, s: self._type_of(stored, t, s))
        n = record.n_reuse
        pred.links = links + [-1] * (n - len(links))
        pred.tags = [labels.canonical(tag)[0] or "" if s >= 0 else "" for tag, s in zip(tags + [""] * (n - len(tags)), pred.links)]
        pred.frame = frame + [0] * (n - len(frame))
        pred.extra = extra
        pred.link_p = [float(dict(row).get(s, 0.0)) if s >= 0 else 0.0 for row, s in zip(rows, pred.links)] + [0.0] * (n - len(rows))
        pred.meta["runs"] = [(c.reuse, c.source, c.tag, round(c.p, 4)) for c in chosen]
        return pred

    def _type_of(self, stored: Dict[str, Any], t: int, s: int) -> str:
        import torch

        with torch.no_grad():
            logits = self.head.word_types(stored["h_r"][t:t + 1].to(self.cfg.device), stored["h_s"][s:s + 1].to(self.cfg.device))
        return WORD_TYPES[int(logits.argmax(dim=-1))]

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.head is None:
            return
        torch.save({"model_name": self.substrate.model_name, "encoder": self.substrate.state_dict(),
                    "head": self.head.state_dict(), "dials": self.dials}, path / "span_pair.pt")

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> "SpanPairScorer":
        import torch

        method = cls(cfg)
        state = torch.load(Path(path) / "span_pair.pt", map_location=cfg.device)
        method.load_pruner()
        method.substrate.load_state_dict(state["encoder"])
        method.head = SpanPairHead(method.substrate.hidden_size, L_max=method.L_max, hidden=int(method.dials["hidden"]),
                                   device=cfg.device)
        method.head.load_state_dict(state["head"])
        method.substrate.eval(); method.head.eval()
        return method
