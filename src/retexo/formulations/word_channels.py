# retexo/formulations/word_channels.py
"""Word-level input channels for the pointer's encoder (failure-mode dry run, chain 2, row 6).

The pointer reads the pair as subword pieces and has to recover from them what the lookup
already knows about every word: its lemma, its part of speech, its morphology. In fold 4 the
lookup names the counterpart of 141 reuse words as *the same lemma* and the pointer still
misses 13 % of those links. ``WordChannels`` adds, to every piece of a word, learned
embeddings of the word's lemma (a hashed vocabulary), part of speech (the first character of
the Collatinus tag) and morphology (the whole tag, hashed) -- summed into the piece's word
embedding like BERT's segment embedding, so the attention can match lemmas across the pair
and learn the slot conventions itself. A word the resources cannot analyse gets the padding
row (zero): the surface embedding stays its only input.

    channels = WordChannels(("lemma", "pos"), hidden=768)
    ids = channels.ids(input_ids.shape, reuse_spans, source_spans, pairs)   # [B, W, n_kinds]
    embeds = encoder.embeddings.word_embeddings(input_ids) + channels(ids)
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch

#: The channels in the order their ids are stored: three per surface form (the lookup), three per
#: passage (Stanza's Latin parse, row 7a: the word's dependency relation, its head's lemma, its UPOS).
KINDS = ("lemma", "pos", "morph", "dep", "head", "upos", "shared", "sharedform")
PARSED = ("dep", "head", "upos")
#: Pair-level kinds (row 6 v2): does the word's lemma / surface form occur on the other side of the pair?
#: (1 = no, 2 = yes) -- the identity signal without a vocabulary, so it generalises to unseen lemmas.
CROSS = ("shared", "sharedform")
#: Table sizes without the padding row.
SIZES = {"lemma": 30000, "pos": 32, "morph": 2048, "dep": 64, "head": 30000, "upos": 20, "shared": 2, "sharedform": 2}
_POS_ALPHABET = "abcdefghijklmnopqrstuvwxyz-"
_UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
         "PUNCT", "SCONJ", "SYM", "VERB", "X")


class WordChannels(torch.nn.Module):
    """Learned lemma / POS / morphology embeddings per word, summed onto its pieces.

    Example:
        ```python
        channels = WordChannels(("lemma",), hidden=8)
        channels.word_ids("armis")            # (id of lemma "arma", 0, 0)
        ```
    """

    def __init__(self, kinds: Sequence[str], hidden: int, init_std: float = 0.02):
        super().__init__()
        self.kinds = tuple(k for k in KINDS if k in kinds)
        if not self.kinds:
            raise ValueError(f"no known channel in {kinds!r}; expected some of {KINDS}")
        self.tables = torch.nn.ModuleDict(
            {k: torch.nn.Embedding(SIZES[k] + 1, hidden, padding_idx=0) for k in self.kinds})
        for table in self.tables.values():
            torch.nn.init.normal_(table.weight, std=init_std)
            with torch.no_grad():
                table.weight[0].zero_()
        self._morphology = None
        self._cache: Dict[str, Tuple[int, ...]] = {}
        self._parsed_cache: Dict[Tuple[str, ...], List[Tuple[int, ...]]] = {}

    # ---------- the resources ----------

    @property
    def morphology(self):
        if self._morphology is None:
            from retexo.resources.morphology import Morphology

            self._morphology = Morphology()
        return self._morphology

    @staticmethod
    def _bucket(text: str, size: int) -> int:
        return zlib.crc32(text.encode("utf-8")) % size + 1

    @property
    def parsed_kinds(self) -> Tuple[str, ...]:
        return tuple(k for k in self.kinds if k in PARSED)

    def passage_ids(self, tokens: Sequence[str]) -> List[Tuple[int, ...]]:
        """Per token the ids of the parsed kinds (dep, head, upos), from the shared Stanza parser."""
        key = tuple(tokens)
        hit = self._parsed_cache.get(key)
        if hit is not None:
            return hit
        from retexo.datasets.synthetic import _dep_parser

        parse = _dep_parser().parse_all([list(tokens)])[" ".join(tokens)]
        out = []
        for i, word in enumerate(tokens):
            upos, rel, head = parse[i] if i < len(parse) else ("X", "dep", -1)
            ids = []
            for kind in self.parsed_kinds:
                if kind == "dep":
                    ids.append(self._bucket(rel, SIZES["dep"]))
                elif kind == "upos":
                    ids.append(_UPOS.index(upos) + 1 if upos in _UPOS else 0)
                else:
                    lemma = ""
                    if 0 <= head < len(tokens):
                        try:
                            lemma = self.morphology.lemma(tokens[head])
                        except Exception:
                            lemma = ""
                    ids.append(self._bucket(lemma, SIZES["head"]) if lemma else 0)
            out.append(tuple(ids))
        self._parsed_cache[key] = out
        return out

    def word_ids(self, word: str) -> Tuple[int, ...]:
        """One id per kind (0 = unknown), cached per surface form."""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        letters = re.sub(r"[^A-Za-z]", "", word)
        ids = []
        form_kinds = [k for k in self.kinds if k not in PARSED and k not in CROSS]
        if not letters:
            ids = [0] * len(form_kinds)
        else:
            label: Optional[str] = None
            if "pos" in self.kinds or "morph" in self.kinds:
                try:
                    label = self.morphology.morpho_label(word)
                except Exception:
                    label = None
            for kind in form_kinds:
                if kind == "lemma":
                    try:
                        lemma = self.morphology.lemma(word)
                    except Exception:
                        lemma = ""
                    ids.append(self._bucket(lemma, SIZES["lemma"]) if lemma else 0)
                elif kind == "pos":
                    head = (label or "")[:1]
                    ids.append(_POS_ALPHABET.index(head) + 1 if head in _POS_ALPHABET else 0)
                else:
                    ids.append(self._bucket(label, SIZES["morph"]) if label else 0)
        out = tuple(ids)
        self._cache[word] = out
        return out

    def _lemma_of(self, word: str) -> str:
        try:
            return self.morphology.lemma(word) if re.search(r"[A-Za-z]", word) else ""
        except Exception:
            return ""

    def cross_ids(self, words: Sequence[str], others: Sequence[str]) -> List[Tuple[int, ...]]:
        """Per word of one side, for the pair-level kinds: 2 where its lemma / surface form occurs on
        the other side, 1 where not, 0 where the word has no letters."""
        kinds = [k for k in self.kinds if k in CROSS]
        other_lemmas = {self._lemma_of(w) for w in others} - {""} if "shared" in kinds else set()
        other_forms = {re.sub(r"[^a-z]", "", w.lower()) for w in others} - {""} if "sharedform" in kinds else set()
        out = []
        for word in words:
            letters = re.sub(r"[^a-z]", "", word.lower())
            ids = []
            for kind in kinds:
                if not letters:
                    ids.append(0)
                elif kind == "shared":
                    lemma = self._lemma_of(word)
                    ids.append(2 if lemma and lemma in other_lemmas else 1)
                else:
                    ids.append(2 if letters in other_forms else 1)
            out.append(tuple(ids))
        return out

    # ---------- the batch ----------

    def ids(self, shape: Tuple[int, int], reuse_spans: Sequence[Sequence[Tuple[int, int]]],
            source_spans: Sequence[Sequence[Tuple[int, int]]],
            pairs: Sequence[Tuple[Sequence[str], Sequence[str]]]) -> torch.Tensor:
        """``[B, W, n_kinds]`` channel ids: every piece of a word carries the word's ids, the
        rest (specials, padding, truncated words) the padding row."""
        batch, width = shape
        out = torch.zeros((batch, width, len(self.kinds)), dtype=torch.long)
        form_cols = [k for k, kind in enumerate(self.kinds) if kind not in PARSED and kind not in CROSS]
        parsed_cols = [k for k, kind in enumerate(self.kinds) if kind in PARSED]
        cross_cols = [k for k, kind in enumerate(self.kinds) if kind in CROSS]
        for row, (source, target) in enumerate(pairs):
            for words, others, spans in ((source, target, source_spans[row] if row < len(source_spans) else ()),
                                         (target, source, reuse_spans[row] if row < len(reuse_spans) else ())):
                parsed = self.passage_ids(words) if parsed_cols else None
                cross = self.cross_ids(words, others) if cross_cols else None
                for w, (word, (start, end)) in enumerate(zip(words, spans)):
                    if end > width:
                        break
                    if form_cols:
                        out[row, start:end, form_cols] = torch.tensor(self.word_ids(word), dtype=torch.long)
                    if parsed is not None and w < len(parsed):
                        out[row, start:end, parsed_cols] = torch.tensor(parsed[w], dtype=torch.long)
                    if cross is not None and w < len(cross):
                        out[row, start:end, cross_cols] = torch.tensor(cross[w], dtype=torch.long)
        return out

    def forward(self, channel_ids: torch.Tensor) -> torch.Tensor:
        total = None
        for k, kind in enumerate(self.kinds):
            part = self.tables[kind](channel_ids[..., k])
            total = part if total is None else total + part
        return total
