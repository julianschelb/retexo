# retexo/formulations/pair_encoding.py
"""
Encoding a (source, reuse) pair for the tagging formulation.

The tagging arm needs more than token ids: it must know which token positions
belong to which *word* of the reuse, so that one operation label and one source
pointer can be attached per word. HuggingFace fast tokenizers supply that
through ``word_ids``; Latin BERT does not ship a usable tokenizer at all, so
this module puts both behind one interface.

Latin BERT matters enough to justify the extra path. It was the strongest
classifier on the underlying benchmark, but its vocabulary is a
``tensor2tensor`` subword encoder that ``AutoTokenizer`` silently reads as
WordPiece, mapping roughly five of every seven Latin words to ``[UNK]`` with no
error raised. Using it through the default tokenizer would look like a working
model trained on destroyed input.
"""

from __future__ import annotations

import re

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

#: Model names whose vocabulary needs the tensor2tensor encoder.
LATIN_BERT_MARKERS = ("latin-bert", "latin_bert", "bamman")


def is_latin_bert(model_name: str) -> bool:
    """Whether ``model_name`` (a hub id or a local directory) is a Latin BERT checkpoint.

    A hub id says so in its name; a local directory (a continued-pretraining
    checkpoint saved by ``Stage0Trainer``) says so by carrying Latin BERT's
    ``vocab.txt`` beside its weights, whatever it is called.
    """
    from pathlib import Path

    lowered = model_name.lower()
    if any(marker in lowered for marker in LATIN_BERT_MARKERS):
        return True
    local = Path(model_name)
    return local.is_dir() and (local / "vocab.txt").exists() and (local / "latin_bert.marker").exists()


#: The reference pre-tokenisation of Latin BERT (``locisimiles.tokenization.latin_bert``, after Bamman's
#: ``convert_to_toks``): lowercase, alphabetic runs only -- punctuation never reaches the encoder, and u/v, i/j
#: are left as written.
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def latin_bert_pieces(encoder, word: str, unk: int = 1) -> List[int]:
    """The subword ids of one record token under the reference pre-tokenisation: lowercased, its
    alphabetic runs encoded (``cano,`` is ``cano``; ``ita-que`` is ``ita`` + ``que``); a token without
    letters (``--``) is ``[UNK]`` so that its word span still exists. Every Latin BERT consumer of the
    harness goes through this one function."""
    pieces: List[int] = []
    for run in _LATIN_WORD_RE.findall(word.lower()):
        try:
            pieces.extend(encoder.encode_word(run))
        except ValueError:                           # unreachable for letters, kept for safety
            pieces.append(unk)
    return pieces or [unk]


def latin_bert_vocab(model_name: str) -> str:
    """The path of Latin BERT's ``vocab.txt``: from the hub for a hub id, from the
    directory for a saved checkpoint."""
    from pathlib import Path

    local = Path(model_name) / "vocab.txt"
    if local.exists():
        return str(local)
    import huggingface_hub

    return huggingface_hub.hf_hub_download(model_name, "vocab.txt")

# =============================================================================
# Interface
# =============================================================================


class PairEncoder(ABC):
    """Encodes word-tokenized pairs and reports where each reuse word landed.

    ``encode`` returns the reuse-side spans, which is what the tagging
    formulation needs. The source-side spans are recorded on
    ``last_source_spans`` for the one experiment that needs a head there too
    (E1, where ``DEL`` has no reuse position to attach to).
    """

    #: Source-side word spans from the most recent ``encode`` call.
    last_source_spans: List[List[Tuple[int, int]]] = []

    @abstractmethod
    def encode(
        self,
        pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
        max_length: int,
    ) -> Tuple[Dict[str, "torch.Tensor"], List[List[Tuple[int, int]]]]:  # noqa: F821
        """Return model inputs and, per example, one span per reuse word.

        A span is ``(start, end)`` over token positions. Words truncated away
        get no span, so callers must not assume the list matches the reuse
        length.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for the run record."""

    @classmethod
    def build(cls, model_name: str) -> "PairEncoder":
        """Choose the encoder a backbone requires.

        Example:
            ```python
            encoder = PairEncoder.build("ashleygong03/bamman-burns-latin-bert")
            batch, spans = encoder.encode([(source_words, reuse_words)], 256)
            ```
        """
        if is_latin_bert(model_name):
            return LatinBertPairEncoder(model_name)
        return HuggingFacePairEncoder(model_name)


#: Backward-compatible module-level alias.
build_pair_encoder = PairEncoder.build


# =============================================================================
# HuggingFace
# =============================================================================


class HuggingFacePairEncoder(PairEncoder):
    """Standard fast tokenizer, using ``word_ids`` for word alignment."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.model_name = model_name
        try:
            # byte-level BPE tokenizers (RoBERTa: LaBerta, PhilBerta) refuse
            # pre-tokenized input without this, and it is inert elsewhere
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        except (TypeError, ValueError):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def name(self) -> str:
        return self.model_name

    def encode(self, pairs, max_length):
        batch = self.tokenizer(
            [list(a) for a, _ in pairs],
            [list(b) for _, b in pairs],
            is_split_into_words=True, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        spans: List[List[Tuple[int, int]]] = []
        source_spans: List[List[Tuple[int, int]]] = []
        for row in range(len(pairs)):
            words = batch.word_ids(row)
            sequences = batch.sequence_ids(row)
            found: Dict[int, List[int]] = {}
            found_source: Dict[int, List[int]] = {}
            for position, (word, sequence) in enumerate(zip(words, sequences)):
                if word is None:
                    continue
                if sequence == 1:
                    found.setdefault(word, []).append(position)
                elif sequence == 0:
                    found_source.setdefault(word, []).append(position)
            spans.append([(min(v), max(v) + 1) for _, v in sorted(found.items())])
            source_spans.append(
                [(min(v), max(v) + 1) for _, v in sorted(found_source.items())]
            )
        self.last_source_spans = source_spans
        return dict(batch), spans


# =============================================================================
# Latin BERT
# =============================================================================


class LatinBertPairEncoder(PairEncoder):
    """Latin BERT's own subword encoder, with word spans built explicitly.

    Raises:
        RuntimeError: If the encoder cannot be located, rather than falling
            back to a tokenizer that would silently produce ``[UNK]``.
    """

    PAD, UNK, CLS, SEP = 0, 1, 2, 3

    def __init__(self, model_name: str):
        from retexo.resources import Morphology  # noqa: F401  (import cost only)

        self.model_name = model_name
        try:
            from locisimiles.tokenization.latin_bert import SubwordTextEncoder
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Latin BERT needs the tensor2tensor subword encoder from "
                "locisimiles.tokenization; AutoTokenizer maps most Latin words to [UNK]"
            ) from exc
        vocab = latin_bert_vocab(model_name)
        self.encoder = SubwordTextEncoder.from_file(vocab)
        self.subtokens = self.encoder._subtokens
        specials = {s: i for i, s in enumerate(self.subtokens) if s.startswith("[")}
        self.CLS = specials.get("[CLS]", self.CLS)
        self.SEP = specials.get("[SEP]", self.SEP)
        self.PAD = specials.get("[PAD]", self.PAD)
        #: E35: label prefix -- per operation, the gloss as a word list; None = off
        self.label_prefix = None
        self.LBL = len(self.subtokens)          # one new token id, appended to the vocabulary
        self.last_label_positions = None

    @property
    def vocab_size_with_labels(self) -> int:
        return len(self.subtokens) + 1

    @property
    def name(self) -> str:
        return self.model_name

    def _word_ids(self, word: str) -> List[int]:
        return latin_bert_pieces(self.encoder, word, self.UNK)

    def encode(self, pairs, max_length):
        import torch

        rows: List[List[int]] = []
        types: List[List[int]] = []
        spans: List[List[Tuple[int, int]]] = []
        source_rows: List[List[Tuple[int, int]]] = []

        label_rows: List[List[int]] = []
        prefix_len = 0
        if self.label_prefix is not None:
            prefix_ids, label_pos = [], []
            for gloss in self.label_prefix:
                label_pos.append(1 + len(prefix_ids))          # after [CLS]
                prefix_ids.append(self.LBL)
                for word in gloss:
                    prefix_ids.extend(self._word_ids(word))
            prefix_ids.append(self.SEP)
            prefix_len = len(prefix_ids)
            max_length = max_length + prefix_len               # the passages keep their budget
        for source, target in pairs:
            ids = [self.CLS]
            if self.label_prefix is not None:
                ids.extend(prefix_ids)
                label_rows.append(list(label_pos))
            source_row: List[Tuple[int, int]] = []
            for word in source:
                pieces = self._word_ids(word)
                source_row.append((len(ids), len(ids) + len(pieces)))
                ids.extend(pieces)
            ids.append(self.SEP)
            boundary = len(ids)
            source_rows.append([s for s in source_row if s[1] <= max_length])

            row_spans: List[Tuple[int, int]] = []
            for word in target:
                pieces = self._word_ids(word)
                if len(ids) + len(pieces) + 1 > max_length:
                    break
                row_spans.append((len(ids), len(ids) + len(pieces)))
                ids.extend(pieces)
            ids.append(self.SEP)

            rows.append(ids[:max_length])
            types.append(([0] * boundary + [1] * (len(ids) - boundary))[:max_length])
            spans.append([s for s in row_spans if s[1] <= max_length])

        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), self.PAD, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        token_types = torch.zeros((len(rows), width), dtype=torch.long)
        self.last_source_spans = source_rows
        self.last_label_positions = label_rows if self.label_prefix is not None else None
        for i, (row, row_types) in enumerate(zip(rows, types)):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention[i, : len(row)] = 1
            token_types[i, : len(row_types)] = torch.tensor(row_types)

        batch = {"input_ids": input_ids, "attention_mask": attention,
                 "token_type_ids": token_types}
        return batch, spans
