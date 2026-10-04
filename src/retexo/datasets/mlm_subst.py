"""E38: contextual substitutions from a Latin masked LM (LaBerta).

The generator's residual SUBST source draws a random same-POS word: a substitution
that fits nowhere. Real allusive substitutions fit the slot (*caelo* -> *polo*,
*foribus* -> *portis*, *generos* -> *natos*): LaBerta's MLM proposes exactly those
when the word is masked in its fragment. One proposer per worker process, CPU,
loaded on first use.
"""
from __future__ import annotations

from retexo.formulations.pair_encoding import latin_bert_pieces

import re
from typing import Optional, Sequence

from retexo.core.normalize import normalize

_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


LATIN_BERT = "ashleygong03/bamman-burns-latin-bert"


class ContextualSubstituter:
    """``model_name`` is a HuggingFace masked LM. Latin BERT (the pointer's own encoder)
    needs the tensor2tensor subword encoder from locisimiles (its AutoTokenizer maps
    Latin to [UNK]); everything else goes through AutoTokenizer."""

    def __init__(self, model_name: str = "bowphs/LaBerta", top_k: int = 12, min_prob: float = 0.005):
        self.model_name, self.top_k, self.min_prob = model_name, top_k, min_prob
        self._tok = self._model = None
        self._enc = None                      # Latin BERT subword encoder
        self.cache = None                     # {(seed text, index): [(cand, p), ...]}
        self.calls = self.hits = 0

    def _load(self):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        torch.set_num_threads(1)
        if self.model_name == LATIN_BERT:
            import huggingface_hub
            from locisimiles.tokenization.latin_bert import SubwordTextEncoder
            self._enc = SubwordTextEncoder.from_file(huggingface_hub.hf_hub_download(self.model_name, "vocab.txt"))
            self._sub = self._enc._subtokens
            specials = {t: i for i, t in enumerate(self._sub) if t.startswith("[")}
            self._ids = (specials["[CLS]"], specials["[SEP]"], specials["[MASK]"])
        else:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).eval()

    def _top(self, tokens, index):
        """(candidate strings, probabilities) at the masked slot."""
        import torch
        if self._enc is not None:
            cls, sep, mask = self._ids
            ids, pos = [cls], None
            for i, w in enumerate(tokens):
                if i == index:
                    pos = len(ids); ids.append(mask)
                else:
                    ids.extend(latin_bert_pieces(self._enc, w))
            ids.append(sep)
            with torch.no_grad():
                logits = self._model(input_ids=torch.tensor([ids[:200]])).logits[0, pos]
            probs = torch.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, self.top_k)
            out = []
            for p, i in zip(top.values.tolist(), top.indices.tolist()):
                piece = self._sub[i]
                if not piece.endswith("_"):          # a word-initial piece of a longer word: skip
                    continue
                out.append((piece[:-1], p))
            return out
        masked = list(tokens); masked[index] = self._tok.mask_token
        enc = self._tok(" ".join(masked), return_tensors="pt", truncation=True, max_length=128)
        pos = (enc["input_ids"][0] == self._tok.mask_token_id).nonzero()
        if len(pos) == 0:
            return []
        with torch.no_grad():
            logits = self._model(**enc).logits[0, int(pos[0, 0])]
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, self.top_k)
        return [(self._tok.decode([i]).strip(), p) for p, i in zip(top.values.tolist(), top.indices.tolist())]

    def propose(self, tokens: Sequence[str], index: int, rng, *, attested=None, resources=None) -> Optional[str]:
        """A word for ``tokens[index]`` that the MLM finds likely in that slot and
        that is a different lemma; None when nothing qualifies. Proposals come from
        the precomputed cache when the seed sentence is in it (GPU pass before the
        workers fork), else from the model on this CPU."""
        self.calls += 1
        lead, word, trail = _PUNCT.match(tokens[index]).groups()
        if len(word) < 3:
            return None
        key = (" ".join(tokens), index)
        if self.cache is not None and key in self.cache:
            top = self.cache[key]
        else:
            if self._model is None:
                self._load()
            top = self._top(tokens, index)
        lemma = resources.lemma_or_surface(word) if resources is not None else normalize(word)
        cands, weights = [], []
        for cand, p in top:
            if p < self.min_prob:
                break
            if not cand.isalpha() or len(cand) < 3:
                continue
            if normalize(cand) == normalize(word):
                continue
            if attested is not None and normalize(cand) not in attested:
                continue
            if resources is not None and resources.lemma_or_surface(cand) == lemma:
                continue
            cands.append(cand); weights.append(p)
        if not cands:
            return None
        self.hits += 1
        pick = rng.choices(cands, weights=weights, k=1)[0]
        if word[:1].isupper():
            pick = pick[:1].upper() + pick[1:]
        return lead + pick + trail

    @classmethod
    def precompute(cls, seeds, model_name: str = "bowphs/LaBerta", *, device: str = "cuda", top_k: int = 12,
                   batch_size: int = 96, log=None):
        """Top-k masked-LM proposals for every word of every seed sentence, on the GPU in
        batches, keyed by (seed text, index). Built once in the parent before the generator
        workers fork, so they inherit it read-only."""
        import os, time
        import torch
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")      # the workers fork after this pass
        t0 = time.time()
        sub = cls(model_name, top_k=top_k)
        sub._load(); model = sub._model.to(device)
        queries = []
        for toks in seeds:
            text = " ".join(toks)
            for i, w in enumerate(toks):
                if len(_PUNCT.match(w).group(2)) >= 3:
                    queries.append((text, i, toks))
        cache = {}
        if sub._enc is not None:                                  # Latin BERT: ids by hand, pad by hand
            cls_id, sep, mask = sub._ids
            for b in range(0, len(queries), batch_size):
                chunk = queries[b:b + batch_size]; rows, poss = [], []
                for text, i, toks in chunk:
                    ids = [cls_id]; pos = None
                    for j, w in enumerate(toks):
                        if j == i: pos = len(ids); ids.append(mask)
                        else: ids.extend(latin_bert_pieces(sub._enc, w))
                    ids = ids[:200] + [sep]; rows.append(ids); poss.append(min(pos, 199))
                L = max(len(r) for r in rows)
                x = torch.zeros(len(rows), L, dtype=torch.long); att = torch.zeros(len(rows), L, dtype=torch.long)
                for k, r in enumerate(rows): x[k, :len(r)] = torch.tensor(r); att[k, :len(r)] = 1
                with torch.no_grad():
                    logits = model(input_ids=x.to(device), attention_mask=att.to(device)).logits
                for k, (text, i, toks) in enumerate(chunk):
                    probs = torch.softmax(logits[k, poss[k]].float(), dim=-1); top = torch.topk(probs, top_k)
                    cache[(text, i)] = [(sub._sub[j][:-1], p) for p, j in zip(top.values.tolist(), top.indices.tolist()) if sub._sub[j].endswith("_")]
        else:
            tok = sub._tok
            for b in range(0, len(queries), batch_size):
                chunk = queries[b:b + batch_size]
                texts = []
                for text, i, toks in chunk:
                    m = list(toks); m[i] = tok.mask_token; texts.append(" ".join(m))
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
                for k, (text, i, toks) in enumerate(chunk):
                    pos = (enc["input_ids"][k] == tok.mask_token_id).nonzero()
                    if len(pos) == 0:
                        cache[(text, i)] = []; continue
                    probs = torch.softmax(logits[k, int(pos[0, 0])].float(), dim=-1); top = torch.topk(probs, top_k)
                    cache[(text, i)] = [(tok.decode([j]).strip(), p) for p, j in zip(top.values.tolist(), top.indices.tolist())]
        model.to("cpu")
        if log:
            log(f"E38: {len(cache):,} masked-LM proposals for {len(seeds):,} seeds ({model_name}) in {time.time() - t0:.0f}s")
        return cache
