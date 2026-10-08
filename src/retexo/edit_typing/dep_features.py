# retexo/edit_typing/dep_features.py
"""E41: dependency evidence for a link (Sultan et al. 2014 made into features).

Stanza's Latin model parses every passage once (GPU, in the parent process; cached by
text), and each candidate cell (t, s) gets four extra features: same UPOS in context,
same dependency relation, the two heads are the same word (form or lemma), both words
are content words. On fold 4 these separate the aligner's real substitution errors
(hallucinated or slot-ambiguous links: heads linked 0.00, same POS 0.20-0.50) from
correct links and annotator omissions (same POS 0.70-0.74, heads linked 0.17-0.34).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from retexo.core.normalize import normalize

DEP_FEATURES: Tuple[str, ...] = (
    "dep_same_upos",
    "dep_same_rel",
    "dep_heads_same",
    "dep_both_content",
)


class DependencyParser:
    """Stanza's Latin dependency parser, cached by sentence text.

    Example:
        ```python
        parser = DependencyParser(device="cuda")
        parses = parser.parse_all(token_lists)
        features = DependencyParser.cell_features(src, tgt, parses[src_text], parses[tgt_text], s, t)
        ```
    """

    CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._nlp = None
        self._parses: Dict[str, List[Tuple[str, str, int]]] = {}

    def _pipeline(self):
        if self._nlp is None:
            import stanza

            self._nlp = stanza.Pipeline(
                "la",
                processors="tokenize,pos,lemma,depparse",
                tokenize_pretokenized=True,
                use_gpu=self.device.startswith("cuda"),
                verbose=False,
            )
        return self._nlp

    def parse_all(self, token_lists: Sequence[Sequence[str]], *, batch: int = 2000, log=None):
        """Parse every unseen sentence; returns {text: [(upos, deprel, head_index), ...]}."""
        import time

        todo = {}
        for toks in token_lists:
            text = " ".join(toks)
            if text not in self._parses and text not in todo:
                todo[text] = list(toks)
        if todo:
            t0 = time.time()
            nlp = self._pipeline()
            keys = list(todo)
            for b in range(0, len(keys), batch):
                chunk = keys[b : b + batch]
                doc = nlp([todo[k] for k in chunk])
                for k, sent in zip(chunk, doc.sentences):
                    self._parses[k] = [
                        (w.upos or "X", (w.deprel or "dep").split(":")[0], w.head - 1)
                        for w in sent.words
                    ]
            if log:
                log(
                    f"E41: parsed {len(todo):,} new passages ({len(self._parses):,} cached) in {time.time() - t0:.0f}s"
                )
        return {(" ".join(toks)): self._parses[" ".join(toks)] for toks in token_lists}

    @staticmethod
    def cell_features(
        src: Sequence[str], tgt: Sequence[str], dep_s, dep_t, s: int, t: int, lemma=None
    ) -> List[float]:
        us, rs, hs = dep_s[s] if s < len(dep_s) else ("X", "dep", -1)
        ut, rt, ht = dep_t[t] if t < len(dep_t) else ("X", "dep", -1)
        heads_same = 0.0
        if hs >= 0 and ht >= 0 and hs < len(src) and ht < len(tgt):
            a, b = src[hs], tgt[ht]
            la = lemma(a) if lemma is not None else ""
            lb = lemma(b) if lemma is not None else ""
            heads_same = float(bool(normalize(a) == normalize(b) or (la and la == lb)))
        return [
            float(us == ut),
            float(rs == rt),
            heads_same,
            float(us in DependencyParser.CONTENT and ut in DependencyParser.CONTENT),
        ]
