# retexo/baselines/sultan_aligner.py
"""Note 11: Sultan, Bethard and Sumner 2014's five-stage rule-based monolingual aligner.

The non-neural competitor of Table 1: link two words when they are similar *and*
their neighbours are linked too, in a strict pipeline of stages from identical
word sequences down to stop words, each stage discarding what earlier stages
already consumed, one-to-one within a stage. The original is English over
Stanford CoreNLP and PPDB; the algorithm is fully specified in the paper's five
algorithm boxes, so it is reimplemented here over the evidence ``80/`` already
computes: ``LinkFeaturizer`` for the word-similarity half (same form, same
lemma, WordNet synonymy, a lemma-vector cosine, in place of PPDB) and
``DependencyParser`` (Stanza's Latin model) for the context half, with a UD
relation-equivalence table standing in for the paper's Stanford-label table.

Emits links directly (interface II); exempt from the shared decoder, since
there is no score matrix and no theta. The shared rule typer names the links
and applies the frame rule, exactly as for every other alignment-only row.

``cfg.extra["lang"]`` selects the resource backend: ``"la"`` (default) is the
Latin path above; ``"en"`` swaps in :class:`EnglishFeaturizer` (spaCy lemmas
and POS, NLTK WordNet synonymy in place of PPDB, spaCy NER for names) and
:class:`EnglishDependencyParser` (spaCy's dependency parser; its labels are
close enough to UD that the same ``EQ_UD`` table covers both, extended with
spaCy's own label spellings). English mode exists for one purpose: checking
the reimplementation against Sultan et al.'s own published number on MSR-RTE,
since nothing published exists for Latin to check against.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction
from retexo.baselines.record import Edge, Record
from retexo.core.normalize import normalize
from retexo.datasets.substitution import STOPWORDS
from retexo.edit_typing.link_features import FEATURE_NAMES, LinkFeaturizer
from retexo.resources import Resources

#: upos to Sultan's four content categories (verb, noun, adjective, adverb); PROPN folds into noun,
#: since Latin proper names inflect and are frequently the reuse word of a substitution.
CONTENT_CATEGORY: Dict[str, str] = {
    "VERB": "verb",
    "NOUN": "noun",
    "PROPN": "noun",
    "ADJ": "adj",
    "ADV": "adv",
}
#: upos tags folded into the stop-word test on top of ``substitution.STOPWORDS`` (Sultan's function
#: words); PRON is deliberately left out of this set (kept a candidate content word), matching the
#: paper's own inconsistent treatment of pronouns and the gold's frequent chance pronoun matches.
FUNCTION_UPOS = frozenset({"ADP", "CCONJ", "SCONJ", "DET", "PART", "AUX"})
#: Relation-equivalence classes standing in for Sultan et al.'s Table 1 (Stanford labels); two
#: relations count as equivalent if they are equal or sit in the same class here. Each class mixes
#: UD spellings (Stanza, the Latin path) and spaCy's English spellings, so one table serves both.
EQ_UD: Tuple[frozenset, ...] = (
    frozenset({"nsubj", "nsubj:pass", "csubj", "nsubjpass", "csubjpass"}),
    frozenset({"obj", "iobj", "obl", "obl:arg", "dobj", "pobj", "dative"}),
    frozenset({"amod", "acl", "acl:relcl", "relcl"}),
    frozenset({"advmod", "obl", "prep"}),
    frozenset({"nmod", "nmod:poss", "poss"}),
)

# =============================================================================
# The dependency parse of one passage
# =============================================================================


class DependencyParse:
    """upos / deprel / head (0-indexed, -1 = root) per token of one passage.

    Example:
        ```python
        parse = DependencyParse.of(tokens, parser.parse_all([tokens])[" ".join(tokens)])
        parse.category(0)       # "verb", "noun", "adj", "adv" or None
        parse.children(2)       # token indices whose head is token 2
        ```
    """

    def __init__(self, upos: Sequence[str], deprel: Sequence[str], head: Sequence[int]):
        self.upos = list(upos)
        self.deprel = list(deprel)
        self.head = list(head)

    @classmethod
    def of(cls, tokens: Sequence[str], parsed: Sequence[Tuple[str, str, int]]) -> DependencyParse:
        """From one entry of ``DependencyParser.parse_all``'s ``{text: [(upos, deprel, head), ...]}``."""
        if not parsed:
            return cls(["X"] * len(tokens), ["dep"] * len(tokens), [-1] * len(tokens))
        upos, deprel, head = zip(*parsed)
        return cls(upos, deprel, head)

    def category(self, i: int) -> Optional[str]:
        """The content category of token ``i``, or ``None`` for a function word."""
        if not 0 <= i < len(self.upos):
            return None
        return CONTENT_CATEGORY.get(self.upos[i])

    def children(self, i: int) -> List[int]:
        return [k for k, h in enumerate(self.head) if h == i]

    @staticmethod
    def same_relation(a: str, b: str) -> bool:
        """Whether two dependency labels are the same or sit in one ``EQ_UD`` class."""
        if a == b:
            return True
        return any(a in group and b in group for group in EQ_UD)


# =============================================================================
# English resources, for the MSR-RTE and MultiMWA-MTRef checks
# =============================================================================

#: Not Latin orthography: lowercase and strip everything but letters, digits and marks; no u/v,
#: i/j or classical-spelling folding, which would corrupt English (``very`` is not ``uery``).
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def plain_normalize(token: str) -> str:
    return _PUNCT.sub("", token.lower())


#: spaCy's English NER labels that count as a name for ``ne_align``.
NAME_ENTITY_TYPES = frozenset({"PERSON", "ORG", "GPE", "LOC"})


class EnglishDependencyParser:
    """spaCy's English pipeline, cached by sentence text, matching ``DependencyParser``'s
    ``parse_all`` interface: ``{text: [(upos, deprel, head), ...]}``, 0-indexed, -1 = root.

    Whitespace tokens go in pre-built (not re-tokenised by spaCy, exactly as ``DependencyParser``
    keeps Stanza's ``tokenize_pretokenized``), so the record's own tokenisation stays authoritative.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.model = model
        self._nlp = None
        self._parses: Dict[str, List[Tuple[str, str, int]]] = {}

    def _pipeline(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.model)
        return self._nlp

    def parse_all(
        self, token_lists: Sequence[Sequence[str]], **_
    ) -> Dict[str, List[Tuple[str, str, int]]]:
        from spacy.tokens import Doc

        nlp = self._pipeline()
        todo: Dict[str, List[str]] = {}
        for toks in token_lists:
            text = " ".join(toks)
            if text and text not in self._parses and text not in todo:
                todo[text] = list(toks)
        if todo:
            keys = list(todo)
            docs = (Doc(nlp.vocab, words=todo[k]) for k in keys)
            for k, doc in zip(keys, nlp.pipe(docs)):
                self._parses[k] = [
                    (t.pos_ or "X", (t.dep_ or "dep").lower(), -1 if t.head.i == t.i else t.head.i)
                    for t in doc
                ]
        return {" ".join(toks): self._parses.get(" ".join(toks), []) for toks in token_lists}


class EnglishFeaturizer:
    """The word-similarity evidence :class:`SultanAligner` needs, for English: same form,
    same lemma (spaCy), WordNet synonymy (NLTK, in place of PPDB) and same part of speech
    (spaCy). No distributional vectors, so the cosine tier is always missing. Names come
    from spaCy's NER. Single-word lookups, cached, matching ``LinkFeaturizer``'s own
    context-free convention on the Latin side.

    Example:
        ```python
        featurizer = EnglishFeaturizer()
        featurizer("car", "automobile")      # a full FEATURE_NAMES-shaped vector
        featurizer.is_name("Shukla")
        ```
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.model = model
        self._nlp = None
        self._word: Dict[str, Tuple[str, str, bool]] = {}  # token -> (lemma, pos, is_name)
        self._pair: Dict[Tuple[str, str], List[float]] = {}

    def _pipeline(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.model)
        return self._nlp

    def _analyse(self, token: str) -> Tuple[str, str, bool]:
        if token not in self._word:
            doc = self._pipeline()(token)
            if len(doc):
                t = doc[0]
                self._word[token] = (t.lemma_.lower(), t.pos_, t.ent_type_ in NAME_ENTITY_TYPES)
            else:
                self._word[token] = ("", "", False)
        return self._word[token]

    def is_name(self, token: str) -> bool:
        return self._analyse(token)[2]

    def __call__(
        self, source: str, target: str, s: int = 0, t: int = 0, n_source: int = 1, n_target: int = 1
    ) -> List[float]:
        key = (source, target)
        if key in self._pair:
            return list(self._pair[key])
        f = {name: 0.0 for name in FEATURE_NAMES}
        f["cos_missing"] = 1.0
        f["wn_missing"] = 1.0
        ns, nt = plain_normalize(source), plain_normalize(target)
        f["same_form"] = float(ns == nt)
        lemma_s, pos_s, _ = self._analyse(source)
        lemma_t, pos_t, _ = self._analyse(target)
        f["same_lemma"] = float(bool(lemma_s) and lemma_s == lemma_t and not f["same_form"])
        f["same_pos"] = float(bool(pos_s) and pos_s == pos_t)
        if lemma_s and lemma_t and lemma_s != lemma_t:
            try:
                from nltk.corpus import wordnet

                synonyms = {
                    lemma.name().lower()
                    for synset in wordnet.synsets(lemma_s)
                    for lemma in synset.lemmas()
                }
                f["wn_syn"] = float(lemma_t in synonyms)
            except LookupError:
                pass
        vec = [f[name] for name in FEATURE_NAMES]
        self._pair[key] = vec
        return list(vec)


# =============================================================================
# The aligner
# =============================================================================


@BaselineRegistry.register
class SultanAligner(Baseline):
    """Sultan et al. 2014's pipeline: identical sequences, names, content words by
    dependency evidence, content words by textual-neighbourhood evidence, then stop
    words, each stage one-to-one and never re-touching an earlier stage's links.

    ``cfg.extra`` dials (defaults in brackets, all tuned on the dev fold by
    :meth:`tune`): ``ppdb_sim`` (0.9, the similarity of a resource-attested pair
    that is not an exact form or lemma match), ``w`` (0.9, the weight of word
    similarity against contextual evidence), ``syn_cos`` (0.65, the lemma-vector
    cosine floor), ``no_competitor`` (on, whether a content pair with no context
    still aligns when nothing else competes for either of its words),
    ``stopword_fallback`` (off by default: an exact-match stop-word pair with no
    competitor still aligns even with no already-aligned neighbour. This
    reimplementation's own extension, tried because Sultan et al.'s own
    already-aligned-neighbour requirement (section 3.3.4) is exactly what a heavily
    reworded MSR-RTE pair often breaks -- measured *against* on a 300-pair sample
    (F1 0.889 to 0.870): stop words carry so little content that "no competitor"
    is not real evidence, only the absence of ambiguity, and Sultan et al.'s
    stricter rule turns out to be doing real work. Left in as an opt-in dial, off
    by default),
    ``offline`` (0, whether the backing Latin ``Resources`` run offline),
    ``lang`` (``"la"``, or ``"en"`` for the MSR-RTE / MultiMWA-MTRef check).

    Example:
        ```python
        method = SultanAligner(BaselineConfig(device="cpu"))
        pred = method.postprocess(record, method.predict([record])[0], {})
        # python run_baseline.py --method sultan --fold 4 --smoke 20 --device cpu
        # python run_baseline.py --method sultan --set msr_rte --extra lang=en
        ```
    """

    name = "sultan"
    emits = "edges"
    trainable = False
    typer = "rule"

    #: Sultan et al.'s own tuned values (their section 3.3.5), the starting point for `tune`.
    PPDB_SIM = 0.9
    W = 0.9
    SYN_COS = 0.65
    WINDOW = 3
    MIN_LEN = 2
    NO_COMPETITOR = True

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.lang = str(cfg.extra.get("lang", "la"))
        if self.lang == "en":
            self.evidence = EnglishFeaturizer()
            self._normalize = plain_normalize
            self._stopwords = None  # lazy: needs the NLTK corpus, not at import time
        else:
            # the cached resources only (``offline=1``): a live WordNet lookup costs four or more requests per
            # lemma at a 25 s timeout, and the shared featurizer of every other row is offline too
            self.evidence = LinkFeaturizer(
                Resources(offline=bool(int(cfg.extra.get("offline", 1))))
            )
            self._normalize = normalize
            self._stopwords = STOPWORDS
        self.ppdb_sim = float(cfg.extra.get("ppdb_sim", self.PPDB_SIM))
        self.w = float(cfg.extra.get("w", self.W))
        self.syn_cos = float(cfg.extra.get("syn_cos", self.SYN_COS))
        self.no_competitor = bool(cfg.extra.get("no_competitor", self.NO_COMPETITOR))
        self.stopword_fallback = bool(cfg.extra.get("stopword_fallback", False))
        self.cross_orientation = bool(cfg.extra.get("cross_orientation", False))
        self._parser = None

    # ---------- Lazy resources ----------

    @property
    def stopwords(self) -> frozenset:
        """The stop-word list: ``substitution.STOPWORDS`` for Latin, NLTK's English list."""
        if self._stopwords is None:
            from nltk.corpus import stopwords

            self._stopwords = frozenset(stopwords.words("english"))
        return self._stopwords

    @property
    def parser(self):
        """The dependency parser, built on first use (never at import time): Stanza's Latin
        model, or spaCy's English pipeline for ``lang="en"``."""
        if self._parser is None:
            if self.lang == "en":
                self._parser = EnglishDependencyParser()
            else:
                from retexo.edit_typing.dep_features import DependencyParser

                self._parser = DependencyParser(device=self.cfg.device)
        return self._parser

    # ---------- word similarity (section 3.1) ----------

    def word_sim(self, source: str, target: str) -> float:
        """1.0 for an exact form, lemma or enclitic-stem match; ``ppdb_sim`` for an
        attested lexical relation (WordNet synonymy, or a lemma-vector cosine at or
        above ``syn_cos`` between words of the same part of speech); else 0.0."""
        phi = dict(zip(FEATURE_NAMES, self.evidence(source, target)))
        if phi["same_form"] or phi["same_lemma"] or phi["enclitic_stem_match"]:
            return 1.0
        if phi["wn_syn"] or (
            not phi["cos_missing"] and phi["cos"] >= self.syn_cos and phi["same_pos"]
        ):
            return self.ppdb_sim
        return 0.0

    def is_stop(self, token: str, upos: Optional[str] = None) -> bool:
        """Punctuation (an empty normalised form), a listed stop word, or a function-word upos."""
        norm = self._normalize(token)
        if not norm:
            return True
        if norm in self.stopwords:
            return True
        return upos in FUNCTION_UPOS

    # ---------- contextual evidence (section 3.2) ----------

    @staticmethod
    def _category_compatible(a: Optional[str], b: Optional[str]) -> bool:
        """Two neighbours are compatible unless both resolve to a content category and
        it differs; a pronoun or function-word neighbour (Sultan's own ``nsubj``
        example is often a pronoun) is never excluded by this test alone."""
        return a is None or b is None or a == b

    def dep_context(
        self,
        S: Sequence[str],
        T: Sequence[str],
        parse_s: DependencyParse,
        parse_t: DependencyParse,
        i: int,
        j: int,
    ) -> List[Tuple[int, int]]:
        """Neighbour pairs (Algorithm 1): a parent-parent or child-child pair of ``i``
        and ``j`` whose relations are the same or equivalent, whose categories are
        compatible, and whose word similarity is positive. With ``cross_orientation``
        (off by default) also a parent-of-``i``/child-of-``j`` or
        child-of-``i``/parent-of-``j`` pair, an attempt at the paper's Figure 3
        orientations this reimplementation does not otherwise reproduce -- measured
        with no effect at all on a 300-pair sample (identical to three decimals with
        it on or off): the pairs it finds are redundant with what parent-parent and
        child-child evidence already supplies. Left in, off by default, rather than
        deleted, since a future change to the same-direction evidence could make it
        relevant again."""
        out: List[Tuple[int, int]] = []
        ps, pt = parse_s.head[i], parse_t.head[j]
        cs, ct = parse_s.children(i), parse_t.children(j)
        if (
            ps >= 0
            and pt >= 0
            and DependencyParse.same_relation(parse_s.deprel[i], parse_t.deprel[j])
            and self._category_compatible(parse_s.category(ps), parse_t.category(pt))
            and self.word_sim(S[ps], T[pt]) > 0
        ):
            out.append((ps, pt))
        for k in cs:
            for t_idx in ct:
                if (
                    DependencyParse.same_relation(parse_s.deprel[k], parse_t.deprel[t_idx])
                    and self._category_compatible(parse_s.category(k), parse_t.category(t_idx))
                    and self.word_sim(S[k], T[t_idx]) > 0
                ):
                    out.append((k, t_idx))
        if self.cross_orientation:
            if ps >= 0:
                for t_idx in ct:
                    if (
                        DependencyParse.same_relation(parse_s.deprel[i], parse_t.deprel[t_idx])
                        and self._category_compatible(parse_s.category(ps), parse_t.category(t_idx))
                        and self.word_sim(S[ps], T[t_idx]) > 0
                    ):
                        out.append((ps, t_idx))
            if pt >= 0:
                for k in cs:
                    if (
                        DependencyParse.same_relation(parse_s.deprel[k], parse_t.deprel[j])
                        and self._category_compatible(parse_s.category(k), parse_t.category(pt))
                        and self.word_sim(S[k], T[pt]) > 0
                    ):
                        out.append((k, pt))
        return out

    def text_context(
        self,
        S: Sequence[str],
        T: Sequence[str],
        parse_s: DependencyParse,
        parse_t: DependencyParse,
        i: int,
        j: int,
    ) -> List[Tuple[int, int]]:
        """Every non-stop neighbour of ``i`` within the window paired with every
        non-stop neighbour of ``j`` (Algorithm 2); positions are interchangeable."""

        def neighbours(tokens, parse, idx):
            lo, hi = max(0, idx - self.WINDOW), min(len(tokens), idx + self.WINDOW + 1)
            return [
                k for k in range(lo, hi) if k != idx and not self.is_stop(tokens[k], parse.upos[k])
            ]

        left = neighbours(S, parse_s, i)
        right = neighbours(T, parse_t, j)
        return [(k, t_idx) for k in left for t_idx in right if self.word_sim(S[k], T[t_idx]) > 0]

    # ---------- the pipeline (section 3.3) ----------

    def punct_align(
        self, S: Sequence[str], T: Sequence[str], used_s: Set[int], used_t: Set[int]
    ) -> List[Tuple[int, int]]:
        """Identical punctuation marks, matched by rank within each mark (Sultan et
        al.'s own first pipeline module, ``alignWords``'s punctuation stage). A lone
        punctuation token -- a sentence-final period above all -- has no run to belong
        to and no lexical evidence, so without this stage it can never align at all;
        on MSR-RTE this single gap accounted for a tenth of every sure link in the
        corpus. Marks of the same type tend to keep their relative order across a
        rewrite, so the first ``.`` of ``S`` pairs with the first ``.`` of ``T``, and
        so on; counts need not match."""
        by_mark_s: Dict[str, List[int]] = {}
        by_mark_t: Dict[str, List[int]] = {}
        for i, w in enumerate(S):
            if i not in used_s and w.strip() and self._normalize(w) == "":
                by_mark_s.setdefault(w, []).append(i)
        for j, w in enumerate(T):
            if j not in used_t and w.strip() and self._normalize(w) == "":
                by_mark_t.setdefault(w, []).append(j)
        out = []
        for mark, idx_s in by_mark_s.items():
            for i, j in zip(idx_s, by_mark_t.get(mark, ())):
                out.append((i, j))
                used_s.add(i)
                used_t.add(j)
        return out

    def ws_align(
        self,
        S: Sequence[str],
        T: Sequence[str],
        used_s: Optional[Set[int]] = None,
        used_t: Optional[Set[int]] = None,
    ) -> List[Tuple[int, int]]:
        """Every identical word sequence of length at least ``MIN_LEN`` containing at
        least one non-stop word, longest first, one-to-one (Algorithm, section 3.3.1)."""
        ns, nt = [self._normalize(w) for w in S], [self._normalize(w) for w in T]
        used_s = set() if used_s is None else used_s
        used_t = set() if used_t is None else used_t
        out: List[Tuple[int, int]] = []
        for length in range(min(len(S), len(T)), self.MIN_LEN - 1, -1):
            for i in range(len(S) - length + 1):
                run = tuple(ns[i : i + length])
                if not any(w and w not in self.stopwords for w in run):
                    continue
                for j in range(len(T) - length + 1):
                    if tuple(nt[j : j + length]) != run:
                        continue
                    span_s, span_t = range(i, i + length), range(j, j + length)
                    if used_s.isdisjoint(span_s) and used_t.isdisjoint(span_t):
                        out.extend(zip(span_s, span_t))
                        used_s.update(span_s)
                        used_t.update(span_t)
        return out

    def ne_align(
        self, S: Sequence[str], T: Sequence[str], used_s: Set[int], used_t: Set[int]
    ) -> List[Tuple[int, int]]:
        """Names (``Entities.is_name``, no recogniser: exact term matches only), most
        similar first, one-to-one (section 3.3.2)."""
        candidates = sorted(
            (
                (self.word_sim(S[i], T[j]), i, j)
                for i in range(len(S))
                if i not in used_s and self.evidence.is_name(S[i])
                for j in range(len(T))
                if j not in used_t and self.evidence.is_name(T[j])
            ),
            key=lambda x: -x[0],
        )
        out = []
        for sim, i, j in candidates:
            if sim <= 0 or i in used_s or j in used_t:
                continue
            out.append((i, j))
            used_s.add(i)
            used_t.add(j)
        return out

    def cw_dep_align(
        self,
        S: Sequence[str],
        T: Sequence[str],
        parse_s: DependencyParse,
        parse_t: DependencyParse,
        used_s: Set[int],
        used_t: Set[int],
    ) -> List[Tuple[int, int]]:
        """Content pairs with dependency evidence, by descending score (Algorithm 3);
        the evidence pair itself is aligned alongside its content pair (lines 19-22)."""
        scored = []
        for i in range(len(S)):
            if i in used_s or parse_s.category(i) is None:
                continue
            for j in range(len(T)):
                if j in used_t or parse_t.category(j) != parse_s.category(i):
                    continue
                sim = self.word_sim(S[i], T[j])
                if sim <= 0:
                    continue
                evidence = self.dep_context(S, T, parse_s, parse_t, i, j)
                context_sim = sum(self.word_sim(S[k], T[t_idx]) for k, t_idx in evidence)
                if context_sim <= 0:
                    continue
                scored.append((self.w * sim + (1 - self.w) * context_sim, i, j, evidence))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, i, j, evidence in scored:
            if i in used_s or j in used_t:
                continue
            out.append((i, j))
            used_s.add(i)
            used_t.add(j)
            for k, t_idx in evidence:
                if k not in used_s and t_idx not in used_t:
                    out.append((k, t_idx))
                    used_s.add(k)
                    used_t.add(t_idx)
        return out

    def cw_text_align(
        self,
        S: Sequence[str],
        T: Sequence[str],
        parse_s: DependencyParse,
        parse_t: DependencyParse,
        used_s: Set[int],
        used_t: Set[int],
    ) -> List[Tuple[int, int]]:
        """Content pairs with textual-neighbourhood evidence (Algorithm 4); a pair
        with no context still aligns when ``no_competitor`` and nothing scores higher
        on either of its words (the "no competitor" rule, section 3.3.3)."""
        scored, plain = [], []
        for i in range(len(S)):
            if i in used_s or parse_s.category(i) is None:
                continue
            for j in range(len(T)):
                if j in used_t or parse_t.category(j) != parse_s.category(i):
                    continue
                sim = self.word_sim(S[i], T[j])
                if sim <= 0:
                    continue
                evidence = self.text_context(S, T, parse_s, parse_t, i, j)
                context_sim = sum(self.word_sim(S[k], T[t_idx]) for k, t_idx in evidence)
                if context_sim > 0:
                    scored.append((self.w * sim + (1 - self.w) * context_sim, i, j))
                else:
                    plain.append((sim, i, j))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, i, j in scored:
            if i in used_s or j in used_t:
                continue
            out.append((i, j))
            used_s.add(i)
            used_t.add(j)
        if self.no_competitor:
            for sim, i, j in sorted(plain, key=lambda x: -x[0]):
                if i in used_s or j in used_t:
                    continue
                beaten = any(
                    s2 > sim
                    for s2, i2, j2 in plain
                    if (i2 == i or j2 == j)
                    and (i2, j2) != (i, j)
                    and i2 not in used_s
                    and j2 not in used_t
                )
                if not beaten:
                    out.append((i, j))
                    used_s.add(i)
                    used_t.add(j)
        return out

    def sw_dep_align(
        self,
        S: Sequence[str],
        T: Sequence[str],
        parse_s: DependencyParse,
        parse_t: DependencyParse,
        used_s: Set[int],
        used_t: Set[int],
        aligned: Dict[int, int],
    ) -> List[Tuple[int, int]]:
        """Stop words whose parent or a child is already aligned to the other side's
        counterpart under the exact same relation label (section 3.3.4)."""
        out = []
        for i in range(len(S)):
            if i in used_s or not self.is_stop(S[i], parse_s.upos[i]):
                continue
            for j in range(len(T)):
                if (
                    j in used_t
                    or not self.is_stop(T[j], parse_t.upos[j])
                    or self.word_sim(S[i], T[j]) <= 0
                ):
                    continue
                ps, pt = parse_s.head[i], parse_t.head[j]
                parent_ok = (
                    ps >= 0 and aligned.get(ps) == pt and parse_s.deprel[i] == parse_t.deprel[j]
                )
                child_ok = any(
                    aligned.get(k) == t_idx and parse_s.deprel[k] == parse_t.deprel[t_idx]
                    for k in parse_s.children(i)
                    for t_idx in parse_t.children(j)
                )
                if parent_ok or child_ok:
                    out.append((i, j))
                    used_s.add(i)
                    used_t.add(j)
                    aligned[i] = j
        return out

    def sw_text_align(
        self,
        S: Sequence[str],
        T: Sequence[str],
        used_s: Set[int],
        used_t: Set[int],
        aligned: Dict[int, int],
    ) -> List[Tuple[int, int]]:
        """Stop words whose immediate left neighbour or immediate right neighbour is
        already aligned to the other side's corresponding neighbour, checked
        separately (section 3.3.4)."""
        out = []
        for i in range(len(S)):
            if i in used_s:
                continue
            for j in range(len(T)):
                if j in used_t or self.word_sim(S[i], T[j]) <= 0:
                    continue
                left = i - 1 >= 0 and j - 1 >= 0 and aligned.get(i - 1) == j - 1
                right = i + 1 < len(S) and j + 1 < len(T) and aligned.get(i + 1) == j + 1
                if left or right:
                    out.append((i, j))
                    used_s.add(i)
                    used_t.add(j)
                    aligned[i] = j
        return out

    def sw_fallback_align(
        self, S: Sequence[str], T: Sequence[str], used_s: Set[int], used_t: Set[int]
    ) -> List[Tuple[int, int]]:
        """A stop word still unaligned after both evidence-based stop-word stages
        aligns anyway if it is an exact form-or-lemma match (``word_sim == 1.0``) for
        exactly one remaining candidate on each side -- no dependency or textual
        evidence, no competitor. Sultan et al.'s own stop-word stage requires an
        already-aligned neighbour (section 3.3.4); a heavily reworded MSR-RTE pair
        often leaves no such neighbour even for an exact repeat like "the"/"the" or
        "is"/"is". This is the reimplementation's own extension beyond the paper,
        gated by ``stopword_fallback`` (default on) precisely because it is not in
        the paper."""
        plain = [
            (i, j)
            for i in range(len(S))
            if i not in used_s and self.is_stop(S[i])
            for j in range(len(T))
            if j not in used_t and self.is_stop(T[j]) and self.word_sim(S[i], T[j]) == 1.0
        ]
        out = []
        for i, j in plain:
            if i in used_s or j in used_t:
                continue
            competitor = any(
                (i2 == i or j2 == j)
                and (i2, j2) != (i, j)
                and i2 not in used_s
                and j2 not in used_t
                for i2, j2 in plain
            )
            if not competitor:
                out.append((i, j))
                used_s.add(i)
                used_t.add(j)
        return out

    def align(
        self, S: Sequence[str], T: Sequence[str], parse_s: DependencyParse, parse_t: DependencyParse
    ) -> List[Tuple[int, int]]:
        """The full pipeline in order; never aligns a word twice (Algorithm 5)."""
        used_s: Set[int] = set()
        used_t: Set[int] = set()
        links = list(self.punct_align(S, T, used_s, used_t))
        links += self.ws_align(S, T, used_s, used_t)
        links += self.ne_align(S, T, used_s, used_t)
        links += self.cw_dep_align(S, T, parse_s, parse_t, used_s, used_t)
        links += self.cw_text_align(S, T, parse_s, parse_t, used_s, used_t)
        aligned = dict(links)
        links += self.sw_dep_align(S, T, parse_s, parse_t, used_s, used_t, aligned)
        links += self.sw_text_align(S, T, used_s, used_t, aligned)
        if self.stopword_fallback:
            links += self.sw_fallback_align(S, T, used_s, used_t)
        return links

    # ---------- Inference ----------

    def predict(self, records: List[Record]) -> List[Prediction]:
        # one batched parse call for the whole list (Stanza chunks internally), not one per record
        parsed = self.parser.parse_all(
            [r.source_tokens for r in records] + [r.reuse_tokens for r in records]
        )
        out = []
        for record in records:
            S, T = record.source_tokens, record.reuse_tokens
            parse_s = DependencyParse.of(S, parsed[" ".join(S)])
            parse_t = DependencyParse.of(T, parsed[" ".join(T)])
            pred = Prediction.empty(len(T))
            seen_t: Set[int] = set()
            for s, t in self.align(S, T, parse_s, parse_t):
                if t in seen_t:
                    pred.extra.append(Edge(t, s, ""))
                else:
                    pred.links[t] = s
                    seen_t.add(t)
            out.append(pred)
        return out

    # ---------- Tuning ----------

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """The paper's own two-dimensional search (``ppdb_sim``, ``w``) plus the two
        Latin dials (``syn_cos``, ``no_competitor``), on the dev fold's token accuracy."""
        from retexo.baselines.scorer import BaselineScorer

        best = {
            "ppdb_sim": self.ppdb_sim,
            "w": self.w,
            "syn_cos": self.syn_cos,
            "no_competitor": self.no_competitor,
        }
        best_acc = -1.0
        for ppdb_sim in (0.8, 0.9, 1.0):
            for w in (0.7, 0.8, 0.9, 1.0):
                for syn_cos in (0.6, 0.65, 0.7):
                    for no_competitor in (True, False):
                        self.ppdb_sim, self.w = ppdb_sim, w
                        self.syn_cos, self.no_competitor = syn_cos, no_competitor
                        acc = BaselineScorer.token_accuracy(dev, self.predict(dev))
                        if acc > best_acc:
                            best_acc = acc
                            best = {
                                "ppdb_sim": ppdb_sim,
                                "w": w,
                                "syn_cos": syn_cos,
                                "no_competitor": no_competitor,
                            }
        self.ppdb_sim, self.w = best["ppdb_sim"], best["w"]
        self.syn_cos, self.no_competitor = best["syn_cos"], best["no_competitor"]
        if log:
            log(f"[sultan] tuned {best}, dev token_acc {best_acc:.3f}")
        return best
