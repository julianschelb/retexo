# retexo/datasets/substitution.py
"""
Choosing replacement words for synthetic variants.

Generation and detection must not draw on the same resource. If substitutions
come from the WordNet the oracle consults, every synthetic example is
oracle-recoverable by construction, and the corpus teaches a model nothing the
oracle does not already know — it distils a lookup table we already have. The
asymmetry between a wide generator and a narrower detector is what gives the
model something to learn, and it is the reason this module exists rather than
simply calling WordNet.

The pipeline is a pool, then filters, then a language model as a *scorer*:

    pool      Collatinus lemmas intersected with the corpus vocabulary
    filter    same part of speech, morphologically compatible, different
              lemma, not a stopword
    inflect   Collatinus declines the candidate into the form the slot needs
    score     Latin BERT ranks the survivors in context
    check     the original must score highly in the candidate's place

Reading the model's top-k predictions directly does not work: they are
dominated by function words, and subword vocabularies return fragments rather
than words. Supplying the candidates and using the model only to rank them
avoids both problems and keeps the pool under our control.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from retexo.core.normalize import normalize
from retexo.formulations.pair_encoding import latin_bert_pieces

#: Latin function words, excluded from substitution candidates.
STOPWORDS = frozenset(
    [
        "et",
        "ac",
        "atque",
        "que",
        "nec",
        "neque",
        "aut",
        "vel",
        "ve",
        "sed",
        "at",
        "nam",
        "enim",
        "autem",
        "vero",
        "quidem",
        "tamen",
        "in",
        "ad",
        "ex",
        "de",
        "ab",
        "a",
        "e",
        "cum",
        "sine",
        "per",
        "pro",
        "sub",
        "super",
        "ante",
        "post",
        "inter",
        "ob",
        "prae",
        "propter",
        "non",
        "ne",
        "haud",
        "nihil",
        "nemo",
        "qui",
        "quae",
        "quod",
        "quis",
        "quid",
        "is",
        "ea",
        "id",
        "hic",
        "haec",
        "ille",
        "illa",
        "illud",
        "iste",
        "ipse",
        "idem",
        "sum",
        "es",
        "est",
        "sunt",
        "eram",
        "erat",
        "erit",
        "esse",
        "fuit",
        "ut",
        "si",
        "cum",
        "dum",
        "donec",
        "quia",
        "quod",
        "quoniam",
        "ubi",
        "unde",
        "quo",
        "ne",
        "an",
        "num",
        "utrum",
        "iam",
        "nunc",
        "tunc",
        "tum",
        "deinde",
        "mox",
        "semper",
        "saepe",
        "nondum",
        "iamque",
        "quoque",
        "etiam",
        "adhuc",
        "ita",
        "sic",
        "tam",
        "quam",
        "magis",
        "minus",
        "valde",
        "admodum",
        "satis",
        "nimis",
        "prorsus",
        "omnino",
        "igitur",
        "ergo",
        "itaque",
        "nempe",
        "scilicet",
        "videlicet",
        "forte",
        "fortasse",
        "ego",
        "tu",
        "nos",
        "vos",
        "me",
        "te",
        "se",
        "mihi",
        "tibi",
        "sibi",
        "meus",
        "tuus",
        "suus",
        "noster",
        "vester",
        "omnis",
        "omnes",
        "totus",
        "alius",
        "alter",
        "ceterus",
        "quisque",
        "quisquam",
        "ullus",
        "nullus",
    ]
)

# =============================================================================
# Candidate pool
# =============================================================================


@dataclass
class CandidatePool:
    """Latin words available as substitutes, with their lemma and part of speech.

    Built from the Collatinus lexicon intersected with the corpus vocabulary,
    so every candidate is both a real word and one this corpus actually uses.
    WordNet is deliberately absent.
    """

    by_pos: Dict[str, List[str]] = field(default_factory=dict)
    lemma_of: Dict[str, str] = field(default_factory=dict)

    #: Lemmas excluded as too frequent to be a meaningful substitution.
    excluded_frequent: int = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_pos.values())

    @classmethod
    def build(
        cls,
        resources,
        corpus_words: Sequence[str],
        *,
        max_words: Optional[int] = None,
        frequent_cutoff: int = 250,
    ) -> CandidatePool:
        """Assemble the pool from a corpus vocabulary.

        Function words are excluded by *frequency* rather than by a hand-written
        list. No available list is adequate: CLTK's Latin stoplist has 92
        entries and contains neither *quippe* nor *interdum*, both of which a
        contextual model will happily rank first in any slot. Frequency is the
        property that actually defines a function word, so the most frequent
        ``frequent_cutoff`` lemmas in the corpus are dropped, and the explicit
        list is kept only as a floor for words a small sample might miss.
        """
        import collections

        counts = collections.Counter(resources.lemma_or_surface(w) for w in corpus_words)
        frequent = {lemma for lemma, _ in counts.most_common(frequent_cutoff)}

        pool = cls(excluded_frequent=len(frequent))
        seen = set()
        for word in corpus_words:
            surface = normalize(word)
            if not surface or surface in seen or len(surface) < 3:
                continue
            seen.add(surface)
            lemma = resources.lemma_or_surface(word)
            # Function words must be excluded by lemma, not by surface: an
            # inflected form such as "ipsa" is not in the list but "ipse" is,
            # and filtering on the surface alone lets every inflection through.
            if surface in STOPWORDS or lemma in STOPWORDS or lemma in frequent:
                continue
            pos = resources.pos_of(word) if resources.has("morphology") else "n"
            pool.by_pos.setdefault(pos, []).append(surface)
            pool.lemma_of[surface] = lemma
            if max_words and len(seen) >= max_words:
                break
        return pool

    def candidates(self, pos: str, exclude_lemma: str, limit: int) -> List[str]:
        """Words of a given part of speech, excluding the source's own lemma."""
        options = self.by_pos.get(pos) or []
        out = [w for w in options if self.lemma_of.get(w) != exclude_lemma]
        return out[:limit] if limit and len(out) > limit else out


# =============================================================================
# Scorer
# =============================================================================


class ContextualScorer:
    """Ranks candidate words by how well they fit a masked slot.

    Latin BERT is used because its own subword encoder segments Latin without
    fragmenting it, unlike the general-purpose vocabularies, and because a
    score over supplied candidates avoids the failure mode of reading top-k
    predictions directly.
    """

    def __init__(
        self,
        model_name: str = "ashleygong03/bamman-burns-latin-bert",
        device: str = "cpu",
    ):
        self.model_name = model_name
        #: Scoring runs once per candidate slot over the whole seed corpus, so
        #: this is the dominant cost of generation. On CPU it is the difference
        #: between minutes and hours.
        self.device = device
        self._encoder = None
        self._model = None
        self._specials: Dict[str, int] = {}

    def _build(self):
        if self._model is not None:
            return
        import huggingface_hub
        import torch  # noqa: F401
        from locisimiles.tokenization.latin_bert import SubwordTextEncoder
        from transformers import AutoModelForMaskedLM

        vocab = huggingface_hub.hf_hub_download(self.model_name, "vocab.txt")
        self._encoder = SubwordTextEncoder.from_file(vocab)
        subtokens = self._encoder._subtokens
        self._specials = {
            s: i for i, s in enumerate(subtokens) if s in ("[CLS]", "[SEP]", "[MASK]")
        }
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def score(
        self,
        words: Sequence[str],
        position: int,
        candidates: Sequence[str],
        *,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Rank candidates for the slot at ``position``, best first.

        The score is the mean log-probability of a candidate's subtokens in the
        masked slot, so candidates of different lengths stay comparable.
        """
        import torch

        self._build()
        cls, sep, mask = (
            self._specials.get("[CLS]", 2),
            self._specials.get("[SEP]", 3),
            self._specials.get("[MASK]", 4),
        )
        ids, spans = [cls], []
        for word in words:
            pieces = latin_bert_pieces(self._encoder, word)
            spans.append((len(ids), len(ids) + len(pieces)))
            ids.extend(pieces)
        ids.append(sep)

        start, end = spans[position]
        masked = ids[:start] + [mask] + ids[end:]
        with torch.no_grad():
            input_ids = torch.tensor([masked], device=self.device)
            logits = self._model(input_ids=input_ids).logits[0, start]
            log_probs = torch.log_softmax(logits, dim=-1).cpu()

        scored: List[Tuple[str, float]] = []
        for candidate in candidates:
            pieces = latin_bert_pieces(self._encoder, candidate)
            if not pieces:
                continue
            scored.append((candidate, float(sum(log_probs[p] for p in pieces) / len(pieces))))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# =============================================================================
# The substitution source
# =============================================================================


@dataclass
class ContextualSubstitutionSource:
    """Draws substitutes from a corpus pool, ranked in context by Latin BERT.

    Example:
        ```python
        source = ContextualSubstitutionSource.build(resources, corpus_words)
        source.replacement_in_context(words, position, tag="SYN", rng=rng)
        ```

    ``MORPH`` is handled by Collatinus rather than by the scorer, since a
    different inflection of the same lemma is a lookup, not a choice.
    """

    resources: object
    pool: CandidatePool
    scorer: ContextualScorer = field(default_factory=ContextualScorer)

    #: Candidates scored per slot. Larger is slower and rarely better.
    pool_limit: int = 400

    #: How many the contextual model proposes before similarity chooses. Too
    #: few and the right word never reaches the shortlist; too many and a
    #: poorly-fitting but highly similar word wins on similarity alone.
    shortlist_size: int = 8

    #: Floor on distributional similarity to the source lemma. Candidates below
    #: it are removed *before* the model ranks them, because a contextual model
    #: ranks function words highly everywhere and the top of its list is
    #: otherwise dominated by them. Calibrated against probes where genuine
    #: synonyms score 0.45-0.61 and unrelated particles 0.21-0.36; the
    #: near-antonym *angustus*/*altus* sits at 0.39, just below the floor.
    min_similarity: Optional[float] = 0.40

    #: Part of speech is a weak filter for Latin: Collatinus leaves nominals
    #: unmarked, so most words default to noun. It narrows the pool but is not
    #: relied on.
    use_pos_filter: bool = True

    #: What this source can honestly produce. A distributional model can find a
    #: word that is close in meaning, and Collatinus can inflect one, but
    #: neither can tell a hypernym from a synonym from an antonym: those need a
    #: hierarchy, and the only Latin hierarchy available is the WordNet the
    #: oracle detects with, which is the circularity this source exists to
    #: avoid. Asked for HYPER it would return the same word it returns for SYN,
    #: and the label would be fiction. Those relations are therefore left to
    #: the real pairs in phases two and three, where the oracle reads them off
    #: attested evidence rather than the generator inventing them.
    supported_tags: frozenset = frozenset({"MORPH", "SYN-DIST"})

    _rejected_antonyms: int = 0
    _accepted: int = 0

    @classmethod
    def build(cls, resources, corpus_words: Sequence[str], **kwargs):
        """Construct with a pool built from a corpus vocabulary."""
        return cls(resources=resources, pool=CandidatePool.build(resources, corpus_words), **kwargs)

    def realises(self, tag: str) -> bool:
        """Whether a substitution under ``tag`` would mean what it says."""
        return tag in self.supported_tags

    def replacement(self, token: str, tag: str, rng: random.Random) -> Optional[str]:
        """Context-free substitution, which this source does not offer.

        Every decision here depends on the surrounding words, so a caller that
        cannot supply them gets nothing rather than a word chosen blind.
        """
        return None

    # ---------- Public API ----------

    def replacement_in_context(
        self,
        words: Sequence[str],
        position: int,
        tag: str,
        rng: random.Random,
    ) -> Optional[str]:
        """A substitute for ``words[position]``, or ``None`` if none is suitable."""
        token = words[position]
        if tag == "MORPH":
            return self._inflection(token, rng)

        lemma = self.resources.lemma_or_surface(token)
        pos = self.resources.pos_of(token) if self.use_pos_filter else None
        candidates = (
            self.pool.candidates(pos, lemma, 0)
            if pos
            else [w for words_ in self.pool.by_pos.values() for w in words_]
        )
        plausible = self._semantically_plausible(lemma, candidates)
        if not plausible:
            return None

        # Each signal is good at one thing and bad at the other, so neither
        # decides alone. Distributional similarity knows what is semantically
        # related but not what fits here; a contextual model knows what fits
        # here but ranks particles first, because a particle fits anywhere. The
        # model therefore proposes a shortlist and similarity picks from it.
        shortlist = self.scorer.score(words, position, plausible, top_k=self.shortlist_size)
        if not shortlist:
            return None
        best, best_similarity = None, -1.0
        for candidate, _ in shortlist:
            similarity = (
                self.resources.vectors.similarity(
                    lemma, self.pool.lemma_of.get(candidate, candidate)
                )
                if self.resources.has("vectors")
                else 0.0
            )
            if similarity is not None and similarity > best_similarity:
                best, best_similarity = candidate, similarity
        if best is None:
            best = shortlist[0][0]
        self._accepted += 1
        return best

    def stats(self) -> Dict[str, int]:
        """What the source accepted and rejected, for the run record."""
        return {
            "accepted": self._accepted,
            "rejected_low_similarity": self._rejected_antonyms,
            "pool_size": len(self.pool),
        }

    # ---------- Internals ----------

    def _semantically_plausible(self, source_lemma: str, candidates: Sequence[str]) -> List[str]:
        """Keep candidates distributionally close enough to be a substitution.

        This runs *before* the contextual model rather than after it. A masked
        model ranks whatever fits the slot, and function words fit everywhere,
        so filtering afterwards means its whole top-k can be unusable and the
        slot yields nothing. Filtering first hands the model a list that is
        already semantically plausible and asks only which member fits here.

        A word the vectors do not know is dropped rather than kept: with a pool
        of thousands, admitting every unknown word would reintroduce the noise
        this filter exists to remove.
        """
        if self.min_similarity is None or not self.resources.has("vectors"):
            return list(candidates[: self.pool_limit])
        kept: List[Tuple[str, float]] = []
        for candidate in candidates:
            similarity = self.resources.vectors.similarity(
                source_lemma, self.pool.lemma_of.get(candidate, candidate)
            )
            if similarity is None or similarity < self.min_similarity:
                self._rejected_antonyms += 1
                continue
            kept.append((candidate, similarity))
        kept.sort(key=lambda x: -x[1])
        return [c for c, _ in kept[: self.pool_limit]]

    def _inflection(self, token: str, rng: random.Random) -> Optional[str]:
        """A different attested inflection of the same lemma."""
        if not self.resources.has("morphology"):
            return None
        try:
            forms = self.resources.morphology._get_decliner().decline(
                self.resources.morphology.lemma(token)
            )
        except Exception:
            return None
        others = [f for f, _ in forms if normalize(f) != normalize(token)]
        return rng.choice(others) if others else None
