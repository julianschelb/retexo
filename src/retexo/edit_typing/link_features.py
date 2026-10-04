# retexo/edit_typing/link_features.py
"""
Symbolic evidence for one link: what the resources say about a word pair.

The pointer decides *which* source word a reuse word came from, and it is good
at that. Deciding *what kind* of change the link records is a different
question, and E2 showed that an encoder alone cannot answer it: synonyms,
hypernyms and antonyms drawn from WordNet scored 0.17-0.38 against chance 0.125,
and the inventory was cut to one SUBST class as a result.

That was a finding about evidence, not about the operations. A lemmatizer knows
whether two forms share a lemma; a wordnet knows whether one lemma is listed as
the other's hypernym; a name list knows whether both tokens are names. The typer
should see all of that beside the contextual vectors and learn how far to trust
each -- which is the "predict coarse, refine symbolically" outcome the roadmap
named for E2 and nobody chose.

Every feature is either a flag or a number in [0, 1], with an explicit
*missing* flag wherever a resource can be silent, so that "the wordnet has no
entry" is never confused with "the wordnet says no".

    featurizer = LinkFeaturizer(resources)
    featurizer(source_word, reuse_word, s_index, t_index, n_source, n_reuse)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from retexo.core.normalize import normalize

# =============================================================================
# The feature vector
# =============================================================================

FEATURE_NAMES: Tuple[str, ...] = (
    "same_form",          # identical after normalization
    "same_lemma",         # same lemma, different surface
    "lemma_missing",      # the lemmatizer returned nothing usable
    "wn_syn",             # target lemma in source lemma's synonyms (either way)
    "wn_hyper",           # target is a hypernym of source
    "wn_hypo",            # target is a hyponym of source
    "wn_ant",             # antonyms
    "wn_deriv",           # derivational family (POS shift)
    "wn_any",             # any of the above
    "wn_missing",         # no wordnet record for the source lemma at all
    "cos",                # lemma-vector cosine, clipped to [0, 1]
    "cos_missing",        # either lemma has no vector
    "both_names",         # both tokens on the proper-name list
    "either_name",
    "enclitic_src",       # source ends in -que / -ue / -ne (a real enclitic)
    "enclitic_tgt",
    "enclitic_stem_match",  # stem of one side matches the other side
    "prefix_ratio",       # shared prefix / longer length
    "edit_ratio",         # 1 - levenshtein / longer length
    "len_ratio",          # shorter / longer
    "same_case",          # both capitalized or both not
    "same_pos",           # first part-of-speech guess agrees
    "rel_position",       # 0.5 + (t/n_t - s/n_s) / 2, so 0.5 = same slot
)

N_FEATURES = len(FEATURE_NAMES)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class LinkFeaturizer:
    """Computes the evidence vector for a (source word, reuse word) pair.

    Resource calls are cached per word and per pair, because the lemmatizer is
    the slow part and the same words recur across a batch of pairs.
    """

    def __init__(self, resources=None):
        self.resources = resources
        self._lemma: Dict[str, str] = {}
        self._pos: Dict[str, str] = {}
        self._name: Dict[str, bool] = {}
        self._enclitic: Dict[str, Optional[Tuple[str, str]]] = {}
        self._wordnet: Dict[str, Dict[str, set]] = {}
        self._pair: Dict[Tuple[str, str], List[float]] = {}

    # ---------- per-word lookups, cached ----------

    def lemma(self, token: str) -> str:
        if token not in self._lemma:
            value = ""
            if self.resources is not None and self.resources.has("morphology"):
                try:
                    value = self.resources.morphology.lemma(token) or ""
                except Exception:
                    value = ""
            self._lemma[token] = value
        return self._lemma[token]

    def pos(self, token: str) -> str:
        if token not in self._pos:
            value = ""
            if self.resources is not None and self.resources.has("morphology"):
                try:
                    candidates = self.resources.pos_candidates(token)
                    value = candidates[0] if candidates else ""
                except Exception:
                    value = ""
            self._pos[token] = value
        return self._pos[token]

    def is_name(self, token: str) -> bool:
        if token not in self._name:
            value = False
            if self.resources is not None and self.resources.has("entities"):
                try:
                    value = bool(self.resources.entities.is_name(token))
                except Exception:
                    value = False
            self._name[token] = value
        return self._name[token]

    def enclitic(self, token: str) -> Optional[Tuple[str, str]]:
        if token not in self._enclitic:
            value = None
            if self.resources is not None and self.resources.has("morphology"):
                try:
                    value = self.resources.morphology.split_enclitic(token)
                except Exception:
                    value = None
            self._enclitic[token] = value
        return self._enclitic[token]

    def wordnet(self, token: str) -> Dict[str, set]:
        """Union over the part-of-speech guesses of the relations of a lemma."""
        lemma = self.lemma(token)
        if lemma in self._wordnet:
            return self._wordnet[lemma]
        record: Dict[str, set] = {"synonyms": set(), "hypernyms": set(),
                                  "hyponyms": set(), "antonyms": set(),
                                  "derivatives": set(), "_present": set()}
        if lemma and self.resources is not None and self.resources.has("wordnet"):
            try:
                for pos in (self.resources.pos_candidates(token) or ("n", "v", "a")):
                    found = self.resources.wordnet.lookup(lemma, pos) or {}
                    if found:
                        record["_present"].add(pos)
                    for key in ("synonyms", "hypernyms", "hyponyms", "antonyms",
                                "derivatives"):
                        record[key].update(normalize(w) for w in found.get(key, ()))
            except Exception:
                pass
        self._wordnet[lemma] = record
        return record

    def cosine(self, a: str, b: str) -> Optional[float]:
        if self.resources is None or not self.resources.has("vectors"):
            return None
        la, lb = self.lemma(a) or normalize(a), self.lemma(b) or normalize(b)
        try:
            return self.resources.vectors.similarity(la, lb)
        except Exception:
            return None

    # ---------- the vector ----------

    def __call__(self, source: str, target: str, s_index: int = 0, t_index: int = 0,
                 n_source: int = 1, n_target: int = 1) -> List[float]:
        key = (source, target)
        if key in self._pair:
            base = list(self._pair[key])
        else:
            base = self._pair_features(source, target)
            self._pair[key] = list(base)
        # position is a property of the link, not of the words
        rel = 0.5 + ((t_index / max(n_target, 1)) - (s_index / max(n_source, 1))) / 2.0
        base.append(min(max(rel, 0.0), 1.0))
        return base

    def _pair_features(self, source: str, target: str) -> List[float]:
        ns, nt = normalize(source), normalize(target)
        same_form = float(ns == nt)
        ls, lt = self.lemma(source), self.lemma(target)
        lemma_missing = float(not ls or not lt)
        same_lemma = float(bool(ls) and ls == lt and not same_form)

        rec_s, rec_t = self.wordnet(source), self.wordnet(target)
        lt_n, ls_n = normalize(lt), normalize(ls)
        wn_syn = float(lt_n in rec_s["synonyms"] or ls_n in rec_t["synonyms"])
        wn_hyper = float(lt_n in rec_s["hypernyms"] or ls_n in rec_t["hyponyms"])
        wn_hypo = float(lt_n in rec_s["hyponyms"] or ls_n in rec_t["hypernyms"])
        wn_ant = float(lt_n in rec_s["antonyms"] or ls_n in rec_t["antonyms"])
        wn_deriv = float(lt_n in rec_s["derivatives"] or ls_n in rec_t["derivatives"])
        wn_any = float(any((wn_syn, wn_hyper, wn_hypo, wn_ant, wn_deriv)))
        wn_missing = float(not rec_s["_present"])

        cos = self.cosine(source, target)
        cos_missing = float(cos is None)
        cos_value = min(max(cos, 0.0), 1.0) if cos is not None else 0.0

        name_s, name_t = self.is_name(source), self.is_name(target)
        enc_s, enc_t = self.enclitic(source), self.enclitic(target)
        stem_match = 0.0
        if enc_s and normalize(enc_s[0]) == nt:
            stem_match = 1.0
        if enc_t and normalize(enc_t[0]) == ns:
            stem_match = 1.0
        if enc_s and lt and self.lemma(enc_s[0]) == lt and not same_form:
            stem_match = 1.0
        if enc_t and ls and self.lemma(enc_t[0]) == ls and not same_form:
            stem_match = 1.0

        longer = max(len(ns), len(nt), 1)
        prefix_ratio = _prefix(ns, nt) / longer
        edit_ratio = 1.0 - _levenshtein(ns, nt) / longer
        len_ratio = min(len(ns), len(nt)) / longer

        cap = lambda w: bool(re.match(r"^[A-Z]", re.sub(r"^[^A-Za-z]+", "", w) or ""))
        same_case = float(cap(source) == cap(target))
        ps, pt = self.pos(source), self.pos(target)
        same_pos = float(bool(ps) and ps == pt)

        return [same_form, same_lemma, lemma_missing,
                wn_syn, wn_hyper, wn_hypo, wn_ant, wn_deriv, wn_any, wn_missing,
                cos_value, cos_missing,
                float(name_s and name_t), float(name_s or name_t),
                float(enc_s is not None), float(enc_t is not None), stem_match,
                prefix_ratio, edit_ratio, len_ratio, same_case, same_pos]


# =============================================================================
# Symbolic typing, for the lookup-only baseline
# =============================================================================

class SymbolicTyper:
    """Names a link from its evidence vector alone, with no learned parameter.

    Returns ``"SUBST"`` when nothing attests a relation -- the honest residual,
    which is what the hand labels call it too.

    Args:
        syn_dist: Cosine floor above which an unattested pair is called
            ``SYN-DIST`` rather than ``SUBST``.
        ne_floor: Cosine floor above which two proper names are called
            ``NE-SUB`` rather than ``SUBST``.

    Example:
        ```python
        typer = SymbolicTyper()
        typer(featurizer(source, target))   # "MORPH", "SYN", "SUBST", ...
        ```
    """

    #: The order a lookup decides in. Identity first, then the cheapest relation
    #: that a resource attests, then the distributional fallbacks. This is what the
    #: EditPlan oracle does by other means, expressed over the same evidence the
    #: typer sees, so that "hybrid minus symbolic" is a like-for-like number.
    ORDER = ("NOP", "MERGE", "SPLIT", "MORPH", "SYN", "ANT", "HYPER",
             "HYPO", "POS", "NE-SUB", "SYN-DIST")

    DEFAULT_SYN_DIST = 0.65
    DEFAULT_NE_FLOOR = 0.50

    def __init__(self, *, syn_dist: float = DEFAULT_SYN_DIST, ne_floor: float = DEFAULT_NE_FLOOR):
        self.syn_dist = syn_dist
        self.ne_floor = ne_floor

    def __call__(self, features: List[float]) -> str:
        f = dict(zip(FEATURE_NAMES, features))
        if f["same_form"]:
            return "NOP"
        if f["enclitic_stem_match"]:
            return "SPLIT" if f["enclitic_src"] and not f["enclitic_tgt"] else "MERGE"
        if f["same_lemma"]:
            return "MORPH"
        if f["wn_syn"]:
            return "SYN"
        if f["wn_ant"]:
            return "ANT"
        if f["wn_hyper"]:
            return "HYPER"
        if f["wn_hypo"]:
            return "HYPO"
        if f["wn_deriv"]:
            return "POS"
        if f["both_names"] and not f["cos_missing"] and f["cos"] >= self.ne_floor:
            return "NE-SUB"
        if not f["cos_missing"] and f["cos"] >= self.syn_dist:
            return "SYN-DIST"
        return "SUBST"
