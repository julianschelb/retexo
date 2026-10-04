# retexo/llm/llm_benchmark.py
"""
Is an LLM a good source of labels?

Renamed from the preliminary experiment module E32; kept under its own name
(rather than archived) because the LLM-baselines note plans to import this
module's benchmark and scoring machinery, unchanged, when the prompt-only and
fine-tuned rows are built. ``TypeJob.to_coarse``/``COARSE`` use the raw
hand-label spelling (``NOP``), not ``retexo.baselines.labels``' canonical
spelling (``COPY``) -- the same NOP/COPY boundary the harness's own adapters
cross deliberately at their edge, not something this rename tries to unify.

Two jobs, both scored automatically so a prompt can be judged in minutes:

**type** (:class:`TypeJob`) -- name the relation on a real link. The test set
is the reader's adjudicated verdicts on fold 4
(`data/gold_full/fine_verdicts_fold4.json`, 36 links), joined to a typed-links
dump that carries the passages and the trained typer's answer on the same
link. So every prompt is scored against a hand label *and* against the model
we already have.

**propose** (:class:`ProposeJob`) -- for a Latin lemma, propose related words
(stage 1 of the generative kickstart). Scored where Latin WordNet does answer
(agreement), and by the generator's own mechanical filters (attested in the
corpus, right part of speech, not the same lemma), which is what would gate a
proposal anyway.

Example:
    ```python
    from retexo.llm.llm_benchmark import TypeJob

    job = TypeJob.from_dump(dump_path, verdicts_path)
    answers = [reply_to_prompt(job.render_prompt(item)) for item in job]
    scores = job.score(answers)
    job.print_table(answers, scores, label="T9")
    ```
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Sequence

from retexo.llm.llm_prompts import TAGS

LENIENT = {"SYN-DIST": "SYN"}
BASELINES = ("hybrid", "gated", "symbolic")

#: The hand labels name only three classes. Any named relation implies the word
#: was replaced, so every fine tag but NOP and MORPH folds into SUBST.
COARSE = ("NOP", "MORPH", "SUBST")


# =============================================================================
# Parsing a reply
# =============================================================================


class TagParser:
    """Reads an LLM reply into an answer, one method per parser named by a
    :class:`~retexo.llm.llm_prompts.PromptVariant`'s ``parse`` field.

    :attr:`PARSERS` maps every parser name used by ``llm_prompts`` to the
    method that implements it, so a caller only needs the name: ``TagParser.PARSERS[variant.parse](reply)``.
    """

    @staticmethod
    def tag(text: str) -> Optional[str]:
        """The first tag mentioned anywhere (the model often answers with one word)."""
        up = text.upper()
        hits = [(up.find(t), t) for t in sorted(TAGS, key=len, reverse=True) if t in up]
        if not hits:
            return None
        # a longer tag containing a shorter one (SYN-DIST vs SYN) wins at the same position
        pos = min(p for p, _ in hits)
        return sorted([t for p, t in hits if p == pos], key=len, reverse=True)[0]

    @classmethod
    def last_tag(cls, text: str) -> Optional[str]:
        """The tag on the last non-empty line, falling back to :meth:`tag` on the whole text."""
        for line in reversed([l.strip() for l in text.strip().splitlines() if l.strip()]):
            found = cls.tag(line)
            if found:
                return found
        return cls.tag(text)

    @staticmethod
    def _json_block(text: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    @classmethod
    def json_relations(cls, text: str) -> Dict[str, List[str]]:
        """The four relation lists of a ``propose`` reply, glossed or not."""
        d = cls._json_block(text) or {}
        out = {}
        for rel in ("SYN", "HYPER", "HYPO", "ANT"):
            got = d.get(rel) or d.get(rel.lower()) or []
            words = []
            for w in got if isinstance(got, list) else []:
                if isinstance(w, str):
                    words.append(w.strip())
                elif isinstance(w, (list, tuple)) and w and isinstance(w[0], str):
                    words.append(w[0].strip())          # ["ensis", "sword"]
            out[rel] = [w for w in words if w]
        return out

    PARSERS: ClassVar[Dict[str, Callable[[str], object]]] = {}


TagParser.PARSERS = {
    "tag": TagParser.tag,
    "last_tag": TagParser.last_tag,
    "json_relations": TagParser.json_relations,
    "json_relations_glossed": TagParser.json_relations,
}
#: Kept for callers that want the mapping without naming the class.
PARSERS = TagParser.PARSERS


# =============================================================================
# type: items, scoring, reporting
# =============================================================================


class TypeJob:
    """The **type** job: items with a gold relation, and scoring against them.

    An item carries the passages, the link, the gold answer, and (for
    comparison) what each of the existing typers already said. Build one with
    :meth:`from_dump`, then score any list of model answers against it.

    Attributes:
        items: One dict per scored link (see :meth:`from_dump`).
        coarse: Whether ``items`` were built (and answers are scored) in the
            hand labels' three-class space instead of the fine relations.

    Example:
        ```python
        job = TypeJob.from_dump(dump_path, verdicts_path, limit=40)
        scores = job.score(["SYN", "MORPH", ...])
        job.print_table(answers, scores, label="T9")
        ```
    """

    def __init__(self, items: List[dict], *, coarse: bool = False):
        self.items = items
        self.coarse = coarse

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:
        return f"TypeJob(n={len(self.items)}, coarse={self.coarse})"

    @staticmethod
    def to_coarse(tag: Optional[str]) -> Optional[str]:
        """A fine tag as the hand labels see it: unchanged, inflected, or replaced."""
        if tag is None:
            return None
        return tag if tag in ("NOP", "MORPH") else "SUBST"

    @staticmethod
    def fold_tag(tag: Optional[str], lenient: bool) -> Optional[str]:
        """``tag``, or its lenient fold (``SYN-DIST`` -> ``SYN``) when ``lenient``."""
        if tag is None:
            return None
        return LENIENT.get(tag, tag) if lenient else tag

    @classmethod
    def from_dump(
        cls, dump: Path, verdicts: Path, *, limit: int = 0, seed: int = 1, coarse: bool = False
    ) -> "TypeJob":
        """The links with a label and their passages, plus the typers' answers.

        Default: the reader's adjudicated fine relations (a few dozen, the honest
        test of naming). ``coarse``: every link the dump carries, labelled by the
        hand labels' three classes -- eight times the sample, and the question a
        label *source* has to get right before anything else.
        """
        rows = [json.loads(l) for l in Path(dump).read_text().splitlines() if l.strip()]
        truth = json.loads(Path(verdicts).read_text())
        items = []
        for r in rows:
            if not r.get("link_found"):
                continue
            key = f"{r['pair']}:{r['source']}>{r['target']}"
            if coarse:
                gold = cls.to_coarse(r.get("gold"))
                if gold is None:
                    continue
            else:
                gold = truth.get(key) or truth.get(f"{r['pair']}:{r['t']}")
                if gold in (None, "?"):
                    continue
            items.append({"key": key, "pair": r["pair"], "ref_type": r.get("ref_type"),
                          "source": r["source"], "target": r["target"],
                          "context_source": r.get("context_source", ""),
                          "context_target": r.get("context_target", ""),
                          "gold": gold,
                          **{b: (cls.to_coarse(r.get(b)) if coarse else r.get(b)) for b in BASELINES}})
        items.sort(key=lambda d: d["key"])
        if limit and len(items) > limit:
            items = random.Random(seed).sample(items, limit)
            items.sort(key=lambda d: d["key"])
        return cls(items, coarse=coarse)

    def score(self, answers: Sequence[Optional[str]]) -> dict:
        """Accuracy of ``answers`` and of every baseline on the same items, strict
        and lenient (SYN-DIST folded into SYN), with the confusion of the answers."""
        items = self.items
        out = {"n": len(items), "parsed": sum(1 for a in answers if a in TAGS), "coarse": self.coarse}
        if self.coarse:
            answers = [self.to_coarse(a) for a in answers]
        for lenient in (False, True):
            key = "lenient" if lenient else "strict"
            right = sum(1 for it, a in zip(items, answers)
                        if a is not None and self.fold_tag(a, lenient) == self.fold_tag(it["gold"], lenient))
            out[f"llm_{key}"] = right / max(len(items), 1)
            for b in BASELINES:
                hit = sum(1 for it in items
                          if it.get(b) and self.fold_tag(it[b], lenient) == self.fold_tag(it["gold"], lenient))
                out[f"{b}_{key}"] = hit / max(len(items), 1)
        conf = Counter()
        per_gold = defaultdict(Counter)
        for it, a in zip(items, answers):
            conf[(it["gold"], a or "UNPARSED")] += 1
            per_gold[it["gold"]][a or "UNPARSED"] += 1
        out["confusion"] = {f"{g}->{p}": n for (g, p), n in sorted(conf.items(), key=lambda kv: -kv[1])}
        out["by_gold"] = {g: dict(c) for g, c in sorted(per_gold.items())}
        return out

    def per_operation(self, answers: Sequence[Optional[str]], *, lenient: bool = False) -> Dict[str, tuple]:
        """Per gold operation: (right, total) for ``answers``."""
        per = defaultdict(lambda: [0, 0])
        for it, a in zip(self.items, answers):
            per[it["gold"]][1] += 1
            per[it["gold"]][0] += int(a is not None and self.fold_tag(a, lenient) == self.fold_tag(it["gold"], lenient))
        return {k: tuple(v) for k, v in per.items()}

    def print_table(self, answers: Sequence[Optional[str]], scores: dict, *, label: str, model: str = "") -> None:
        """One table for this run: the row for ``label`` and the rows for the
        typers we already have, with accuracy per operation beside the totals."""
        items = self.items
        ops = [op for op, _ in sorted(Counter(it["gold"] for it in items).items(), key=lambda kv: -kv[1])]
        rows = [(label, self.per_operation(answers), scores["llm_strict"], scores["llm_lenient"])]
        for who, name in (("gated", "Our typer (gated)"), ("symbolic", "Lemma rule (symbolic)"),
                          ("hybrid", "Our typer (ungated)")):
            rows.append((name, self.per_operation([it.get(who) for it in items]),
                         scores[who + "_strict"], scores[who + "_lenient"]))
        width = max(len(r[0]) for r in rows) + 2
        head = (f"\n  {'':<{width}}{'Accuracy (strict)':>19}{'Accuracy (lenient)':>20}   "
                + "  ".join(f"{op:>9}" for op in ops))
        print(f"  parsed {scores['parsed']}/{scores['n']} replies" + (f"   model: {model}" if model else ""))
        print(head)
        for name, per, strict, lenient in rows:
            cells = "  ".join(f"{(f'{per[op][0]}/{per[op][1]}' if op in per else '-'):>9}" for op in ops)
            print(f"  {name:<{width}}{strict:>19.3f}{lenient:>20.3f}   {cells}")
        print("  gold counts: " + ", ".join(f"{op} {sum(1 for it in items if it['gold'] == op)}" for op in ops))
        worst = [f"{k} {v}" for k, v in list(scores["confusion"].items())[:6] if not k.split("->")[0] == k.split("->")[1]]
        print("  commonest errors (gold -> said): " + ", ".join(worst[:5]))


# =============================================================================
# propose: items and scoring
# =============================================================================


class ProposeJob:
    """The **propose** job: Latin lemmas Latin WordNet answers for, so a
    proposal can be checked against something.

    WordNet's Latin is thin and partly wrong (its antonyms of *mors* are
    *abortum, adgnatio*), so agreement with it is a weak proxy for a prompt's
    quality, not a measure of a proposal's truth -- the hand check is.

    Example:
        ```python
        job = ProposeJob.from_resources(resources, lemmas, limit=32)
        scores = job.score(answers, attested=corpus_vocabulary)
        ```
    """

    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:
        return f"ProposeJob(n={len(self.items)})"

    @classmethod
    def from_resources(
        cls, resources, lemmas: Sequence[str], *, limit: int = 32, seed: int = 1, pos: str = "n"
    ) -> "ProposeJob":
        """Sample lemmas from ``lemmas`` that Latin WordNet has a record for."""
        rng = random.Random(seed)
        pool = list(dict.fromkeys(lemmas))
        rng.shuffle(pool)
        fields = {"SYN": "synonyms", "HYPER": "hypernyms", "HYPO": "hyponyms", "ANT": "antonyms"}
        items = []
        for lemma in pool:
            try:
                record = resources.wordnet.lookup(lemma, pos)
            except Exception:
                continue
            known = {rel: sorted({w.lower() for w in record.get(field, []) if w})
                     for rel, field in fields.items() if record.get(field)}
            if known:
                items.append({"lemma": lemma, "pos": pos, "known": known})
            if len(items) >= limit:
                break
        return cls(items)

    def score(self, answers: Sequence[Dict[str, List[str]]], *, attested: Optional[set] = None) -> dict:
        """Agreement with WordNet where WordNet answers, and the share of proposals
        that are attested in the corpus at all (the first mechanical filter)."""
        proposed = confirmed = in_corpus = 0
        per_rel = defaultdict(lambda: {"proposed": 0, "confirmed": 0})
        covered = 0
        for it, ans in zip(self.items, answers):
            for rel, words in (ans or {}).items():
                known = {w.lower() for w in it["known"].get(rel, [])}
                for w in words:
                    proposed += 1
                    per_rel[rel]["proposed"] += 1
                    if attested is not None and w.lower() in attested:
                        in_corpus += 1
                    if w.lower() in known:
                        confirmed += 1; per_rel[rel]["confirmed"] += 1
            if any((ans or {}).get(rel) for rel in it["known"]):
                covered += 1
        return {"items": len(self.items), "proposals": proposed,
                "wordnet_confirmed": confirmed / max(proposed, 1),
                "attested_in_corpus": (in_corpus / max(proposed, 1)) if attested is not None else None,
                "items_with_an_answer": covered / max(len(self.items), 1),
                "per_relation": {r: {**v, "rate": v["confirmed"] / max(v["proposed"], 1)}
                                 for r, v in sorted(per_rel.items())}}


# =============================================================================
# Extra input for the model: what the resources say, and a word-by-word gloss
# =============================================================================


class EvidenceAnnotator:
    """Turns a :class:`~retexo.edit_typing.link_features.LinkFeaturizer`'s
    evidence for one link into prose, for a prompt that shows its work (T12-T15).

    Example:
        ```python
        annotator = EvidenceAnnotator(featurizer)
        job.items and annotator.attach(job.items, gold_pairs)
        ```
    """

    def __init__(self, featurizer):
        self.featurizer = featurizer

    def sentences(
        self, source_word: str, target_word: str, *,
        s_index: int = 0, t_index: int = 0, n_source: int = 1, n_target: int = 1,
    ) -> str:
        """The typer's own evidence for this cell, in words. Same 23 features the
        model reads as a vector, so an LLM given this block sees what the typer sees."""
        from retexo.edit_typing.link_features import FEATURE_NAMES

        featurizer = self.featurizer
        f = dict(zip(FEATURE_NAMES, featurizer(source_word, target_word, s_index, t_index,
                                                n_source, n_target)))
        ls, lt = featurizer.lemma(source_word), featurizer.lemma(target_word)
        rels = [name for name, key in (("synonym", "wn_syn"), ("more general", "wn_hyper"),
                                       ("more specific", "wn_hypo"), ("opposite", "wn_ant"),
                                       ("same derivational family", "wn_deriv")) if f.get(key, 0) > 0]
        lines = [
            f"- lemma of the source word: {ls or 'not found'}",
            f"- lemma of the reuse word: {lt or 'not found'}",
            "- same word form: " + ("yes" if f["same_form"] else "no"),
            "- same lemma, different form: " + ("yes" if f["same_lemma"] else "no"),
            "- Latin WordNet: " + (", ".join(rels) if rels else
                                   ("no record for these lemmas" if f.get("wn_missing", 0) else "no relation recorded")),
            "- similarity of the lemma vectors: " + ("not available" if f.get("cos_missing", 0)
                                                     else f"{f['cos']:.2f} (0 = unrelated, 1 = identical)"),
            f"- spelling: shared prefix {f['prefix_ratio']:.2f}, edit similarity {f['edit_ratio']:.2f},"
            f" length ratio {f['len_ratio']:.2f}",
            "- part of speech agrees: " + ("yes" if f["same_pos"] else "no"),
            "- proper names: " + ("both" if f["both_names"] else "one" if f["either_name"] else "neither"),
            f"- position in the passage: {f['rel_position']:.2f} (0.5 = the same slot)",
        ]
        return "\n".join(lines)

    def attach(self, items: Sequence[dict], gold_pairs) -> None:
        """Fill each item's ``flags`` field. ``gold_pairs`` maps pair id -> the GoldPair,
        for the passage lengths the position feature needs."""
        for it in items:
            g = gold_pairs.get(it["pair"])
            n_s = len(g.source_tokens) if g is not None else 1
            n_t = len(g.target_tokens) if g is not None else 1
            it["flags"] = self.sentences(it["source"], it["target"],
                                         s_index=it.get("s", 0), t_index=it.get("t", 0),
                                         n_source=n_s, n_target=n_t)

    @staticmethod
    def parse_gloss(text: str) -> str:
        """The model's word-by-word rendering, cleaned of any preamble."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        keep = [l for l in lines if "=" in l or "-" in l or ":" in l]
        return "\n".join(keep[:40]) if keep else text.strip()[:1200]
