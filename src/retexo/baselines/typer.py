# retexo/baselines/typer.py
"""The shared post-hoc typer: from links to operations, the same for every row.

Given a link, the operation is nearly free (definition 2.3): the same form
after normalisation is COPY, the same lemma is MORPH, a link with different
lemmas is SUBST refined to a lexical relation by lookup (SYN, POS, NE-SUB),
and an unlinked word is INS or DEL. Every alignment-only row passes through
this rule so that architecture is the only difference between rows. The rule
is an ordered decision list in the ERRANT tradition over the 23-dimensional
evidence vector ``link_features.LinkFeaturizer`` computes, with the two
outcomes ``lemma_missing`` and ``no_rel_found`` recorded where a resource is
silent (Moritz et al. 2016), so that coverage is a number the record carries.

Beside the rule: the frame rule for alignment-only rows (formula templates
from the training folds, a keyword list, and the adjacency rule that a citing
formula ends at most two tokens before the next link), the lookup-derivation
of V2 and V3 labels on the gold links (the same function on gold and on
predictions, so the two never drift), the gate the full system uses (lookup
where it attests, the model where it is silent), the trained head for the
Table 2 ablation as two functions over ``ChangeDetector``, and the ceiling row
of Table 1, ``GoldLinks``, which hands the gold links to the rule.

The rule exists three times in the project (``link_features.SymbolicTyper``,
``run_e24``'s lemma-derived arm, ``run_e26.symbolic_types``); this module is
the one with a name per function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from retexo.baselines import BaselineRegistry, labels
from retexo.baselines.base import Baseline, BaselineConfig, Prediction
from retexo.baselines.record import Record, extra_edges, links_of
from retexo.core.normalize import normalize
from retexo.edit_typing.link_features import FEATURE_NAMES, SymbolicTyper
from retexo.edit_typing.repair import Repairer

DEFAULT_SYN_COS = (
    SymbolicTyper.DEFAULT_SYN_DIST
)  # link_features.py: 0.65, calibrated in calibrate_threshold.py
DEFAULT_NE_FLOOR = SymbolicTyper.DEFAULT_NE_FLOOR  # link_features.py: 0.50
CLAUSE_END = (":", ".", ";", "?", "!")
DEFAULT_FRAME_KEYWORDS = (
    "ait",
    "inquit",
    "dicit",
    "dixit",
    "scribit",
    "scripsit",
    "testatur",
    "loquitur",
    "canit",
    "cecinit",
    "legimus",
    "legitur",
    "dicens",
    "dicente",
)


class RuleTyper:
    """From links to operations, the same rule for every row.

    Example:
        ```python
        tags, outcomes, details = RuleTyper.rule_type(record, links, featurizer)
        frame = RuleTyper.frame_rule(record, links, keywords={"ait"})
        ```
    """

    @staticmethod
    def _feature_dict(phi: Sequence[float]) -> Dict[str, float]:
        return dict(zip(FEATURE_NAMES, phi))

    @classmethod
    def rule_type(
        cls,
        record: Record,
        links: Sequence[int],
        featurizer,
        *,
        syn_cos: float = DEFAULT_SYN_COS,
        ne_floor: float = DEFAULT_NE_FLOOR,
        spelling_fold: bool = True,
    ) -> Tuple[List[str], List[str], List[str]]:
        """The ordered decision list per reuse word: ``(tags, outcomes, details)``.

        Order: unlinked gives ``""`` (INS or FRAME are read off the links and
        the frame by the scorer); same form after ``normalize`` (or the same
        spelling key when ``spelling_fold`` is on) gives COPY; an enclitic on
        exactly one side with matching stems gives SPLIT or MERGE; the same
        lemma gives MORPH; then the lexical tests in order: WordNet synonymy
        or a lemma-vector cosine at or above ``syn_cos`` (SYN), a derivational
        relative (POS), two names (NE-SUB), else SUBST with ``no_rel_found``.
        A hypernym, hyponym or antonym edge is written to ``details`` only.
        The featurizer may be any callable
        ``(source_word, reuse_word, s, t, n_source, n_reuse) -> phi`` over
        ``FEATURE_NAMES``; a stub suffices for tests.
        """
        n_t, n_s = record.n_reuse, record.n_source
        tags, outcomes, details = [""] * n_t, [""] * n_t, [""] * n_t
        for t, s in enumerate(links):
            if s is None or s < 0 or s >= n_s:
                continue
            src, tgt = record.source_tokens[s], record.reuse_tokens[t]
            if normalize(src) == normalize(tgt) or (
                spelling_fold and Repairer.spelling_key(src) == Repairer.spelling_key(tgt)
            ):
                tags[t] = "COPY"
                continue
            f = cls._feature_dict(featurizer(src, tgt, s, t, n_s, n_t))
            if f.get("enclitic_stem_match"):
                tags[t] = (
                    "SPLIT" if f.get("enclitic_src") and not f.get("enclitic_tgt") else "MERGE"
                )
                continue
            if f.get("same_lemma"):
                tags[t] = "MORPH"
                continue
            if f.get("lemma_missing"):
                outcomes[t] = "lemma_missing"
            if f.get("wn_syn") or (
                not f.get("cos_missing") and f.get("cos", 0.0) >= syn_cos and f.get("same_pos", 1)
            ):
                tags[t] = "SYN"
            elif f.get("wn_deriv"):
                tags[t] = "POS"
            elif (
                f.get("both_names")
                and not f.get("cos_missing")
                and f.get("cos", 0.0) >= ne_floor
                or f.get("both_names")
                and f.get("cos_missing")
            ):
                tags[t] = "NE-SUB"
            else:
                tags[t] = "SUBST"
                if not outcomes[t]:
                    outcomes[t] = "no_rel_found"
            for name, relation in (("wn_hyper", "HYPER"), ("wn_hypo", "HYPO"), ("wn_ant", "ANT")):
                if f.get(name) and not details[t]:
                    details[t] = relation
        return tags, outcomes, details

    # ---------- the frame rule ----------

    @staticmethod
    def _sourceless_runs(links: Sequence[int]) -> List[Tuple[int, int]]:
        runs, start = [], None
        for t, s in enumerate(list(links) + [0]):
            unlinked = t < len(links) and (s is None or s < 0)
            if unlinked and start is None:
                start = t
            elif not unlinked and start is not None:
                runs.append((start, t - 1))
                start = None
        return runs

    @classmethod
    def frame_rule(
        cls,
        record: Record,
        links: Sequence[int],
        templates: Sequence[Sequence[str]] = (),
        keywords: Optional[Set[str]] = None,
        *,
        min_len: int = 2,
        max_gap: int = 2,
        one_span: bool = True,
    ) -> List[int]:
        """FRAME flags on sourceless runs: templates, keywords, then adjacency.

        A run is a candidate if it matches a formula template word for word
        (``attest.FrameLexicon``) or contains a keyword; a keyword run is
        extended left to the sentence start (``Repairer.frame_extend`` without
        the colon rule); then ``Repairer.frame_adjacency`` keeps the runs that
        end at most ``max_gap`` tokens before the next link, one span at most.
        """
        n_t = record.n_reuse
        words = [normalize(w) for w in record.reuse_tokens]
        keywords = {
            normalize(k) for k in (keywords if keywords is not None else DEFAULT_FRAME_KEYWORDS)
        }
        flags = [0] * n_t
        keyed = {tuple(normalize(w) for w in tpl) for tpl in templates if len(tpl) >= min_len}
        for a, b in cls._sourceless_runs(links):
            run = words[a : b + 1]
            for length in range(len(run), min_len - 1, -1):
                for start in range(0, len(run) - length + 1):
                    if tuple(run[start : start + length]) in keyed:
                        for t in range(a + start, a + start + length):
                            flags[t] = 1
            hits = [a + i for i, w in enumerate(run) if w in keywords]
            if hits:
                # the formula runs from the start of the run to the first clause-final
                # token after the keyword; without one, the keyword and a following
                # capitalised token (the author's name: *ut ait Maro*)
                k = hits[0]
                end = next(
                    (u for u in range(k, b + 1) if record.reuse_tokens[u].endswith(CLAUSE_END)),
                    None,
                )
                if end is None:
                    end = k + 1 if k + 1 <= b and record.reuse_tokens[k + 1][:1].isupper() else k
                for t in range(a, end + 1):
                    flags[t] = 1
        if any(flags):
            flags = Repairer.frame_extend(
                record.reuse_tokens, list(links), flags, colon_rule=False, extend_left=True
            )
            flags = [f if (links[t] is None or links[t] < 0) else 0 for t, f in enumerate(flags)]
        return Repairer.frame_adjacency(list(links), flags, max_gap=max_gap, one_span=one_span)

    # ---------- derivation on the gold, coverage, the gate ----------

    @classmethod
    def derive_fine(cls, record: Record, featurizer, **dials) -> Record:
        """Write V3 operations on the gold's SUBST edges by lookup; human COPY and MORPH stay.

        Also writes ``detail`` on every refined edge and the provenance fields
        ``fine_ops``, ``lemma_missing`` and ``no_rel_found`` (reuse indices),
        so the coverage of the inventory is a number the record carries.
        """
        links, tags, frame, _ = links_of(record)
        rule_tags, outcomes, details = cls.rule_type(record, links, featurizer, **dials)
        by_r = {}
        for edge in record.links:
            by_r.setdefault(edge.r, []).append(edge)
        for t, s in enumerate(links):
            if s < 0:
                continue
            for edge in by_r.get(t, []):
                if edge.s != s:
                    continue
                if edge.op == "SUBST" and rule_tags[t] in labels.LEXICAL:
                    edge.op = rule_tags[t]
                if details[t] and not edge.detail:
                    edge.detail = details[t]
        record.provenance["fine_ops"] = "resource-lookup"
        record.provenance["lemma_missing"] = [
            t for t, o in enumerate(outcomes) if o == "lemma_missing"
        ]
        record.provenance["no_rel_found"] = [
            t for t, o in enumerate(outcomes) if o == "no_rel_found"
        ]
        return record

    @classmethod
    def annotate_regimes(cls, records: Sequence[Record], featurizer, **dials) -> None:
        """The lookup column beside the gold (definition section 3.6), without touching an operation.

        Per reuse word ``annotation["regime"]`` says what the resources found on
        the gold link -- ``form`` (the same form), ``lemma`` (the same lemma),
        ``relation`` (a WordNet, derivation or name relation), ``lemma_missing``,
        ``no_rel_found`` -- or ``none`` where there is no link; the provenance
        lists ``lemma_missing`` / ``no_rel_found`` feed ``coverage``. Cheap: one
        ``rule_type`` pass per record; the scorer's ``by_regime`` reads it.
        """
        for record in records:
            links, _, _, _ = links_of(record)
            rule_tags, outcomes, _ = cls.rule_type(record, links, featurizer, **dials)
            regimes = []
            for t, s in enumerate(links):
                if s < 0:
                    regimes.append("none")
                elif outcomes[t]:
                    regimes.append(outcomes[t])
                elif rule_tags[t] == "COPY":
                    regimes.append("form")
                elif rule_tags[t] in ("MORPH", "SPLIT", "MERGE"):
                    regimes.append("lemma")
                else:
                    regimes.append("relation")
            record.annotation["regime"] = regimes
            record.annotation["lookup_op"] = list(rule_tags)
            record.provenance["lemma_missing"] = [
                t for t, o in enumerate(outcomes) if o == "lemma_missing"
            ]
            record.provenance["no_rel_found"] = [
                t for t, o in enumerate(outcomes) if o == "no_rel_found"
            ]

    @staticmethod
    def coverage(records: Sequence[Record]) -> Dict[str, float]:
        """Share of gold links a level can name, from the provenance of ``derive_fine``."""
        n_links = sum(len(r.links) for r in records) or 1
        missing = sum(len(r.provenance.get("lemma_missing", [])) for r in records)
        no_rel = sum(len(r.provenance.get("no_rel_found", [])) for r in records)
        per_op: Dict[str, int] = {}
        for r in records:
            for e in r.links:
                per_op[e.op] = per_op.get(e.op, 0) + 1
        out = {
            "links": float(n_links),
            "lemma_missing": missing / n_links,
            "no_rel_found": no_rel / n_links,
            "V2_named": 1.0 - (missing + no_rel) / n_links,
        }
        out.update({f"share_{op}": n / n_links for op, n in sorted(per_op.items())})
        return out

    @staticmethod
    def attested_flags(record: Record, links: Sequence[int], featurizer) -> List[bool]:
        """Per reuse word, whether a resource attests the link's type (``attest.OPEN_TYPES``)."""
        from retexo.edit_typing.attest import Attester

        out = []
        for t, s in enumerate(links):
            if s is None or s < 0:
                out.append(True)
                continue
            phi = featurizer(
                record.source_tokens[s],
                record.reuse_tokens[t],
                s,
                t,
                record.n_source,
                record.n_reuse,
            )
            _, ok = Attester.attest_type(phi)
            out.append(bool(ok))
        return out

    @staticmethod
    def gate(
        rule_tags: Sequence[str], attested: Sequence[bool], model_tags: Sequence[str]
    ) -> List[str]:
        """Lookup where it attests, the model on the residual (``run_e26.gate``)."""
        return [r if ok else m for r, ok, m in zip(rule_tags, attested, model_tags)]

    # ---------- the trained head (Table 2 ablation) ----------

    @staticmethod
    def fit_typer_head(
        records: Sequence[Record],
        links_per_record: Sequence[Sequence[int]],
        featurizer,
        cfg: BaselineConfig,
        *,
        synthetic=None,
        use_evidence: bool = True,
        log=None,
    ):
        """Fine-tune the encoder plus the typer MLP on the given links; returns the ``ChangeDetector``."""
        from dataclasses import replace

        from retexo.baselines.record import record_to_example
        from retexo.formulations.change_detector import ChangeDetector, ChangeDetectorConfig

        examples = []
        for record, links in zip(records, links_per_record):
            example = record_to_example(record, featurizer)
            examples.append(replace(example, alignments=list(links)))
        if synthetic:
            examples = list(synthetic) + examples
        config = ChangeDetectorConfig(
            base_model=cfg.base_model,
            pooling="mean",
            device=cfg.device,
            seed=cfg.seed,
            epochs=max(1, cfg.epochs if not cfg.smoke else 1),
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_length,
            # the examples spell the copy ``NOP`` (``GoldPair`` / ``fine_from_gold``); a head whose classes said
            # ``COPY`` never saw a copy target (dry run 2026-09-16: every copy typed SUBST) -- ``type_with_head``
            # canonicalises NOP back to COPY on the way out
            pointer=False,
            operations=(),
            fine_operations=("NOP",) + tuple(op for op in labels.EDGE_OPS if op != "COPY"),
            feature_dim=len(FEATURE_NAMES),
            use_link_features=use_evidence,
            frame_head=False,
            typer_hidden=int(cfg.extra.get("typer_hidden", 256)),
            typer_lr=float(cfg.extra.get("typer_lr", 1e-3)),
            group_loss_weight=float(cfg.extra.get("group_loss_weight", 1.0)),
        )
        model = ChangeDetector(config)
        model.fit(examples, log=log)
        return model

    @staticmethod
    def type_with_head(model, record: Record, links: Sequence[int], featurizer) -> List[str]:
        """The trained head's tag per reuse word at the given links, in the record's vocabulary."""
        from retexo.baselines.record import record_to_example

        example = record_to_example(record, featurizer)
        tags = model.predict_typed([example], [list(links)], featurizer)[0]
        out = []
        for t, tag in enumerate(tags):
            if links[t] is None or links[t] < 0:
                out.append("")
            else:
                op, _ = labels.canonical(tag)
                out.append(op or "SUBST")
        return out


# =============================================================================
# The ceiling row
# =============================================================================


@BaselineRegistry.register
class GoldLinks(Baseline):
    """Table 1's first row: the gold links through the rule typer.

    Example:
        ```python
        method = GoldLinks(BaselineConfig(device="cpu"))
        pred = method.postprocess(record, method.predict([record])[0], {})
        # python run_baseline.py --method gold_links --fold 4 --smoke 20 --device cpu
        ```
    """

    name = "gold_links"
    emits = "edges"
    trainable = False
    typer = "rule"

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """The rule typer on the gold links; the gold frames stay, no frame rule."""
        from retexo.baselines.adapters import PredictionAdapter

        return PredictionAdapter.type_prediction(
            pred, record, self.typer, self.featurizer, frame_rule="none"
        )

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            links, _, frame, _ = links_of(record)
            pred = Prediction.empty(record.n_reuse)
            pred.links = list(links)
            pred.frame = list(frame)
            pred.extra = extra_edges(record)
            out.append(pred)
        return out


class FrameKeywords:
    """The citing-formula keyword list."""

    @staticmethod
    def load(path: Optional[Path] = None) -> Set[str]:
        """A file with one keyword per line, else the built-in seeds."""
        path = Path(path) if path else Path("retexo/resources/frame_keywords.txt")
        if path.exists():
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        return set(DEFAULT_FRAME_KEYWORDS)


#: Backward-compatible module-level aliases; ``typ.rule_type(...)`` etc. still work.
rule_type = RuleTyper.rule_type
frame_rule = RuleTyper.frame_rule
derive_fine = RuleTyper.derive_fine
coverage = RuleTyper.coverage
attested_flags = RuleTyper.attested_flags
gate = RuleTyper.gate
fit_typer_head = RuleTyper.fit_typer_head
type_with_head = RuleTyper.type_with_head
load_keywords = FrameKeywords.load
