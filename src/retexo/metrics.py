# retexo/metrics.py
"""
Scoring predicted scripts.

A script is compared with the gold script as a *set of operations*, matched on
two criteria and reported as precision, recall and F1 for each. This follows
the scheme ERRANT established for grammatical error correction, where edits are
matched by span and error type, and which Seq2Edits reports as span-based
precision, recall and F0.5.

The choice needs stating because the editing literature offers no alternative:
Seq2Edits, GECToR, FELIX and CoEdIT all evaluate the *text* their edits
produce, since there the script is a means and the corrected sentence is the
product. Here the script is the product, so a text-level metric would measure
the wrong object.

Precision and recall are kept separate because they diagnose opposite failures
that a single figure hides. High precision with low recall is a model
predicting few operations and defaulting to copies; low precision with high
recall is a model firing everywhere and drowning the script in insertions and
deletions — the degenerate solution the cost model exists to prevent.

**exec-match does not compare formulations.** A tagging model is given the
reuse as input and predicts an alignment over it, so replaying its output
reproduces the reuse by construction and an untrained model scores 1.0. It is
therefore reported only for formulations that write the script, and is ``None``
elsewhere rather than a number that measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from retexo.core.scriba import Scriba
from retexo.core.script import EditScript

# =============================================================================
# Scores
# =============================================================================


@dataclass(frozen=True)
class PRF:
    """Precision, recall and F1 over a set comparison."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    @classmethod
    def from_counts(cls, matched: int, predicted: int, gold: int) -> "PRF":
        """Build from match counts, treating an empty side as zero."""
        precision = matched / predicted if predicted else 0.0
        recall = matched / gold if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return cls(precision, recall, f1)


@dataclass
class ScriptMetrics:
    """Every score for one set of predictions."""

    n: int = 0
    invalid: int = 0
    alignment: PRF = field(default_factory=PRF)
    typed: PRF = field(default_factory=PRF)
    per_operation: Dict[str, PRF] = field(default_factory=dict)
    copy_only: PRF = field(default_factory=PRF)
    degenerate: PRF = field(default_factory=PRF)
    macro_typed_f1: float = 0.0
    link_strict: float = 0.0
    link_lenient: float = 0.0
    exec_match: Optional[float] = None

    def as_flat_dict(self, prefix: str = "test_") -> Dict[str, float]:
        """Flat scalars, in the repo's metric-key convention."""
        flat: Dict[str, float] = {
            f"{prefix}n": self.n,
            f"{prefix}invalid_rate": self.invalid / self.n if self.n else 0.0,
            f"{prefix}alignment_precision": self.alignment.precision,
            f"{prefix}alignment_recall": self.alignment.recall,
            f"{prefix}alignment_f1": self.alignment.f1,
            f"{prefix}typed_precision": self.typed.precision,
            f"{prefix}typed_recall": self.typed.recall,
            f"{prefix}typed_f1": self.typed.f1,
            f"{prefix}copy_only_f1": self.copy_only.f1,
            f"{prefix}degenerate_f1": self.degenerate.f1,
            f"{prefix}macro_typed_f1": self.macro_typed_f1,
            f"{prefix}link_strict": self.link_strict,
            f"{prefix}link_lenient": self.link_lenient,
            f"{prefix}link_gap": self.link_lenient - self.link_strict,
        }
        if self.exec_match is not None:
            flat[f"{prefix}exec_match"] = self.exec_match
        for tag, prf in sorted(self.per_operation.items()):
            key = tag.lower().replace("-", "_")
            flat[f"{prefix}op_{key}_f1"] = prf.f1
        return flat


# =============================================================================
# Evaluation
# =============================================================================


class ScriptScorer:
    """Scores predicted scripts against gold scripts.

    Example:
        ```python
        metrics = ScriptScorer.evaluate(predictions, gold)
        print(metrics.as_flat_dict())
        ```
    """

    @staticmethod
    def _aligned_items(script: EditScript) -> Tuple[Set[tuple], Set[tuple]]:
        """The operations of a script as two comparable sets.

        The first ignores the operation tag and captures only which source
        position produced which reuse position; the second includes the tag.
        Marker operations carry no reuse position and are excluded, since they
        annotate a region rather than account for one.
        """
        alignment: Set[tuple] = set()
        typed: Set[tuple] = set()
        for op in script.operations:
            sources = op.source_indices or (None,)
            targets = op.target_indices or (None,)
            if sources == (None,) and targets == (None,):
                continue
            for position, target in enumerate(targets):
                source = sources[min(position, len(sources) - 1)]
                alignment.add((source, target))
                typed.add((source, target, op.tag))
        return alignment, typed

    @staticmethod
    def _copy_only(script: EditScript) -> EditScript:
        """The trivial script: copy aligned positions, insert or delete the rest."""
        from retexo.operations import EditOperation

        source, target = script.source_tokens, script.target_tokens
        shared = min(len(source), len(target))
        operations = [
            EditOperation("NOP", (i,), (i,), (source[i],), (target[i],))
            for i in range(shared)
        ]
        operations += [
            EditOperation("INS", (), (j,), (), (target[j],))
            for j in range(shared, len(target))
        ]
        operations += [
            EditOperation("DEL", (i,), (), (source[i],), ())
            for i in range(shared, len(source))
        ]
        return EditScript(list(source), list(target), operations, script.registry)

    @staticmethod
    def _degenerate(script: EditScript) -> EditScript:
        """Delete every source token, insert every reuse token, relate nothing.

        The script that explains any pair whatsoever and therefore explains none.
        It is reported on every evaluation because it is the floor a real system
        has to clear, and because it is invisible to a text-level metric: replaying
        it reproduces the reuse exactly. Where the gold scripts are dominated by
        insertions and deletions, this scores well without containing a single
        claim about how the two passages are related.
        """
        from retexo.operations import EditOperation

        source, target = script.source_tokens, script.target_tokens
        operations = [
            EditOperation("DEL", (i,), (), (token,), ())
            for i, token in enumerate(source)
        ]
        operations += [
            EditOperation("INS", (), (j,), (), (token,))
            for j, token in enumerate(target)
        ]
        return EditScript(list(source), list(target), operations, script.registry)

    @staticmethod
    def _links(script: EditScript) -> Dict[int, int]:
        """Reuse position to source position, for operations carrying both."""
        links: Dict[int, int] = {}
        for op in script.operations:
            if op.source_indices and op.target_indices:
                for position, target in enumerate(op.target_indices):
                    links[target] = op.source_indices[
                        min(position, len(op.source_indices) - 1)
                    ]
        return links

    @classmethod
    def evaluate(
        cls,
        predictions: Sequence[Optional[EditScript]],
        gold: Sequence[EditScript],
        scriba: Optional[Scriba] = None,
        *,
        generative: bool = True,
    ) -> ScriptMetrics:
        """Score predictions against gold scripts.

        Args:
            generative: Whether the formulation writes the script. When ``False``,
                ``exec_match`` is left unset, since replay is trivially satisfied
                for a model that is handed the reuse.

        A ``None`` prediction counts as invalid and contributes no matches, so the
        invalid rate cannot be improved by declining to answer.
        """
        scriba = scriba or Scriba()
        result = ScriptMetrics(n=len(gold))

        matched_a = predicted_a = gold_a = 0
        matched_t = predicted_t = gold_t = 0
        copy_m = copy_p = copy_g = 0
        degen_m = degen_p = 0
        per_tag: Dict[str, List[int]] = {}
        strict_hits = strict_total = lenient_hits = 0
        exec_hits = 0

        for predicted, reference in zip(predictions, gold):
            gold_align, gold_typed = cls._aligned_items(reference)
            gold_a += len(gold_align)
            gold_t += len(gold_typed)

            baseline_align, baseline_typed = cls._aligned_items(cls._copy_only(reference))
            copy_m += len(baseline_typed & gold_typed)
            copy_p += len(baseline_typed)
            copy_g += len(gold_typed)

            _, degenerate_typed = cls._aligned_items(cls._degenerate(reference))
            degen_m += len(degenerate_typed & gold_typed)
            degen_p += len(degenerate_typed)

            for _, _, tag in gold_typed:
                per_tag.setdefault(tag, [0, 0, 0])[2] += 1

            gold_links = cls._links(reference)
            strict_total += len(gold_links)

            if predicted is None:
                result.invalid += 1
                continue

            pred_align, pred_typed = cls._aligned_items(predicted)
            predicted_a += len(pred_align)
            predicted_t += len(pred_typed)
            matched_a += len(pred_align & gold_align)
            matched_t += len(pred_typed & gold_typed)

            for _, _, tag in pred_typed:
                per_tag.setdefault(tag, [0, 0, 0])[1] += 1
            for item in pred_typed & gold_typed:
                per_tag[item[2]][0] += 1

            if generative and scriba.verify(
                predicted, reference.source_tokens, reference.target_tokens
            ):
                exec_hits += 1

            pred_links = cls._links(predicted)
            pred_sources = set(pred_links.values())
            for target, source in gold_links.items():
                if pred_links.get(target) == source:
                    strict_hits += 1
                if source in pred_sources:
                    lenient_hits += 1

        result.alignment = PRF.from_counts(matched_a, predicted_a, gold_a)
        result.typed = PRF.from_counts(matched_t, predicted_t, gold_t)
        result.copy_only = PRF.from_counts(copy_m, copy_p, copy_g)
        result.degenerate = PRF.from_counts(degen_m, degen_p, gold_t)
        result.per_operation = {
            tag: PRF.from_counts(m, p, g) for tag, (m, p, g) in per_tag.items()
        }
        #: Averaged over operations rather than over instances, so a model that
        #: wins on the two majority tags cannot hide behind a micro-average.
        scored = [prf.f1 for tag, prf in result.per_operation.items() if per_tag[tag][2]]
        result.macro_typed_f1 = sum(scored) / len(scored) if scored else 0.0
        result.link_strict = strict_hits / strict_total if strict_total else 0.0
        result.link_lenient = lenient_hits / strict_total if strict_total else 0.0
        if generative:
            result.exec_match = exec_hits / result.n if result.n else 0.0
        return result


#: Backward-compatible module-level alias; several attic scripts and notebooks
#: still call ``metrics.evaluate(...)`` directly.
evaluate = ScriptScorer.evaluate
