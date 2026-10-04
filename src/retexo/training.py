# retexo/training.py
"""
The three-phase training schedule.

A model is not fine-tuned once but three times, on progressively more realistic
and more adversarial data:

1. **Synthetic** variants, where the script is correct by construction and
   difficulty is set directly.
2. **Real reuse**, scripts the oracle derives from annotated pairs, which is
   where the model meets attested quotation practice.
3. **Reuse against non-reuse**, mixing those pairs one-to-one with pairs that
   are not reuses at all.

The third phase is the one most easily left out and the one the detection claim
depends on. A model that has only ever seen genuine reuse will produce a
confident, inexpensive script for any pair it is shown, because it has never
encountered a pair with nothing to explain. Its cost would then fail to
separate reuse from non-reuse, which is precisely what the distance is claimed
to do.

Targets for negatives need no new output format: run on an unrelated pair, the
oracle already produces the correct answer, an expensive script of deletions
and insertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from retexo.datasets.dataset import LabelledPair
from retexo.formulations.base import ScriptExample, ScriptModel
from retexo.datasets.localize import Localized

# =============================================================================
# Phases
# =============================================================================


@dataclass(frozen=True)
class Phase:
    """One training phase: a name and the examples it uses."""

    name: str
    examples: Sequence[ScriptExample]

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class ScheduleReport:
    """What each phase actually trained on, for the run record."""

    sizes: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, List[float]] = field(default_factory=dict)

    def as_flat_dict(self) -> Dict[str, float]:
        """Flat scalars in the repo's metric-key convention."""
        flat: Dict[str, float] = {}
        for name, size in self.sizes.items():
            flat[f"phase_{name}_examples"] = size
        for name, losses in self.losses.items():
            if losses:
                flat[f"phase_{name}_final_loss"] = losses[-1]
        return flat


# =============================================================================
# The schedule
# =============================================================================


class TrainingSchedule:
    """Builds the three phases and runs a model through them in order.

    Example:
        ```python
        phases = TrainingSchedule.build_phases(synthetic, positives, negatives)
        report = TrainingSchedule.train_in_phases(model, phases, log=print)
        ```
    """

    @staticmethod
    def oracle_examples(
        pairs: Sequence[LabelledPair],
        oracle,
        *,
        origin: str = "oracle",
        on_error: str = "skip",
        localize_spans: bool = True,
    ) -> List[ScriptExample]:
        """Derive a script for each pair with the oracle.

        Args:
            on_error: ``"skip"`` drops pairs the oracle cannot plan; ``"raise"``
                surfaces the failure. Skipping is the default because a single
                unparseable passage should not end a run, but the count is
                reported so silent loss is visible.
            localize_spans: Narrow each pair to its reused span before planning.
                On by default because a benchmark pair is two passages and the
                reuse is often a fragment of the longer one, so supervision built
                from whole passages teaches that reuse is mostly insertion. See
                :mod:`retexo.datasets.localize`.
        """
        out: List[ScriptExample] = []
        for pair in pairs:
            source, target = pair.source.split(), pair.target.split()
            if not source or not target:
                continue
            span = Localized.find(source, target, enabled=localize_spans)
            try:
                script = oracle.plan_tokens(span.source, span.target)
            except Exception:
                if on_error == "raise":
                    raise
                continue
            out.append(
                ScriptExample(span.source, span.target, script, origin=origin)
            )
        return out

    @staticmethod
    def teacher_examples(
        pairs: Sequence[LabelledPair],
        oracle,
        *,
        origin: str = "teacher",
        on_error: str = "skip",
        aligner=None,
        aligner_threshold: float = 0.40,
    ) -> List[ScriptExample]:
        """Relabel pairs with the teacher: full passages, FRAME outside the window.

        The counterpart of :meth:`oracle_examples` for the redesign. Passages
        are kept whole, so the examples match what a model sees at inference,
        and the material outside the located window is described as FRAME per
        contiguous run instead of being trimmed away or charged per token.
        """
        from retexo.datasets.teacher import RelabellingTeacher

        teacher = RelabellingTeacher(aligner=aligner, aligner_threshold=aligner_threshold)
        out: List[ScriptExample] = []
        for pair in pairs:
            source, target = pair.source.split(), pair.target.split()
            if not source or not target:
                continue
            try:
                script, fragment_span = teacher.relabel_pair(source, target, oracle)
            except Exception:
                if on_error == "raise":
                    raise
                continue
            out.append(ScriptExample(
                source, target, script, origin=origin,
                framed=any(op.tag == "FRAME" for op in script.operations),
                fragment_span=fragment_span,
            ))
        return out

    @staticmethod
    def build_phases(
        synthetic: Sequence[ScriptExample],
        positives: Sequence[ScriptExample],
        negatives: Sequence[ScriptExample],
    ) -> List[Phase]:
        """Assemble the three phases from their parts.

        The third phase interleaves positives and negatives rather than
        concatenating them, so that a batch is unlikely to be all of one kind.
        """
        mixed: List[ScriptExample] = []
        for index in range(max(len(positives), len(negatives))):
            if index < len(positives):
                mixed.append(positives[index])
            if index < len(negatives):
                mixed.append(negatives[index])
        return [
            Phase("synthetic", list(synthetic)),
            Phase("real", list(positives)),
            Phase("mixed", mixed),
        ]

    @staticmethod
    def train_in_phases(
        model: ScriptModel,
        phases: Sequence[Phase],
        *,
        log: Optional[Callable[[str], None]] = None,
    ) -> ScheduleReport:
        """Fine-tune a model through every phase in order.

        Each phase continues from the previous one's weights, so the schedule
        is cumulative rather than three independent runs.
        """
        report = ScheduleReport()
        for phase in phases:
            if not phase.examples:
                if log:
                    log(f"    phase {phase.name}: empty, skipped")
                report.sizes[phase.name] = 0
                continue
            before = len(model.log.losses)
            model.fit(phase.examples)
            report.sizes[phase.name] = len(phase)
            report.losses[phase.name] = model.log.losses[before:]
            if log:
                final = report.losses[phase.name]
                log(f"    phase {phase.name}: {len(phase)} examples, "
                    f"loss {final[0]:.3f} -> {final[-1]:.3f}" if final
                    else f"    phase {phase.name}: {len(phase)} examples")
        return report


#: Backward-compatible module-level aliases; attic scripts call these directly.
oracle_examples = TrainingSchedule.oracle_examples
teacher_examples = TrainingSchedule.teacher_examples
build_phases = TrainingSchedule.build_phases
train_in_phases = TrainingSchedule.train_in_phases
