# formulations/change_detector/__init__.py
"""
E0: does a reuse token differ from its source token, yes or no.

The simplest thing the tagging architecture can be asked, and deliberately so.
With one-to-one operations the answer is computable by string comparison, which
makes this a **plumbing check with a known ceiling** rather than a test of
whether the model understands Latin: if it cannot reach a score that ``!=``
reaches, the pair encoding, the word-to-subword alignment or the label mapping
is broken, and every later result would inherit the fault.

Only one head, and no pointer. The pointer is not in the loop here, so its
128-position cap cannot contaminate the measurement.

The one variable is where a word's representation comes from. 59% of Latin
words split into more than one subword under XLM-R, and the tagging formulation
labels the **first** piece — so the model judges *uirumque* from ``▁u``, and
*primarius* from ``▁primari`` while its source counterpart ``▁primus`` is one
unsplit token. ``pooling="mean"`` averages the word's pieces instead.

The detector now lives in a package; every public name the flat module
exported is re-exported here, so ``from retexo.formulations.change_detector
import X`` keeps working for every consumer:

- ``ChangeDetectorConfig`` -- the knobs (``config.py``)
- ``ChangeExample`` -- one labelled pair (``example.py``)
- ``ChangeDetector`` -- the model (``detector.py``, composed from the
  mixins in ``encoding.py``, ``losses.py``, ``training.py`` and
  ``prediction.py``)
- ``LEXICAL_TAGS`` and ``GROUP_TARGET`` -- the typer's constants
  (``constants.py``)
- ``evidence_veto`` -- the module-level alias of
  ``ChangeDetector.evidence_veto``
"""

from __future__ import annotations

from retexo.formulations.change_detector.config import ChangeDetectorConfig
from retexo.formulations.change_detector.constants import GROUP_TARGET, LEXICAL_TAGS
from retexo.formulations.change_detector.detector import ChangeDetector, evidence_veto
from retexo.formulations.change_detector.example import ChangeExample

__all__ = ["ChangeDetector", "ChangeDetectorConfig", "ChangeExample"]
