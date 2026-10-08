# datasets/synthetic/__init__.py
"""
The typed synthetic generator: shaped like real reuse, typed like real reuse.

Two preliminary experiments, merged. **E4** (``attic/retexo/e4.py``)
established the shape: four operations decidable without a lexicon (``COPY``,
``SUBST``, ``INS``, ``DEL``), generated to the measured statistics of real
reuse -- short scattered fragments inside real prose, not label balance alone
-- because a generator matching only the label mix reached synthetic macro
0.98 while gold macro sat at 0.075. **E24** kept that shape and added
everything it left out: typed links (the substitute's tag -- ``MORPH``,
``SYN``, ``HYPER``, ``HYPO``, ``ANT``, ``SYN-DIST``, ``NE-SUB``, ``POS`` --
carried through with its evidence vector), ``SPLIT``/``MERGE`` (an enclitic
detached or attached), ``FRAME`` (an attribution formula drawn from the
*training folds'* hand-labelled frames), and switchable ``REORDER``.
``QUOTE``, ``ADAPT`` and ``DISPERSE`` need no construction: they fall out of
the alignment at decode time, on synthetic and real pairs alike.

The two are one system in practice -- E24 always calls through to E4's
distributions and its acceptance calibration -- so they live in one module
under one name. What did not survive the merge: E4's own single-typed
generator ``generate`` (superseded by ``generate_typed`` below), its report
dataclass, and its realism-comparison helpers' now-dead call sites; those stay
in ``attic/retexo/e4.py`` for the record. Everything importable from here
today was load-bearing before the merge.

    pool, report = generate_typed(seeds, context, frames=frame_pool(...), ...)

or, through the thin config object:

    generator = TypedGenerator(GeneratorConfig(frames=frame_pool(...)))
    pool, report = generator.generate(seeds, context)

The one module is a package since the readability split, and every name it
exported is re-exported here, so ``from retexo.datasets.synthetic import X``
keeps working for every consumer. The submodules, one concern each:

- ``shape`` -- the untyped shape (E4): the measured targets, the spelling
  variant, the realism report, the degenerate baseline.
- ``inventory`` -- the typed inventory (E24): fine tags, substitution
  weights, event rates, the rare and relation-bearing tags.
- ``substitution`` -- attestation, acceptance calibration, the untyped
  probe and the typed substitution source.
- ``pools`` -- the frame templates and the pool selection with its rare-tag
  floor.
- ``one_pair`` -- one typed pair: the report dataclass, the spelling and
  enclitic events, the pair constructor ``_make_one_typed``.
- ``parallel`` -- the multiprocess generation (``generate_typed``).
- ``evidence`` -- the per-pair evidence grid (``featurize_pairs``) and the
  switchable state it reads.
- ``gold_examples`` -- a hand-labelled pair as a typed example
  (``fine_from_gold``).
- ``generator`` -- the reusable object: ``GeneratorConfig`` and
  ``TypedGenerator``.

The switchable evidence state -- ``DEP_FEATURES_ON``, ``FRAME_CHANNEL_ON``,
the shared ``_dep_parser`` -- stays settable through this package exactly as
it was through the flat module: reads and writes are routed to ``evidence``,
the module whose functions read them.
"""

from __future__ import annotations

import sys
import types

from retexo.datasets.synthetic import evidence
from retexo.datasets.synthetic.evidence import (
    _FEAT,
    _feat_chunk,
    _feat_init,
    featurize_pairs,
    frame_support_channel,
)
from retexo.datasets.synthetic.generator import GeneratorConfig, TypedGenerator
from retexo.datasets.synthetic.gold_examples import _link_only_coarse, fine_from_gold
from retexo.datasets.synthetic.inventory import (
    ENCLITIC_RATE,
    ENCLITICS,
    FINE_OPERATIONS,
    FRAME_RATE,
    RARE_TAGS,
    RELATED_TAGS,
    RESIDUAL_MAX_COS,
    TYPED_WEIGHTS,
)
from retexo.datasets.synthetic.one_pair import (
    _SPELLING_RULES,
    TypedReport,
    _attach_enclitic,
    _make_one_typed,
    spelling_variant,
)
from retexo.datasets.synthetic.parallel import (
    _MLM_CACHE,
    _WORKER,
    _chunk,
    _init_worker,
    generate_typed,
)
from retexo.datasets.synthetic.pools import balanced_subset, fine_mix, frame_pool, select_pool
from retexo.datasets.synthetic.shape import (
    _ANYWHERE,
    _INITIAL,
    _TRAILING,
    COPY_VARIANT_SHARE,
    FRAGMENT_LENGTHS,
    FRAGMENTS,
    OPERATIONS,
    TARGET,
    _draw,
    _fragments,
    _variant,
    all_insert_baseline,
    coarse,
    log_realism,
    realism_report,
)
from retexo.datasets.synthetic.substitution import (
    _make_substitute,
    _make_typed_substitute,
    attested_forms,
    calibrate_acceptance,
)

__all__ = [
    "OPERATIONS",
    "TARGET",
    "FRAGMENTS",
    "FRAGMENT_LENGTHS",
    "COPY_VARIANT_SHARE",
    "attested_forms",
    "calibrate_acceptance",
    "coarse",
    "realism_report",
    "log_realism",
    "all_insert_baseline",
    "FINE_OPERATIONS",
    "TYPED_WEIGHTS",
    "RESIDUAL_MAX_COS",
    "FRAME_RATE",
    "ENCLITIC_RATE",
    "ENCLITICS",
    "RARE_TAGS",
    "select_pool",
    "balanced_subset",
    "fine_mix",
    "frame_pool",
    "TypedReport",
    "generate_typed",
    "featurize_pairs",
    "fine_from_gold",
    "GeneratorConfig",
    "TypedGenerator",
]

# The switchable evidence state (the two flags, the shared dependency parser)
# is module state that callers set through this package -- ``synthetic.DEP_FEATURES_ON
# = True`` in the baselines, a monkeypatched ``_dep_parser`` in the tests -- while
# the functions that read it live in ``evidence``. The package routes reads and
# writes there, so the flat-module semantics are unchanged.

_STATE = ("DEP_FEATURES_ON", "FRAME_CHANNEL_ON", "_DEP_PARSER", "_dep_parser")


class _StateModule(types.ModuleType):
    """The package's module type, routing the switchable evidence state to ``evidence``."""

    def __getattr__(self, name):
        if name in _STATE:
            return getattr(evidence, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _STATE:
            setattr(evidence, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _StateModule
