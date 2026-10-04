# retexo/baselines/__init__.py
"""The baseline package: one API for every method row of the paper.

Every method of Table 1 and Table 2 is a ``Baseline`` (``base.py``) that reads
the record of the definition's section 1.3 (``record.py``), emits a
``Prediction``, and is decoded, typed and scored by shared code (``adapters``,
``decoder``, ``typer``, ``scorer``). Methods register themselves under their
``name`` when their module is imported; the driver ``run_baseline.py`` imports
this package and looks the method up in ``REGISTRY``.

The package holds no model logic of its own. It wraps what ``retexo``
already has and adds only what the paper's tables need and the notes in
``Literature/Implementation Notes`` specify.
"""

from __future__ import annotations

from typing import Dict, Type


class BaselineRegistry:
    """One API for every method row of the paper.

    Methods register themselves under their ``name`` when their module is
    imported (``@BaselineRegistry.register`` on the class); the driver
    ``run_baseline.py`` looks the method up here.

    Example:
        ```python
        BaselineRegistry.load_all()
        method = BaselineRegistry.get("copy_input")
        ```
    """

    #: Every registered baseline, by its ``name``.
    methods: Dict[str, Type] = {}

    #: Modules that register a baseline on import, in ``load_all`` order.
    MODULES = ("floors", "typer", "em_aligner", "sultan_aligner", "sim_aligner",
               "span_aligner", "typed_pointer", "tagger", "span_pair",
               "refinement", "nmt_aligner", "full_system", "llm", "stored_rows")

    @classmethod
    def register(cls, baseline_cls):
        """Class decorator: enter ``baseline_cls`` under its ``name`` attribute."""
        if not getattr(baseline_cls, "name", ""):
            raise ValueError(f"baseline class {baseline_cls.__name__} has no name")
        if baseline_cls.name in cls.methods and cls.methods[baseline_cls.name] is not baseline_cls:
            raise ValueError(f"baseline name {baseline_cls.name!r} registered twice")
        cls.methods[baseline_cls.name] = baseline_cls
        return baseline_cls

    @classmethod
    def get(cls, name: str):
        """The registered class for ``name``; a clear error otherwise."""
        if name not in cls.methods:
            raise KeyError(f"unknown baseline {name!r}; known: {sorted(cls.methods)}")
        return cls.methods[name]

    @classmethod
    def load_all(cls) -> Dict[str, Type]:
        """Import every method module so that ``methods`` is complete."""
        import importlib

        for module in cls.MODULES:
            try:
                importlib.import_module(f"retexo.baselines.{module}")
            except ModuleNotFoundError as error:
                if error.name != f"retexo.baselines.{module}":
                    raise
        return cls.methods


#: Backward-compatible module-level aliases.
REGISTRY = BaselineRegistry.methods
register = BaselineRegistry.register
get = BaselineRegistry.get
load_all = BaselineRegistry.load_all
