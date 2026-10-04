# retexo/__init__.py
"""
Typed edit scripts for Latin intertextual reuse.

Building a variant and checking it round-trips:

```python
from retexo import VariantBuilder, Scriba

builder = VariantBuilder("uox faucibus haesit")
builder.keep(0).syn(1, "gutture").keep(2)
variant, script = builder.build()

Scriba().verify(script, "uox faucibus haesit".split(), variant)   # True
Scriba().execute(script.invert(), variant)                        # back to the source
```
"""
from __future__ import annotations

__version__ = "0.1.0"

from retexo.core.builder import VariantBuilder
from retexo.core.oracle import (
    Aligner,
    EditPlanOracle,
    GreedyAligner,
    OptimalAligner,
    OracleConfig,
)
from retexo.resources import Resources
from retexo.core.normalize import normalize
from retexo.operations import (
    ALL_OPERATIONS,
    EditOperation,
    Level,
    Operation,
    OperationRegistry,
    Role,
)
from retexo.core.scriba import Scriba, Validity
from retexo.core.script import CostModel, EditScript

__all__ = [
    "ALL_OPERATIONS",
    "Aligner",
    "EditPlanOracle",
    "GreedyAligner",
    "OptimalAligner",
    "OracleConfig",
    "Resources",
    "CostModel",
    "EditOperation",
    "EditScript",
    "Level",
    "Operation",
    "OperationRegistry",
    "Role",
    "Scriba",
    "Validity",
    "VariantBuilder",
    "normalize",
]
