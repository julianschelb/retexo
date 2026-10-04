"""Where retexo looks for data, caches and runs.

Everything that is not code (the benchmark and annotation files, the resource caches, the run folders) lives under one
directory, the working directory by default. Set ``RETEXO_HOME`` to point somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """The directory that holds ``data/``, ``resources_cache/`` and ``runs/``."""
    return Path(os.environ.get("RETEXO_HOME") or Path.cwd()).resolve()


__all__ = ["home"]
