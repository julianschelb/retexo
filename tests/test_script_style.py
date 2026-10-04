"""The script-style test modules, each run as ``python tests/test_x.py``: the exit code is the result."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import script_style_modules

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.parametrize("module", script_style_modules(), ids=lambda p: p.stem)
def test_script_runs_clean(module):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    done = subprocess.run([sys.executable, str(module)], capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, f"{module.name} failed:\n{done.stdout[-1500:]}\n{done.stderr[-1500:]}"
