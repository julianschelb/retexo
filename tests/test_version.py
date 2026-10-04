"""The version is written in two places; they must agree, or a release publishes the wrong number."""

import re
from pathlib import Path

import retexo

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'(?m)^version = "([^"]+)"', text).group(1)
    assert retexo.__version__ == declared


def test_the_changelog_mentions_the_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"v{retexo.__version__}" in changelog
