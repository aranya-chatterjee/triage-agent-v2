"""Smoke tests: prove the project installs and imports correctly."""

import pytest

import triage
from triage.config import require


def test_package_imports():
    assert triage.__version__ == "0.1.0"


def test_require_returns_value():
    assert require("abc", "X") == "abc"


def test_require_explains_missing_key():
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN is not set"):
        require(None, "GITHUB_TOKEN")
