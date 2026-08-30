from __future__ import annotations

import pathlib
import sys

import pytest

# Make `functions/` importable so `rmsync` resolves as it does in Lambda.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "functions"))

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def page_rm() -> bytes:
    return (FIXTURES / "synthetic_page.rm").read_bytes()


@pytest.fixture
def blank_rm() -> bytes:
    return (FIXTURES / "synthetic_blank.rm").read_bytes()


@pytest.fixture
def legacy_v5_rm() -> bytes:
    return (FIXTURES / "synthetic_legacy_v5.rm").read_bytes()
