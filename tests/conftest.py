"""Test setup: isolate every test run from the real .env, cache and clips folder.

clipper.config reads the environment at import time, so this must run before
anything imports clipper - conftest.py is loaded first by pytest.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="stream_clipper_tests_"))
os.environ["CLIPPER_ENV"] = str(_TMP / "missing.env")  # never load the real .env
os.environ["CLIPPER_CACHE"] = str(_TMP / "cache")
os.environ["CLIPPER_OUT"] = str(_TMP / "out")
for key in ("TWITCH_CHANNEL", "TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET",
            "TWITCH_TOKEN", "TWITCH_REFRESH_TOKEN"):
    os.environ.pop(key, None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clipper import cli, config  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Each test starts with an empty processed_streams.json."""
    config.PROCESSED_FILE.unlink(missing_ok=True)
    yield
    config.PROCESSED_FILE.unlink(missing_ok=True)


@pytest.fixture
def make_args():
    """Real CLI defaults, with overrides: make_args("--max-clips", "2")."""
    return lambda *argv: cli.parse_args(list(argv))
