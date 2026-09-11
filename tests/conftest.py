"""Test-suite fixtures — stage 06 of the ICM pipeline (.icm/).

Sets ``AUK_TEST_MOCK_ENGINE=1`` before any worker module is imported so the
engine boots in mock mode: no GPU, no ``torch``, no ``auk`` package.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["AUK_TEST_MOCK_ENGINE"] = "1"

import pytest  # noqa: E402

import schema_validator  # noqa: E402


@pytest.fixture
def b64_clip() -> str:
    """A tiny valid base64 blob (16 bytes)."""
    import base64

    return base64.b64encode(b"RIFF____WAVEfmt " + b"\x00" * 8).decode("ascii")


@pytest.fixture
def instruct_payload() -> dict[str, object]:
    """Golden instruct_tts payload — the minimal valid request."""
    return {"task": "instruct_tts", "instruction": "A warm narrator reads the news."}


@pytest.fixture
def normalized(instruct_payload):
    return schema_validator.normalize(instruct_payload)
