"""Stage 03 acceptance — engine.py mock path against the stage 03 output spec
(.icm/stages/03-engine-and-model-lifecycle/output/engine-and-model-lifecycle.md)."""

import io
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

import engine
import schema_validator

REPO_ROOT = Path(__file__).resolve().parent.parent

META_KEYS = {"task_executed", "model_variant", "nfe", "sample_rate", "duration_seconds", "size_bytes"}


def _req(**overrides):
    payload = {"instruction": "A warm narrator reads the news."}
    payload.update(overrides)
    return schema_validator.normalize(payload)


def test_mock_engine_imports_without_torch_or_auk():
    """The mock env path boots on a bare interpreter: no torch, no auk, no numpy."""
    code = (
        "import sys; import engine; "
        "assert engine._MODE == 'mock'; "
        "assert not any(m.startswith('torch') for m in sys.modules), 'torch leaked'; "
        "assert 'auk' not in sys.modules, 'auk leaked'; "
        "assert 'numpy' not in sys.modules, 'numpy leaked'; "
        "print('OK')"
    )
    env = dict(os.environ, AUK_TEST_MOCK_ENGINE="1")
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_synthesize_returns_wav_24k_mono():
    wav, meta = engine.synthesize(_req(gen_seconds=1.0))
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2  # 16-bit PCM


def test_metadata_shape():
    wav, meta = engine.synthesize(_req(gen_seconds=1.0))
    assert set(meta) == META_KEYS
    assert meta["sample_rate"] == 24000
    assert meta["size_bytes"] == len(wav)
    assert isinstance(meta["duration_seconds"], float)
    assert 0.9 <= meta["duration_seconds"] <= 1.1
    assert meta["task_executed"] == "instruct_tts"


def test_variant_dispatch():
    _, flash_meta = engine.synthesize(_req(model_variant="flash"))
    assert flash_meta["model_variant"] == "flash" and flash_meta["nfe"] == 4
    _, base_meta = engine.synthesize(_req(model_variant="base"))
    assert base_meta["model_variant"] == "base" and base_meta["nfe"] == 32


def test_seed_changes_samples_deterministically():
    wav_a1, _ = engine.synthesize(_req(gen_seconds=0.5, seed=7))
    wav_a2, _ = engine.synthesize(_req(gen_seconds=0.5, seed=7))
    wav_b, _ = engine.synthesize(_req(gen_seconds=0.5, seed=8))
    assert wav_a1 == wav_a2  # same seed → identical bytes
    assert wav_a1 != wav_b  # different seed → different bytes


def test_task_executed_follows_request():
    _, meta = engine.synthesize(_req(task="enhancement", instruction="clean", audio="c2hvcnQtY2xpcA=="))
    assert meta["task_executed"] == "enhancement"


def test_temp_file_helpers_round_trip():
    import tempfile

    tmp_dir = tempfile.gettempdir()
    before = set(os.listdir(tmp_dir))
    path = engine._write_temp_audio(b"fake-wav-bytes")
    try:
        assert os.path.exists(path) and path.endswith(".wav")
        assert set(os.listdir(tmp_dir)) - before == {os.path.basename(path)}
    finally:
        engine._remove_temp(path)
    assert not os.path.exists(path)
    assert set(os.listdir(tmp_dir)) == before  # cleanup leaves nothing behind


def test_engine_error_wraps_failures(monkeypatch):
    def boom(_req):
        raise RuntimeError("weights exploded")

    monkeypatch.setattr(engine, "_mock_generate", boom)
    with pytest.raises(engine.EngineError) as excinfo:
        engine.synthesize(_req())
    assert excinfo.value.code == "inference_failed"
    assert "weights exploded" in excinfo.value.message


def test_mock_wav_carries_no_disk_artifacts(tmp_path, monkeypatch):
    """Mock synthesize writes nothing to disk (no hot-path I/O)."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    engine.synthesize(_req())
    assert list(tmp_path.iterdir()) == []
