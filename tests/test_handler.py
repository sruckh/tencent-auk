"""Stage 04 acceptance — handler.py flow, error mapping, crash-dump scrubbing,
and scanner compliance against the stage 04 output spec."""

import json
from pathlib import Path

import engine
import handler
import storage

REPO_ROOT = Path(__file__).resolve().parent.parent

META = {
    "task_executed": "instruct_tts",
    "model_variant": "flash",
    "nfe": 4,
    "sample_rate": 24000,
    "duration_seconds": 1.0,
    "size_bytes": 44,
}
WAV = b"RIFF____WAVE"
BASE64_FIELDS = {"delivery": "base64", "audio_base64": "UklGRg==", "size_bytes": len(WAV)}


def _stub_engine(monkeypatch, wav=WAV, meta=META, exc=None):
    def fake_synthesize(req):
        if exc is not None:
            raise exc
        return wav, meta

    monkeypatch.setattr(engine, "synthesize", fake_synthesize)


def _stub_storage(monkeypatch, fields=None, exc=None, calls=None):
    def fake_deliver(wav_bytes, job_id, response_delivery, cfg=None, client_factory=None):
        if calls is not None:
            calls.append((wav_bytes, job_id, response_delivery))
        if exc is not None:
            raise exc
        return dict(fields or BASE64_FIELDS)

    monkeypatch.setattr(storage, "deliver", fake_deliver)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_handler_success_envelope(monkeypatch):
    _stub_engine(monkeypatch)
    _stub_storage(monkeypatch)
    result = handler.handler({"id": "job-42", "input": {"instruction": "hello"}})
    assert "error" not in result
    assert result["delivery"] == "base64"
    for key, value in META.items():
        assert result[key] == value
    assert result["task_executed"] == "instruct_tts"


def test_handler_passes_job_id_and_delivery_preference(monkeypatch):
    calls = []
    _stub_engine(monkeypatch)
    _stub_storage(monkeypatch, calls=calls)
    payload = {"instruction": "hello", "response_delivery": "s3"}
    handler.handler({"id": "job-42", "input": payload})
    wav_bytes, job_id, response_delivery = calls[0]
    assert job_id == "job-42"
    assert response_delivery == "s3"
    assert wav_bytes == WAV


# ---------------------------------------------------------------------------
# Error mapping — every pinned code
# ---------------------------------------------------------------------------

def test_handler_missing_input_envelope():
    result = handler.handler({"id": "j1"})
    assert result["error"]["code"] == "invalid_payload"


def test_handler_non_dict_input_envelope():
    result = handler.handler({"id": "j1", "input": ["nope"]})
    assert result["error"]["code"] == "invalid_payload"


def test_handler_missing_instruction(monkeypatch):
    _stub_engine(monkeypatch)
    _stub_storage(monkeypatch)
    result = handler.handler({"id": "j1", "input": {"task": "instruct_tts"}})
    assert result["error"]["code"] == "missing_required_field"
    assert result["error"]["field"] == "instruction"


def test_handler_validation_error_no_engine_call(monkeypatch):
    def explode(req):
        raise AssertionError("engine must not run for invalid payloads")

    monkeypatch.setattr(engine, "synthesize", explode)
    result = handler.handler({"id": "j1", "input": {"instruction": "x", "model_variant": "v9"}})
    assert result["error"]["code"] == "unsupported_model_variant"


def test_handler_engine_failure(monkeypatch):
    _stub_engine(monkeypatch, exc=engine.EngineError("inference_failed", "upstream blew up"))
    result = handler.handler({"id": "j1", "input": {"instruction": "x"}})
    assert result == {"error": {"code": "inference_failed", "message": "upstream blew up"}}


def test_handler_delivery_failure(monkeypatch):
    _stub_engine(monkeypatch)
    _stub_storage(monkeypatch, exc=storage.DeliveryError("delivery_failed", "b2 down"))
    result = handler.handler({"id": "j1", "input": {"instruction": "x"}})
    assert result == {"error": {"code": "delivery_failed", "message": "b2 down"}}


def test_handler_s3_credentials_missing(monkeypatch):
    _stub_engine(monkeypatch)
    _stub_storage(monkeypatch, exc=storage.DeliveryError("s3_credentials_missing", "no creds"))
    result = handler.handler({"id": "j1", "input": {"instruction": "x", "response_delivery": "s3"}})
    assert result["error"]["code"] == "s3_credentials_missing"


# ---------------------------------------------------------------------------
# Daemon isolation + crash-dump scrubbing
# ---------------------------------------------------------------------------

def test_handler_daemon_survives_unexpected(monkeypatch, capsys):
    def explode(req):
        raise RuntimeError("catastrophic state corruption")

    monkeypatch.setattr(engine, "synthesize", explode)
    result = handler._safe_handler({"id": "j-crash", "input": {"instruction": "x"}})
    assert result["error"]["code"] == "inference_failed"
    out = capsys.readouterr().out
    assert "Traceback" in out
    assert "j-crash" in out  # job id present for operators
    assert "END CRASH DUMP" in out


def test_crash_dump_scrubbed(monkeypatch, capsys):
    """A Secret travelling through a failure path renders masked, never raw."""
    secret = storage.Secret("super-secret-aws-key")

    def explode(req):
        raise RuntimeError(f"boto3 rejected {secret}")

    monkeypatch.setattr(engine, "synthesize", explode)
    result = handler._safe_handler({"id": "j2", "input": {"instruction": "x"}})
    assert result["error"]["code"] == "inference_failed"
    out = capsys.readouterr().out
    assert "super-secret-aws-key" not in out
    assert "Secret(***" in out


def test_scanner_strings_present():
    source = (REPO_ROOT / "handler.py").read_text()
    assert "import runpod" in source
    assert "runpod.serverless.start(" in source


# ---------------------------------------------------------------------------
# Local harness (in-process half; subprocess half in test_runpod_local.py)
# ---------------------------------------------------------------------------

def test_run_local_test_prints_json(monkeypatch, capsys):
    _stub_engine(monkeypatch)
    _stub_storage(monkeypatch)
    code = handler._run_local_test(
        ["handler.py", "--test_input", json.dumps({"input": {"instruction": "hi"}})]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["delivery"] == "base64"


def test_run_local_test_validation_failure_is_output(monkeypatch, capsys):
    code = handler._run_local_test(["handler.py", "--test_input", json.dumps({"input": {}})])
    assert code == 0  # job failure is output, not process crash
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["error"]["code"] == "missing_required_field"


def test_run_local_test_bad_json_exits_nonzero(capsys):
    assert handler._run_local_test(["handler.py", "--test_input", "{broken"]) == 1
    assert handler._run_local_test(["handler.py", "--test_input"]) == 1


def test_startup_logs_delivery_config(capsys, monkeypatch):
    """Boot log exposes delivery state without ever rendering a credential."""
    import storage

    monkeypatch.setattr(handler.engine, "_log_system_info", lambda: None)
    monkeypatch.setattr(storage, "_CFG", storage.StorageConfig.from_env({}))
    handler._log_system_info()
    out = capsys.readouterr().out
    assert "[auk-worker] delivery: default=auto, s3_configured=False" in out
    assert "Secret" in out or "AKIA" not in out  # no credential material either way
