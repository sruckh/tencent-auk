"""Stage 04 acceptance — the `--test_input` local harness as a real subprocess
(no RunPod SDK, no GPU): `python3 handler.py --test_input '<json>'`."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, AUK_TEST_MOCK_ENGINE="1", PYTHONUNBUFFERED="1")


def _run(payload: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "handler.py", "--test_input", payload],
        cwd=REPO_ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _last_json(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


def test_runpod_local_success():
    proc = _run(json.dumps({"input": {"instruction": "hello from the harness"}}))
    assert proc.returncode == 0, proc.stderr
    result = _last_json(proc.stdout)
    assert result["delivery"] == "base64"
    assert result["task_executed"] == "instruct_tts"
    assert result["sample_rate"] == 24000
    assert "[auk-engine] mock mode" in proc.stdout  # startup health line


def test_runpod_local_validation_failure():
    proc = _run(json.dumps({"input": {}}))
    assert proc.returncode == 0  # job failure is output, not a crash
    result = _last_json(proc.stdout)
    assert result["error"]["code"] == "missing_required_field"


def test_runpod_local_zero_shot_golden():
    clip = "c2hvcnQtY2xpcA=="  # base64 of "short-clip"
    proc = _run(
        json.dumps(
            {
                "input": {
                    "task": "zero_shot_tts",
                    "instruction": "Say it with this voice.",
                    "prompt_audio": clip,
                    "prompt_text": "short clip",
                    "seed": 42,
                }
            }
        )
    )
    assert proc.returncode == 0, proc.stderr
    result = _last_json(proc.stdout)
    assert result["task_executed"] == "zero_shot_tts"
    assert result["model_variant"] == "flash"


def test_runpod_local_bad_json_exits_nonzero():
    proc = _run("{broken json")
    assert proc.returncode == 1
