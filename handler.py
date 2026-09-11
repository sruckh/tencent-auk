"""RunPod serverless handler for the Tencent AuK worker.

Job lifecycle per `.icm/references/runpod-invariants.md` (stage 04 of the ICM
pipeline in `.icm/`): module-scope engine bootstrap via ``import engine``, the
structured error envelope for every handled failure, scrubbed crash dumps for
unhandled ones, and a self-contained ``--test_input`` local harness.

Static scanner compliance: this file literally contains ``import runpod`` and
the ``runpod.serverless.start(...)`` call.
"""

from __future__ import annotations

import json
import sys
import traceback

try:
    import runpod  # pyright: ignore[reportMissingImports]  # platform SDK — production image installs it (stage 05)
except ImportError:  # CI / local harness runs without the SDK
    runpod = None  # type: ignore[assignment]

import engine  # noqa: E402  — module-scope bootstrap happens HERE, at import
import schema_validator  # noqa: E402
import storage  # noqa: E402


def _log_delivery_config() -> None:
    """Boot probe (module scope — the engine line comes from engine.py's own
    bootstrap, this one from ours): `s3_configured` proves whether endpoint
    credentials reached the worker env; never renders their values."""
    cfg = storage._CFG
    print(
        f"[auk-worker] delivery: default={cfg.delivery_default}, "
        f"s3_configured={cfg.s3_configured()}, bucket={cfg.bucket or '-'}, "
        f"endpoint={'set' if cfg.endpoint_url else 'unset'}",
        flush=True,
    )


def _log_system_info() -> None:  # noqa: F810 — explicit startup hook
    """Startup health verification; operators may re-invoke it."""
    engine._log_system_info()
    _log_delivery_config()


_log_delivery_config()  # runs unconditionally at import — deploy probe


def _crash_dump(job_id: str, exc: BaseException) -> None:
    """Print the full traceback + context to stdout before returning the error
    envelope. Scrubbed by construction: credentials only ever travel as
    `Secret` objects (masked repr) and audio bytes are never interpolated."""
    print(f"=== AUK WORKER CRASH DUMP — job_id={job_id} ===", flush=True)
    print(f"exception: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc(file=sys.stdout)
    print("=== END CRASH DUMP ===", flush=True)


def handler(job: dict[str, object]) -> dict[str, object]:
    """One RunPod job → one response dict. Never raises for a job failure."""
    job = job or {}
    job_id = str(job.get("id") or "local")
    payload = job.get("input")

    try:
        if not isinstance(payload, dict):
            raise schema_validator.ValidationError(
                "invalid_payload", "job must carry a JSON object under 'input'", field="input"
            )
        req = schema_validator.normalize(payload)
    except schema_validator.ValidationError as exc:
        return exc.to_dict()

    try:
        wav_bytes, metadata = engine.synthesize(req)
    except engine.EngineError as exc:
        return {"error": {"code": exc.code, "message": exc.message}}

    try:
        delivery = storage.deliver(wav_bytes, job_id, req.response_delivery)
    except storage.DeliveryError as exc:
        return {"error": {"code": exc.code, "message": exc.message}}

    return {**delivery, **metadata}


def _log_result_shape(job_id: str, result: dict[str, object]) -> None:
    """Deploy probe: prove the exact payload the SDK will serialize. Logs
    field names/types and JSON byte size — never audio data, URLs, or the
    presign signature. Catches the two silent killers of job-done delivery:
    non-serializable values and NaN/Infinity (json.dumps emits bare ``NaN``,
    which strict gateways reject with 400)."""
    try:
        rendered = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        print(
            f"[auk-worker] result job_id={job_id} NOT JSON-SERIALIZABLE: "
            f"{type(exc).__name__}: {exc}; fields={sorted(result)}",
            flush=True,
        )
        return
    flags = []
    if "NaN" in rendered or "Infinity" in rendered:
        flags.append("CONTAINS-NAN/INFINITY")
    print(
        f"[auk-worker] result job_id={job_id} json_bytes={len(rendered)} "
        f"fields={{ {', '.join(f'{k}: {type(v).__name__}' for k, v in sorted(result.items()))} }}"
        f"{' | ' + ' '.join(flags) if flags else ''}",
        flush=True,
    )


def _safe_handler(job: dict[str, object]) -> dict[str, object]:
    """Daemon-safe wrapper: an unexpected exception produces a crash dump and a
    structured error — the process never dies from a job (PRD NFR 2)."""
    job_id = str((job or {}).get("id") or "local")
    result: dict[str, object]
    try:
        result = handler(job)
    except Exception as exc:  # noqa: BLE001 — last-resort isolation
        _crash_dump(job_id, exc)
        result = {
            "error": {
                "code": "inference_failed",
                "message": "unhandled worker exception (see crash dump)",
            }
        }
    _log_result_shape(job_id, result)
    return result


# ---------------------------------------------------------------------------
# Local harness + server start (no RunPod SDK needed for --test_input)
# ---------------------------------------------------------------------------

def _run_local_test(argv: list[str]) -> int:
    """``python3 handler.py --test_input '<json>'`` — run one job, print the
    JSON result, exit 0 (job-level failures are output, not process crashes)."""
    idx = argv.index("--test_input")
    raw = argv[idx + 1] if len(argv) > idx + 1 else None
    if raw is None:
        print("--test_input requires a JSON argument", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"--test_input argument is not valid JSON: {exc}", file=sys.stderr)
        return 1
    result = _safe_handler({"id": "local-test", "input": payload})
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    if "--test_input" in sys.argv:
        sys.exit(_run_local_test(sys.argv))
    if runpod is None:
        print("runpod SDK is not installed; production images pin it (stage 05). "
              "For local runs use: python3 handler.py --test_input '<json>'", file=sys.stderr)
        sys.exit(1)
    runpod.serverless.start({"handler": _safe_handler})
