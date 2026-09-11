"""Defensive schema validation for the Tencent AuK RunPod worker.

Converts loose JSON job payloads into an immutable :class:`NormalizedRequest`
per the pins in `.icm/references/payload-contracts.md` (stage 02 of the ICM
pipeline in `.icm/`).  Fail-fast: the first failing rule raises
:class:`ValidationError` with a machine-readable code from the closed set in
the API specification.
"""

from __future__ import annotations

import base64
import os
import urllib.request
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Pinned limits and ranges (stage 01 decisions / API Spec §2, §5)
# ---------------------------------------------------------------------------

MAX_AUDIO_BYTES = 15 * 1024 * 1024          # 15 MB, decoded bytes — never base64 length
MIN_GEN_SECONDS = 0.5
MAX_GEN_SECONDS = 300.0
FLASH_NFE_RANGE = (1, 8)
BASE_NFE_RANGE = (16, 64)
BASE_CFG_RANGE = (1.0, 5.0)

TASKS = {
    "zero_shot_tts",
    "instruct_tts",
    "content_edit",
    "acoustic_edit",
    "paralinguistic_edit",
    "enhancement",
    "separation",
    "auto",
}
VARIANTS = ("flash", "base")
DELIVERY_OPTIONS = ("auto", "s3", "base64")
URL_SCHEMES = ("http://", "https://")
DATA_URL_MARKERS = (";base64,",)
URL_CONNECT_TIMEOUT_S = 10

# Task matrix: required / forbidden fields per resolved task
_TASK_REQUIRES_AUDIO = {"content_edit", "acoustic_edit", "paralinguistic_edit", "enhancement", "separation"}


class ValidationError(Exception):
    """Structured, machine-readable validation failure."""

    code: str
    message: str
    field: str | None

    def __init__(self, code: str, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def to_dict(self) -> dict[str, object]:
        error: dict[str, object] = {"code": self.code, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        return {"error": error}


@dataclass(frozen=True)
class NormalizedRequest:
    """Immutable, engine-ready view of one validated request."""

    task: str
    instruction: str
    audio_bytes: bytes | None
    prompt_audio_bytes: bytes | None
    prompt_text: str | None
    gen_seconds: float | None
    gen_text: str | None
    model_variant: str
    nfe: int
    cfg_scale: float
    seed: int | None
    response_delivery: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF      # kana
        or 0x3400 <= code <= 0x4DBF   # CJK ext A
        or 0x4E00 <= code <= 0x9FFF   # CJK unified
        or 0xF900 <= code <= 0xFAFF   # CJK compat
    )


def estimate_duration_from_text(text: str) -> float:
    """Duration heuristic: ~4 CJK chars/s, ~15 latin chars/s, floor 1.0 s."""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = sum(1 for ch in text if not _is_cjk(ch) and not ch.isspace())
    estimate = cjk / 4.0 + other / 15.0
    return round(min(max(estimate, 1.0), MAX_GEN_SECONDS), 3)


def _default_fetch_url(url: str) -> bytes:
    """Stream a remote audio URL with the pinned 10 s timeout + 15 MB cap."""
    request = urllib.request.Request(url, headers={"User-Agent": "auk-runpod-worker/1.0"})
    with urllib.request.urlopen(request, timeout=URL_CONNECT_TIMEOUT_S) as response:  # noqa: S310
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_AUDIO_BYTES:
                raise ValidationError("audio_too_large", f"remote audio exceeds {MAX_AUDIO_BYTES} bytes", field="audio")
            chunks.append(chunk)
        return b"".join(chunks)


def _ingest_audio(value: object, field: str, fetch_url: Callable[[str], bytes]) -> bytes:
    """Single-pass ingest of one audio field: URL → download, else → base64."""
    if not isinstance(value, str) or not value:
        raise ValidationError("invalid_base64", f"{field} must be a base64 string or an http(s) URL", field=field)
    if value.startswith(URL_SCHEMES):
        try:
            data = fetch_url(value)
        except ValidationError:
            raise
        except Exception as exc:  # timeouts, HTTP errors, socket errors
            raise ValidationError("audio_download_failed", f"failed to download {field}: {type(exc).__name__}", field=field) from exc
        if len(data) > MAX_AUDIO_BYTES:
            raise ValidationError("audio_too_large", f"decoded {field} exceeds {MAX_AUDIO_BYTES} bytes", field=field)
        return data
    payload = value
    for marker in DATA_URL_MARKERS:
        idx = payload.find(marker)
        if idx != -1:
            payload = payload[idx + len(marker):]
            break
    try:
        # Single-pass decode: these bytes are reused by engine and storage.
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValidationError("invalid_base64", f"{field} is not valid base64", field=field) from exc
    if len(data) > MAX_AUDIO_BYTES:
        raise ValidationError("audio_too_large", f"decoded {field} exceeds {MAX_AUDIO_BYTES} bytes", field=field)
    return data


def _require_str(payload: dict[str, object], field: str, *, required: bool = False, nonempty: bool = False) -> str | None:
    value = payload.get(field)
    if value is None:
        if required:
            raise ValidationError("missing_required_field", f"{field} is required", field=field)
        return None
    if not isinstance(value, str) or (nonempty and not value.strip()):
        code = "missing_required_field" if required else "invalid_payload"
        raise ValidationError(code, f"{field} must be a non-empty string", field=field)
    return value


def _require_number(payload: dict[str, object], field: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("invalid_payload", f"{field} must be a number", field=field)
    return float(value)


def _require_int(payload: dict[str, object], field: str, lo: int, hi: int, default: int) -> int:
    value = payload.get(field)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValidationError("invalid_payload", f"{field} must be an integer in [{lo}, {hi}]", field=field)
    return value


def _default_variant() -> str:
    env = os.environ.get("DEFAULT_MODEL_VARIANT", "flash")
    return env if env in VARIANTS else "flash"


def _resolve_variant_params(payload: dict[str, object]) -> tuple[str, int, float]:
    variant = payload.get("model_variant")
    if variant is None:
        variant = _default_variant()
    elif variant not in VARIANTS:
        raise ValidationError("unsupported_model_variant", f"model_variant must be one of {list(VARIANTS)}", field="model_variant")

    if variant == "flash":
        nfe = _require_int(payload, "nfe", *FLASH_NFE_RANGE, default=4)
        # Flash: guidance is distilled into the weights — CFG is always 0.0.
        if payload.get("cfg_scale") is not None:
            _require_number(payload, "cfg_scale")
        cfg_scale = 0.0
    else:
        nfe = _require_int(payload, "nfe", *BASE_NFE_RANGE, default=32)
        cfg_scale = _require_number(payload, "cfg_scale")
        if cfg_scale is None:
            cfg_scale = 2.0
        elif not BASE_CFG_RANGE[0] <= cfg_scale <= BASE_CFG_RANGE[1]:
            raise ValidationError("invalid_payload", f"cfg_scale must be within {BASE_CFG_RANGE} for base", field="cfg_scale")
    return variant, nfe, cfg_scale


def _resolve_task(payload: dict[str, object]) -> str:
    task = payload.get("task")
    if task is None or task == "auto":
        if payload.get("prompt_audio") is not None:
            return "zero_shot_tts"
        if payload.get("audio") is not None:
            # Edit / enhance / separate intents are indistinguishable — refuse
            # to guess (stage 01 decision 1).
            raise ValidationError("invalid_payload", "task must be explicit when audio is provided and task is auto", field="task")
        return "instruct_tts"
    if not isinstance(task, str) or task not in TASKS:
        raise ValidationError("invalid_payload", f"task must be one of {sorted(TASKS)}", field="task")
    return task


def _enforce_task_matrix(payload: dict[str, object], task: str) -> None:
    has_audio = payload.get("audio") is not None
    has_prompt = payload.get("prompt_audio") is not None

    if task == "zero_shot_tts":
        if not has_prompt:
            raise ValidationError("missing_required_field", "zero_shot_tts requires prompt_audio", field="prompt_audio")
        if has_audio:
            raise ValidationError("invalid_payload", "zero_shot_tts forbids audio", field="audio")
    elif task == "instruct_tts":
        for field in ("audio", "prompt_audio", "prompt_text"):
            if payload.get(field) is not None:
                raise ValidationError("invalid_payload", f"instruct_tts forbids {field}", field=field)
    else:  # edit / enhance / separate family
        if not has_audio:
            raise ValidationError("missing_required_field", f"{task} requires audio", field="audio")
        if has_prompt:
            raise ValidationError("invalid_payload", f"{task} forbids prompt_audio", field="prompt_audio")

    if payload.get("prompt_text") is not None and not has_prompt:
        raise ValidationError("invalid_payload", "prompt_text requires prompt_audio", field="prompt_text")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def normalize(payload: object, *, fetch_url: Callable[[str], bytes] | None = None) -> NormalizedRequest:
    """Validate one job payload and return the immutable request.

    Accepts either the full RunPod job (``{"input": {...}}``) or the bare
    input dict.  Raises :class:`ValidationError` on the first failing rule.
    """
    fetch = fetch_url or _default_fetch_url

    if not isinstance(payload, dict):
        raise ValidationError("invalid_payload", "payload must be a JSON object")
    payload_in = payload.get("input", payload)
    if not isinstance(payload_in, dict):
        raise ValidationError("invalid_payload", "'input' must be a JSON object", field="input")

    instruction = _require_str(payload_in, "instruction", required=True, nonempty=True)
    assert instruction is not None  # narrowed: required=True raises otherwise
    task = _resolve_task(payload_in)
    _enforce_task_matrix(payload_in, task)

    model_variant, nfe, cfg_scale = _resolve_variant_params(payload_in)

    audio_bytes = _ingest_audio(payload_in["audio"], "audio", fetch) if payload_in.get("audio") is not None else None
    prompt_audio_bytes = _ingest_audio(payload_in["prompt_audio"], "prompt_audio", fetch) if payload_in.get("prompt_audio") is not None else None
    prompt_text = _require_str(payload_in, "prompt_text")
    gen_text = _require_str(payload_in, "gen_text")

    gen_seconds = _require_number(payload_in, "gen_seconds")
    if gen_seconds is not None:
        if not MIN_GEN_SECONDS <= gen_seconds <= MAX_GEN_SECONDS:
            raise ValidationError(
                "invalid_payload",
                f"gen_seconds must be within [{MIN_GEN_SECONDS}, {MAX_GEN_SECONDS}]",
                field="gen_seconds",
            )
    elif gen_text:
        gen_seconds = estimate_duration_from_text(gen_text)
    # else: edit tasks measure the source clip in the engine; synthesis tasks
    # pass None through to AuK's own default (stage 01 decision 5).

    seed = payload_in.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValidationError("invalid_payload", "seed must be a non-negative integer", field="seed")

    response_delivery = payload_in.get("response_delivery") or "auto"
    if response_delivery not in DELIVERY_OPTIONS:
        raise ValidationError("invalid_payload", f"response_delivery must be one of {list(DELIVERY_OPTIONS)}", field="response_delivery")

    return NormalizedRequest(
        task=task,
        instruction=instruction,
        audio_bytes=audio_bytes,
        prompt_audio_bytes=prompt_audio_bytes,
        prompt_text=prompt_text,
        gen_seconds=gen_seconds,
        gen_text=gen_text,
        model_variant=model_variant,
        nfe=nfe,
        cfg_scale=cfg_scale,
        seed=seed,
        response_delivery=response_delivery,
    )
