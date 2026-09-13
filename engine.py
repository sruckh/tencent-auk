"""In-memory AuK engine for the RunPod worker.

Module-scope bootstrap (stage 03 of the ICM pipeline in `.icm/`): the model
loads once at import time — never per job — per `.icm/references/runpod-invariants.md`.
Set ``AUK_TEST_MOCK_ENGINE=1`` before importing to swap in a pure-stdlib mock
(no GPU, no ``torch``, no ``auk`` package) for CI and the local harness.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import wave
from typing import Any

SAMPLE_RATE = 24000            # pinned: 24 kHz, mono, 16-bit PCM WAV
MOCK_ENV = "AUK_TEST_MOCK_ENGINE"
ENCODER_BF16_ENV = "AUK_ENCODER_BF16"
CKPT_ENV = "CKPT_ROOT"
DEFAULT_HUB_CACHE = "/runpod-volume/huggingface-cache/hub"
DEFAULT_CKPT_ROOT = "/runpod-volume/ckpts"
_COMPONENTS = {
    "base": ("tencent/AuK", "auk_base.safetensors"),
    "flash": ("tencent/AuK-Flash", "auk_flash.safetensors"),
    "encoder": ("Qwen/Qwen2.5-Omni-3B", None),
}

_MODE_MOCK = "mock"
_MODE_REAL = "real"
_MODE = _MODE_MOCK if os.environ.get(MOCK_ENV) == "1" else _MODE_REAL
_CKPT_ROOT: str | None = None  # resolved during real-mode bootstrap
_ENGINES: dict[str, Any] = {}


class EngineError(Exception):
    """Structured engine failure (code from the closed error-code set)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Checkpoint resolution — filesystem checks are also testable in mock mode
# ---------------------------------------------------------------------------

def _component_complete(path: Path, checkpoint: str | None) -> bool:
    """Reject partial snapshots, including missing encoder weight shards."""
    def present(name: str) -> bool:
        candidate = path / name
        return candidate.is_file() and candidate.stat().st_size > 0

    try:
        if checkpoint:
            return all(present(name) for name in ("config.yaml", checkpoint, "vae.safetensors"))
        if not all(present(name) for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json")):
            return False
        if not (present("tokenizer.json") or (present("vocab.json") and present("merges.txt"))):
            return False
        if present("model.safetensors.index.json"):
            index = json.loads((path / "model.safetensors.index.json").read_text())
            shards = index["weight_map"].values()
            return bool(index["weight_map"]) and all(
                isinstance(name, str) and Path(name).name == name and present(name)
                for name in shards
            )
        return present("model.safetensors")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _native_snapshot(hub: Path, repo_id: str, checkpoint: str | None) -> Path | None:
    """Prefer refs/main, then the first complete snapshot (sorted by name)."""
    model = hub / ("models--" + repo_id.replace("/", "--"))
    snapshots = model / "snapshots"
    candidates = []
    try:
        revision = (model / "refs" / "main").read_text().strip()
        if revision and revision not in (".", "..") and Path(revision).name == revision:
            candidates.append(snapshots / revision)
    except OSError:
        pass
    if snapshots.is_dir():
        candidates.extend(sorted(snapshots.iterdir()))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir() and _component_complete(candidate, checkpoint):
            # Explicitly offline: no metadata request even with a working network.
            try:
                from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]  # production dep

                resolved = Path(snapshot_download(
                    repo_id=repo_id, revision=candidate.name, cache_dir=str(hub),
                    local_files_only=True,
                ))
            except Exception:
                continue  # unconfirmable natively → next candidate, then explicit tier
            if resolved.resolve() == candidate.resolve() and _component_complete(resolved, checkpoint):
                return resolved
    return None


def _download_component(repo_id: str, target: Path, hub: Path, checkpoint: str | None, timeout: float) -> None:
    """Isolate the explicitly online fallback from the offline inference process."""
    if os.environ.get("AUK_OFFLINE") == "1":
        raise EngineError("inference_failed", f"Cached model missing: {repo_id}; AUK_OFFLINE=1")
    patterns = ["config.yaml", checkpoint, "vae.safetensors"] if checkpoint else [
        "*.json", "*.safetensors", "*.txt", "*.jinja",
    ]
    env = dict(os.environ, HF_HUB_CACHE=str(hub), HF_HUB_OFFLINE="0")
    # The Hub reads offline settings at import time. A bounded child permits
    # downloading without ever toggling the parent's offline client state.
    script = (
        "import json,sys; from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2], "
        "local_dir=sys.argv[3], allow_patterns=json.loads(sys.argv[4]), "
        "local_files_only=False, max_workers=4, etag_timeout=10)"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script, repo_id, str(hub), str(target), json.dumps(patterns)],
            env=env, check=True, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError("inference_failed", f"Checkpoint download failed: {repo_id} ({type(exc).__name__})") from exc


def _resolve_checkpoints() -> dict[str, Path]:
    """Resolve all components before model construction; never download per job."""
    hub = Path(os.environ.get("HF_HUB_CACHE", DEFAULT_HUB_CACHE)).resolve()
    root = Path(os.environ.get(CKPT_ENV, DEFAULT_CKPT_ROOT)).resolve()
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Reserve at least five minutes of the platform's init window for loading.
    try:
        init_timeout = float(os.environ.get("RUNPOD_INIT_TIMEOUT", "1200"))
    except ValueError:
        init_timeout = 1200.0
    budget = min(900.0, max(0.0, init_timeout - 300.0))
    deadline = time.monotonic() + budget
    resolved = {}
    for variant, (repo_id, checkpoint) in _COMPONENTS.items():
        path = _native_snapshot(hub, repo_id, checkpoint)
        if path is None:
            path = root / repo_id.split("/")[1]
            if not _component_complete(path, checkpoint):
                if os.environ.get("AUK_OFFLINE") == "1":
                    raise EngineError("inference_failed", f"Cached model missing: {repo_id}; AUK_OFFLINE=1")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EngineError("inference_failed", "Checkpoint download startup budget exhausted")
                _download_component(repo_id, path, hub, checkpoint, remaining)
                if not _component_complete(path, checkpoint):
                    raise EngineError("inference_failed", f"Incomplete checkpoint component: {repo_id}")
        resolved[variant] = path
    return resolved


# ---------------------------------------------------------------------------
# Shared metadata helper
# ---------------------------------------------------------------------------

def _metadata(wav_bytes: bytes, req, n_samples: int, sample_rate: int = SAMPLE_RATE) -> dict[str, object]:
    """Fixed synthesis-metadata field set (payload-contracts.md §Response)."""
    return {
        "task_executed": req.task,
        "model_variant": req.model_variant,
        "nfe": 4 if req.model_variant == "flash" else req.nfe,
        "sample_rate": sample_rate,
        "duration_seconds": round(n_samples / sample_rate, 3),
        "size_bytes": len(wav_bytes),
    }


# ---------------------------------------------------------------------------
# Temp-file helpers (the one sanctioned disk write: upstream consumes paths)
# ---------------------------------------------------------------------------

def _write_temp_audio(audio_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


def _remove_temp(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# ===========================================================================
# MOCK MODE — stdlib only; the whole test suite and --test_input run here
# ===========================================================================

if _MODE == _MODE_MOCK:

    def _default_variant() -> str:
        env = os.environ.get("DEFAULT_MODEL_VARIANT", "flash")
        return env if env in ("flash", "base") else "flash"

    def _mock_frequency(instruction: str) -> float:
        digest = int(hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8], 16)
        return 300.0 + (digest % 400)

    def _mock_generate(req) -> int:
        """Deterministic sine WAV built in-memory; returns sample count."""
        seconds = req.gen_seconds if req.gen_seconds else 1.0
        seconds = min(max(seconds, 0.5), 5.0)
        n_samples = int(seconds * SAMPLE_RATE)
        freq = _mock_frequency(req.instruction)
        phase = ((req.seed or 0) % 360) / 360.0 * 2.0 * math.pi
        frames = bytearray()
        for i in range(n_samples):
            value = math.sin(2.0 * math.pi * freq * (i / SAMPLE_RATE) + phase)
            frames += struct.pack("<h", max(-32768, min(32767, int(value * 12000))))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(bytes(frames))
        _MOCK.last_wav = buf.getvalue()
        return n_samples

    class _MockEngine:
        def __init__(self) -> None:
            self.last_wav = b""

    _MOCK = _MockEngine()

    def _log_system_info() -> None:
        print(
            f"[auk-engine] mock mode ({MOCK_ENV}=1) — variant={_default_variant()}, "
            f"sample_rate={SAMPLE_RATE}; no GPU, no model loaded",
            flush=True,
        )

    _log_system_info()

    def synthesize(req) -> tuple[bytes, dict[str, object]]:
        """Mock path: same contract as the real path, stdlib only."""
        try:
            n_samples = _mock_generate(req)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError("inference_failed", f"mock synthesis failed: {exc}") from exc
        return _MOCK.last_wav, _metadata(_MOCK.last_wav, req, n_samples)

# ===========================================================================
# REAL MODE — torch + upstream auk package; production only
# ===========================================================================

else:

    import torch  # pyright: ignore[reportMissingImports]  # noqa: E402 — production dep; mock mode never imports it

    def _default_variant() -> str:
        env = os.environ.get("DEFAULT_MODEL_VARIANT", "flash")
        return env if env in ("flash", "base") else "flash"

    def _build_engine(variant: str):
        from auk.infer.infer_auk import AukInfer  # pyright: ignore[reportMissingImports]  # real mode only

        path = _CHECKPOINTS[variant]
        checkpoint = _COMPONENTS[variant][1]
        assert checkpoint is not None
        handle = AukInfer(
            str(path / "config.yaml"), str(path / checkpoint),
            device=_DEVICE, dtype="bf16", qwen_path=str(_CHECKPOINTS["encoder"]),
        )
        _downcast_encoder(handle)
        return handle

    def _downcast_encoder(handle) -> None:
        """Return the frozen Qwen text encoder to bf16, undoing upstream's fp32 upcast.

        Upstream loads the encoder with ``torch_dtype=torch.bfloat16`` and then,
        a few lines later, calls ``model.to(torch.float32)`` over the whole
        CFMEdit — which holds the encoder — so it lands on the GPU in fp32 at
        twice the necessary size, once per variant. The ``dtype="bf16"`` this
        module passes only selects the autocast dtype used at generation, so it
        never undoes that.

        Only the encoder is touched, and that is deliberate:

        * ``CFMEdit.sample`` derives the ODE state dtype from
          ``next(self.parameters()).dtype``. ``self.transformer`` is registered
          first, so that is the *transformer's* dtype — leaving it fp32 keeps the
          whole integration trajectory in fp32 rather than demoting it to bf16.
        * The VAE is left alone too: it decodes outside the autocast block, and
          ``denormalize`` forces ``.float()`` as the explicit fp32 handoff.

        The encoder itself gains nothing from fp32 storage: it is frozen, runs
        once per job under ``torch.no_grad()`` inside autocast, so its matmuls
        are already bf16 — this only stops holding a second fp32 copy of the
        weights. Disable with ``AUK_ENCODER_BF16=0`` if output must be
        byte-comparable to the fp32-weight baseline."""
        if os.environ.get(ENCODER_BF16_ENV, "1") == "0":
            print(f"[auk-engine] encoder bf16 downcast disabled ({ENCODER_BF16_ENV}=0)", flush=True)
            return
        encoder = getattr(getattr(handle, "model", None), "text_encoder", None)
        if encoder is None:  # unexpected upstream shape — leave it untouched
            return
        encoder.to(torch.bfloat16)
        # ``.to(bf16)`` allocates new params and leaves the fp32 blocks as garbage
        # in torch's caching allocator, where the driver still counts them as
        # used — measured at 7.6 GiB stranded across two variants. Nothing has
        # run yet, so flushing the cache here is free; the next real allocation
        # would otherwise have to find that memory the hard way.
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — never let housekeeping break bootstrap
            pass

    def _evict_engine() -> None:
        """Free the resident variant so the next build starts from empty VRAM.

        CFMEdit owns the Qwen thinker, the VAE and the DiT, so dropping the
        handle drops the whole stack; empty_cache() returns the freed segments
        to the driver — the caching allocator would otherwise hold them
        reserved and the next build would OOM against its own freed memory."""
        if not _ENGINES:
            return
        print(
            f"[auk-engine] evicting resident {sorted(_ENGINES)} "
            "before building another variant",
            flush=True,
        )
        _ENGINES.clear()
        torch.cuda.empty_cache()

    def _get_engine(variant: str):
        """Single-resident engine handle: never two AukInfers in VRAM.

        Each AukInfer is a full encoder+VAE+DiT stack, so two resident
        variants OOM every card this worker targets; a job that asks for the
        other variant evicts the resident one first (swap-on-switch)."""
        if variant not in _ENGINES:
            _evict_engine()
            _ENGINES[variant] = _build_engine(variant)
        return _ENGINES[variant]

    def _log_system_info() -> None:
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                device = f"cuda:0 {props.name} {props.total_memory / (1024 ** 3):.1f} GiB"
            else:
                device = "cpu (dev fallback)"
        except Exception as exc:  # never block startup on telemetry
            device = f"unknown ({exc})"
        print(
            f"[auk-engine] real mode — python/torch ready, device={device}, "
            f"ckpt_root={_CKPT_ROOT}, default_variant={_default_variant()}, "
            f"loaded={sorted(_ENGINES)}",
            flush=True,
        )

    # -- module-scope bootstrap (import time, never per job) ----------------
    _CHECKPOINTS = _resolve_checkpoints()
    _CKPT_ROOT = str(Path(os.environ.get(CKPT_ENV, DEFAULT_CKPT_ROOT)).resolve())  # pyright: ignore[reportConstantRedefinition]
    torch.set_float32_matmul_precision("high")
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _vram(tag: str) -> None:
        """Deploy probe: actual VRAM at a point in the job, never estimated.

        The per-model cost was an estimate and the placement threshold a policy
        (stage 01), so an OOM was previously unattributable from the code alone
        — it took this probe to establish the real figures (2026-09-11). Reports
        free/total at the moment of interest plus torch's allocated/peak; note
        torch_alloc covers only the caching allocator, so `free` is the number
        that answers "will this fit?". Never raises — telemetry must not break
        inference."""
        try:
            free, total = torch.cuda.mem_get_info()
            print(
                f"[auk-engine] vram {tag}: free={free / 2**30:.1f} of {total / 2**30:.1f} GiB"
                f" | torch_alloc={torch.cuda.memory_allocated() / 2**30:.1f}"
                f" peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )
        except Exception:  # noqa: BLE001 — telemetry is best-effort
            pass

    # Single-resident placement (revised 2026-09-13). Each AukInfer is a full
    # encoder+VAE+DiT stack — the 96 GiB probe once measured 42.7 GiB with
    # both resident — and two stacks OOM a 32 GiB card. Exactly one model
    # lives in VRAM: the default builds at import (cold start); a job asking
    # for the other variant evicts the resident one first (_get_engine).
    primary = _default_variant()
    _ENGINES[primary] = _build_engine(primary)
    _vram(f"after '{primary}' build")
    _log_system_info()

    def synthesize(req) -> tuple[bytes, dict[str, object]]:
        """Call the pinned upstream API; preserve its returned rate in the WAV."""
        import soundfile as sf  # pyright: ignore[reportMissingImports]  # production dep

        tmp_path = None
        try:
            audio_bytes = req.prompt_audio_bytes if req.task == "zero_shot_tts" else req.audio_bytes
            if audio_bytes:
                tmp_path = _write_temp_audio(audio_bytes)
            instruction = req.instruction
            if req.prompt_text:
                # The cookbook has no transcript parameter. Carry it through
                # the supported text content, explicitly labeled as reference.
                instruction += "\nReference audio transcript: " + json.dumps(req.prompt_text, ensure_ascii=False)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    *([{"type": "audio", "audio": tmp_path}] if tmp_path else []),
                ],
            }]
            # Text-only synthesis needs a positive target length (COOKBOOK).
            # With audio, None intentionally preserves the source duration.
            seconds = req.gen_seconds
            if seconds is None and tmp_path is None:
                seconds = 7.0
            handle = _get_engine(req.model_variant)
            # Bracket the generation: the "before" line shows what the resident
            # model actually costs, the "after" line its peak during the job.
            _vram(f"before generate task={req.task} variant={req.model_variant}")
            try:
                waveform, sample_rate = handle.generate(
                    messages,
                    gen_seconds=seconds,
                    nfe=4 if req.model_variant == "flash" else req.nfe,
                    cfg_strength=0.0 if req.model_variant == "flash" else req.cfg_scale,
                    seed=req.seed,
                )
            finally:
                _vram(f"after generate task={req.task} variant={req.model_variant}")
            if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
                raise ValueError("Invalid upstream sample rate")
            if waveform.ndim not in (1, 2) or (waveform.ndim == 2 and waveform.shape[0] != 1):
                raise ValueError("Expected upstream mono waveform")
            data = waveform.detach().cpu().float().numpy().reshape(-1)
            if len(data) == 0:
                raise ValueError("Empty upstream waveform")
            buf = io.BytesIO()
            sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()
            return wav_bytes, _metadata(wav_bytes, req, len(data), sample_rate)
        except EngineError:
            raise
        except Exception as exc:
            # Exception text can contain audio paths, prompt text or credentials,
            # so only the type is reported — except an ImportError's module name,
            # which is safe and makes a missing dependency diagnosable from the
            # wire error alone (e.g. qwen_omni_utils -> librosa).
            detail = type(exc).__name__
            if isinstance(exc, ImportError) and getattr(exc, "name", None):
                detail += f": {exc.name}"
            raise EngineError("inference_failed", f"synthesis failed: {detail}") from exc
        finally:
            _remove_temp(tmp_path)
