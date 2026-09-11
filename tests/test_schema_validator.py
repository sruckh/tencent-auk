"""Stage 02 acceptance — schema_validator.py against
.icm/references/payload-contracts.md and the stage 02 output spec."""

import base64
import dataclasses

import pytest

import schema_validator
from schema_validator import MAX_AUDIO_BYTES, ValidationError, normalize


def raises_code(code, field=None):
    class _Matcher:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                pytest.fail(f"expected ValidationError({code})")
            assert isinstance(exc, ValidationError)
            assert exc.code == code, f"expected {code}, got {exc.code}: {exc.message}"
            if field is not None:
                assert exc.field == field, f"expected field {field}, got {exc.field}"
            assert exc.to_dict()["error"]["code"] == code
            return True

    return _Matcher()


# ---------------------------------------------------------------------------
# Golden payloads — task matrix + auto resolution
# ---------------------------------------------------------------------------

def test_normalize_zero_shot_golden(b64_clip):
    req = normalize(
        {
            "prompt_audio": b64_clip,
            "prompt_text": "Exact reference transcript.",
            "instruction": "Say the following with the same voice: hello.",
        }
    )
    assert req.task == "zero_shot_tts"
    assert req.prompt_audio_bytes == base64.b64decode(b64_clip)
    assert req.prompt_text == "Exact reference transcript."
    assert req.audio_bytes is None


def test_normalize_instruct_golden(instruct_payload):
    req = normalize(instruct_payload)
    assert req.task == "instruct_tts"
    assert req.model_variant == "flash"
    assert req.nfe == 4
    assert req.cfg_scale == 0.0


def test_normalize_explicit_task_wins(b64_clip):
    req = normalize(
        {
            "task": "enhancement",
            "instruction": "Remove the background hiss.",
            "audio": b64_clip,
        }
    )
    assert req.task == "enhancement"


def test_normalize_auto_resolves_prompt_audio(b64_clip):
    req = normalize({"instruction": "Clone this voice.", "prompt_audio": b64_clip})
    assert req.task == "zero_shot_tts"


@pytest.mark.parametrize("task", ["content_edit", "acoustic_edit", "paralinguistic_edit", "enhancement", "separation"])
def test_normalize_edit_tasks_require_audio(task, b64_clip):
    with raises_code("missing_required_field", field="audio"):
        normalize({"task": task, "instruction": "Do the thing."})
    req = normalize({"task": task, "instruction": "Do the thing.", "audio": b64_clip})
    assert req.task == task
    assert req.audio_bytes is not None


def test_normalize_auto_with_audio_is_ambiguous(b64_clip):
    with raises_code("invalid_payload", field="task"):
        normalize({"instruction": "Fix it.", "audio": b64_clip})


def test_normalize_zero_shot_requires_prompt_audio():
    with raises_code("missing_required_field", field="prompt_audio"):
        normalize({"task": "zero_shot_tts", "instruction": "Clone."})


def test_normalize_zero_shot_forbids_audio(b64_clip):
    with raises_code("invalid_payload", field="audio"):
        normalize(
            {
                "task": "zero_shot_tts",
                "instruction": "Clone.",
                "prompt_audio": b64_clip,
                "audio": b64_clip,
            }
        )


def test_normalize_prompt_text_without_prompt_audio_rejected(instruct_payload):
    with raises_code("invalid_payload", field="prompt_text"):
        normalize({**instruct_payload, "prompt_text": "orphan transcript"})


def test_normalize_unknown_task_rejected():
    with raises_code("invalid_payload", field="task"):
        normalize({"task": "sing_a_song", "instruction": "Sing."})


def test_normalize_missing_instruction_rejected():
    with raises_code("missing_required_field", field="instruction"):
        normalize({"task": "instruct_tts"})


@pytest.mark.parametrize("payload", [None, [], "hello", {"input": "not-a-dict"}, 42])
def test_normalize_invalid_envelope_rejected(payload):
    with raises_code("invalid_payload"):
        normalize(payload)


def test_normalize_accepts_envelope_and_bare_dict(b64_clip):
    inner = {"instruction": "hi", "prompt_audio": b64_clip}
    assert normalize({"input": inner}).task == normalize(inner).task == "zero_shot_tts"


def test_normalize_returns_frozen(instruct_payload):
    req = normalize(instruct_payload)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.instruction = "mutated"


# ---------------------------------------------------------------------------
# Ingest — single-pass decode, 15 MB decoded cap, URLs
# ---------------------------------------------------------------------------

def test_audio_exactly_15mb_accepted():
    clip = base64.b64encode(b"\x00" * MAX_AUDIO_BYTES).decode("ascii")
    req = normalize({"task": "enhancement", "instruction": "clean", "audio": clip})
    assert len(req.audio_bytes) == MAX_AUDIO_BYTES


def test_audio_one_byte_over_rejected():
    clip = base64.b64encode(b"\x00" * (MAX_AUDIO_BYTES + 1)).decode("ascii")
    with raises_code("audio_too_large", field="audio"):
        normalize({"task": "enhancement", "instruction": "clean", "audio": clip})


def test_limit_applies_to_decoded_not_base64_length():
    # Exactly 15 MB decoded → ~20 MB base64; an encoded-length limit would
    # wrongly reject this, a decoded-length limit accepts it.
    clip = base64.b64encode(b"\x00" * MAX_AUDIO_BYTES).decode("ascii")
    assert len(clip) > MAX_AUDIO_BYTES
    req = normalize({"task": "enhancement", "instruction": "clean", "audio": clip})
    assert len(req.audio_bytes) == MAX_AUDIO_BYTES


def test_data_url_prefix_stripped(b64_clip):
    req = normalize({"instruction": "hi", "prompt_audio": f"data:audio/wav;base64,{b64_clip}"})
    assert req.prompt_audio_bytes == base64.b64decode(b64_clip)


def test_invalid_base64_rejected():
    with raises_code("invalid_base64", field="prompt_audio"):
        normalize({"instruction": "hi", "prompt_audio": "!!!not-base64!!!"})


def test_non_http_scheme_is_not_a_url():
    with raises_code("invalid_base64", field="audio"):
        normalize({"task": "enhancement", "instruction": "x", "audio": "ftp://example.com/a.wav"})


def test_url_ingest_downloads():
    seen = {}

    def fake_fetch(url):
        seen["url"] = url
        return b"remote-bytes"

    req = normalize(
        {"task": "enhancement", "instruction": "x", "audio": "https://example.com/src.wav"},
        fetch_url=fake_fetch,
    )
    assert seen["url"] == "https://example.com/src.wav"
    assert req.audio_bytes == b"remote-bytes"


def test_url_ingest_timeout_maps_to_download_failed():
    def boom(url):
        raise TimeoutError("timed out")

    with raises_code("audio_download_failed", field="audio"):
        normalize({"task": "enhancement", "instruction": "x", "audio": "https://example.com/src.wav"}, fetch_url=boom)


def test_url_ingest_oversize_maps_to_audio_too_large():
    def big(url):
        return b"\x00" * (MAX_AUDIO_BYTES + 1)

    with raises_code("audio_too_large", field="audio"):
        normalize({"task": "enhancement", "instruction": "x", "audio": "https://example.com/src.wav"}, fetch_url=big)


def test_single_pass_decode(monkeypatch, b64_clip):
    calls = {"n": 0}
    real_decode = base64.b64decode

    def counting(value, *a, **kw):
        calls["n"] += 1
        return real_decode(value, *a, **kw)

    monkeypatch.setattr(schema_validator.base64, "b64decode", counting)
    normalize(
        {
            "instruction": "hi",
            "prompt_audio": b64_clip,
            "gen_text": "hello world",
        }
    )
    assert calls["n"] == 1  # one clip, decoded exactly once


# ---------------------------------------------------------------------------
# Parameter guardrails — variants, NFE, CFG, seed
# ---------------------------------------------------------------------------

def test_unknown_variant_rejected():
    with raises_code("unsupported_model_variant", field="model_variant"):
        normalize({"instruction": "hi", "model_variant": "turbo"})


def test_flash_forces_cfg_zero():
    req = normalize({"instruction": "hi", "model_variant": "flash", "cfg_scale": 3.0})
    assert req.cfg_scale == 0.0


def test_flash_nfe_out_of_range_rejected():
    with raises_code("invalid_payload", field="nfe"):
        normalize({"instruction": "hi", "model_variant": "flash", "nfe": 32})


def test_base_defaults_applied():
    req = normalize({"instruction": "hi", "model_variant": "base"})
    assert req.nfe == 32
    assert req.cfg_scale == 2.0


def test_base_nfe_and_cfg_out_of_range_rejected():
    with raises_code("invalid_payload", field="nfe"):
        normalize({"instruction": "hi", "model_variant": "base", "nfe": 4})
    with raises_code("invalid_payload", field="cfg_scale"):
        normalize({"instruction": "hi", "model_variant": "base", "cfg_scale": 9.0})


def test_env_default_variant(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_VARIANT", "base")
    assert normalize({"instruction": "hi"}).model_variant == "base"
    monkeypatch.setenv("DEFAULT_MODEL_VARIANT", "garbage")
    assert normalize({"instruction": "hi"}).model_variant == "flash"


def test_negative_seed_rejected():
    with raises_code("invalid_payload", field="seed"):
        normalize({"instruction": "hi", "seed": -1})


# ---------------------------------------------------------------------------
# Duration — explicit bounds and the gen_text heuristic
# ---------------------------------------------------------------------------

def test_gen_seconds_bounds():
    assert normalize({"instruction": "hi", "gen_seconds": 0.5}).gen_seconds == 0.5
    assert normalize({"instruction": "hi", "gen_seconds": 300}).gen_seconds == 300.0
    for bad in (0.4, 300.1, "six", True):
        with raises_code("invalid_payload", field="gen_seconds"):
            normalize({"instruction": "hi", "gen_seconds": bad})


def test_duration_heuristic_gen_text():
    assert normalize({"instruction": "hi", "gen_text": "你好世界你好世界"}).gen_seconds == 2.0
    latin = "a" * 30
    assert normalize({"instruction": "hi", "gen_text": latin}).gen_seconds == 2.0
    assert normalize({"instruction": "hi", "gen_text": "a"}).gen_seconds == 1.0  # floor


def test_gen_seconds_none_passthrough():
    req = normalize({"instruction": "hi"})
    assert req.gen_seconds is None


def test_gen_seconds_beats_gen_text():
    req = normalize({"instruction": "hi", "gen_seconds": 5, "gen_text": "word " * 50})
    assert req.gen_seconds == 5.0


def test_invalid_response_delivery_rejected():
    with raises_code("invalid_payload", field="response_delivery"):
        normalize({"instruction": "hi", "response_delivery": "fax"})


def test_response_delivery_passthrough():
    assert normalize({"instruction": "hi", "response_delivery": "s3"}).response_delivery == "s3"
    assert normalize({"instruction": "hi"}).response_delivery == "auto"
