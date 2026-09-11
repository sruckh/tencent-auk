"""Run the real worker branch against stand-ins; only AST-read upstream source."""
import ast
import base64
import importlib.util
import io
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest

import schema_validator
from tests.test_checkpoints import explicit

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "src/src/auk/infer/infer_auk.py"


class Samples(list[float]):
    def reshape(self, size):
        assert size == -1
        return self


class Tensor:
    ndim = 2

    def __init__(self, count=12000):
        self.shape = (1, count)
        self.values = Samples([0.1] * count)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def real_engine(tmp_path, monkeypatch):
    explicit(tmp_path / "ckpts")
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path / "ckpts"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("AUK_TEST_MOCK_ENGINE", "0")
    monkeypatch.setenv("AUK_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DEFAULT_MODEL_VARIANT", "flash")
    state = SimpleNamespace(builds=[], calls=[], sample_rate=48000, fail=False, tensor=Tensor())
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="fake GPU", total_memory=24 * 1024**3),
        ),
        set_float32_matmul_precision=lambda value: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class AukInfer:
        def __init__(self, config_path, ckpt_path, *, device=None, dtype="bf16", qwen_path=None):
            assert Path(config_path).is_file()
            assert Path(ckpt_path).is_file()
            assert Path(ckpt_path).with_name("vae.safetensors").is_file()
            assert isinstance(qwen_path, str) and Path(qwen_path).is_dir()
            assert device == "cuda" and dtype == "bf16"
            assert engine_env("HF_HUB_OFFLINE") == "1"
            state.builds.append((config_path, ckpt_path, qwen_path))

        def generate(self, messages, *, audio=None, gen_seconds=None, nfe=32, cfg_strength=2.0,
                     sway_sampling_coef=-1.0, t_grid=None, seed=None):
            clips = [item["audio"] for msg in messages for item in msg["content"] if item["type"] == "audio"]
            state.calls.append(SimpleNamespace(messages=messages, clips=clips,
                clip_bytes=[Path(p).read_bytes() for p in clips], seconds=gen_seconds,
                nfe=nfe, cfg=cfg_strength, seed=seed))
            if state.fail:
                raise RuntimeError("private-transcript-do-not-leak")
            return state.tensor, state.sample_rate

    for name in ("auk", "auk.infer", "auk.infer.infer_auk"):
        module = ModuleType(name)
        setattr(module, "AukInfer", AukInfer)
        monkeypatch.setitem(sys.modules, name, module)

    def write(buffer, data, rate, *, format, subtype):
        assert format == "WAV" and subtype == "PCM_16"
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(b"".join(struct.pack("<h", int(x * 32767)) for x in data))

    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=write))
    spec = importlib.util.spec_from_file_location("_real_engine_gate", ROOT / "engine.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, state


def engine_env(name):
    import os
    return os.environ.get(name)


def req(**kwargs):
    return schema_validator.normalize({"instruction": "Say the target words.", **kwargs})


def test_constructor_and_generate_calls_match_vendored_signature():
    tree = ast.parse(UPSTREAM.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AukInfer")
    signatures = {n.name: n.args for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in signatures["__init__"].args] == ["self", "config_path", "ckpt_path"]
    assert {a.arg for a in signatures["__init__"].kwonlyargs} >= {"device", "dtype", "qwen_path"}
    assert {a.arg for a in signatures["generate"].kwonlyargs} >= {"gen_seconds", "nfe", "cfg_strength", "seed"}
    worker = ast.parse((ROOT / "engine.py").read_text())
    for node in ast.walk(worker):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
            assert len(node.args) == 1
            assert {kw.arg for kw in node.keywords} <= {a.arg for a in signatures["generate"].kwonlyargs}
    assert "cfg_scale" not in {a.arg for a in signatures["generate"].kwonlyargs}


def test_flash_recipe_is_pinned_in_source():
    tree = ast.parse(UPSTREAM.read_text())
    generate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate")
    flash = next(n for n in ast.walk(generate) if isinstance(n, ast.If) and ast.unparse(n.test) == "self.is_flash")
    assignments = {ast.unparse(n.targets[0]): ast.literal_eval(n.value)
                   for n in flash.body if isinstance(n, ast.Assign)}
    assert assignments["nfe"] == 4 and assignments["cfg_strength"] == 0.0


def test_constructor_resolves_each_variant_and_encoder(real_engine):
    module, state = real_engine
    assert len(state.builds) == 2
    assert {Path(row[1]).name for row in state.builds} == {"auk_flash.safetensors", "auk_base.safetensors"}
    assert all(Path(row[2]).name == "Qwen2.5-Omni-3B" for row in state.builds)
    module.synthesize(req(gen_seconds=0.5))
    assert len(state.builds) == 2


@pytest.mark.parametrize("variant,nfe,cfg", [("flash", 8, 5), ("base", 64, 3)])
def test_per_request_controls_and_returned_sample_rate(real_engine, variant, nfe, cfg):
    module, state = real_engine
    wav, meta = module.synthesize(req(model_variant=variant, nfe=nfe, cfg_scale=cfg, seed=73, gen_seconds=0.5))
    call = state.calls[-1]
    assert call.seed == 73 and call.seconds == 0.5
    assert call.nfe == (4 if variant == "flash" else nfe)
    assert call.cfg == (0 if variant == "flash" else cfg)
    assert meta["nfe"] == call.nfe
    assert meta["sample_rate"] == 48000 and meta["duration_seconds"] == 0.25
    assert meta["size_bytes"] == len(wav)
    with wave.open(io.BytesIO(wav)) as audio:
        assert audio.getframerate() == 48000
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.getnframes() == 12000


def test_zero_shot_uses_prompt_audio_and_supported_transcript_text(real_engine):
    module, state = real_engine
    clip = b"reference bytes"
    module.synthesize(req(task="zero_shot_tts", prompt_audio=base64.b64encode(clip).decode(),
                          prompt_text="Exact reference words.", gen_seconds=3))
    call = state.calls[-1]
    assert call.clip_bytes == [clip]
    content = call.messages[0]["content"]
    assert content[0]["text"].startswith("Say the target words.")
    assert 'Reference audio transcript: "Exact reference words."' in content[0]["text"]
    assert set(content[1]) == {"type", "audio"}
    assert not Path(call.clips[0]).exists()


def test_source_edit_preserves_none_duration(real_engine):
    module, state = real_engine
    module.synthesize(req(task="content_edit", audio="YXVkaW8="))
    assert state.calls[-1].seconds is None
    assert state.calls[-1].clip_bytes == [b"audio"]
    assert not Path(state.calls[-1].clips[0]).exists()


def test_text_only_gets_explicit_default_duration(real_engine):
    module, state = real_engine
    module.synthesize(req())
    assert state.calls[-1].seconds == 7.0
    assert state.calls[-1].clips == []


def test_failure_scrubs_message_and_cleans_temp_audio(real_engine):
    module, state = real_engine
    state.fail = True
    with pytest.raises(module.EngineError) as exc:
        module.synthesize(req(task="enhancement", audio="YXVkaW8="))
    assert exc.value.code == "inference_failed"
    assert "private-transcript" not in str(exc.value)
    assert not Path(state.calls[-1].clips[0]).exists()


@pytest.mark.parametrize("rate", [0, -1, True, 48000.5])
def test_invalid_rate_rejected(real_engine, rate):
    module, state = real_engine
    state.sample_rate = rate
    with pytest.raises(module.EngineError):
        module.synthesize(req())


def test_non_mono_rejected_instead_of_flattening_channels(real_engine):
    module, state = real_engine
    state.tensor.shape = (2, 6000)
    with pytest.raises(module.EngineError):
        module.synthesize(req())


def test_empty_audio_rejected(real_engine):
    module, state = real_engine
    state.tensor = Tensor(0)
    with pytest.raises(module.EngineError):
        module.synthesize(req())
