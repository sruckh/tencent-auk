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


class _FakeEncoder:
    """Stand-in for the frozen Qwen text encoder, tracking its own dtype.

    Upstream upcasts it to fp32 inside CFMEdit.to(torch.float32); these tests
    assert the worker puts it back to bf16 without touching the transformer."""

    def __init__(self):
        self.dtype = "float32"
        self.requests_grad = True

    def to(self, dtype):
        self.dtype = str(dtype).replace("torch.", "")
        return self


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
def make_real_engine(tmp_path, monkeypatch):
    """Factory: bootstrap config (free VRAM, second-build failure) is frozen
    at call time because the module bootstraps at import."""
    explicit(tmp_path / "ckpts")
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path / "ckpts"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("AUK_TEST_MOCK_ENGINE", "0")
    monkeypatch.setenv("AUK_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DEFAULT_MODEL_VARIANT", "flash")
    state = SimpleNamespace(builds=[], calls=[], sample_rate=48000, fail=False, tensor=Tensor(),
                            free_gib=40.0, fail_second_build_with=None,
                            torch_alloc_gib=12.1, torch_peak_gib=13.4)

    class OutOfMemoryError(RuntimeError):
        pass

    torch = SimpleNamespace(
        OutOfMemoryError=OutOfMemoryError,
        # _downcast_encoder targets torch.bfloat16; the stand-in only needs the
        # attributes to resolve, since _FakeEncoder.to records a string.
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="fake GPU", total_memory=96 * 1024**3),
            mem_get_info=lambda: (int(state.free_gib * 1024**3), 96 * 1024**3),
            memory_allocated=lambda: int(state.torch_alloc_gib * 1024**3),
            max_memory_allocated=lambda: int(state.torch_peak_gib * 1024**3),
            empty_cache=lambda: None,
        ),
        set_float32_matmul_precision=lambda value: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class AukInfer:
        def __init__(self, config_path, ckpt_path, *, device=None, dtype="bf16", qwen_path=None):
            if state.fail_second_build_with is not None and len(state.builds) >= 1:
                raise state.fail_second_build_with
            assert Path(config_path).is_file()
            assert Path(ckpt_path).is_file()
            assert Path(ckpt_path).with_name("vae.safetensors").is_file()
            assert isinstance(qwen_path, str) and Path(qwen_path).is_dir()
            assert device == "cuda" and dtype == "bf16"
            assert engine_env("HF_HUB_OFFLINE") == "1"
            state.builds.append((config_path, ckpt_path, qwen_path))
            self.model = SimpleNamespace(text_encoder=_FakeEncoder())
            self.dtype = "bf16" 

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

    def _make(*, free_gib=40.0, fail_second_build=False):
        if fail_second_build:
            state.fail_second_build_with = OutOfMemoryError("CUDA out of memory")
        state.free_gib = free_gib
        spec = importlib.util.spec_from_file_location("_real_engine_gate", ROOT / "engine.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, state

    return _make


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


def test_constructor_resolves_each_variant_and_encoder(make_real_engine):
    module, state = make_real_engine()
    assert len(state.builds) == 2
    assert {Path(row[1]).name for row in state.builds} == {"auk_flash.safetensors", "auk_base.safetensors"}
    assert all(Path(row[2]).name == "Qwen2.5-Omni-3B" for row in state.builds)
    module.synthesize(req(gen_seconds=0.5))
    assert len(state.builds) == 2


def test_encoder_downcast_to_bf16_on_build(make_real_engine):
    """Upstream loads the encoder bf16 then upcasts the whole CFMEdit to fp32,
    so it lands on the GPU at twice the necessary size, once per variant. The
    worker puts the frozen encoder back to bf16."""
    module, state = make_real_engine()
    handles = module._ENGINES
    assert len(handles) == 2
    for handle in handles.values():
        assert handle.model.text_encoder.dtype == "bfloat16"


def test_encoder_downcast_is_opt_out(make_real_engine, monkeypatch, capsys):
    """AUK_ENCODER_BF16=0 keeps upstream's fp32 weights, for byte-comparable
    output against the unmodified baseline."""
    monkeypatch.setenv("AUK_ENCODER_BF16", "0")
    module, state = make_real_engine()
    assert module._ENGINES["flash"].model.text_encoder.dtype == "float32"
    assert "encoder bf16 downcast disabled" in capsys.readouterr().out


def test_downcast_leaves_the_transformer_and_vae_alone():
    """The other half of the contract: CFMEdit.sample derives the ODE state
    dtype from next(self.parameters()).dtype, and self.transformer is registered
    before text_encoder — so downcasting the transformer would demote the whole
    integration trajectory to bf16. Asserted structurally against the worker
    source, since the real transformer needs a GPU."""
    tree = ast.parse((ROOT / "engine.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_downcast_encoder")
    casts = [ast.unparse(n.func) for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in {"to", "half", "bfloat16", "float"}]
    assert casts == ["encoder.to"]  # the encoder only — never self.model / self.vae_model

    # The downcast strands the old fp32 blocks in torch's caching allocator
    # (measured: 15.1 GiB freed logically, only 7.5 GiB returned to the driver),
    # so the handler must flush the cache. Asserted structurally because the
    # fake torch's empty_cache is a lambda and cannot fail the way the real one
    # would if this regressed.
    flushed = [ast.unparse(n.func) for n in ast.walk(fn)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "empty_cache"]
    assert flushed == ["torch.cuda.empty_cache"]


def test_low_free_vram_builds_primary_only(make_real_engine):
    """A card too small for two AukInfer stacks keeps the second lazy.

    Measured per-model cost is ~21.4 GiB, so 5 GiB free after the primary build
    means no room for a second (~24 GB-class behaviour)."""
    module, state = make_real_engine(free_gib=5.0)
    assert len(state.builds) == 1
    wav, meta = module.synthesize(req(gen_seconds=0.5))
    assert meta["model_variant"] == "flash" and len(state.builds) == 1


def test_vram_probe_reports_build_and_generation(make_real_engine, capsys):
    """The deploy probe that makes an OOM attributable: the measured footprint
    after the variant build, and free VRAM bracketing the generation. The
    per-model cost is an estimate and the placement threshold is a policy, so
    without these numbers an OOM cannot be told from a genuine overrun."""
    module, state = make_real_engine(free_gib=11.5)
    capsys.readouterr()  # drop import-time bootstrap output
    module.synthesize(req(gen_seconds=0.5))
    out = capsys.readouterr().out
    assert "vram before generate task=instruct_tts variant=flash" in out
    assert "vram after generate task=instruct_tts variant=flash" in out
    assert "free=11.5 of 96.0 GiB" in out
    assert "torch_alloc=12.1 peak=13.4 GiB" in out


def test_vram_probe_never_breaks_inference(make_real_engine, capsys, monkeypatch):
    """Telemetry is best-effort: a failing CUDA query must not fail the job."""
    module, state = make_real_engine(free_gib=11.5)

    def boom():
        raise RuntimeError("CUDA driver lost")

    monkeypatch.setattr(module.torch.cuda, "mem_get_info", boom)
    wav, meta = module.synthesize(req(gen_seconds=0.5))  # must still succeed
    assert meta["model_variant"] == "flash"
    assert "vram before generate" not in capsys.readouterr().out


def test_second_variant_loads_lazily_on_request(make_real_engine):
    module, state = make_real_engine(free_gib=5.0)
    module.synthesize(req(model_variant="base", gen_seconds=0.5))
    assert len(state.builds) == 2
    assert Path(state.builds[1][1]).name == "auk_base.safetensors"


def test_oom_on_second_build_degrades_to_single_variant(make_real_engine):
    module, state = make_real_engine(free_gib=40.0, fail_second_build=True)
    assert len(state.builds) == 1  # import completed with the primary only
    wav, meta = module.synthesize(req(gen_seconds=0.5))
    assert meta["model_variant"] == "flash"


@pytest.mark.parametrize("variant,nfe,cfg", [("flash", 8, 5), ("base", 64, 3)])
def test_per_request_controls_and_returned_sample_rate(make_real_engine, variant, nfe, cfg):
    module, state = make_real_engine()
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


def test_zero_shot_uses_prompt_audio_and_supported_transcript_text(make_real_engine):
    module, state = make_real_engine()
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


def test_source_edit_preserves_none_duration(make_real_engine):
    module, state = make_real_engine()
    module.synthesize(req(task="content_edit", audio="YXVkaW8="))
    assert state.calls[-1].seconds is None
    assert state.calls[-1].clip_bytes == [b"audio"]
    assert not Path(state.calls[-1].clips[0]).exists()


def test_text_only_gets_explicit_default_duration(make_real_engine):
    module, state = make_real_engine()
    module.synthesize(req())
    assert state.calls[-1].seconds == 7.0
    assert state.calls[-1].clips == []


def test_failure_scrubs_message_and_cleans_temp_audio(make_real_engine):
    module, state = make_real_engine()
    state.fail = True
    with pytest.raises(module.EngineError) as exc:
        module.synthesize(req(task="enhancement", audio="YXVkaW8="))
    assert exc.value.code == "inference_failed"
    assert "private-transcript" not in str(exc.value)
    assert not Path(state.calls[-1].clips[0]).exists()


@pytest.mark.parametrize("rate", [0, -1, True, 48000.5])
def test_invalid_rate_rejected(make_real_engine, rate):
    module, state = make_real_engine()
    state.sample_rate = rate
    with pytest.raises(module.EngineError):
        module.synthesize(req())


def test_non_mono_rejected_instead_of_flattening_channels(make_real_engine):
    module, state = make_real_engine()
    state.tensor.shape = (2, 6000)
    with pytest.raises(module.EngineError):
        module.synthesize(req())


def test_empty_audio_rejected(make_real_engine):
    module, state = make_real_engine()
    state.tensor = Tensor(0)
    with pytest.raises(module.EngineError):
        module.synthesize(req())
