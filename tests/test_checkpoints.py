"""Checkpoint gates use tiny local fixtures, never model downloads or Hub imports."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import engine


def component(path, checkpoint):
    path.mkdir(parents=True, exist_ok=True)
    names = ("config.yaml", checkpoint, "vae.safetensors") if checkpoint else (
        "config.json", "tokenizer_config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors",
    )
    for name in names:
        (path / name).write_text("fixture")
    return path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    hub, root = tmp_path / "hub", tmp_path / "ckpts"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("CKPT_ROOT", str(root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("AUK_OFFLINE", "1")
    monkeypatch.setenv("RUNPOD_INIT_TIMEOUT", "1200")
    calls = []

    def snapshot_download(*, repo_id, revision, cache_dir, local_files_only):
        assert local_files_only is True
        assert cache_dir == str(hub)
        calls.append((repo_id, revision))
        return hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    return hub, root, calls


def native(hub, variant, revision="a" * 40, ref=True):
    repo_id, checkpoint = engine._COMPONENTS[variant]
    model = hub / ("models--" + repo_id.replace("/", "--"))
    path = component(model / "snapshots" / revision, checkpoint)
    if ref:
        (model / "refs").mkdir(exist_ok=True)
        (model / "refs" / "main").write_text(revision + "\n")
    return path


def explicit(root):
    return {
        variant: component(root / repo_id.split("/")[1], checkpoint)
        for variant, (repo_id, checkpoint) in engine._COMPONENTS.items()
    }


def test_native_cache_wins_over_explicit(cache, monkeypatch):
    hub, root, calls = cache
    explicit(root)
    expected = {variant: native(hub, variant) for variant in engine._COMPONENTS}
    monkeypatch.setattr(engine, "_download_component", lambda *args: pytest.fail("unexpected download"))
    assert engine._resolve_checkpoints() == expected
    assert len(calls) == 3


@pytest.mark.parametrize("ref", [None, "absent", "../escape", "", ".."])
def test_missing_stale_or_unsafe_ref_falls_back(cache, ref):
    hub, root, _ = cache
    explicit(root)
    first = native(hub, "base", "a" * 40, ref=False)
    native(hub, "base", "z" * 40, ref=False)
    if ref is not None:
        refs = first.parent.parent / "refs"
        refs.mkdir()
        (refs / "main").write_text(ref)
    assert engine._resolve_checkpoints()["base"] == first


def test_main_ref_preferred_over_first_directory(cache):
    hub, root, _ = cache
    explicit(root)
    native(hub, "base", "a" * 40, ref=False)
    preferred = native(hub, "base", "z" * 40)
    assert engine._resolve_checkpoints()["base"] == preferred


def test_partial_snapshot_does_not_mask_complete_snapshot(cache):
    hub, root, _ = cache
    explicit(root)
    partial = native(hub, "base", "a" * 40)
    (partial / "vae.safetensors").unlink()
    valid = native(hub, "base", "b" * 40, ref=False)
    assert engine._resolve_checkpoints()["base"] == valid


def test_complete_explicit_cache_works_offline(cache):
    _, root, calls = cache
    expected = explicit(root)
    assert engine._resolve_checkpoints() == expected
    assert calls == []


def test_offline_missing_fails_without_download(cache, monkeypatch):
    monkeypatch.setattr(engine, "_download_component", lambda *args: pytest.fail("network attempted"))
    with pytest.raises(engine.EngineError, match="AUK_OFFLINE=1"):
        engine._resolve_checkpoints()


def test_download_only_missing_component_and_reuse(cache, monkeypatch):
    hub, root, _ = cache
    monkeypatch.setenv("AUK_OFFLINE", "0")
    native(hub, "flash")
    native(hub, "encoder")
    calls = []

    def download(repo_id, target, actual_hub, checkpoint, timeout):
        assert repo_id == "tencent/AuK"
        assert target == root / "AuK"
        assert actual_hub == hub
        assert 0 < timeout <= 900
        calls.append(repo_id)
        component(target, checkpoint)

    monkeypatch.setattr(engine, "_download_component", download)
    first = engine._resolve_checkpoints()
    assert engine._resolve_checkpoints() == first
    assert calls == ["tencent/AuK"]


def test_partial_download_is_rejected(cache, monkeypatch):
    monkeypatch.setenv("AUK_OFFLINE", "0")
    monkeypatch.setattr(engine, "_download_component", lambda *args: None)
    with pytest.raises(engine.EngineError, match="Incomplete checkpoint"):
        engine._resolve_checkpoints()


def test_encoder_requires_all_indexed_shards(tmp_path):
    path = component(tmp_path / "encoder", None)
    (path / "model.safetensors").unlink()
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
    (path / "one.safetensors").write_text("one")
    assert not engine._component_complete(path, None)
    (path / "two.safetensors").write_text("two")
    assert engine._component_complete(path, None)
    (path / "two.safetensors").write_text("")
    assert not engine._component_complete(path, None)


def test_corrupt_encoder_index_rejected(tmp_path):
    path = component(tmp_path / "encoder", None)
    (path / "model.safetensors.index.json").write_text("not json")
    assert not engine._component_complete(path, None)


def test_download_child_is_bounded_and_parent_remains_offline(cache, monkeypatch):
    hub, root, _ = cache
    monkeypatch.setenv("AUK_OFFLINE", "0")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    engine._download_component("tencent/AuK", root / "AuK", hub, "auk_base.safetensors", 123)
    args, kwargs = calls[0]
    assert "snapshot_download" in args[0][2]
    assert "local_dir=sys.argv[3]" in args[0][2]
    assert "local_files_only=False" in args[0][2]
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "0"
    assert kwargs["env"]["HF_HUB_CACHE"] == str(hub)
    assert kwargs["timeout"] == 123
    assert kwargs["check"] is True
    assert engine.os.environ["HF_HUB_OFFLINE"] == "1"


def test_download_kill_switch_checks_before_child(cache, monkeypatch):
    hub, root, _ = cache
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("child started"))
    with pytest.raises(engine.EngineError, match="AUK_OFFLINE=1"):
        engine._download_component("tencent/AuK", root / "AuK", hub, "auk_base.safetensors", 10)


def test_download_timeout_structured(cache, monkeypatch):
    hub, root, _ = cache
    monkeypatch.setenv("AUK_OFFLINE", "0")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("download", 10)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(engine.EngineError, match="TimeoutExpired"):
        engine._download_component("tencent/AuK", root / "AuK", hub, "auk_base.safetensors", 10)


def test_no_download_when_startup_budget_exhausted(cache, monkeypatch):
    monkeypatch.setenv("AUK_OFFLINE", "0")
    monkeypatch.setenv("RUNPOD_INIT_TIMEOUT", "300")
    monkeypatch.setattr(engine, "_download_component", lambda *args: pytest.fail("download attempted"))
    with pytest.raises(engine.EngineError, match="budget exhausted"):
        engine._resolve_checkpoints()


def test_hub_failure_during_native_confirmation_falls_through(cache, monkeypatch):
    hub, root, _ = cache
    explicit(root)
    native(hub, "base")

    def broken(**kwargs):
        raise RuntimeError("inconsistent hub metadata")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", broken)
    assert engine._resolve_checkpoints()["base"] == root / "AuK"


def test_non_numeric_init_timeout_uses_default_budget(cache, monkeypatch):
    _, root, _ = cache
    monkeypatch.setenv("RUNPOD_INIT_TIMEOUT", "not-a-number")
    expected = explicit(root)
    assert engine._resolve_checkpoints() == expected
