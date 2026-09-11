"""Stage 05 acceptance — spec-level container contract checks (no Docker
build in CI): the Dockerfile and requirements.txt carry the pinned stack."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_pins():
    text = (REPO_ROOT / "requirements.txt").read_text()
    assert "runpod==" in text
    assert "boto3==" in text
    assert "botocore>=1.36" in text  # hard floor — B2 checksum fix
    assert "huggingface_hub>=" in text
    assert "soundfile>=" in text
    # Upstream inference deps, worker-owned because the vendored package
    # installs --no-deps (torch range deviation — see stage 05 spec).
    for dep in ("transformers>=", "qwen-omni-utils>=", "omegaconf>=",
                "torchdiffeq", "x_transformers>=", "safetensors"):
        assert dep in text
    packages = [line.split("#")[0].strip() for line in text.splitlines() if line and not line.startswith("#")]
    # torchdiffeq is a legit upstream dep — exclude only the torch GPU family.
    assert not any(line.startswith(("torch==", "torch>", "torch<", "torchvision",
                                    "torchaudio", "numpy", "flash", "pytest", "hf_transfer"))
                   for line in packages)


def test_python_version_and_package_layout_match_upstream():
    import tomllib
    package = tomllib.loads((REPO_ROOT / "src/pyproject.toml").read_text())
    assert package["project"]["requires-python"] == ">=3.10"
    assert (REPO_ROOT / "src/.python-version").read_text().strip() == "3.10"
    assert "python3.12 -m venv" in (REPO_ROOT / "Dockerfile").read_text()
    assert package["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert (REPO_ROOT / "src/src/auk/infer/infer_auk.py").is_file()
    assert (REPO_ROOT / "src/LICENSE").is_file()
    assert not (REPO_ROOT / "src/.git").exists()


def test_dockerfile_base_image():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "FROM nvidia/cuda:12.8.2-devel-ubuntu24.04" in text


def test_dockerfile_audio_codecs_lean():
    text = (REPO_ROOT / "Dockerfile").read_text()
    for package in ("ffmpeg", "libsndfile1", "sox"):
        assert package in text
    assert "--no-install-recommends" in text


def test_dockerfile_no_flash_attn():
    text = (REPO_ROOT / "Dockerfile").read_text()
    source = (REPO_ROOT / "src/src/auk/infer/infer_auk.py").read_text()
    assert 'model_arc["attn_backend"] = "torch"' in source  # inference path needs no flash-attn
    assert "flash_attn" not in text
    assert "pip install flash" not in text
    assert "&& pip check" not in text  # deviation accepted — a real check run would fail the build


def test_dockerfile_pytorch_cuda_index():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0" in text
    assert "download.pytorch.org/whl/cu128" in text
    import tomllib
    project = tomllib.loads((REPO_ROOT / "src/pyproject.toml").read_text())["project"]
    # Recorded deviation: the image's 2.8 line is OUTSIDE upstream's tested
    # 2.7 ABI range (exclusive <2.8 bound) — handled by --no-deps install.
    assert "torch>=2.7,<2.8" in set(project["dependencies"])


def test_dockerfile_upstream_package_and_worker_files():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "COPY src/ /app/src/" in text
    assert "pip install --no-deps /app/src" in text  # no resolution: torch deviation
    assert "from auk.infer.infer_auk import AukInfer" in text  # import smoke, image build only
    assert text.index("pip install --no-deps /app/src") < text.index("COPY schema_validator.py")
    assert "COPY schema_validator.py engine.py storage.py handler.py /app/" in text


def test_dockerfile_env_and_volume():
    text = (REPO_ROOT / "Dockerfile").read_text()
    for pinned in (
        "CKPT_ROOT=/runpod-volume/ckpts",
        "HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "AUK_OFFLINE=0",
        "DEFAULT_MODEL_VARIANT=flash",
        "RUNPOD_INIT_TIMEOUT=1200",
        "AUDIO_DELIVERY=auto",
        "S3_ADDRESSING_STYLE=path",
        # Fragmentation remedy for the 24 GB-class OOM; the 2.8 spelling, which
        # a later rename would silently invalidate (see auk-architecture.md).
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    ):
        assert pinned in text
    assert "VOLUME /runpod-volume" in text
    assert "HF_HOME=" not in text
    assert "/runpod-volume/hf-cache" not in text


def test_dockerfile_launch_contract():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert 'CMD ["python3", "-u", "handler.py"]' in text


def test_dockerfile_no_baked_credentials():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "AWS_SECRET_ACCESS_KEY=" not in text
    assert "AWS_ACCESS_KEY_ID=" not in text
