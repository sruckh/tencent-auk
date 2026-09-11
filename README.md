<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="tencent-auk — RunPod Serverless worker: one JSON job in, AuK speech audio out">
</p>

# tencent-auk

A RunPod Serverless worker for [Tencent AuK](https://github.com/Tencent-Hunyuan/AuK) (Audio Swiss Knife): one
JSON job in, speech audio out. Eight task modes cover zero-shot voice cloning, instruct TTS, speech content
editing, acoustic editing, paralinguistic editing, and audio enhancement/separation — served by a warm,
in-memory engine with dual model variants (**AuK-Flash**, a fixed 4-step distilled DiT, and **AuK-Base**, a
flow-matching DiT) and delivered as inline base64 or a presigned B2/S3 URL.

## Quick start

**1. Deploy.** RunPod builds the image from this repo's `Dockerfile`. Before the first deploy:

- Attach a **network volume** to the endpoint.
- Enable **native model caching** for all three model repos — `tencent/AuK`, `tencent/AuK-Flash`, and
  `Qwen/Qwen2.5-Omni-3B` — so RunPod pre-downloads them into the volume before the worker starts.
- Set the runtime secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as endpoint environment variables
  (never baked into the image).

**2. Call it.** RunPod exposes the serverless API at `https://api.runpod.ai/v2/<endpoint-id>`:

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "instruct_tts",
      "instruction": "A warm narrator reads the news.",
      "gen_seconds": 6.0
    }
  }'
```

**3. Get audio.** With `AUDIO_DELIVERY=auto` and no S3 credentials configured, the result arrives inline as
base64 WAV (`size_bytes`, `duration_seconds`, `sample_rate`, and the effective `nfe` ride along as metadata).

## Endpoints

The worker registers **one** RunPod handler; the platform routes below are provisioned automatically for every
serverless endpoint at `https://api.runpod.ai/v2/<endpoint-id>`:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/run` | queue a job (async; returns `id` + status) |
| `POST` | `/runsync` | queue and block until the result is ready |
| `GET` | `/status/:id` | poll a job's status / fetch its result |
| `POST` | `/cancel/:id` | cancel a queued or running job |
| `GET` | `/health` | worker liveness + concurrency snapshot |

There are no worker-internal HTTP routes — the entire API surface is the `input` dict of the job payload.

## The job contract (`input`)

Fail-fast validation with a closed error set; first failing rule wins.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `task` | string | `"auto"` | one of the 8 modes; see matrix below |
| `instruction` | string | — | **required** — natural-language directive |
| `audio` | string | — | source clip for edit/enhance/separate; base64 or `http(s)://` URL |
| `prompt_audio` | string | — | zero-shot reference clip; base64 or URL |
| `prompt_text` | string | — | exact transcript of `prompt_audio`; valid only alongside it |
| `gen_seconds` | number | heuristic | `0.5..300.0`; or text via `gen_text` heuristic |
| `gen_text` | string | — | text feeding the duration heuristic |
| `model_variant` | string | `DEFAULT_MODEL_VARIANT` | `flash` \| `base` |
| `nfe` | int | 4 (flash) / 32 (base) | flash accepts `1..8`; base `16..64` |
| `cfg_scale` | number | 0.0 (flash) / 2.0 (base) | forced to `0.0` for flash; base `1.0..5.0` |
| `seed` | int | — | ≥ 0; forwarded for reproducibility |
| `response_delivery` | string | `"auto"` | `auto` \| `s3` \| `base64` |

Audio fields accept base64 (optionally `data:audio/…;base64,`-prefixed) or `http(s)://` URLs. Each clip is
decoded **exactly once**; the 15 MB cap applies to decoded bytes; URL ingest streams with a 10 s timeout.

## Task modes

| Task | `instruction` | `audio` | `prompt_audio` | `prompt_text` | What it does |
|------|:---:|:---:|:---:|:---:|------|
| `zero_shot_tts` | required | forbidden | **required** | optional | clone a voice from a reference clip |
| `instruct_tts` | required | forbidden | forbidden | forbidden | voice design from the instruction alone |
| `content_edit` | required | **required** | forbidden | forbidden | word replace / insert / delete, lyric swap |
| `acoustic_edit` | required | **required** | forbidden | forbidden | semitone pitch, speed scale, dB volume |
| `paralinguistic_edit` | required | **required** | forbidden | forbidden | emotion, timbre, de-accent, whisper, nonverbal cues |
| `enhancement` | required | **required** | forbidden | forbidden | denoise, dereverb |
| `separation` | required | **required** | forbidden | forbidden | multi-speaker / singing extraction, target-speaker extraction |
| `auto` | required | resolver | resolver | resolver | see below |

**`task: "auto"` resolution** — explicit task wins; otherwise `prompt_audio` → `zero_shot_tts`; neither audio
field → `instruct_tts`; bare `audio` → `invalid_payload` (edit intents are indistinguishable — the worker
refuses to guess).

## Responses

One top-level object; metadata (`task_executed`, `model_variant`, `nfe`, `sample_rate`, `duration_seconds`,
`size_bytes`) is merged in. `sample_rate` is the rate the model returns per request (the test mock runs 24 kHz).

**Base64** — `delivery="base64"`, `audio_base64`, `size_bytes`.

**S3** — `delivery="s3"`, `audio_url`, `bucket`, `key`, `size_bytes`, `url_expires_in`, `url_expires_at`.
Keys follow `{prefix}{YYYY}/{MM}/{DD}/{sanitized_job_id}-{uuid4}.wav`; presigned GETs default to 24 h.

**Failure** — structured envelope, no credential material, no audio bytes:

```json
{"error": {"code": "missing_required_field", "message": "…", "field": "instruction"}}
```

Error codes (closed set): `invalid_payload` · `missing_required_field` · `invalid_base64` · `audio_too_large` ·
`audio_download_failed` · `unsupported_model_variant` · `inference_failed` · `s3_credentials_missing` ·
`delivery_failed`.

## Environmental variables

Values shown are the baked-in image defaults (`Dockerfile`); override per deployment via RunPod endpoint
environment variables. The two credential variables are **runtime-only** — never baked into the image.

### Engine & checkpoints

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_MODEL_VARIANT` | `flash` | variant loaded first; the other loads lazily |
| `CKPT_ROOT` | `/runpod-volume/ckpts` | designated checkpoint layout, tier 2 of resolution |
| `HF_HUB_CACHE` | `/runpod-volume/huggingface-cache/hub` | native RunPod model-cache location, tier 1 |
| `HF_HUB_OFFLINE` | `1` | parent inference process never talks to the Hub |
| `TRANSFORMERS_OFFLINE` | `1` | transformers likewise resolves locally only |
| `AUK_OFFLINE` | `0` | `1` forbids the download fallback entirely (fail fast) |
| `RUNPOD_INIT_TIMEOUT` | `1200` | platform init budget; downloads bounded to `min(900, this − 300)` s |

### Platform / process

| Variable | Default | Purpose |
|----------|---------|---------|
| `RUNPOD_LOG_LEVEL` | `INFO` | SDK log level |
| `PYTHONUNBUFFERED` | `1` | unbuffered stdout (crash dumps, health logs) |
| `PYTHONDONTWRITEBYTECODE` | `1` | no `.pyc` writes in the container |

### Storage & delivery

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUDIO_DELIVERY` | `auto` | deployment default; `auto` without S3 creds → base64 |
| `S3_ENDPOINT_URL` | `https://s3.us-west-004.backblazeb2.com` | B2 S3-compatible endpoint |
| `S3_BUCKET` | `auk-audio-production` | delivery bucket (placeholder — override per environment) |
| `S3_REGION` | `us-west-004` | B2 region |
| `S3_KEY_PREFIX` | `auk/` | key prefix |
| `S3_PRESIGN_EXPIRY_SECONDS` | `86400` | presigned GET lifetime (24 h) |
| `S3_ADDRESSING_STYLE` | `path` | B2 requires path-style |
| `AWS_ACCESS_KEY_ID` | — | **runtime secret** — endpoint env only |
| `AWS_SECRET_ACCESS_KEY` | — | **runtime secret** — endpoint env only |

### Development / test

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUK_TEST_MOCK_ENGINE` | — | `1` swaps the engine for a stdlib-only mock: no GPU, no torch, no `auk`, no network |

## How it works

<p align="center">
  <img src="./assets/readme/architecture.svg" width="100%" alt="Bootstrap resolves checkpoints offline-first from the network volume and builds warm engines; each job is validated, synthesized, and delivered">
</p>

**Bootstrap (import time, once).** The engine resolves all three components — `tencent/AuK`, `tencent/AuK-Flash`,
`Qwen/Qwen2.5-Omni-3B` — offline-first from the native cache: `refs/main` → snapshot, falling back to the first
complete snapshot in sorted order. Complete-file checks (including every shard in the safetensors index) reject
partial downloads. A complete `CKPT_ROOT` directory is reused next; only then may a missing component use one
bounded `snapshot_download(..., local_dir=...)` into the designated location. `AUK_OFFLINE=1` forbids that
fallback. The inference process itself never talks to the network — downloads run in a bounded bootstrap child.

**Per job (hot path).** Fail-fast validation → the already-decoded clip (if any) is written to one temp file →
cookbook-style messages with the optional transcript carried as labeled reference text → `generate(messages,
gen_seconds, nfe, cfg_strength, seed)` → the returned `(waveform, sample_rate)` is serialized as mono 16-bit PCM
WAV in memory → delivery. Temp files are removed in `finally`; the handler never dies from a malformed job.

**Variants.** AuK-Flash runs its distilled recipe — exactly 4 steps, CFG 0.0 — regardless of the accepted input
range `1..8`; metadata reports the effective `nfe=4`. AuK-Base honors `nfe` 16..64 (default 32) and `cfg_scale`
1.0..5.0 (default 2.0). At ≥ 20 GiB VRAM both variants load eagerly; below that the default loads first and the
other loads lazily on first use.

## Development

```bash
AUK_TEST_MOCK_ENGINE=1 python3 -m pytest -v tests/   # 139 tests, stdlib-only, no GPU/network
AUK_TEST_MOCK_ENGINE=1 python3 handler.py --test_input '{"input": {"instruction": "hello"}}'
```

- **Vendored upstream**: complete clean export of [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK)
  at `bc84b758c7566dfce9bef0c49fba443456093b40` (MIT) under `src/` — imported by the built image, never edited.
  API: `from auk.infer.infer_auk import AukInfer, save_audio`.
- **Torch note**: the image pins torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0 (cu128) — a recorded
  deviation from upstream's declared `torch>=2.7,<2.8`, so the vendored package installs `--no-deps` with its
  remaining inference deps owned by `requirements.txt`. No flash-attention wheel: inference pins
  `attn_backend="torch"`.
- **Contracts**: the `.icm/` workspace pins every behavioral contract; `python3 /root/.claude/skills/icm/scripts/audit .icm`
  validates the workspace.

## License

The vendored upstream AuK code is MIT-licensed (retained in `src/LICENSE`). This repository does not yet carry a
separate license file for the worker code.
