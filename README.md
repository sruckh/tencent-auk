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
      "instruction": "Based on the following description: \"A warm, professional narrator with a medium pace.\", generate speech content \"RunPod serverless is up and running.\".",
      "gen_seconds": 2.5
    }
  }'
```

AuK has **no separate text field** — the words to speak travel inside `instruction` using the upstream cookbook
templates: instruct TTS is `Based on the following description: "…", generate speech content "…".` and zero-shot
is `Say the following with the same voice: "…".`

**3. Get audio.** With credentials configured, `auto` delivery uploads to B2/S3 and returns a presigned URL;
without credentials it falls back to inline base64 WAV (`size_bytes`, `duration_seconds`, `sample_rate`, and the
effective `nfe` ride along as metadata). Prefer S3 for anything longer than a few seconds — RunPod's result
gateway rejects very large inline payloads.

### Cold starts & `/runsync`

The first job after a worker spawn waits through model loading (~90 s), which exceeds the default `/runsync`
sync window — the gateway returns `IN_PROGRESS`, and the worker's later delivery to the expired sync record is
rejected (the job then reports completed with **no output**). For the first job after any cold boot, either
raise the window with `?wait=300000` (5 min, milliseconds) or use `/run` + `/status/:id` polling; a warm worker
answers `/runsync` in a couple of seconds with no options needed.

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
| `response_delivery` | string | `"auto"` | `auto` \| `s3` \| `base64`; `auto` prefers presigned S3 when credentials are configured (RunPod rejects very large inline results), else base64 |

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

The worker returns **one flat object**. RunPod wraps it in the job envelope under `output`, so a client always
reads `job.output.*`:

```json
{
  "id": "a5a102c6-…-u2",
  "status": "COMPLETED",
  "delayTime": 125,
  "executionTime": 1595,
  "output": {
    "delivery": "s3",
    "audio_url": "https://s3.us-west-004.backblazeb2.com/Tencent-AuK/…?X-Amz-Signature=…",
    "bucket": "Tencent-AuK",
    "key": "AuK2026/09/12/a5a102c6-…-b98f49f1-….wav",
    "size_bytes": 48044,
    "duration_seconds": 1,
    "sample_rate": 24000,
    "model_variant": "flash",
    "nfe": 4,
    "task_executed": "instruct_tts",
    "url_expires_in": 86400,
    "url_expires_at": "2026-09-13T01:06:05Z"
  }
}
```

`sample_rate` is the rate the model actually returns per request (24 kHz in production; the test mock agrees).

**Base64** — `delivery="base64"`, plus `audio_base64` and `size_bytes`. Only safe for short clips: RunPod's
result gateway caps the inline response (a third-party source reports ~10 MB on `/run`, ~20 MB on `/runsync` —
**unverified against RunPod's own docs**), and exceeding it fails the delivery silently. Prefer S3.

**S3** — `delivery="s3"`, plus `audio_url`, `bucket`, `key`, `size_bytes`, `url_expires_in`, `url_expires_at`.
`audio_url` is a presigned GET, valid for `url_expires_in` seconds (default 24 h) — download or play it before
then. Keys follow `{prefix}{YYYY}/{MM}/{DD}/{sanitized_job_id}-{uuid4}.wav`, where the prefix is concatenated
**verbatim with no separator**: `auk/` yields `auk/2026/…`, but `AuK` yields `AuK2026/…`. `auto` picks this
whenever credentials are configured.

### Failures — read the error as a JSON *string*

A failed job reports `status: "FAILED"`. **`error` is a JSON string, not an object** — this is a RunPod platform
requirement, not a worker choice. `runpod-python` hoists any top-level `error` off the handler's return value
into the job's own error field, and the job-done gateway rejects a non-string there with `400 Bad Request`
(runpod-python#309). The worker therefore serializes its structured envelope into a string at the wire boundary:

```json
{
  "id": "…",
  "status": "FAILED",
  "error": "{\"code\": \"missing_required_field\", \"message\": \"instruction is required\", \"field\": \"instruction\"}"
}
```

**Parse it:**

```js
const detail = typeof job.error === "string" ? JSON.parse(job.error) : job.error;
// detail.code, detail.message, detail.field (field is optional)
```

Do not write `job.error.code` — it is `undefined`. The structured `code`/`message`/`field` survive verbatim
inside the string, so nothing is lost; it only needs one `JSON.parse`.

Error codes (closed set): `invalid_payload` · `missing_required_field` · `invalid_base64` · `audio_too_large` ·
`audio_download_failed` · `unsupported_model_variant` · `inference_failed` · `s3_credentials_missing` ·
`delivery_failed`.

Messages are safe to display: the engine and storage interpolate only the exception's **type name**, never its
own text, and credentials travel as masked `Secret` objects. No error ever carries audio bytes, a presigned URL,
or credential material.

A job can also come back `COMPLETED` with **no `output`** if the result could not be delivered to RunPod at all
(for example the `/runsync` window expired mid-job — see **Cold starts & `/runsync`**). Treat "completed but no `output`" as a
failure and retry rather than reading fields off it.

## Calling it from a front end

**The RunPod API key must never reach the browser.** Every request authenticates with
`Authorization: Bearer <RUNPOD_API_KEY>`, and an endpoint id + key pair can spend your GPU minutes — shipping
it in client-side code exposes it to anyone who opens devtools. Route browser calls through a small backend you
control:

```
browser  ──POST /api/tts──▶  your backend  ──Bearer key──▶  api.runpod.ai/v2/<endpoint-id>
         ◀──{ audioUrl }──                ◀──job JSON──
```

The backend additionally lets you hold the API key server-side, rate-limit callers, and swap the endpoint id
without a front-end redeploy.

**Prefer async `/run` + poll for anything user-facing.** `/runsync` holds the connection open, so a cold start
(~90 s, see below) blocks it and can time out. `/run` returns an id immediately, which suits a loading state:

```js
// backend
const r = await fetch(`https://api.runpod.ai/v2/${ENDPOINT}/run`, {
  method: "POST",
  headers: { Authorization: `Bearer ${RUNPOD_API_KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({ input: { task: "instruct_tts", instruction, gen_seconds: 2.5 } }),
});
const { id } = await r.json();

// poll until terminal — IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED | CANCELLED | TIMED_OUT
while (true) {
  const job = await (await fetch(`https://api.runpod.ai/v2/${ENDPOINT}/status/${id}`, {
    headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` },
  })).json();
  if (job.status === "COMPLETED" && job.output) return { ok: true, ...job.output };
  if (job.status === "COMPLETED") return { ok: false, code: "no_output" };   // delivery lost
  if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(job.status)) {
    const detail = typeof job.error === "string" ? JSON.parse(job.error) : job.error;
    return { ok: false, ...detail };
  }
  await new Promise((res) => setTimeout(res, 1000));   // back off in production
}
```

**Playing the result.** Both delivery modes produce something an `<audio>` element can use directly:

```js
// `result` is the flattened success object from the poll above ({ ok: true, ...job.output })

// S3 (the default whenever credentials are configured) — presigned, valid ~24 h
audioEl.src = result.audio_url;

// base64 fallback — wrap the bytes in a Blob
const bytes = Uint8Array.from(atob(result.audio_base64), (c) => c.charCodeAt(0));
audioEl.src = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
```

`audio_url` expires (`url_expires_at` is an ISO-8601 UTC timestamp). Download or cache the audio if the user
might replay it long after the request — or ask for `response_delivery: "base64"` when the clip is short.

**Design the UI for a cold start.** A freshly booted worker loads models before the first job (~90 s); warm
workers answer in ~1.5–2.5 s. Either keep a worker warm (`workers_min: 1`) or show honest progress rather than a
spinner that looks hung. Sending `"response_delivery": "s3"` is worthwhile for any clip beyond a few seconds.

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
| `AUK_ENCODER_BF16` | `1` | `0` keeps upstream's fp32 encoder weights (see **How it works → Variants**); set only when output must be byte-comparable to the unmodified baseline |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | allocator fragmentation headroom. Note the spelling is version-specific — torch 2.8 uses `PYTORCH_CUDA_ALLOC_CONF`; `PYTORCH_ALLOC_CONF` is a later rename and is a silent no-op here |

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
1.0..5.0 (default 2.0).

**GPU sizing matters more than the variant count.** Measured on an RTX PRO 6000 with both variants resident:

```
[auk-engine] vram before generate task=instruct_tts variant=flash: free=59.1 of 95.0 GiB | torch_alloc=27.6 peak=27.6 GiB
[auk-engine] vram after  generate task=instruct_tts variant=flash: free=59.0 of 95.0 GiB | torch_alloc=28.1 peak=30.5 GiB
```

So each resident variant costs **~13.8 GiB** — every `AukInfer` owns its own encoder + VAE + DiT — plus a
**~2.9 GiB** per-job working set. Placement is decided at startup: the default variant loads eagerly and the
second joins only if `SECOND_VARIANT_MIN_FREE_GIB` (25) stays free.

| Card | One variant + job (~16.7 GiB) | Both variants + job (~30.5 GiB) |
|------|:---:|:---:|
| 24 GB class (RTX 4090, 23.5 GiB) | yes | no |
| 48 GB class (A40 / L40S) | yes | yes |
| 80 GB class (A100) | yes | yes |
| 96 GB (RTX PRO 6000) | yes | yes |

A request for a variant that is not resident builds it on demand and fails with a structured `inference_failed`
if it does not fit.

These figures already include an encoder fix: upstream loads the Qwen text encoder in bf16 and then casts the
whole model — encoder included — to fp32, so the worker returns the encoder to bf16 at startup (opt out with
`AUK_ENCODER_BF16=0`). That is what brought the per-variant cost down from ~21.4 GiB. Note the downcast strands
the old fp32 blocks in torch's caching allocator, so the worker flushes the cache immediately after; **if you
remove or reorder that flush, expect the freed memory not to reach the driver** — measured at 7.6 GiB stranded
across two variants.

Treat these numbers as a starting point, not a guarantee: read the `[auk-engine] vram` lines from your own
deploy before sizing a pool.

## Development

```bash
AUK_TEST_MOCK_ENGINE=1 python3 -m pytest -v tests/   # 163 tests, stdlib-only, no GPU/network
AUK_TEST_MOCK_ENGINE=1 python3 handler.py --test_input '{"input": {"instruction": "hello"}}'
```

- **Vendored upstream**: complete clean export of [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK)
  at `bc84b758c7566dfce9bef0c49fba443456093b40` (MIT) under `src/` — imported by the built image, never edited.
  API: `from auk.infer.infer_auk import AukInfer, save_audio`.
- **Torch note**: the image pins torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0 (cu128) — a recorded
  deviation from upstream's declared `torch>=2.7,<2.8`, so the vendored package installs `--no-deps` with its
  remaining inference deps owned by `requirements.txt`. No flash-attention wheel: inference pins
  `attn_backend="torch"`.
- **Where the contracts live**: the behavioral contracts (payload shapes, error codes, placement policy,
  storage field sets) are pinned in a local Interpretable-Context-Methodology workspace under `.icm/`. That tree
  is **not committed** — it is development context, not repository or image content — so a fresh clone will not
  contain it. In a clone, the authoritative sources are the code, `tests/`, and this README.

## License

The vendored upstream AuK code is MIT-licensed (retained in `src/LICENSE`). This repository does not yet carry a
separate license file for the worker code.
