"""Stage 04 acceptance — storage.py against
.icm/references/s3-storage.md and the stage 04 output spec."""

import base64
import datetime as dt

import pytest

import storage
from storage import DeliveryError, Secret, StorageConfig, build_key, deliver, resolve_delivery, sanitize_job_id

CREDS_ENV = {
    "AUDIO_DELIVERY": "auto",
    "S3_ENDPOINT_URL": "https://s3.us-west-004.backblazeb2.com",
    "S3_BUCKET": "auk-audio-production",
    "S3_REGION": "us-west-004",
    "AWS_ACCESS_KEY_ID": "AKIA-LIVE-KEY",
    "AWS_SECRET_ACCESS_KEY": "super-secret-aws-key",
    "S3_KEY_PREFIX": "auk/",
    "S3_PRESIGN_EXPIRY_SECONDS": "86400",
    "S3_ADDRESSING_STYLE": "path",
}

WAV = b"RIFF____WAVEfake-pcm-data"


def cfg(**overrides) -> StorageConfig:
    env = {**CREDS_ENV, **overrides}
    for key, value in list(overrides.items()):
        if value is None:
            env.pop(key, None)
    return StorageConfig.from_env(env)


# ---------------------------------------------------------------------------
# Secret shield (ADR 003 §4)
# ---------------------------------------------------------------------------

def test_secret_masks_repr_and_str():
    secret = Secret("super-secret-aws-key")
    assert str(secret) == "Secret(***)"
    assert repr(secret) == "Secret(***)"
    assert secret.reveal() == "super-secret-aws-key"


def test_secret_never_leaks_through_config_repr():
    config = cfg()
    rendered = f"{config!r} {config.access_key_id!r} {config}"
    assert "super-secret-aws-key" not in rendered
    assert "AKIA-LIVE-KEY" not in rendered
    assert "Secret(***" in rendered


def test_from_env_defaults():
    config = StorageConfig.from_env({})
    assert config.delivery_default == "auto"
    assert config.key_prefix == "auk/"
    assert config.presign_expiry == 86400
    assert config.addressing_style == "path"
    assert not config.s3_configured()


def test_invalid_env_delivery_falls_back_to_auto():
    assert cfg(AUDIO_DELIVERY="fax").delivery_default == "auto"


# ---------------------------------------------------------------------------
# Delivery resolution semantics
# ---------------------------------------------------------------------------

def test_resolve_delivery_explicit_base64():
    assert resolve_delivery("base64", cfg()) == "base64"


def test_resolve_delivery_forced_s3():
    assert resolve_delivery("s3", cfg()) == "s3"


def test_resolve_delivery_forced_s3_without_creds_raises():
    with pytest.raises(DeliveryError) as excinfo:
        resolve_delivery("s3", cfg(AWS_ACCESS_KEY_ID=None, AWS_SECRET_ACCESS_KEY=None))
    assert excinfo.value.code == "s3_credentials_missing"


def test_resolve_delivery_auto_falls_back_to_base64():
    config = cfg(AWS_ACCESS_KEY_ID=None, AWS_SECRET_ACCESS_KEY=None)
    assert config.delivery_default == "auto"
    assert resolve_delivery("auto", config) == "base64"  # auto never raises


def test_resolve_delivery_auto_uses_env_default():
    assert resolve_delivery("auto", cfg(AUDIO_DELIVERY="s3")) == "s3"
    assert resolve_delivery(None, cfg(AUDIO_DELIVERY="base64")) == "base64"


def test_resolve_delivery_auto_env_s3_without_creds_falls_back():
    config = cfg(AUDIO_DELIVERY="s3", AWS_SECRET_ACCESS_KEY=None)
    assert resolve_delivery("auto", config) == "base64"


def test_resolve_delivery_auto_prefers_s3_when_configured():
    """Pin s3-storage.md: auto prefers presigned S3 when credentials exist."""
    assert resolve_delivery("auto", cfg()) == "s3"
    assert resolve_delivery(None, cfg()) == "s3"


def test_resolve_delivery_deployment_s3_without_creds_raises():
    """AUDIO_DELIVERY=s3 adopts forced-s3 semantics — fail loudly."""
    config = cfg(AUDIO_DELIVERY="s3", AWS_ACCESS_KEY_ID=None, AWS_SECRET_ACCESS_KEY=None)
    with pytest.raises(DeliveryError) as excinfo:
        resolve_delivery(None, config)
    assert excinfo.value.code == "s3_credentials_missing"


# ---------------------------------------------------------------------------
# Key building + sanitization
# ---------------------------------------------------------------------------

def test_sanitize_job_id():
    assert sanitize_job_id("job_123") == "job_123"
    assert sanitize_job_id("../../etc/passwd") == "_.._etc_passwd"
    assert "/" not in sanitize_job_id("a/b\\c:d")
    assert "/" not in sanitize_job_id("a..b/../..")
    assert sanitize_job_id("") == "job"
    assert not sanitize_job_id(".../..").startswith(".")


def test_sanitize_job_id_traversal_safe():
    safe = sanitize_job_id("../../etc/passwd")
    key = f"auk/2026/09/10/{safe}-x.wav"
    # No path component may equal "." or ".." — traversal needs a separator,
    # and the sanitized id contains none.
    assert ".." not in key.split("/")
    assert "." not in key.split("/")
    assert key.count("/") == 4  # no traversal past the prefix


def test_build_key_template():
    now = dt.datetime(2026, 9, 10, 13, 45, 0, tzinfo=dt.timezone.utc)
    key = build_key("job_123", cfg(), now=now, uuid4="d8e9f00d")
    assert key == "auk/2026/09/10/job_123-d8e9f00d.wav"


def test_build_key_sanitizes_job_id():
    now = dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc)
    key = build_key("../evil/job", cfg(), now=now, uuid4="u")
    assert key == "auk/2026/09/10/_evil_job-u.wav"
    assert not key.rsplit("/", 1)[-1].startswith(".")  # final segment hides nothing above the prefix


# ---------------------------------------------------------------------------
# deliver() — field sets, upload, presign, failure mapping
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self, record, fail=False):
        self._record = record
        self._fail = fail

    def put_object(self, **kwargs):
        if self._fail:
            raise RuntimeError("connection reset")
        self._record["put"] = kwargs

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):
        self._record["presign"] = {"op": operation, "params": Params, "expires_in": ExpiresIn}
        return f"https://signed.example/{Params['Key']}?X-Amz-Signature=tok"


def make_factory(record, fail=False):
    return lambda cfg_: record.setdefault("client", FakeClient(record, fail=fail)) or record["client"]


def test_deliver_base64_fields():
    result = deliver(WAV, "job_1", "base64", cfg=cfg())
    assert set(result) == {"delivery", "audio_base64", "size_bytes"}
    assert result["delivery"] == "base64"
    assert base64.b64decode(result["audio_base64"]) == WAV
    assert result["size_bytes"] == len(WAV)


def test_deliver_auto_without_creds_returns_base64():
    config = cfg(AWS_ACCESS_KEY_ID=None, AWS_SECRET_ACCESS_KEY=None)
    result = deliver(WAV, "job_1", "auto", cfg=config)
    assert result["delivery"] == "base64"


def test_deliver_auto_uses_s3_when_configured():
    record = {}
    result = deliver(WAV, "job_1", "auto", cfg=cfg(), client_factory=make_factory(record))
    assert result["delivery"] == "s3"
    assert record["put"]["Body"] == WAV
    assert result["audio_url"].startswith("https://signed.example/")


def test_deliver_s3_fields_and_upload():
    record = {}
    result = deliver(WAV, "job_123", "s3", cfg=cfg(), client_factory=make_factory(record))
    assert set(result) == {
        "delivery", "audio_url", "bucket", "key",
        "size_bytes", "url_expires_in", "url_expires_at",
    }
    assert result["delivery"] == "s3"
    assert result["bucket"] == "auk-audio-production"
    assert result["size_bytes"] == len(WAV)
    assert result["url_expires_in"] == 86400
    assert result["url_expires_at"].endswith("Z")  # ISO-8601 UTC
    assert result["audio_url"].startswith("https://signed.example/")

    put = record["put"]
    assert put["Bucket"] == "auk-audio-production"
    assert put["Body"] == WAV
    assert put["ContentType"] == "audio/wav"
    assert put["Key"].startswith("auk/") and put["Key"].endswith(".wav")

    presign = record["presign"]
    assert presign["op"] == "get_object"
    assert presign["params"] == {"Bucket": "auk-audio-production", "Key": put["Key"]}
    assert presign["expires_in"] == 86400
    assert result["key"] == put["Key"]


def test_deliver_s3_b2_config_pins():
    config = cfg()
    pins = storage._s3_client_config(config)
    assert pins["signature_version"] == "s3v4"
    assert pins["request_checksum_calculation"] == "when_required"
    assert pins["response_checksum_validation"] == "when_required"
    assert pins["s3"] == {"addressing_style": "path"}


def test_deliver_s3_presign_expiry_override():
    record = {}
    config = cfg(S3_PRESIGN_EXPIRY_SECONDS="3600")
    result = deliver(WAV, "j", "s3", cfg=config, client_factory=make_factory(record))
    assert result["url_expires_in"] == 3600
    assert record["presign"]["expires_in"] == 3600


def test_deliver_upload_failure_maps_to_delivery_failed():
    record = {}
    with pytest.raises(DeliveryError) as excinfo:
        deliver(WAV, "job_1", "s3", cfg=cfg(), client_factory=make_factory(record, fail=True))
    assert excinfo.value.code == "delivery_failed"


def test_deliver_key_uses_fresh_uuid(monkeypatch):
    """Two uploads for the same job never share a key (uuid4 per upload)."""
    keys = []
    config = cfg()
    for _ in range(2):
        record = {}
        result = deliver(WAV, "same-job", "s3", cfg=config, client_factory=make_factory(record))
        keys.append(result["key"])
    assert keys[0] != keys[1]
