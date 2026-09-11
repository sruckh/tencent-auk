"""Audio delivery for the Tencent AuK RunPod worker.

Implements the S3/B2 presigned-URL contract in `.icm/references/s3-storage.md`
(stage 04 of the ICM pipeline in `.icm/`): the ``Secret`` credential shield,
the ``AUDIO_DELIVERY`` resolution semantics, the pinned key template, and the
mandatory Backblaze B2 botocore configuration.
"""

from __future__ import annotations

import base64
import datetime as _dt
import os
import re
import uuid
from typing import Mapping


class DeliveryError(Exception):
    """Structured delivery failure (code from the closed error-code set)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Secret:
    """Credential wrapper that never renders its value (ADR 003 §4)."""

    __slots__ = ("_value",)

    _value: str

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The only sanctioned read — at the point of use (boto3.client)."""
        return self._value

    def __str__(self) -> str:
        return "Secret(***)"

    def __repr__(self) -> str:
        return "Secret(***)"


DELIVERY_OPTIONS = ("auto", "s3", "base64")
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_job_id(job_id: str) -> str:
    """Directory-traversal guard: keep ``[A-Za-z0-9._-]`` only."""
    cleaned = _UNSAFE_KEY_CHARS.sub("_", str(job_id)).lstrip("./")
    return cleaned or "job"


def build_key(job_id: str, cfg: "StorageConfig", *, now: _dt.datetime | None = None, uuid4: str | None = None) -> str:
    """Pinned template: ``{prefix}{YYYY}/{MM}/{DD}/{sanitized_job_id}-{uuid4}.wav``."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    uid = uuid4 or str(uuid.uuid4())
    return f"{cfg.key_prefix}{now:%Y}/{now:%m}/{now:%d}/{sanitize_job_id(job_id)}-{uid}.wav"


class StorageConfig:
    """Env-backed storage settings; credentials are `Secret`-wrapped at read."""

    __slots__ = (
        "delivery_default",
        "endpoint_url",
        "bucket",
        "region",
        "key_prefix",
        "presign_expiry",
        "addressing_style",
        "access_key_id",
        "secret_access_key",
    )

    def __init__(
        self,
        *,
        delivery_default: str = "auto",
        endpoint_url: str | None = None,
        bucket: str | None = None,
        region: str | None = None,
        key_prefix: str = "auk/",
        presign_expiry: int = 86400,
        addressing_style: str = "path",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        if delivery_default not in DELIVERY_OPTIONS:
            delivery_default = "auto"
        self.delivery_default = delivery_default
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.region = region
        self.key_prefix = key_prefix
        self.presign_expiry = int(presign_expiry)
        self.addressing_style = addressing_style
        self.access_key_id = Secret(access_key_id) if access_key_id else None
        self.secret_access_key = Secret(secret_access_key) if secret_access_key else None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "StorageConfig":
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            delivery_default=source.get("AUDIO_DELIVERY", "auto"),
            endpoint_url=source.get("S3_ENDPOINT_URL"),
            bucket=source.get("S3_BUCKET"),
            region=source.get("S3_REGION"),
            key_prefix=source.get("S3_KEY_PREFIX", "auk/"),
            presign_expiry=int(source.get("S3_PRESIGN_EXPIRY_SECONDS", "86400")),
            addressing_style=source.get("S3_ADDRESSING_STYLE", "path"),
            access_key_id=source.get("AWS_ACCESS_KEY_ID"),
            secret_access_key=source.get("AWS_SECRET_ACCESS_KEY"),
        )

    def s3_configured(self) -> bool:
        return bool(self.endpoint_url and self.bucket and self.access_key_id and self.secret_access_key)


# Module-scope production config (tests inject their own StorageConfig).
_CFG = StorageConfig.from_env()


def _s3_client_config(cfg: StorageConfig) -> dict[str, object]:
    """The mandatory B2 fix, as a pure mapping (pinned verbatim in
    `.icm/references/s3-storage.md`; production passes it to
    ``botocore.config.Config(**_s3_client_config(cfg))``)."""
    return {
        "signature_version": "s3v4",
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
        "s3": {"addressing_style": cfg.addressing_style},
    }


def default_client_factory(cfg: StorageConfig):
    """Production client. ``boto3``/``botocore`` are imported lazily so CI can
    import this module and exercise every non-S3 path without them."""
    if not cfg.s3_configured() or cfg.access_key_id is None or cfg.secret_access_key is None:
        raise DeliveryError("s3_credentials_missing", "storage credentials are not configured")
    import boto3  # deferred: delivery-time dependency

    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id.reveal(),        # point of use
        aws_secret_access_key=cfg.secret_access_key.reveal(),  # point of use
        region_name=cfg.region,
        config=_s3_client_config(cfg),
    )


def resolve_delivery(response_delivery: str | None, cfg: StorageConfig) -> str:
    """``auto`` defers to the env default and never raises; forced ``s3``
    without credentials raises ``s3_credentials_missing`` (fail loudly)."""
    requested = response_delivery or "auto"
    if requested == "s3":
        if not cfg.s3_configured():
            raise DeliveryError("s3_credentials_missing", "S3 delivery requested but storage credentials are not configured")
        return "s3"
    if requested == "base64":
        return "base64"

    default = cfg.delivery_default
    if default == "s3" and cfg.s3_configured():
        return "s3"
    return "base64"  # auto never raises on missing credentials


def _expires_at(presign_expiry: int, now: _dt.datetime) -> str:
    return (now + _dt.timedelta(seconds=presign_expiry)).strftime("%Y-%m-%dT%H:%M:%SZ")


def deliver(
    wav_bytes: bytes,
    job_id: str,
    response_delivery: str | None,
    cfg: StorageConfig | None = None,
    client_factory=None,
) -> dict[str, object]:
    """Deliver one WAV: S3 presigned field set or base64 fallback.

    Returns exactly the pinned field sets from
    `.icm/references/s3-storage.md`; raises :class:`DeliveryError` only for
    forced-S3-without-credentials and upload failures.
    """
    cfg = cfg or _CFG
    resolved = resolve_delivery(response_delivery, cfg)

    if resolved == "base64":
        return {
            "delivery": "base64",
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "size_bytes": len(wav_bytes),
        }

    try:
        factory = client_factory or default_client_factory
        client = factory(cfg)
        key = build_key(job_id, cfg)
        client.put_object(Bucket=cfg.bucket, Key=key, Body=wav_bytes, ContentType="audio/wav")
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg.bucket, "Key": key},
            ExpiresIn=cfg.presign_expiry,
        )
    except DeliveryError:
        raise
    except Exception as exc:
        raise DeliveryError("delivery_failed", f"upload failed: {type(exc).__name__}") from exc

    return {
        "delivery": "s3",
        "audio_url": url,
        "bucket": cfg.bucket,
        "key": key,
        "size_bytes": len(wav_bytes),
        "url_expires_in": cfg.presign_expiry,
        "url_expires_at": _expires_at(cfg.presign_expiry, _dt.datetime.now(_dt.timezone.utc)),
    }
