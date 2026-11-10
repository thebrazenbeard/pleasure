"""Offline, privacy-minimized Tattler host evidence for experimental trials.

This parser never contacts a host. Its output cannot establish model identity,
neural activity, an inference/optimizer step, reward, or phenomenal experience.
Only supplied `/api/v1/system` samples are accepted; network observations and
process details are never copied into the returned evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from json import loads
from math import isfinite
from re import fullmatch


MAX_JSON_BYTES = 256 * 1024
MAX_SAMPLES = 120
MAX_WINDOW = timedelta(hours=1)
# 256 PiB is intentionally far beyond ordinary host RAM while keeping even
# MAX_SAMPLES summed values finite. A reported KiB count is not unbounded JSON.
MAX_MEMORY_KB = 1 << 48
SCHEMA = "PLEASURE_HOST_TELEMETRY_V1"
_METRICS = (
    ("cpu_percent", "percent", 100.0),
    ("io_wait_percent", "percent", 100.0),
    ("mem_total_kb", "KiB", MAX_MEMORY_KB),
    ("mem_available_kb", "KiB", MAX_MEMORY_KB),
)


def _utc(raw: str, name: str) -> datetime:
    if not isinstance(raw, str) or len(raw) > 48 or not fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?Z", raw
    ):
        raise ValueError(f"{name} must be a UTC RFC3339 timestamp ending in Z")
    # Go encodes nanoseconds while datetime supports microseconds. The original
    # timestamp is retained in the evidence instead of claiming ns precision.
    normalized = raw[:-1]
    if "." in normalized:
        whole, fraction = normalized.split(".", 1)
        normalized = whole + "." + fraction[:6]
    try:
        return datetime.fromisoformat(normalized + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid UTC timestamp") from exc


def _identifier(raw: str, name: str, limit: int = 128) -> str:
    if not isinstance(raw, str) or len(raw) > limit or not fullmatch(r"[A-Za-z0-9._:-]+", raw):
        raise ValueError(f"{name} must be a bounded opaque identifier")
    return raw


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key rejected")
        result[key] = value
    return result


def _no_nonfinite(raw: str) -> None:
    raise ValueError("non-finite JSON number rejected")


def _metric(sample: dict, name: str, max_value: float | None) -> tuple[str, float | None]:
    unavailable = sample.get("unavailable_metrics", [])
    if name in unavailable:
        return "unavailable", None
    if name not in sample or sample[name] is None:
        return "missing", None
    value = sample[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if name.endswith("_kb") and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer number of KiB")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite nonnegative number")
    if value < 0 or (max_value is not None and value > max_value):
        raise ValueError(f"{name} outside reported range")
    return "reported", float(value)


def parse_tattler_system_samples(
    raw_json: bytes | str,
    *,
    trial_id: str,
    opaque_host_sha256: str,
    source_revision: str,
    trial_start_utc: str,
    trial_end_utc: str,
    retrieved_at_utc: str,
) -> dict:
    """Summarize bounded, supplied Tattler system samples without raw host data.

    The host digest is supplied by the caller and is **not authenticated**.
    Timestamp overlap is a comparison of declared clocks, not clock attestation.
    """
    trial_id = _identifier(trial_id, "trial_id")
    source_revision = _identifier(source_revision, "source_revision")
    if not isinstance(opaque_host_sha256, str) or not fullmatch(r"[0-9a-f]{64}", opaque_host_sha256):
        raise ValueError("opaque_host_sha256 must be a lowercase SHA-256 digest")
    start = _utc(trial_start_utc, "trial_start_utc")
    end = _utc(trial_end_utc, "trial_end_utc")
    _utc(retrieved_at_utc, "retrieved_at_utc")
    if not start < end or end - start > MAX_WINDOW:
        raise ValueError("trial window must be positive and at most one hour")

    if isinstance(raw_json, bytes):
        raw_bytes = raw_json
    elif isinstance(raw_json, str):
        raw_bytes = raw_json.encode("utf-8")
    else:
        raise TypeError("raw_json must be UTF-8 bytes or text")
    if len(raw_bytes) > MAX_JSON_BYTES:
        raise ValueError("Tattler response exceeds byte limit")
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Tattler response must be valid UTF-8") from exc
    try:
        samples = loads(decoded, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_nonfinite)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid Tattler system JSON") from exc
    if not isinstance(samples, list) or len(samples) > MAX_SAMPLES:
        raise ValueError("Tattler system response must be a bounded array")

    matched: list[tuple[str, datetime, dict]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or type(sample.get("schema_version")) is not int or sample["schema_version"] != 1:
            raise ValueError(f"sample {index} has unsupported schema")
        observed_raw = sample.get("observed_at")
        at = _utc(observed_raw, f"sample {index} observed_at")
        unavailable = sample.get("unavailable_metrics", [])
        if not isinstance(unavailable, list) or len(unavailable) > 64 or any(
            not isinstance(item, str) or len(item) > 64 for item in unavailable
        ):
            raise ValueError(f"sample {index} has invalid unavailable_metrics")
        validated = {name: _metric(sample, name, ceiling) for name, _, ceiling in _METRICS}
        total_status, total = validated["mem_total_kb"]
        available_status, available = validated["mem_available_kb"]
        if total_status == available_status == "reported" and available > total:
            raise ValueError(f"sample {index} reports available memory above total memory")
        if start <= at <= end:
            matched.append((observed_raw, at, sample))

    metrics: dict[str, dict] = {}
    for name, unit, ceiling in _METRICS:
        counts = {"reported": 0, "unavailable": 0, "missing": 0}
        values: list[float] = []
        for _, _, sample in matched:
            status, value = _metric(sample, name, ceiling)
            counts[status] += 1
            if value is not None:
                values.append(value)
        metrics[name] = {
            "unit": unit,
            "reported_samples": counts["reported"],
            "unavailable_samples": counts["unavailable"],
            "missing_samples": counts["missing"],
            "min_reported": min(values) if values else None,
            "max_reported": max(values) if values else None,
            "mean_reported": sum(values) / len(values) if values else None,
        }

    stamps = sorted((at, raw) for raw, at, _ in matched)
    max_gap = max(((b[0] - a[0]).total_seconds() for a, b in zip(stamps, stamps[1:])), default=None)
    return {
        "schema": SCHEMA,
        "evidence_class": "HOST_TELEMETRY",
        "qualification": "SAMPLED_HOST_ONLY" if matched else "UNKNOWN",
        "model_internal_effect": "UNKNOWN",
        "phenomenology": "UNRESOLVED",
        "source": "tattler./api/v1/system",
        "source_revision_declared": source_revision,
        "trial_id": trial_id,
        "opaque_host_sha256_declared": opaque_host_sha256,
        "source_payload_sha256": sha256(raw_bytes).hexdigest(),
        "trial_start_utc": trial_start_utc,
        "trial_end_utc": trial_end_utc,
        "retrieved_at_utc": retrieved_at_utc,
        "clock_correlation": "DECLARED_UTC_UNVERIFIED",
        "source_sample_count": len(samples),
        "overlapping_sample_count": len(matched),
        "first_sample_utc": stamps[0][1] if stamps else None,
        "last_sample_utc": stamps[-1][1] if stamps else None,
        "max_observed_gap_seconds": max_gap,
        "metrics": metrics,
    }
