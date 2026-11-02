"""Versioned safetensors custody for an inference-coupled appraisal head.

This persists only the separately trained quality-prediction head. It does not
train or mutate the underlying language model, certify a checkpoint publisher,
or demonstrate subjective experience. A caller must own the model exclusively
and opt into export/import. A new process may import only after reconstructing
the *same* frozen model, action tokens, head architecture and snapshot.

Both save and load check tensor bytes and model state before and after work.
The artifact cannot carry Python objects or arbitrary instructions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from pleasure_experiment.open_weight_adapter import OpenWeightBinding, _named_digest
from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal
from pleasure_experiment.open_weight_local_model import LocalModelManifest
from pleasure_experiment.open_weight_persistence import (
    _safe_path, _sha, _sha_file, _tensor_digest,
)


FORMAT = "pleasure-open-weight-appraisal-head"
FORMAT_VERSION = 1
CAPTURE_STAGE = "TRAINED_HEAD_CAPTURED_NOT_STUDY_VALIDATED"
FEATURE_SCHEMA = "action-append-final-hidden-state-sigmoid-linear-v1"
EVIDENCE_CLASS = "SOURCE_LOCAL_PERSISTED_APPRAISAL_HEAD_UNAUTHENTICATED"
MAX_HEAD_BYTES = 1024**2 * 2
MAX_MANIFEST_BYTES = 1024**2
MAX_HEAD_ELEMENTS = 16_384
_FILES = frozenset({"head.safetensors", "manifest.json"})
_TOP_FIELDS = frozenset({
    "format", "format_version", "capture_stage", "binding", "snapshot_sha256",
    "action_token_ids", "hidden_size", "updates_at_export", "head_weights_sha256",
    "frozen_model_sha256", "tensors", "head_safetensors_sha256", "feature_schema",
})
_TENSOR_FIELDS = frozenset({"name", "dtype", "shape", "sha256"})


def _feature_schema_for(probe: TorchHiddenAppraisal) -> str:
    """Allow precisely the audited architectures; never trust duck typing.

    Both variants export the same two linear tensors, but those bytes have
    incompatible meanings under their different deterministic feature maps.
    The schema is therefore bound to the *concrete* head type on both ends.
    """
    if type(probe) is TorchHiddenAppraisal:
        return FEATURE_SCHEMA
    # Delayed import keeps the legacy artifact loader independent of optional
    # torch/cache features and rejects all unreviewed user-defined subclasses.
    from pleasure_experiment.open_weight_calibrated_appraisal import (
        FEATURE_SCHEMA as CALIBRATED_FEATURE_SCHEMA, TorchCalibratedAppraisal,
    )
    if type(probe) is TorchCalibratedAppraisal:
        return CALIBRATED_FEATURE_SCHEMA
    raise TypeError("appraisal persistence requires an exact allowlisted head type")


def _allowed_feature_schemas() -> tuple[str, str]:
    from pleasure_experiment.open_weight_calibrated_appraisal import (
        FEATURE_SCHEMA as CALIBRATED_FEATURE_SCHEMA,
    )
    return FEATURE_SCHEMA, CALIBRATED_FEATURE_SCHEMA


@dataclass(frozen=True)
class LocalAppraisalArtifact:
    artifact_directory: str
    format_version: int
    binding: OpenWeightBinding
    snapshot_sha256: str
    head_weights_sha256: str
    frozen_model_sha256: str
    safetensors_sha256: str
    manifest_sha256: str
    hidden_size: int
    action_token_ids: tuple[int, int]
    updates_at_export: int
    loaded_runtime_instance_id: str | None = None
    evidence_class: str = EVIDENCE_CLASS
    externally_authenticated: bool = False
    hosted_model_modified: bool = False
    study_validated: bool = False
    phenomenology: str = "UNRESOLVED"


def _view(
    probe: TorchHiddenAppraisal,
    binding: OpenWeightBinding,
    snapshot: LocalModelManifest,
) -> tuple[Any, str, dict[str, Any], tuple[int, int], int]:
    _feature_schema_for(probe)
    if type(binding) is not OpenWeightBinding or type(snapshot) is not LocalModelManifest:
        raise TypeError("binding and snapshot manifest must be verified local records")
    if (binding.checkpoint_sha256 != snapshot.checkpoint_sha256 or
        binding.tokenizer_sha256 != snapshot.tokenizer_sha256):
        raise ValueError("appraisal binding and source snapshot content differ")
    _sha(snapshot.snapshot_sha256, "snapshot digest")
    torch = importlib.import_module("torch")
    model = probe._model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("appraisal must reference a torch model")
    if probe._pending is not None or probe._unbound is not None:
        raise ValueError("cannot persist appraisal head while TRAIN feedback is pending")
    if (type(probe._tokens) is not tuple or len(probe._tokens) != 2 or
        type(probe._tokens[0]) is not int or type(probe._tokens[1]) is not int or
        probe._tokens[0] == probe._tokens[1]):
        raise ValueError("invalid appraisal action token configuration")
    tensors = probe.export_tensors()
    if set(tensors) != {"weight", "bias"}:
        raise ValueError("only the exact two linear-head tensors may be exported")
    if (tensors["weight"].ndim != 2 or tensors["weight"].shape[0] != 1 or
        tensors["bias"].shape != (1,) or
        not 1 <= tensors["weight"].shape[1] <= 8192):
        raise ValueError("appraisal head has an unsupported architecture")
    size = sum(t.numel() for t in tensors.values())
    if size > MAX_HEAD_ELEMENTS:
        raise ValueError("appraisal head exceeds tensor budget")
    for tensor in tensors.values():
        if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or
            tensor.device.type != "cpu" or not tensor.is_contiguous() or
            not bool(torch.isfinite(tensor).all())):
            raise ValueError("appraisal tensor must be finite contiguous CPU float32")
    named = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, buffer) for name, buffer in model.named_buffers()
    )
    if not named or any(t.device.type == "meta" for _, t in named):
        raise ValueError("appraisal base model is empty or has meta parameters")
    head_storages = tuple(
        param.untyped_storage().data_ptr() for param in probe._head.parameters()
    )
    model_storages = {
        param.untyped_storage().data_ptr() for _, param in named
    }
    if (len(set(head_storages)) != len(head_storages) or
        any(address in model_storages for address in head_storages)):
        raise ValueError("appraisal head shares tensor storage with frozen model state")
    frozen_sha = _named_digest(torch, named)
    return torch, frozen_sha, tensors, probe._tokens, tensors["weight"].shape[1]


def _records(torch: Any, tensors: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
         "sha256": _tensor_digest(torch, tensor)}
        for name, tensor in tensors.items()
    ]


def _receipt(
    location: Path, payload: dict[str, Any], manifest_sha: str,
    *, loaded_runtime_instance_id: str | None = None,
) -> LocalAppraisalArtifact:
    return LocalAppraisalArtifact(
        artifact_directory=str(location), format_version=FORMAT_VERSION,
        binding=OpenWeightBinding(**payload["binding"]),
        snapshot_sha256=payload["snapshot_sha256"],
        head_weights_sha256=payload["head_weights_sha256"],
        frozen_model_sha256=payload["frozen_model_sha256"],
        safetensors_sha256=payload["head_safetensors_sha256"],
        manifest_sha256=manifest_sha, hidden_size=payload["hidden_size"],
        action_token_ids=tuple(payload["action_token_ids"]),
        updates_at_export=payload["updates_at_export"],
        loaded_runtime_instance_id=loaded_runtime_instance_id,
    )


def save_versioned_appraisal_head(
    probe: TorchHiddenAppraisal, *, binding: OpenWeightBinding,
    snapshot_manifest: LocalModelManifest, destination_directory: str | Path,
    expected_head_sha256: str | None = None,
    allow_persistent_export: bool = False,
) -> LocalAppraisalArtifact:
    """Publish a brand-new versioned head; never overwrite an existing path."""
    if allow_persistent_export is not True:
        raise PermissionError("explicit appraisal head export permission required")
    destination = _safe_path(destination_directory, must_exist=False)
    torch, base_sha, tensors, tokens, hidden_size = _view(
        probe, binding, snapshot_manifest,
    )
    head_sha = probe.head_digest
    if (expected_head_sha256 is not None and
        head_sha != _sha(expected_head_sha256, "expected head digest")):
        raise ValueError("learned appraisal head hash differs from expected readback")
    if _named_digest(torch, tuple(tensors.items())) != head_sha:
        raise RuntimeError("head changed while capturing safetensors source")
    records = _records(torch, tensors)
    payload: dict[str, Any] = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "capture_stage": CAPTURE_STAGE, "binding": asdict(binding),
        "snapshot_sha256": snapshot_manifest.snapshot_sha256,
        "action_token_ids": list(tokens), "hidden_size": hidden_size,
        "feature_schema": _feature_schema_for(probe),
        "updates_at_export": probe.updates, "head_weights_sha256": head_sha,
        "frozen_model_sha256": base_sha, "tensors": records,
    }
    safetensors = importlib.import_module("safetensors.torch")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent,
    ))
    try:
        head_file = staging / "head.safetensors"
        safetensors.save_file(tensors, str(head_file))
        head_file.chmod(0o600)
        with head_file.open("rb+") as stream:
            os.fsync(stream.fileno())
        payload["head_safetensors_sha256"] = _sha_file(
            head_file, max_bytes=MAX_HEAD_BYTES,
        )
        manifest_file = staging / "manifest.json"
        with manifest_file.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest_file.chmod(0o600)
        manifest_sha = _sha_file(manifest_file, max_bytes=MAX_MANIFEST_BYTES)
        if (probe.head_digest != head_sha or
            _named_digest(torch, tuple(
                probe._model.named_parameters()
            ) + tuple(("buffer:" + n, b) for n, b in probe._model.named_buffers())) != base_sha):
            raise RuntimeError("model or appraisal head changed during export")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("appraisal destination appeared during export")
        os.rename(staging, destination)
        return _receipt(destination, payload, manifest_sha)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _strict_manifest(source: Path) -> tuple[dict[str, Any], str]:
    entries = list(source.iterdir())
    if (set(item.name for item in entries) != _FILES or
        any(item.is_symlink() or
            (hasattr(item, "is_junction") and item.is_junction()) or not item.is_file()
            for item in entries)):
        raise ValueError("appraisal artifact requires exactly two regular files")
    manifest_sha = _sha_file(source / "manifest.json", max_bytes=MAX_MANIFEST_BYTES)
    try:
        payload = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (UnicodeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("appraisal manifest is not UTF-8 JSON") from exc
    if type(payload) is not dict or set(payload) != _TOP_FIELDS:
        raise ValueError("appraisal manifest schema mismatch")
    if (payload["format"] != FORMAT or
        type(payload["format_version"]) is not int or
        payload["format_version"] != FORMAT_VERSION or
        payload["capture_stage"] != CAPTURE_STAGE or
        type(payload["feature_schema"]) is not str or
        payload["feature_schema"] not in _allowed_feature_schemas()):
        raise ValueError("unsupported appraisal artifact format or claims")
    return payload, manifest_sha


def load_versioned_appraisal_head(
    probe: TorchHiddenAppraisal, *, binding: OpenWeightBinding,
    snapshot_manifest: LocalModelManifest, source_directory: str | Path,
    allow_in_place_mutation: bool = False,
) -> LocalAppraisalArtifact:
    """Verify and import into an exclusively owned newly constructed head.

    The base model remains untouched, and the head's transactional importer
    restores its original state even if a late tensor readback fails.
    """
    if allow_in_place_mutation is not True:
        raise PermissionError("explicit appraisal head import permission required")
    source = _safe_path(source_directory, must_exist=True)
    torch, model_before, original_tensors, tokens, hidden_size = _view(
        probe, binding, snapshot_manifest,
    )
    original_head = probe.head_digest
    payload, manifest_sha = _strict_manifest(source)
    if payload["feature_schema"] != _feature_schema_for(probe):
        raise ValueError("appraisal artifact feature schema differs from target head")
    source_binding = payload["binding"]
    if (type(source_binding) is not dict or
        set(source_binding) != set(asdict(binding))):
        raise ValueError("malformed appraisal source model identity")
    persisted_binding = OpenWeightBinding(**source_binding)
    if (persisted_binding.model_id != binding.model_id or
        persisted_binding.model_revision != binding.model_revision or
        persisted_binding.checkpoint_sha256 != binding.checkpoint_sha256 or
        persisted_binding.tokenizer_sha256 != binding.tokenizer_sha256 or
        payload["snapshot_sha256"] != snapshot_manifest.snapshot_sha256 or
        type(payload["hidden_size"]) is not int or
        payload["hidden_size"] != hidden_size or
        type(payload["action_token_ids"]) is not list or
        payload["action_token_ids"] != list(tokens)):
        raise ValueError("appraisal source and fresh-model identity/config differ")
    if (type(payload["updates_at_export"]) is not int or
        not 0 <= payload["updates_at_export"] <= 4096):
        raise ValueError("invalid appraisal updates receipt")
    head_sha = _sha(payload["head_weights_sha256"], "appraisal head digest")
    if model_before != _sha(payload["frozen_model_sha256"], "frozen model digest"):
        raise ValueError("fresh model differs from persisted frozen model")
    rows = payload["tensors"]
    if (type(rows) is not list or len(rows) != 2 or
        any(type(row) is not dict or set(row) != _TENSOR_FIELDS for row in rows)):
        raise ValueError("appraisal tensor records malformed")
    for (name, tensor), row in zip(original_tensors.items(), rows):
        if (row["name"] != name or row["shape"] != list(tensor.shape) or
            row["dtype"] != str(tensor.dtype)):
            raise ValueError("appraisal tensor shape/type mismatch")
        _sha(row["sha256"], "appraisal tensor digest")
    head_file = source / "head.safetensors"
    expected_file_sha = _sha(payload["head_safetensors_sha256"], "head file digest")
    if _sha_file(head_file, max_bytes=MAX_HEAD_BYTES) != expected_file_sha:
        raise ValueError("appraisal head safetensors file digest mismatch")
    safetensors = importlib.import_module("safetensors.torch")
    loaded = safetensors.load_file(str(head_file), device="cpu")
    if set(loaded) != set(original_tensors):
        raise ValueError("safetensors appraisal tensor names differ")
    for name, row in zip(original_tensors, rows):
        tensor = loaded[name]
        if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or
            tensor.device.type != "cpu" or not tensor.is_contiguous() or
            str(tensor.dtype) != row["dtype"] or
            list(tensor.shape) != row["shape"] or
            not bool(torch.isfinite(tensor).all()) or
            _tensor_digest(torch, tensor) != row["sha256"]):
            raise ValueError("appraisal tensor failed digest/shape/readback checks")
    if _named_digest(torch, tuple((name, loaded[name]) for name in original_tensors)) != head_sha:
        raise ValueError("appraisal named-tensor hash mismatch")
    try:
        probe.import_tensors(loaded, expected_digest=head_sha)
        if (probe.head_digest != head_sha or
            _named_digest(torch, tuple(
                probe._model.named_parameters()
            ) + tuple(("buffer:" + n, b) for n, b in probe._model.named_buffers())) != model_before or
            _sha_file(head_file, max_bytes=MAX_HEAD_BYTES) != expected_file_sha or
            _sha_file(source / "manifest.json", max_bytes=MAX_MANIFEST_BYTES) != manifest_sha):
            raise RuntimeError("appraisal import failed exact post-import verification")
    except Exception:
        probe.import_tensors(original_tensors, expected_digest=original_head)
        if probe.head_digest != original_head:
            raise RuntimeError("appraisal failed-import rollback did not restore head")
        raise
    return _receipt(
        source, payload, manifest_sha,
        loaded_runtime_instance_id=binding.runtime_instance_id,
    )
