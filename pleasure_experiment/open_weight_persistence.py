"""Versioned, offline-only LoRA tensor export and fresh-model import.

The artifact contains only explicit trainable LoRA tensors in safetensors plus
a strict JSON manifest. It cannot authenticate a publisher, evaluator, host,
or subjective experience. The caller owns the complete, exclusively accessed
offline model during export/import, and must supply a separately loaded fresh
base model for a cross-process persistence check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, _named_digest, _tensor_byte_chunks,
)
from pleasure_experiment.open_weight_local_model import (
    LocalModelManifest, PreparedLocalLoRA,
)


FORMAT = "pleasure-open-weight-lora"
FORMAT_VERSION = 1
EVIDENCE_CLASS = "SOURCE_LOCAL_PERSISTED_LORA_TENSOR_READBACK_UNAUTHENTICATED"
MAX_ADAPTER_BYTES = 128 * 1024**2
MAX_MANIFEST_BYTES = 1024**2
MAX_TENSORS = 2048
MAX_SCALARS = 10_000_000
_FILES = frozenset({"adapter.safetensors", "manifest.json"})
_TOP_FIELDS = frozenset({
    "format", "format_version", "capture_stage", "binding", "snapshot_sha256", "rank", "alpha",
    "trainable_parameter_names", "trainable_weights_sha256",
    "frozen_backbone_sha256", "tensors", "adapter_safetensors_sha256",
})
_TENSOR_FIELDS = frozenset({"name", "dtype", "shape", "sha256"})


@dataclass(frozen=True)
class LocalLoRAArtifact:
    artifact_directory: str
    format_version: int
    binding: OpenWeightBinding
    snapshot_sha256: str
    trainable_weights_sha256: str
    frozen_backbone_sha256: str
    safetensors_sha256: str
    manifest_sha256: str
    tensor_count: int
    rank: int
    alpha: int
    loaded_runtime_instance_id: str | None = None
    evidence_class: str = EVIDENCE_CLASS
    externally_authenticated: bool = False
    hosted_model_modified: bool = False
    study_validated: bool = False
    phenomenology: str = "UNRESOLVED"

    @property
    def digest(self) -> str:
        """Exact named trainable-tensor digest for an upstream run receipt."""
        return self.trainable_weights_sha256


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_path(value: str | Path, *, must_exist: bool) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("artifact path must be a string or Path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.name in ("", ".", ".."):
        raise ValueError("artifact directory must be absolute without parent traversal")
    # Canonical casing / Windows 8.3 expansion may change spelling; check
    # redirecting entries directly, not equality of raw and resolved paths.
    parts = (path,) + tuple(path.parents)
    for item in parts:
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("artifact directory or ancestor is a link or junction")
    if must_exist:
        if not path.is_dir():
            raise ValueError("artifact directory does not exist")
        return path.resolve(strict=True)
    if path.exists():
        raise FileExistsError("artifact destination already exists; overwrite forbidden")
    if not path.parent.is_dir():
        raise ValueError("artifact parent directory must already exist")
    return path.parent.resolve(strict=True) / path.name


def _sha_file(path: Path, *, max_bytes: int) -> str:
    size = path.stat().st_size
    if not 1 <= size <= max_bytes:
        raise ValueError("artifact file exceeds the specified byte budget")
    result = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    if path.stat().st_size != size:
        raise RuntimeError("artifact file changed while hashing")
    return result.hexdigest()


def _tensor_digest(torch: Any, tensor: Any) -> str:
    digest = sha256()
    for chunk in _tensor_byte_chunks(torch, tensor):
        digest.update(chunk)
    return digest.hexdigest()


def _model_view(prepared: PreparedLocalLoRA, snapshot: LocalModelManifest):
    if type(prepared) is not PreparedLocalLoRA or type(snapshot) is not LocalModelManifest:
        raise TypeError("prepared local LoRA and verified snapshot manifest required")
    if (type(prepared.binding) is not OpenWeightBinding or
        prepared.binding.checkpoint_sha256 != snapshot.checkpoint_sha256 or
        prepared.binding.tokenizer_sha256 != snapshot.tokenizer_sha256):
        raise ValueError("prepared model and snapshot content identities differ")
    _sha(snapshot.snapshot_sha256, "snapshot digest")
    if (type(prepared.rank) is not int or not 1 <= prepared.rank <= 16 or
        type(prepared.alpha) is not int or not 1 <= prepared.alpha <= 64):
        raise ValueError("LoRA rank and alpha are invalid")
    torch = importlib.import_module("torch")
    if not isinstance(prepared.model, torch.nn.Module):
        raise TypeError("prepared model must be torch.nn.Module")
    all_params = tuple(prepared.model.named_parameters())
    names = prepared.trainable_parameter_names
    if (type(names) is not tuple or not names or len(names) > MAX_TENSORS or
        len(set(names)) != len(names) or
        any(type(name) is not str or
            (".lora_A." not in name and ".lora_B." not in name) for name in names)):
        raise ValueError("only exact named LoRA matrices may be persisted")
    by_name = dict(all_params)
    if len(by_name) != len(all_params) or not set(names).issubset(by_name):
        raise ValueError("selected LoRA parameter names do not resolve exactly")
    selected = tuple((name, by_name[name]) for name in names)
    if (any(param.device.type == "meta" or not param.is_floating_point() or
            not param.is_contiguous() for _, param in selected) or
        sum(p.numel() for _, p in selected) > MAX_SCALARS):
        raise ValueError("unsupported or overbudget LoRA tensor")
    frozen = tuple((name, param) for name, param in all_params if name not in names)
    if not frozen:
        raise ValueError("cannot persist an adapter without a frozen backbone")
    frozen_state = frozen + tuple(
        ("buffer:" + name, buf) for name, buf in prepared.model.named_buffers()
    )
    if any(param.device.type == "meta" or not param.is_contiguous()
           for _, param in frozen_state):
        raise ValueError("unsupported frozen backbone storage")
    # Distinct named Parameters can still share the same underlying storage.
    # Never export/import a purported trainable-only tensor if writing it
    # would also mutate a frozen parameter or another selected LoRA matrix.
    selected_storages = tuple(
        p.untyped_storage().data_ptr() for _, p in selected
    )
    frozen_storages = {
        p.untyped_storage().data_ptr() for _, p in frozen_state
    }
    if (len(set(selected_storages)) != len(selected_storages) or
        any(ptr in frozen_storages for ptr in selected_storages)):
        raise ValueError("LoRA tensor shares storage with frozen or selected model state")
    return torch, selected, frozen_state


def _tensor_records(torch: Any, named: tuple[tuple[str, Any], ...]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, param in named:
        if not bool(torch.isfinite(param.detach()).all()):
            raise ValueError("nonfinite LoRA adapter weights cannot be persisted")
        result.append({
            "name": name, "dtype": str(param.dtype),
            "shape": list(param.shape), "sha256": _tensor_digest(torch, param),
        })
    return result


def _receipt(
    path: Path, payload: dict[str, Any], manifest_sha: str,
    *, loaded_runtime_instance_id: str | None = None,
) -> LocalLoRAArtifact:
    return LocalLoRAArtifact(
        artifact_directory=str(path), format_version=FORMAT_VERSION,
        binding=OpenWeightBinding(**payload["binding"]),
        snapshot_sha256=payload["snapshot_sha256"],
        trainable_weights_sha256=payload["trainable_weights_sha256"],
        frozen_backbone_sha256=payload["frozen_backbone_sha256"],
        safetensors_sha256=payload["adapter_safetensors_sha256"],
        manifest_sha256=manifest_sha, tensor_count=len(payload["tensors"]),
        rank=payload["rank"], alpha=payload["alpha"],
        loaded_runtime_instance_id=loaded_runtime_instance_id,
    )


def save_versioned_lora_adapter(
    prepared: PreparedLocalLoRA, *, snapshot_manifest: LocalModelManifest,
    destination_directory: str | Path, expected_trainable_sha256: str | None = None,
    allow_persistent_export: bool = False,
) -> LocalLoRAArtifact:
    """Atomically publish a new, non-overwriting versioned LoRA directory.

    Call this *before* a one-use training runner restores its original weights.
    The optional expected digest binds the export to a runner's final readback.
    """
    if allow_persistent_export is not True:
        raise PermissionError("explicit LoRA artifact export permission required")
    destination = _safe_path(destination_directory, must_exist=False)
    torch, selected, frozen_state = _model_view(prepared, snapshot_manifest)
    before = _named_digest(torch, selected)
    if expected_trainable_sha256 is not None:
        if before != _sha(expected_trainable_sha256, "expected trainable digest"):
            raise ValueError("export does not match the learned model tensor readback")
    frozen_before = _named_digest(torch, frozen_state)
    records = _tensor_records(torch, selected)
    tensors = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in selected
    }
    safetensors = importlib.import_module("safetensors.torch")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent,
    ))
    try:
        weights_file = staging / "adapter.safetensors"
        safetensors.save_file(tensors, str(weights_file))
        weights_file.chmod(0o600)
        with weights_file.open("rb+") as written_weights:
            os.fsync(written_weights.fileno())
        payload: dict[str, Any] = {
            "format": FORMAT, "format_version": FORMAT_VERSION,
            "capture_stage": "TRAINED_TENSORS_CAPTURED_NOT_STUDY_VALIDATED",
            "binding": asdict(prepared.binding),
            "snapshot_sha256": snapshot_manifest.snapshot_sha256,
            "rank": prepared.rank, "alpha": prepared.alpha,
            "trainable_parameter_names": list(prepared.trainable_parameter_names),
            "trainable_weights_sha256": before,
            "frozen_backbone_sha256": frozen_before,
            "tensors": records,
            "adapter_safetensors_sha256": _sha_file(
                weights_file, max_bytes=MAX_ADAPTER_BYTES,
            ),
        }
        with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as target:
            json.dump(payload, target, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        (staging / "manifest.json").chmod(0o600)
        manifest_sha = _sha_file(
            staging / "manifest.json", max_bytes=MAX_MANIFEST_BYTES,
        )
        # Rehash both live model partitions to catch concurrent mutation
        # while cloning/serializing. A successful save never alters them.
        if (_named_digest(torch, selected) != before or
            _named_digest(torch, frozen_state) != frozen_before):
            raise RuntimeError("live model changed during adapter artifact export")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("artifact destination appeared during export")
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
        raise ValueError("artifact directory must contain only two regular files")
    manifest_path = source / "manifest.json"
    manifest_sha = _sha_file(manifest_path, max_bytes=MAX_MANIFEST_BYTES)
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is invalid UTF-8 JSON") from exc
    if type(parsed) is not dict or set(parsed) != _TOP_FIELDS:
        raise ValueError("unsupported LoRA artifact manifest fields")
    if parsed["format"] != FORMAT or type(parsed["format_version"]) is not int or (
        parsed["format_version"] != FORMAT_VERSION
    ):
        raise ValueError("unsupported LoRA artifact format version")
    if parsed["capture_stage"] != "TRAINED_TENSORS_CAPTURED_NOT_STUDY_VALIDATED":
        raise ValueError("LoRA artifact asserts an unsupported validation stage")
    return parsed, manifest_sha


def load_versioned_lora_adapter(
    prepared: PreparedLocalLoRA, *, snapshot_manifest: LocalModelManifest,
    source_directory: str | Path, allow_in_place_mutation: bool = False,
) -> LocalLoRAArtifact:
    """Verify and import into a caller-owned separately prepared fresh model.

    The caller must not pass a concurrently served model. A failed import
    restores all selected adapter parameters, even on a partial copy.
    """
    if allow_in_place_mutation is not True:
        raise PermissionError("explicit in-place LoRA reload permission required")
    source = _safe_path(source_directory, must_exist=True)
    torch, selected, frozen_state = _model_view(prepared, snapshot_manifest)
    payload, manifest_sha = _strict_manifest(source)
    binding_payload = payload["binding"]
    if type(binding_payload) is not dict or set(binding_payload) != set(asdict(prepared.binding)):
        raise ValueError("LoRA artifact binding fields are malformed")
    source_binding = OpenWeightBinding(**binding_payload)
    if (source_binding.model_id != prepared.binding.model_id or
        source_binding.model_revision != prepared.binding.model_revision or
        source_binding.checkpoint_sha256 != prepared.binding.checkpoint_sha256 or
        source_binding.tokenizer_sha256 != prepared.binding.tokenizer_sha256 or
        payload["snapshot_sha256"] != snapshot_manifest.snapshot_sha256 or
        type(payload["rank"]) is not int or payload["rank"] != prepared.rank or
        type(payload["alpha"]) is not int or payload["alpha"] != prepared.alpha):
        raise ValueError("LoRA artifact model, snapshot or configuration identity mismatch")
    expected_names = list(prepared.trainable_parameter_names)
    if (type(payload["trainable_parameter_names"]) is not list or
        payload["trainable_parameter_names"] != expected_names):
        raise ValueError("LoRA artifact parameter selection differs from the new model")
    records = payload["tensors"]
    if (type(records) is not list or len(records) != len(selected) or
        any(type(row) is not dict or set(row) != _TENSOR_FIELDS for row in records)):
        raise ValueError("malformed LoRA tensor manifest")
    for (name, tensor), row in zip(selected, records):
        if (row["name"] != name or row["dtype"] != str(tensor.dtype) or
            row["shape"] != list(tensor.shape)):
            raise ValueError("LoRA tensor name, dtype or shape mismatch")
        _sha(row["sha256"], "per-tensor digest")
    _sha(payload["trainable_weights_sha256"], "trainable digest")
    frozen_before = _named_digest(torch, frozen_state)
    if frozen_before != _sha(payload["frozen_backbone_sha256"], "frozen digest"):
        raise ValueError("fresh model backbone differs from the artifact")
    adapter_path = source / "adapter.safetensors"
    if (_sha_file(adapter_path, max_bytes=MAX_ADAPTER_BYTES) !=
        _sha(payload["adapter_safetensors_sha256"], "safetensors digest")):
        raise ValueError("adapter safetensors content differs from the manifest")
    safetensors = importlib.import_module("safetensors.torch")
    loaded = safetensors.load_file(str(adapter_path), device="cpu")
    if set(loaded) != set(expected_names):
        raise ValueError("adapter safetensors contains missing or unexpected tensors")
    for (name, param), row in zip(selected, records):
        tensor = loaded[name]
        if (str(tensor.dtype) != row["dtype"] or list(tensor.shape) != row["shape"] or
            _tensor_digest(torch, tensor) != row["sha256"] or
            not bool(torch.isfinite(tensor).all())):
            raise ValueError("adapter safetensors tensor failed manifest verification")
    # Restore all selected tensors on any failure, including post-copy hash
    # validation; the frozen backbone is read-only throughout.
    baseline = tuple((name, param.detach().clone()) for name, param in selected)
    try:
        with torch.no_grad():
            for name, param in selected:
                param.copy_(loaded[name].to(param.device))
        if (_named_digest(torch, selected) != payload["trainable_weights_sha256"] or
            _named_digest(torch, frozen_state) != frozen_before):
            raise RuntimeError("reloaded adapter/frozen model digest verification failed")
        if (_sha_file(adapter_path, max_bytes=MAX_ADAPTER_BYTES) !=
            payload["adapter_safetensors_sha256"] or
            _sha_file(source / "manifest.json", max_bytes=MAX_MANIFEST_BYTES) !=
            manifest_sha):
            raise RuntimeError("LoRA artifact files changed during model import")
    except Exception:
        with torch.no_grad():
            for (name, param), (_, saved) in zip(selected, baseline):
                param.copy_(saved)
        if (_named_digest(torch, selected) != _named_digest(torch, baseline) or
            _named_digest(torch, frozen_state) != frozen_before):
            raise RuntimeError("failed LoRA import could not be rolled back exactly")
        raise
    return _receipt(
        source, payload, manifest_sha,
        loaded_runtime_instance_id=prepared.binding.runtime_instance_id,
    )
