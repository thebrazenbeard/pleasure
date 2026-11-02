"""Versioned, safe custody for a *joint* LoRA representation and appraisal.

Only explicit CPU float32 LoRA/head matrices and bounded replay metadata are
saved. The pretrained base stays fixed, and safetensors is the sole weight
format: never load pickles, arbitrary Python, or an unverified model source.
Import restores stateless-SGD training count and prior chosen-receipt IDs so a
new process can continue learning without accidentally replaying old rewards.

Evidence is source-local; a SHA does not authenticate the host, publisher,
external reward, or any subjective state. Artifacts are not study results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import ctypes
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from pleasure_experiment.open_weight_adapter import OpenWeightBinding, _named_digest
from pleasure_experiment.open_weight_local_model import LocalModelManifest, PreparedLocalLoRA
from pleasure_experiment.open_weight_persistence import _safe_path, _sha, _sha_file, _tensor_digest


FORMAT = "pleasure-joint-open-weight-lora-appraisal"
FORMAT_VERSION = 1
CAPTURE_STAGE = "TRAINED_JOINT_WEIGHTS_NOT_STUDY_VALIDATED"
EVIDENCE_CLASS = "SOURCE_LOCAL_JOINT_LORA_APPRAISAL_PERSISTENCE_UNAUTHENTICATED"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TENSORS_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 256
MAX_ELEMENTS = 8_000_000
MAX_UPDATES = 4096
FILES = frozenset({"weights.safetensors", "manifest.json"})
FIELDS = frozenset({
    "format", "format_version", "capture_stage", "binding", "snapshot_sha256",
    "frozen_base_sha256", "action_token_ids", "hidden_size", "feature_schema",
    "lora_config", "optimizer_spec", "updates_at_export", "seen_evidence_ids",
    "seen_episode_ids", "replay_ledger_sha256", "joint_weights_sha256",
    "head_weights_sha256", "lora_weights_sha256", "tensors",
    "weights_safetensors_sha256", "optimizer_state_format",
})
TENSOR_FIELDS = frozenset({"name", "shape", "dtype", "sha256"})
OPTIMIZER_STATE_FORMAT = "stateless_sgd_v1_no_momentum"
_EPISODE = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")


@dataclass(frozen=True)
class JointAppraisalArtifact:
    artifact_directory: str
    format_version: int
    binding: OpenWeightBinding
    snapshot_sha256: str
    frozen_base_sha256: str
    head_weights_sha256: str
    lora_weights_sha256: str
    joint_weights_sha256: str
    safetensors_sha256: str
    manifest_sha256: str
    feature_schema: str
    action_token_ids: tuple[int, int]
    updates_at_export: int
    loaded_runtime_instance_id: str | None = None
    evidence_class: str = EVIDENCE_CLASS
    externally_authenticated: bool = False
    hosted_model_modified: bool = False
    study_validated: bool = False
    phenomenology: str = "UNRESOLVED"


def _hash(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _known_probe(probe: object) -> None:
    from pleasure_experiment.open_weight_joint_appraisal import TorchJointLoRAAppraisal
    if type(probe) is not TorchJointLoRAAppraisal:
        raise TypeError("an exact locally constructed TorchJointLoRAAppraisal is required")


def _ledger(probe: Any) -> dict[str, object]:
    state = probe.export_training_state()
    if type(state) is not dict or set(state) != {
        "updates", "seen_evidence_ids", "seen_episode_ids",
    }:
        raise ValueError("joint learner training ledger schema invalid")
    n = state["updates"]
    receipts = state["seen_evidence_ids"]
    episodes = state["seen_episode_ids"]
    if (type(n) is not int or not 0 <= n <= MAX_UPDATES or
        type(receipts) is not tuple or type(episodes) is not tuple or
        len(receipts) != len(episodes) or len(receipts) != n or
        len(set(receipts)) != n or len(set(episodes)) != n or
        any(type(v) is not str or _sha(v, "chosen evidence") != v for v in receipts) or
        any(type(v) is not str or _EPISODE.fullmatch(v) is None for v in episodes)):
        raise ValueError("joint learner replay ledger invalid or inconsistent")
    return {
        "updates": n, "seen_evidence_ids": list(receipts),
        "seen_episode_ids": list(episodes),
    }


def _configuration(probe: Any) -> tuple[dict[str, object], dict[str, object]]:
    cfg = probe.lora_config
    opt = probe.optimizer_spec
    if (type(cfg) is not dict or set(cfg) != {"rank", "alpha", "target_modules"} or
        type(cfg["rank"]) is not int or not 1 <= cfg["rank"] <= 16 or
        type(cfg["alpha"]) is not int or not 1 <= cfg["alpha"] <= 64 or
        type(cfg["target_modules"]) not in (list, tuple) or
        not 1 <= len(cfg["target_modules"]) <= 16 or
        any(type(t) is not str or not t or len(t) > 80
            for t in cfg["target_modules"]) or
        len(set(cfg["target_modules"])) != len(cfg["target_modules"])):
        raise ValueError("unsupported exact LoRA configuration")
    if (type(opt) is not dict or
        set(opt) != {"algorithm", "learning_rate", "max_grad_norm",
                     "max_delta_l2", "l2_penalty", "max_updates", "train_lora"} or
        opt["algorithm"] != "stateless-SGD" or type(opt["train_lora"]) is not bool):
        raise ValueError("unsupported stateless SGD recipe")
    from math import isfinite
    for field in ("learning_rate", "max_grad_norm", "max_delta_l2", "l2_penalty"):
        value = opt[field]
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("nonfinite or negative stateless SGD recipe")
    if (type(opt["max_updates"]) is not int or
        not 1 <= opt["max_updates"] <= MAX_UPDATES):
        raise ValueError("joint learner update budget unsupported")
    return ({"rank": cfg["rank"], "alpha": cfg["alpha"],
             "target_modules": list(cfg["target_modules"])}, dict(opt))


def _view(probe: Any, binding: OpenWeightBinding,
          snapshot: LocalModelManifest) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    _known_probe(probe)
    if (type(binding) is not OpenWeightBinding or
        type(snapshot) is not LocalModelManifest):
        raise TypeError("verified binding and snapshot records required")
    prepared = probe._prepared
    if (type(prepared) is not PreparedLocalLoRA or prepared.binding != binding or
        binding.checkpoint_sha256 != snapshot.checkpoint_sha256 or
        binding.tokenizer_sha256 != snapshot.tokenizer_sha256):
        raise ValueError("joint head source model and snapshot binding differ")
    _sha(snapshot.snapshot_sha256, "snapshot digest")
    if probe.pending:
        raise ValueError("cannot persist joint weights with pending feedback")
    tokens = probe.action_token_ids
    if (type(tokens) is not tuple or len(tokens) != 2 or
        any(type(t) is not int or not 0 <= t < 2**31 for t in tokens) or
        tokens[0] == tokens[1] or prepared.action_token_ids != tokens):
        raise ValueError("joint action-token order is invalid")
    torch = importlib.import_module("torch")
    if not isinstance(prepared.model, torch.nn.Module):
        raise TypeError("prepared PEFT LoRA model must be torch module")
    named_model = tuple(prepared.model.named_parameters())
    config, optimizer = _configuration(probe)
    trained = tuple((name, param) for name, param in named_model if param.requires_grad)
    expected_names = (prepared.trainable_parameter_names
                      if optimizer["train_lora"] else ())
    if (tuple(name for name, _ in trained) != expected_names or
        any(".lora_A." not in name and ".lora_B." not in name
            for name, _ in trained)):
        raise ValueError("LoRA trainability differs from explicit optimizer control arm")
    frozen_storages = {
        param.untyped_storage().data_ptr()
        for name, param in named_model if name not in prepared.trainable_parameter_names
    }
    frozen_storages.update(
        value.untyped_storage().data_ptr()
        for _, value in prepared.model.named_buffers()
    )
    live_trainable_storages = [
        param.untyped_storage().data_ptr() for _, param in trained
    ] + [param.untyped_storage().data_ptr() for param in probe._head.parameters()]
    if (len(set(live_trainable_storages)) != len(live_trainable_storages) or
        any(address in frozen_storages for address in live_trainable_storages)):
        raise ValueError("trainable LoRA/head aliases frozen base or another tensor")
    tensors = probe.export_tensors()
    if (type(tensors) is not dict or not 3 <= len(tensors) <= MAX_TENSORS or
        set(tensors).intersection({"head.weight", "head.bias"}) !=
        {"head.weight", "head.bias"} or
        any(type(name) is not str or not name.startswith("lora.")
            for name in tensors if not name.startswith("head.")) or
        any(name.startswith("head.") and name not in ("head.weight", "head.bias")
            for name in tensors) or
        any(len(name) > 512 for name in tensors)):
        raise ValueError("joint LoRA/head export must contain exact named tensors")
    weight, bias = tensors["head.weight"], tensors["head.bias"]
    if (weight.ndim != 2 or weight.shape[0] != 1 or bias.shape != (1,) or
        not 1 <= weight.shape[1] <= 8192):
        raise ValueError("unsupported joint appraisal linear-head architecture")
    if sum(t.numel() for t in tensors.values()) > MAX_ELEMENTS:
        raise ValueError("joint trained tensor size budget exceeded")
    for name, tensor in tensors.items():
        if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu" or
            tensor.dtype != torch.float32 or tensor.requires_grad or
            not tensor.is_contiguous() or not 1 <= tensor.ndim <= 2 or
            tensor.numel() == 0 or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"joint trained tensor {name} must be finite CPU float32")
    lora_names = {"lora." + name for name in prepared.trainable_parameter_names}
    if set(tensors) - {"head.weight", "head.bias"} != lora_names:
        raise ValueError("exported LoRA tensor names differ from explicit prepared targets")
    ledger = _ledger(probe)
    if ledger["updates"] != probe.updates or ledger["updates"] > optimizer["max_updates"]:
        raise ValueError("joint training ledger and optimizer update budget differ")
    if (type(probe.feature_schema) is not str or not 1 <= len(probe.feature_schema) <= 128):
        raise ValueError("unsupported joint appraisal feature schema")
    base_sha = _sha(probe.base_digest, "frozen base digest")
    head_sha = _sha(probe.head_digest, "head digest")
    lora_sha = _sha(probe.lora_digest, "LoRA digest")
    joint_sha = _sha(probe.joint_digest, "joint digest")
    if joint_sha != _named_digest(torch, tuple(tensors.items())):
        raise RuntimeError("joint head/LoRA tensors differ from claimed digest")
    payload = {
        "binding": asdict(binding), "snapshot_sha256": snapshot.snapshot_sha256,
        "frozen_base_sha256": base_sha, "head_weights_sha256": head_sha,
        "lora_weights_sha256": lora_sha, "joint_weights_sha256": joint_sha,
        "action_token_ids": list(tokens), "hidden_size": weight.shape[1],
        "feature_schema": probe.feature_schema, "lora_config": config,
        "optimizer_spec": optimizer, "updates_at_export": ledger["updates"],
        "seen_evidence_ids": ledger["seen_evidence_ids"],
        "seen_episode_ids": ledger["seen_episode_ids"],
        "replay_ledger_sha256": _hash(ledger),
    }
    return torch, tensors, payload


def _records(torch: Any, tensors: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
         "sha256": _tensor_digest(torch, tensor)}
        for name, tensor in tensors.items()
    ]


def _receipt(location: Path, payload: dict[str, Any], manifest_sha: str,
             *, loaded_runtime_instance_id: str | None = None) -> JointAppraisalArtifact:
    return JointAppraisalArtifact(
        artifact_directory=str(location), format_version=FORMAT_VERSION,
        binding=OpenWeightBinding(**payload["binding"]),
        snapshot_sha256=payload["snapshot_sha256"],
        frozen_base_sha256=payload["frozen_base_sha256"],
        head_weights_sha256=payload["head_weights_sha256"],
        lora_weights_sha256=payload["lora_weights_sha256"],
        joint_weights_sha256=payload["joint_weights_sha256"],
        safetensors_sha256=payload["weights_safetensors_sha256"],
        manifest_sha256=manifest_sha, feature_schema=payload["feature_schema"],
        action_token_ids=tuple(payload["action_token_ids"]),
        updates_at_export=payload["updates_at_export"],
        loaded_runtime_instance_id=loaded_runtime_instance_id,
    )


def _atomic_no_replace(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError("joint artifact destination already exists")
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("atomic no-replace directory rename unavailable")
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno), str(target))
    elif os.name == "nt":
        # Windows directory rename refuses an existing destination.
        os.rename(source, target)
    else:
        raise RuntimeError("atomic no-replace directory rename unavailable")


def save_versioned_joint_appraisal(
    probe: Any, *, binding: OpenWeightBinding,
    snapshot_manifest: LocalModelManifest, destination_directory: str | Path,
    expected_joint_sha256: str | None = None,
    allow_persistent_export: bool = False,
) -> JointAppraisalArtifact:
    if allow_persistent_export is not True:
        raise PermissionError("explicit joint LoRA/head export permission required")
    destination = _safe_path(destination_directory, must_exist=False)
    torch, tensors, details = _view(probe, binding, snapshot_manifest)
    if (expected_joint_sha256 is not None and
        _sha(expected_joint_sha256, "expected joint tensor digest") !=
        details["joint_weights_sha256"]):
        raise ValueError("joint trained tensor SHA differs from expected")
    payload = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "capture_stage": CAPTURE_STAGE,
        "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
        **details, "tensors": _records(torch, tensors),
    }
    safetensors = importlib.import_module("safetensors.torch")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.pending-", dir=destination.parent,
    ))
    try:
        weights = staging / "weights.safetensors"
        safetensors.save_file(tensors, str(weights))
        weights.chmod(0o600)
        with weights.open("rb+") as out:
            os.fsync(out.fileno())
        payload["weights_safetensors_sha256"] = _sha_file(
            weights, max_bytes=MAX_TENSORS_BYTES,
        )
        manifest = staging / "manifest.json"
        with manifest.open("x", encoding="utf-8", newline="\n") as out:
            json.dump(payload, out, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        manifest.chmod(0o600)
        manifest_sha = _sha_file(manifest, max_bytes=MAX_MANIFEST_BYTES)
        _, verify_tensors, verify_details = _view(probe, binding, snapshot_manifest)
        if (verify_details != details or
            _named_digest(torch, tuple(verify_tensors.items())) !=
            details["joint_weights_sha256"]):
            raise RuntimeError("joint model or replay state changed during export")
        _atomic_no_replace(staging, destination)
        return _receipt(destination, payload, manifest_sha)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _reject_duplicate_keys(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ValueError("joint artifact manifest has duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON manifest number rejected")


def _strict_manifest(source: Path) -> tuple[dict[str, Any], str]:
    entries = list(source.iterdir())
    if (set(entry.name for entry in entries) != FILES or
        any(entry.is_symlink() or
            (hasattr(entry, "is_junction") and entry.is_junction()) or
            not entry.is_file() for entry in entries)):
        raise ValueError("joint artifact directory requires exactly two regular files")
    digest = _sha_file(source / "manifest.json", max_bytes=MAX_MANIFEST_BYTES)
    try:
        source_bytes = (source / "manifest.json").read_bytes()
        payload = json.loads(
            source_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("joint artifact manifest is not strict UTF-8 JSON") from exc
    if (type(payload) is not dict or set(payload) != FIELDS or
        payload["format"] != FORMAT or
        type(payload["format_version"]) is not int or
        payload["format_version"] != FORMAT_VERSION or
        payload["capture_stage"] != CAPTURE_STAGE or
        payload["optimizer_state_format"] != OPTIMIZER_STATE_FORMAT):
        raise ValueError("unsupported joint artifact format or schema")
    return payload, digest


def load_versioned_joint_appraisal(
    probe: Any, *, binding: OpenWeightBinding,
    snapshot_manifest: LocalModelManifest, source_directory: str | Path,
    allow_in_place_mutation: bool = False,
) -> JointAppraisalArtifact:
    """Transactional exact import into caller-owned LoRA/head, including ledger."""
    if allow_in_place_mutation is not True:
        raise PermissionError("explicit joint head/LoRA import permission required")
    source = _safe_path(source_directory, must_exist=True)
    torch, original_tensors, original = _view(probe, binding, snapshot_manifest)
    if (original["updates_at_export"] != 0 or
        original["seen_evidence_ids"] or original["seen_episode_ids"]):
        raise ValueError("joint artifact import requires a pristine TRAIN ledger; "
                         "cannot roll back an already updated learner")
    payload, manifest_sha = _strict_manifest(source)
    if type(payload["binding"]) is not dict or set(payload["binding"]) != set(asdict(binding)):
        raise ValueError("joint artifact source binding malformed")
    old_binding = OpenWeightBinding(**payload["binding"])
    if (old_binding.model_id != binding.model_id or
        old_binding.model_revision != binding.model_revision or
        old_binding.checkpoint_sha256 != binding.checkpoint_sha256 or
        old_binding.tokenizer_sha256 != binding.tokenizer_sha256):
        raise ValueError("fresh model identity differs from joint artifact")
    for field in (
        "snapshot_sha256", "frozen_base_sha256", "action_token_ids",
        "hidden_size", "feature_schema", "lora_config", "optimizer_spec",
    ):
        if payload[field] != original[field]:
            raise ValueError(f"fresh model or joint recipe differs in {field}")
    # Validate all receipt history before mutating any LoRA or head tensor.
    ledger = {
        "updates": payload["updates_at_export"],
        "seen_evidence_ids": payload["seen_evidence_ids"],
        "seen_episode_ids": payload["seen_episode_ids"],
    }
    if (type(ledger["updates"]) is not int or
        not 0 <= ledger["updates"] <= MAX_UPDATES or
        type(ledger["seen_evidence_ids"]) is not list or
        type(ledger["seen_episode_ids"]) is not list or
        len(ledger["seen_evidence_ids"]) != ledger["updates"] or
        len(ledger["seen_episode_ids"]) != ledger["updates"] or
        len(set(ledger["seen_evidence_ids"])) != ledger["updates"] or
        len(set(ledger["seen_episode_ids"])) != ledger["updates"] or
        any(type(v) is not str or _sha(v, "receipt") != v
            for v in ledger["seen_evidence_ids"]) or
        any(type(v) is not str or _EPISODE.fullmatch(v) is None
            for v in ledger["seen_episode_ids"]) or
        payload["replay_ledger_sha256"] != _hash(ledger)):
        raise ValueError("joint replay ledger failed exact digest and counter check")
    for field in ("head_weights_sha256", "lora_weights_sha256", "joint_weights_sha256"):
        _sha(payload[field], field)
    rows = payload["tensors"]
    if (type(rows) is not list or len(rows) != len(original_tensors) or
        any(type(row) is not dict or set(row) != TENSOR_FIELDS for row in rows)):
        raise ValueError("joint artifact tensor schema malformed")
    for (name, tensor), row in zip(original_tensors.items(), rows):
        if (row["name"] != name or row["dtype"] != str(tensor.dtype) or
            row["shape"] != list(tensor.shape)):
            raise ValueError("joint artifact named tensor shape/dtype mismatch")
        _sha(row["sha256"], "tensor digest")
    weights = source / "weights.safetensors"
    expected_file_sha = _sha(payload["weights_safetensors_sha256"], "weights file SHA")
    if _sha_file(weights, max_bytes=MAX_TENSORS_BYTES) != expected_file_sha:
        raise ValueError("joint safetensors file digest mismatch")
    safetensors = importlib.import_module("safetensors.torch")
    loaded = safetensors.load_file(str(weights), device="cpu")
    if set(loaded) != set(original_tensors):
        raise ValueError("joint safetensors names differ from prepared targets")
    for (name, template), row in zip(original_tensors.items(), rows):
        candidate = loaded[name]
        if (type(candidate) is not torch.Tensor or
            candidate.dtype != torch.float32 or candidate.device.type != "cpu" or
            not candidate.is_contiguous() or candidate.shape != template.shape or
            not bool(torch.isfinite(candidate).all()) or
            _tensor_digest(torch, candidate) != row["sha256"]):
            raise ValueError("joint safetensors tensor failed byte/shape readback")
    if _named_digest(torch, tuple((name, loaded[name]) for name in original_tensors)) != payload["joint_weights_sha256"]:
        raise ValueError("joint named-tensor digest mismatch")
    prior_state = _ledger(probe)
    try:
        probe.import_tensors(
            loaded, expected_digest=payload["joint_weights_sha256"],
            updates_at_export=ledger["updates"],
            seen_evidence_ids=tuple(ledger["seen_evidence_ids"]),
            seen_episode_ids=tuple(ledger["seen_episode_ids"]),
        )
        if (probe.joint_digest != payload["joint_weights_sha256"] or
            probe.lora_digest != payload["lora_weights_sha256"] or
            probe.head_digest != payload["head_weights_sha256"] or
            probe.base_digest != original["frozen_base_sha256"] or
            _ledger(probe) != ledger or
            _sha_file(weights, max_bytes=MAX_TENSORS_BYTES) != expected_file_sha or
            _sha_file(source / "manifest.json", max_bytes=MAX_MANIFEST_BYTES) != manifest_sha):
            raise RuntimeError("joint imported tensor/ledger state failed exact readback")
    except Exception:
        probe.import_tensors(
            original_tensors, expected_digest=original["joint_weights_sha256"],
            updates_at_export=prior_state["updates"],
            seen_evidence_ids=tuple(prior_state["seen_evidence_ids"]),
            seen_episode_ids=tuple(prior_state["seen_episode_ids"]),
        )
        if probe.joint_digest != original["joint_weights_sha256"]:
            raise RuntimeError("joint failed-import rollback changed model state")
        raise
    return _receipt(source, payload, manifest_sha,
                    loaded_runtime_instance_id=binding.runtime_instance_id)
