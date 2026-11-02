"""Opt-in, offline-only safetensors checkpoint loader for the open-weight pilot.

No network download, arbitrary model Python, pickle loading, file mutation, or
training occurs here. A caller supplies an already-local checkpoint directory.
File manifests are checked before and after loading; these are source-local
integrity checks, NOT publisher signatures or provider authentication.

Explicit LoRA preparation, if requested, changes only the supplied model
object in RAM and returns its exact named trainable tensor set. Actual learning
still requires TorchChosenOutcomeAdapter and a separately bound outcome loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
from typing import Any

from pleasure_experiment.open_weight_adapter import OpenWeightBinding


_SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z")
_ID = re.compile(r"[A-Za-z0-9._:/-]{1,160}\Z")
_TARGET = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")
_TOKENIZER_FILES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "sentencepiece.bpe.model", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt",
})
_PASSIVE_FILES = frozenset({"README.md", "LICENSE", "LICENSE.txt", ".gitattributes"})
_MODEL_FILES = frozenset({
    "config.json", "generation_config.json", "model.safetensors.index.json",
})
MAX_SNAPSHOT_BYTES = 8 * 1024**3


def _bounded_id(value: str, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded identifier")
    return value


def _digest_records(rows: tuple["LocalFileEvidence", ...]) -> str:
    payload = [(row.name, row.size, row.sha256) for row in rows]
    return sha256(json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def _safe_local_directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("snapshot directory must be a path")
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError("snapshot directory must be an absolute path")
    if ".." in supplied.parts:
        raise ValueError("snapshot directory must not contain parent traversal")
    # Reject symlinks in the selected path or an ancestor: a manifest cannot
    # pin an attacker-controlled resolution that changes between operations.
    # On Windows, junctions are another form of redirecting filesystem entry.
    # Raw Path equality with resolve() is not useful for this: drive spelling,
    # canonical casing, and short-name expansion may legitimately change.
    for item in (supplied,) + tuple(supplied.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("snapshot directory or ancestor is a symlink or junction")
    if not supplied.is_dir():
        raise ValueError("snapshot directory must exist and be a directory")
    resolved = supplied.resolve(strict=True)
    return resolved


def _allowed_file(name: str) -> bool:
    if name in _TOKENIZER_FILES or name in _PASSIVE_FILES or name in _MODEL_FILES:
        return True
    return name == "model.safetensors" or _SHARD.fullmatch(name) is not None


def _check_ignored_hf_download_metadata(path: Path) -> None:
    """Permit only inert Hugging Face local_dir metadata, never load it.

    The download cache is deliberately omitted from the model's content
    manifest because it has no inference role. Unsafe links, nested content
    outside the recognized metadata tree and unbounded files fail closed.
    """
    if path.is_symlink() or not path.is_dir():
        raise ValueError("unsafe Hugging Face metadata sidecar")
    files_seen = 0
    bytes_seen = 0
    for member in path.rglob("*"):
        if member.is_symlink() or (not member.is_dir() and not member.is_file()):
            raise ValueError("unsafe Hugging Face metadata sidecar")
        relative = member.relative_to(path).parts
        if not relative or relative[0] != "huggingface":
            raise ValueError("unrecognized Hugging Face metadata sidecar")
        if member.is_file():
            files_seen += 1
            bytes_seen += member.stat().st_size
            if files_seen > 2048 or bytes_seen > 8 * 1024**2:
                raise ValueError("Hugging Face metadata sidecar exceeds bounds")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"{label} exceeds one MiB")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid UTF-8 JSON") from exc
    if type(parsed) is not dict:
        raise ValueError(f"{label} must contain a JSON object")
    # Deliberately reject remote-code mappings even though the loader also
    # explicitly sets trust_remote_code=False.
    if "auto_map" in parsed:
        raise ValueError(f"{label} requests unsupported dynamic model/tokenizer code")
    return parsed


@dataclass(frozen=True)
class LocalFileEvidence:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class LocalModelManifest:
    snapshot_directory: str
    files: tuple[LocalFileEvidence, ...]
    checkpoint_sha256: str
    tokenizer_sha256: str
    snapshot_sha256: str
    total_bytes: int
    evidence_class: str = "LOCAL_FILES_SHA256_UNAUTHENTICATED"


@dataclass(frozen=True)
class LoadedLocalCausalModel:
    model: Any
    tokenizer: Any
    binding: OpenWeightBinding
    action_token_ids: tuple[int, int]
    manifest: LocalModelManifest
    action_prefix: str
    action_suffixes: tuple[str, str]
    externally_authenticated: bool = False
    hosted_model_modified: bool = False


@dataclass(frozen=True)
class PreparedLocalLoRA:
    model: Any
    binding: OpenWeightBinding
    action_token_ids: tuple[int, int]
    trainable_parameter_names: tuple[str, ...]
    rank: int
    alpha: int
    evidence_class: str = "IN_MEMORY_EXPLICIT_LORA_NOT_TRAINED"
    externally_authenticated: bool = False
    hosted_model_modified: bool = False


def inspect_local_snapshot(
    snapshot_directory: str | Path, *, max_total_bytes: int = 1024**3,
) -> LocalModelManifest:
    """Hash actual local model/tokenizer files, rejecting unsafe formats.

    A local digest cannot prove which publisher produced files. All files must
    reside directly in the selected directory, with no links or nested
    scripts. Only safetensors model weights are accepted.
    """
    if type(max_total_bytes) is not int or not 1 <= max_total_bytes <= MAX_SNAPSHOT_BYTES:
        raise ValueError("max_total_bytes must be a bounded positive integer")
    root = _safe_local_directory(snapshot_directory)
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    if not entries:
        raise ValueError("checkpoint snapshot directory is empty")
    names = set()
    total = 0
    rows: list[LocalFileEvidence] = []
    for path in entries:
        if path.name == ".cache":
            _check_ignored_hf_download_metadata(path)
            continue
        if path.is_symlink() or not path.is_file() or not _allowed_file(path.name):
            raise ValueError(f"unsupported snapshot entry: {path.name}")
        if path.name in names:
            raise ValueError("duplicate snapshot entry")
        names.add(path.name)
        size = path.stat().st_size
        if size < 0 or total + size > max_total_bytes:
            raise ValueError("snapshot exceeds explicit file byte budget")
        total += size
        digest = sha256()
        with path.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        rows.append(LocalFileEvidence(path.name, size, digest.hexdigest()))
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise ValueError("model config and tokenizer config are both required")
    if not (names & {"tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model"} or
            {"vocab.json", "merges.txt"}.issubset(names)):
        raise ValueError("local tokenizer vocabulary files are missing")
    safes = {name for name in names if name.endswith(".safetensors")}
    if not safes:
        raise ValueError("safetensors checkpoint weights are required")
    _read_json(root / "config.json", label="model config")
    _read_json(root / "tokenizer_config.json", label="tokenizer config")
    if "model.safetensors.index.json" in names:
        index = _read_json(root / "model.safetensors.index.json", label="shard index")
        mapping = index.get("weight_map")
        if type(mapping) is not dict or not mapping:
            raise ValueError("shard index requires a nonempty weight_map")
        if (any(type(name) is not str or not name for name in mapping.keys()) or
            any(type(name) is not str for name in mapping.values())):
            raise ValueError("shard index contains unsupported weight mapping types")
        referenced = set(mapping.values())
        if referenced != safes:
            raise ValueError("shard index references missing or extra weight files")
    elif safes != {"model.safetensors"}:
        raise ValueError("sharded checkpoints require an explicit safetensors index")
    all_rows = tuple(rows)
    checkpoint = tuple(row for row in all_rows if row.name in _MODEL_FILES or
                       row.name.endswith(".safetensors"))
    tokenizer = tuple(row for row in all_rows if row.name in _TOKENIZER_FILES)
    return LocalModelManifest(
        snapshot_directory=str(root), files=all_rows,
        checkpoint_sha256=_digest_records(checkpoint),
        tokenizer_sha256=_digest_records(tokenizer),
        snapshot_sha256=_digest_records(all_rows), total_bytes=total,
    )


def validate_single_token_actions(
    tokenizer: Any, *, action_prefix: str,
    action_suffixes: tuple[str, str] = (" A", " B"),
) -> tuple[int, int]:
    """Require both alternatives to be exact one-token continuations.

    This check is for the *supplied prefix only*. A trial's task encoder must
    use the same ending/prefix conditions or separately recheck tokenization.
    """
    if (type(action_prefix) is not str or not action_prefix or
        len(action_prefix) > 4096):
        raise ValueError("action_prefix must be a bounded nonempty string")
    if (type(action_suffixes) is not tuple or len(action_suffixes) != 2 or
        any(type(s) is not str or not s or len(s) > 32 for s in action_suffixes)
        or action_suffixes[0] == action_suffixes[1]):
        raise ValueError("two distinct bounded action suffixes are required")
    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer must implement encode")
    prefix = tokenizer.encode(action_prefix, add_special_tokens=False)
    if type(prefix) is not list or not prefix or any(type(v) is not int for v in prefix):
        raise ValueError("tokenizer returned an unsupported prefix encoding")
    found = []
    for suffix in action_suffixes:
        whole = tokenizer.encode(action_prefix + suffix, add_special_tokens=False)
        if (type(whole) is not list or len(whole) != len(prefix) + 1 or
            whole[:-1] != prefix or type(whole[-1]) is not int or whole[-1] < 0):
            raise ValueError("A/B alternatives are not exact single-token continuations")
        found.append(whole[-1])
    if found[0] == found[1]:
        raise ValueError("A and B resolve to the same action token")
    return found[0], found[1]


def load_verified_local_causal_model(
    snapshot_directory: str | Path, *, model_id: str, model_revision: str,
    runtime_instance_id: str, action_prefix: str = "Answer:",
    action_suffixes: tuple[str, str] = (" A", " B"),
    max_total_bytes: int = 1024**3,
) -> LoadedLocalCausalModel:
    """Load only already-local safetensors, rechecking all bytes after load."""
    model_id = _bounded_id(model_id, "model_id")
    model_revision = _bounded_id(model_revision, "model_revision")
    runtime_instance_id = _bounded_id(runtime_instance_id, "runtime_instance_id")
    before = inspect_local_snapshot(snapshot_directory, max_total_bytes=max_total_bytes)
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError("optional transformers dependency unavailable") from exc
    # Deliberately no from_pretrained(repo_id) or network-based snapshot_download.
    # No trust_remote_code, torch.load, pickle or .bin checkpoint fallback.
    tok = transformers.AutoTokenizer.from_pretrained(
        before.snapshot_directory, local_files_only=True, trust_remote_code=False,
    )
    action_ids = validate_single_token_actions(
        tok, action_prefix=action_prefix, action_suffixes=action_suffixes,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        before.snapshot_directory, local_files_only=True,
        trust_remote_code=False, use_safetensors=True,
    )
    after = inspect_local_snapshot(snapshot_directory, max_total_bytes=max_total_bytes)
    if after != before:
        raise RuntimeError("checkpoint/tokenizer files changed during local load")
    binding = OpenWeightBinding(
        model_id=model_id, model_revision=model_revision,
        runtime_instance_id=runtime_instance_id,
        checkpoint_sha256=before.checkpoint_sha256,
        tokenizer_sha256=before.tokenizer_sha256,
    )
    return LoadedLocalCausalModel(
        model=model, tokenizer=tok, binding=binding,
        action_token_ids=action_ids, manifest=before,
        action_prefix=action_prefix, action_suffixes=action_suffixes,
    )


def prepare_explicit_lora(
    loaded: LoadedLocalCausalModel, *, target_modules: tuple[str, ...],
    rank: int = 2, alpha: int = 4, max_trainable_parameters: int = 1_000_000,
) -> PreparedLocalLoRA:
    """Inject opt-in LoRA into the supplied in-memory causal LM, no training."""
    if type(loaded) is not LoadedLocalCausalModel:
        raise TypeError("loaded local model must be a LoadedLocalCausalModel")
    if (type(target_modules) is not tuple or not target_modules or
        len(target_modules) > 16 or len(set(target_modules)) != len(target_modules)
        or any(type(v) is not str or _TARGET.fullmatch(v) is None
               for v in target_modules)):
        raise ValueError("target_modules must be a nonempty unique bounded tuple")
    if type(rank) is not int or not 1 <= rank <= 16:
        raise ValueError("rank must be between 1 and 16")
    if type(alpha) is not int or not 1 <= alpha <= 64:
        raise ValueError("alpha must be between 1 and 64")
    if (type(max_trainable_parameters) is not int or
        not 1 <= max_trainable_parameters <= 10_000_000):
        raise ValueError("invalid trainable tensor budget")
    available = tuple(name for name, _ in loaded.model.named_modules())
    if not all(any(name == target or name.endswith("." + target) for name in available)
               for target in target_modules):
        raise ValueError("explicit LoRA target module does not exist in model")
    try:
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise RuntimeError("optional PEFT LoRA dependency unavailable") from exc
    config = peft.LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=list(target_modules),
        lora_dropout=0.0, bias="none", task_type=peft.TaskType.CAUSAL_LM,
    )
    prepared = peft.get_peft_model(loaded.model, config)
    named = tuple((name, p) for name, p in prepared.named_parameters() if p.requires_grad)
    if (not named or any(".lora_A." not in name and ".lora_B." not in name
                         for name, _ in named)):
        raise RuntimeError("PEFT did not produce exclusively trainable LoRA matrices")
    if sum(p.numel() for _, p in named) > max_trainable_parameters:
        raise RuntimeError("in-memory LoRA exceeds explicit trainable tensor budget")
    return PreparedLocalLoRA(
        model=prepared, binding=loaded.binding,
        action_token_ids=loaded.action_token_ids,
        trainable_parameter_names=tuple(name for name, _ in named),
        rank=rank, alpha=alpha,
    )
