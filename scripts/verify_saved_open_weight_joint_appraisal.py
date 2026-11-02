#!/usr/bin/env python3
"""Fresh-process import, withheld-offer inference, and one TRAIN-only continuation.

This is a local reproducibility/custody check, not a proof of sentience or a
host-authenticated result. The two v3 EVAL offers families are *never resolved*.
Only a separately reserved synthetic TRAIN evaluator supplies one payoff.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.isolated_outcome_evaluator_v3 import (
    V3OutcomeEvaluator, V3_WITHIN_FAMILIES, V3_TRAIN_FAMILIES,
)
from pleasure_experiment.isolated_outcome_evaluator import TRAIN_ARM
from pleasure_experiment.open_weight_joint_appraisal import TorchJointLoRAAppraisal
from pleasure_experiment.open_weight_joint_persistence import (
    load_versioned_joint_appraisal, save_versioned_joint_appraisal,
)
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, prepare_explicit_lora,
    validate_single_token_actions,
)
from pleasure_experiment.open_weight_stochastic import SeededActionSampler
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION = re.compile(r"[a-f0-9]{40}\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
MAX_CHILD_OUTPUT = 1024 * 1024
MAX_ARTIFACT_FILE = 32 * 1024 * 1024
STUDY_SEEDS = frozenset((163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227))
DEFAULT_EVAL_SEED = 39749121
DEFAULT_CONTINUATION_SEED = 39749139
FRESH_TRAIN_ARM = "joint_on"
PINNED_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
PINNED_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PINNED_SAFETENSORS_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check imported joint LoRA/head in a NEW Python process, then save one genuine TRAIN continuation",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--head-dir", required=True)
    parser.add_argument("--continuation-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--runtime-instance-id", default="offline-joint-fresh-continuation-v3")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--continuation-seed", type=int, default=DEFAULT_CONTINUATION_SEED)
    parser.add_argument("--heldout-family", choices=V3_WITHIN_FAMILIES, default=V3_WITHIN_FAMILIES[0])
    parser.add_argument("--appraisal-gain", type=float, default=12.0)
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--expected-joint-sha256")
    parser.add_argument("--allow-reload", action="store_true")
    parser.add_argument("--allow-continuation-training", action="store_true")
    parser.add_argument("--allow-persistent-export", action="store_true")
    parser.add_argument("--_fresh-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_parent-pid", type=int, help=argparse.SUPPRESS)
    return parser


def _validate(args: argparse.Namespace) -> None:
    for field in ("allow_reload", "allow_continuation_training", "allow_persistent_export"):
        if getattr(args, field) is not True:
            raise PermissionError(f"explicit --{field.replace('_', '-')} required")
    if args.revision != PINNED_REVISION or args.model_id != PINNED_MODEL_ID:
        raise ValueError("continuation requires exact preregistered SmolLM2 model and revision")
    if (type(args.runtime_instance_id) is not str or
        not 1 <= len(args.runtime_instance_id) <= 128):
        raise ValueError("bounded fresh runtime instance ID required")
    for field in ("seed", "continuation_seed"):
        value = getattr(args, field)
        if type(value) is not int or not 0 <= value < 2**32 or value in STUDY_SEEDS:
            raise ValueError(f"{field} must be a reserved uint32 seed outside preregistered study")
    if args.seed == args.continuation_seed:
        raise ValueError("continuation TRAIN evaluator must use a different seed")
    if args.heldout_family not in V3_WITHIN_FAMILIES:
        raise ValueError("heldout family must be a v3 latent EVAL family")
    if (type(args.appraisal_gain) not in (int, float) or
        not math.isfinite(args.appraisal_gain) or
        not 0 < args.appraisal_gain <= 16):
        raise ValueError("appraisal gain must be finite in (0,16]")
    if (type(args.max_snapshot_bytes) is not int or
        not 1 <= args.max_snapshot_bytes <= 1024**3):
        raise ValueError("local snapshot byte budget must be within 1 GiB")
    if args.expected_joint_sha256 is not None and (
        type(args.expected_joint_sha256) is not str or
        SHA.fullmatch(args.expected_joint_sha256) is None
    ):
        raise ValueError("expected joint digest must be a lowercase SHA256")
    if (type(args._fresh_worker) is not bool or
        (args._parent_pid is not None and
         (type(args._parent_pid) is not int or args._parent_pid <= 0))):
        raise ValueError("invalid internal fresh-process identity")
    source = Path(args.head_dir).resolve()
    dest = Path(args.continuation_dir).resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("continuation destination must not overlap source artifact")
    if dest.exists():
        raise FileExistsError("continuation artifact destination already exists (no replace)")
    if not source.is_dir():
        raise FileNotFoundError("source versioned joint artifact does not exist")


def _worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-I", "-B", str(Path(__file__).resolve()),
        "--_fresh-worker", "--_parent-pid", str(os.getpid()),
        "--snapshot", args.snapshot, "--head-dir", args.head_dir,
        "--continuation-dir", args.continuation_dir,
        "--model-id", args.model_id, "--revision", args.revision,
        "--runtime-instance-id", args.runtime_instance_id,
        "--seed", str(args.seed), "--continuation-seed", str(args.continuation_seed),
        "--heldout-family", args.heldout_family,
        "--appraisal-gain", str(args.appraisal_gain),
        "--max-snapshot-bytes", str(args.max_snapshot_bytes),
        "--allow-reload", "--allow-continuation-training", "--allow-persistent-export",
    ]
    if args.expected_joint_sha256 is not None:
        command.extend(("--expected-joint-sha256", args.expected_joint_sha256))
    return command


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(constant: str) -> object:
    raise ValueError(f"nonfinite JSON value: {constant}")


def _fresh_process(args: argparse.Namespace) -> dict[str, object]:
    if args._parent_pid is not None:
        raise ValueError("only internal invocation may declare parent PID")
    # Parent performs an independently loaded reference inference on the same
    # deterministic seed/family/index v3 offers; these are never resolved.
    reference = _parent_reference(args)
    result = subprocess.run(
        _worker_command(args), cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding="utf-8", timeout=900, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fresh continuation subprocess failed ({result.returncode}): {result.stderr.strip()[:512]}"
        )
    if len(result.stdout.encode("utf-8")) > MAX_CHILD_OUTPUT:
        raise RuntimeError("fresh child JSON exceeds size cap")
    try:
        report = json.loads(result.stdout, object_pairs_hook=_strict_pairs,
                            parse_constant=_reject_constant)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("fresh child did not return one canonical JSON report") from exc
    if (type(report) is not dict or
        type(report.get("fresh_process_pid")) is not int or
        report["fresh_process_pid"] == os.getpid() or
        report.get("parent_process_pid") != os.getpid() or
        report.get("spawned_fresh_model_process") is not True or
        report.get("tensor_import_verified") is not True or
        report.get("source_artifact_unchanged") is not True or
        report.get("continuation_training_saved") is not True or
        report.get("heldout_outcomes_not_resolved") is not True or
        report.get("distinct_evaluator_workers") is not True):
        raise RuntimeError("fresh subprocess identity, custody, or persistence verification failed")
    _verify_numeric_parity(reference, report.get("four_new_unresolved_heldout_offers"))
    if (report.get("imported_joint_sha256") != reference["joint_sha256"] or
        report.get("imported_head_sha256") != reference["head_sha256"] or
        report.get("imported_lora_sha256") != reference["lora_sha256"] or
        report.get("frozen_base_sha256_unchanged") != reference["base_sha256"]):
        raise RuntimeError("fresh worker and parent reference weight digests disagree")
    report["parent_reference_four_unresolved_heldout_offers"] = reference["rows"]
    report["parent_child_numeric_prediction_parity_verified"] = True
    return report


def _artifact_sha256(directory: str | Path) -> str:
    """Hash exact source bytes, forbidding artifact links and unexpected entries."""
    source = Path(directory)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("artifact directory must be a real directory")
    if {p.name for p in source.iterdir()} != {"manifest.json", "weights.safetensors"}:
        raise ValueError("source versioned artifact has unexpected files")
    digest = sha256()
    for name in ("manifest.json", "weights.safetensors"):
        file = source / name
        if not file.is_file() or file.is_symlink() or file.stat().st_size > MAX_ARTIFACT_FILE:
            raise ValueError("artifact file is unsafe, linked, or over size limit")
        digest.update(name.encode("ascii") + b"\0")
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned_snapshot(loaded: object) -> None:
    if (loaded.binding.model_id != PINNED_MODEL_ID or
        loaded.binding.model_revision != PINNED_REVISION or
        not any(row.name == "model.safetensors" and
                row.sha256 == PINNED_SAFETENSORS_SHA256
                for row in loaded.manifest.files)):
        raise ValueError("local snapshot is not exact pinned pretrained safetensors bytes")


def _encode_offer(loaded: object, offer: object) -> tuple[int, ...]:
    prompt = public_prompt(offer)
    if validate_single_token_actions(
        loaded.tokenizer, action_prefix=prompt,
        action_suffixes=loaded.action_suffixes,
    ) != loaded.action_token_ids:
        raise RuntimeError("fresh snapshot changed A/B action token order")
    ids = tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))
    if not ids:
        raise ValueError("empty fresh evaluator prompt")
    return ids


def _score_offer(probe: TorchJointLoRAAppraisal, offer: object,
                 ids: tuple[int, ...], gain: float) -> dict[str, object]:
    on = probe.score(ids, gain=gain, partition="EVAL", episode_id=offer.episode_id)
    off = probe.score(ids, gain=0.0, partition="EVAL", episode_id=offer.episode_id)
    if (on.base_digest != off.base_digest or
        on.head_digest != off.head_digest or
        on.lora_digest != off.lora_digest or
        on.joint_digest != off.joint_digest or
        (off.base_prob_a, off.base_prob_b) != (off.prob_a, off.prob_b)):
        raise RuntimeError("disabled-head baseline or inference tensor identity changed")
    return {
        "episode_id": offer.episode_id,
        "input_ids_sha256": on.input_ids_sha256,
        "joint_sha256": on.joint_digest,
        "head_sha256": on.head_digest,
        "lora_sha256": on.lora_digest,
        "base_sha256": on.base_digest,
        "on_prob_a": on.prob_a,
        "on_prob_b": on.prob_b,
        "disabled_head_prob_a": off.prob_a,
        "disabled_head_prob_b": off.prob_b,
        "predicted_quality_a": on.predicted_quality_a,
        "predicted_quality_b": on.predicted_quality_b,
    }


def _parent_reference(args: argparse.Namespace) -> dict[str, object]:
    """Independent parent-process reference, with EVAL issue/score only.

    v3 offers derive from seed+partition+family+pair index, not the worker's
    session nonce, so a wholly separate process reconstructs exactly the same
    input IDs. The reference loads the persisted tensor artifact, not a caller-
    supplied prediction file. This costs a second local model load intentionally.
    """
    source_sha = _artifact_sha256(args.head_dir)
    importlib.import_module("torch").set_num_threads(2)
    parent_runtime = f"joint-v3-parent-reference-{os.getpid()}"
    if parent_runtime == args.runtime_instance_id:
        raise ValueError("parent reference and worker runtime IDs must differ")
    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=parent_runtime,
        max_total_bytes=args.max_snapshot_bytes,
    )
    _assert_pinned_snapshot(loaded)
    loaded.model.eval()
    prepared = prepare_explicit_lora(loaded, target_modules=("q_proj", "v_proj"),
                                     rank=2, alpha=4)
    probe = TorchJointLoRAAppraisal(
        prepared, hidden_size=loaded.model.config.hidden_size,
        learning_rate=.05, allow_training=True, max_updates=48, train_lora=True,
    )
    artifact = load_versioned_joint_appraisal(
        probe, binding=loaded.binding, snapshot_manifest=loaded.manifest,
        source_directory=args.head_dir, allow_in_place_mutation=True,
    )
    if (artifact.binding.runtime_instance_id in
        (parent_runtime, args.runtime_instance_id) or
        probe.head_digest != artifact.head_weights_sha256 or
        probe.lora_digest != artifact.lora_weights_sha256 or
        probe.joint_digest != artifact.joint_weights_sha256 or
        probe.base_digest != artifact.frozen_base_sha256):
        raise RuntimeError("parent reference checkpoint identity or tensor import mismatch")
    with V3OutcomeEvaluator(
        seed=args.seed, evaluator_id=EVALUATOR,
        arms=(FRESH_TRAIN_ARM,), training_pairs_per_family=1,
        heldout_pairs_per_generator=4,
    ) as evaluator:
        offers = tuple(evaluator.issue(
            partition="EVAL", arm=FRESH_TRAIN_ARM,
            family_id=args.heldout_family, pair_index=index, side=0,
        ) for index in range(4))
        rows = [_score_offer(probe, offer, _encode_offer(loaded, offer),
                             args.appraisal_gain) for offer in offers]
    if _artifact_sha256(args.head_dir) != source_sha:
        raise RuntimeError("source artifact changed during parent reference scoring")
    return {"rows": rows, "joint_sha256": probe.joint_digest,
            "head_sha256": probe.head_digest,
            "lora_sha256": probe.lora_digest, "base_sha256": probe.base_digest}


def _verify_numeric_parity(reference: dict[str, object], child_rows: object) -> None:
    """Fail closed on *behavioral* mismatch; hashes by themselves are not enough."""
    expected = reference["rows"]
    if (type(expected) is not list or len(expected) != 4 or
        type(child_rows) is not list or len(child_rows) != 4):
        raise RuntimeError("parent/child must score exactly four identical EVAL offers")
    text_fields = (
        "episode_id", "input_ids_sha256", "joint_sha256", "head_sha256",
        "lora_sha256", "base_sha256",
    )
    numeric_fields = (
        "on_prob_a", "on_prob_b", "disabled_head_prob_a",
        "disabled_head_prob_b", "predicted_quality_a", "predicted_quality_b",
    )
    for index, (parent, child) in enumerate(zip(expected, child_rows)):
        if type(parent) is not dict or type(child) is not dict:
            raise RuntimeError(f"EVAL offer {index}: malformed prediction payload")
        if set(parent) != set(child) or any(
            type(parent[field]) is not str or parent[field] != child[field]
            for field in text_fields
        ):
            raise RuntimeError(f"EVAL offer {index}: input or tensor identity mismatch")
        for field in numeric_fields:
            a, b = parent[field], child[field]
            if (type(a) not in (int, float) or type(b) not in (int, float) or
                not math.isfinite(a) or not math.isfinite(b) or
                not math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7)):
                raise RuntimeError(f"EVAL offer {index}: cross-process {field} differs")


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    if args._parent_pid is None or args._parent_pid != os.getppid():
        raise PermissionError("fresh worker must have its declared live parent PID")
    source_sha = _artifact_sha256(args.head_dir)
    importlib.import_module("torch").set_num_threads(2)
    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )
    _assert_pinned_snapshot(loaded)
    loaded.model.eval()
    prepared = prepare_explicit_lora(
        loaded, target_modules=("q_proj", "v_proj"), rank=2, alpha=4,
    )
    probe = TorchJointLoRAAppraisal(
        prepared, hidden_size=loaded.model.config.hidden_size,
        learning_rate=.05, allow_training=True, max_updates=48, train_lora=True,
    )
    frozen_before = probe.base_digest
    before_initial = (probe.head_digest, probe.lora_digest, probe.joint_digest)
    artifact = load_versioned_joint_appraisal(
        probe, binding=loaded.binding, snapshot_manifest=loaded.manifest,
        source_directory=args.head_dir, allow_in_place_mutation=True,
    )
    if artifact.binding.runtime_instance_id == loaded.binding.runtime_instance_id:
        raise RuntimeError("source and fresh resumed runtime IDs must differ")
    if (probe.head_digest != artifact.head_weights_sha256 or
        probe.lora_digest != artifact.lora_weights_sha256 or
        probe.joint_digest != artifact.joint_weights_sha256 or
        probe.base_digest != artifact.frozen_base_sha256 or
        probe.base_digest != frozen_before):
        raise RuntimeError("loaded head, LoRA or frozen-base SHA differs from persisted artifact")
    if (args.expected_joint_sha256 is not None and
        args.expected_joint_sha256 != probe.joint_digest):
        raise ValueError("imported joint digest differs from explicitly pinned expected digest")
    ledger_before = probe.export_training_state()
    if (ledger_before["updates"] != artifact.updates_at_export or
        ledger_before["updates"] >= 48):
        raise RuntimeError("imported replay counter invalid or continuation budget exhausted")

    # No EVAL resolve, commit, or posthoc_evaluate calls anywhere in this path.
    with V3OutcomeEvaluator(
        seed=args.seed, evaluator_id=EVALUATOR, arms=(FRESH_TRAIN_ARM,),
        training_pairs_per_family=1, heldout_pairs_per_generator=4,
    ) as evaluator:
        eval_pid = evaluator.worker_pid
        offers = tuple(evaluator.issue(
            partition="EVAL", arm=FRESH_TRAIN_ARM,
            family_id=args.heldout_family, pair_index=index, side=0,
        ) for index in range(4))
        if len(set(offer.episode_id for offer in offers)) != 4:
            raise RuntimeError("EVAL offers reused episode identities")
        eval_rows = [_score_offer(probe, offer, _encode_offer(loaded, offer),
                                  args.appraisal_gain) for offer in offers]

    # A fresh, distinct seed and evaluator worker reserve an entirely NEW TRAIN
    # offer. The chosen action and its propensity are recorded pre-outcome.
    with V3OutcomeEvaluator(
        seed=args.continuation_seed, evaluator_id=EVALUATOR,
        arms=(FRESH_TRAIN_ARM,), training_pairs_per_family=1,
        heldout_pairs_per_generator=1,
    ) as train_evaluator:
        train_pid = train_evaluator.worker_pid
        if train_pid == eval_pid or train_pid in (os.getpid(), args._parent_pid):
            raise RuntimeError("continuation and heldout evaluators not independent")
        offer = train_evaluator.issue(
            partition="TRAIN", arm=TRAIN_ARM,
            family_id=V3_TRAIN_FAMILIES[0], pair_index=0, side=0,
        )
        if offer.episode_id in ledger_before["seen_episode_ids"]:
            raise RuntimeError("continuation TRAIN episode replays saved ledger")
        preview = probe.score(
            _encode_offer(loaded, offer), gain=args.appraisal_gain,
            partition="TRAIN", defer_choice=True, episode_id=offer.episode_id,
        )
        sampler = SeededActionSampler(seed=args.continuation_seed, exploration_rate=.4)
        trace = sampler.sample(
            preview.prob_a, preview.prob_b,
            input_ids_sha256=preview.input_ids_sha256, partition="TRAIN",
            episode_id=offer.episode_id, policy_id="joint_fresh_resume_v3",
        )
        chosen = probe.bind_choice(preview, trace.chosen_action)
        ticket = train_evaluator.commit(
            offer, chosen.chosen_action, trace.decision_digest,
            arm=TRAIN_ARM, model_id=loaded.binding.model_id,
            runtime_instance_id=loaded.binding.runtime_instance_id,
        )
        receipt = train_evaluator.resolve(ticket, arm=TRAIN_ARM)
        if (receipt.partition != "TRAIN" or receipt.episode_id != offer.episode_id or
            receipt.task_id != offer.task_id or receipt.family_id != offer.family_id or
            receipt.action != trace.chosen_action or
            receipt.decision_digest != trace.decision_digest or
            receipt.runtime_instance_id != loaded.binding.runtime_instance_id or
            receipt.model_id != loaded.binding.model_id or
            receipt.evaluator_id != EVALUATOR or
            receipt.evidence_id in ledger_before["seen_evidence_ids"]):
            raise RuntimeError("fresh TRAIN chosen-only receipt identity mismatch or replay")
        update = probe.observe_chosen(
            chosen, measured_quality=receipt.measured_quality,
            evidence_id=receipt.evidence_id,
        )

    ledger_after = probe.export_training_state()
    if (ledger_after["updates"] != ledger_before["updates"] + 1 or
        set(ledger_after["seen_evidence_ids"]) !=
        set(ledger_before["seen_evidence_ids"]) | {receipt.evidence_id} or
        set(ledger_after["seen_episode_ids"]) !=
        set(ledger_before["seen_episode_ids"]) | {offer.episode_id} or
        update.update_index != ledger_after["updates"] or
        not update.optimizer_step_executed or
        not update.head_changed or not update.lora_changed or
        probe.base_digest != frozen_before):
        raise RuntimeError("continuation update or replay ledger/frozen-base verification failed")
    if _artifact_sha256(args.head_dir) != source_sha:
        raise RuntimeError("source artifact mutated during fresh-process continuation")
    new_artifact = save_versioned_joint_appraisal(
        probe, binding=loaded.binding, snapshot_manifest=loaded.manifest,
        destination_directory=args.continuation_dir,
        expected_joint_sha256=probe.joint_digest,
        allow_persistent_export=args.allow_persistent_export,
    )
    if (_artifact_sha256(args.head_dir) != source_sha or
        new_artifact.joint_weights_sha256 != probe.joint_digest or
        new_artifact.head_weights_sha256 != probe.head_digest or
        new_artifact.lora_weights_sha256 != probe.lora_digest or
        new_artifact.frozen_base_sha256 != frozen_before or
        new_artifact.updates_at_export != ledger_after["updates"] or
        _artifact_sha256(args.continuation_dir) == source_sha):
        raise RuntimeError("new artifact readback, base integrity, or no-replace check failed")
    return {
        "fresh_process_pid": os.getpid(),
        "parent_process_pid": os.getppid(),
        "spawned_fresh_model_process": True,
        "independent_heldout_worker_pid": eval_pid,
        "independent_continuation_worker_pid": train_pid,
        "distinct_evaluator_workers": True,
        "new_runtime_instance_id_declared": loaded.binding.runtime_instance_id,
        "origin_runtime_instance_id_declared": artifact.binding.runtime_instance_id,
        "model_id_declared": loaded.binding.model_id,
        "model_revision_declared": loaded.binding.model_revision,
        "source_artifact_sha256_unchanged": source_sha,
        "source_artifact_unchanged": True,
        "snapshot_sha256": loaded.manifest.snapshot_sha256,
        "fresh_initial_untrained_head_sha256": before_initial[0],
        "fresh_initial_untrained_lora_sha256": before_initial[1],
        "fresh_initial_untrained_joint_sha256": before_initial[2],
        "imported_head_sha256": artifact.head_weights_sha256,
        "imported_lora_sha256": artifact.lora_weights_sha256,
        "imported_joint_sha256": artifact.joint_weights_sha256,
        "frozen_base_sha256_unchanged": frozen_before,
        "tensor_import_verified": True,
        "fixed_heldout_v3_family": args.heldout_family,
        "four_new_unresolved_heldout_offers": eval_rows,
        "heldout_outcomes_not_resolved": True,
        "continuation_seed_outside_study": args.continuation_seed,
        "continuation_train_episode_id": offer.episode_id,
        "continuation_train_evidence_id": receipt.evidence_id,
        "continuation_sampled_action": trace.chosen_action,
        "continuation_behavior_prob_a": trace.behavior_prob_a,
        "continuation_behavior_prob_b": trace.behavior_prob_b,
        "continuation_chosen_propensity": trace.chosen_propensity,
        "continuation_decision_digest": trace.decision_digest,
        "updates_before": ledger_before["updates"],
        "updates_after": ledger_after["updates"],
        "new_head_sha256": new_artifact.head_weights_sha256,
        "new_lora_sha256": new_artifact.lora_weights_sha256,
        "new_joint_sha256": new_artifact.joint_weights_sha256,
        "continuation_artifact_directory": new_artifact.artifact_directory,
        "continuation_artifact_manifest_sha256": new_artifact.manifest_sha256,
        "continuation_training_saved": True,
        "same_user_process_local_not_externally_authenticated": True,
        "quality_gain_established_by_reload_alone": False,
        "hosted_gpt_modified": False,
        "subjective_pleasure_proven": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate(args)
    return _run_worker(args) if args._fresh_worker else _fresh_process(args)


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError, ImportError,
            OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"fresh joint continuation rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
