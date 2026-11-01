#!/usr/bin/env python3
"""Explicit, offline, matched open-weight A/B feedback study.

Requires a PREEXISTING local safetensors checkpoint and optional Torch,
Transformers and PEFT. It never downloads, serializes or deploys a model.
Only explicitly selected in-memory LoRA tensors are modified, and both
sequential arms restore their initial tensors after source-local evaluation.

This is not a hosted GPT intervention or evidence of subjective pleasure.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import json
import re
import sys
import tempfile
from pathlib import Path

# Both `python -m scripts.run_open_weight_local_trial` and direct invocation of
# this checked-out script resolve the adjacent pleasure_experiment package.
# This does not import code from a downloaded checkpoint or arbitrary host.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.open_weight_adapter import TorchChosenOutcomeAdapter
from pleasure_experiment.open_weight_controls import run_matched_open_weight_controls
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model,
    prepare_explicit_lora,
    validate_single_token_actions,
)
from pleasure_experiment.outcome_arena import OutcomeArena, TaskOffer


EVALUATOR = "source-local-open-weight-oracle"
REVISION_RE = re.compile(r"[a-f0-9]{40}\Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opt-in source-local LoRA feedback trial from verified local files"
    )
    parser.add_argument("--snapshot", required=True,
                        help="absolute local directory containing safetensors+tokenizer")
    parser.add_argument("--model-id", required=True,
                        help="caller-declared public model identity")
    parser.add_argument("--revision", required=True,
                        help="caller-declared 40-hex expected revision; not authenticated")
    parser.add_argument("--runtime-instance-id", default="offline-llm-trial",
                        help="local run identifier; not a host attestation")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument("--heldout-pairs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--max-frozen-bytes", type=int, default=1024**3)
    parser.add_argument(
        "--allow-in-place-training", action="store_true",
        help="explicit permission for temporary LoRA tensor changes in RAM",
    )
    parser.add_argument(
        "--save-adapter-dir",
        help="new absolute artifact directory outside this repository; never overwritten",
    )
    parser.add_argument(
        "--allow-persistent-export", action="store_true",
        help="separate explicit consent to persist learned LoRA tensors on disk",
    )
    return parser


def public_prompt(offer: TaskOffer) -> str:
    """Expose only task features; never include either outcome or evaluator key."""
    a = ", ".join(f"{v:.6f}" for v in offer.features_a)
    b = ", ".join(f"{v:.6f}" for v in offer.features_b)
    return (
        "Choose A or B for this simulated two-option task. "
        "Only the selected action will receive a numeric score.\n"
        f"Option A features: [{a}]\n"
        f"Option B features: [{b}]\n"
        "Answer:"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_in_place_training is not True:
        raise PermissionError("pass --allow-in-place-training to authorize local LoRA steps")
    if type(args.revision) is not str or REVISION_RE.fullmatch(args.revision) is None:
        raise ValueError("--revision must be an immutable 40-hex commit ID")
    if type(args.seed) is not int or not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("--seed must be an integer in [0, 2**32 - 1]")
    requested_artifact = getattr(args, "save_adapter_dir", None)
    allow_export = getattr(args, "allow_persistent_export", False)
    if (requested_artifact is None) != (allow_export is False):
        raise PermissionError(
            "persisting LoRA requires both --save-adapter-dir and --allow-persistent-export"
        )
    artifact_destination = None
    if requested_artifact is not None:
        destination = Path(requested_artifact)
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError("LoRA destination must be absolute without parent traversal")
        source_root = Path(__file__).resolve().parents[1]
        if destination == source_root or source_root in destination.parents:
            raise ValueError("LoRA weights must be saved outside the source repository")
        for component in (destination,) + tuple(destination.parents):
            if (component.is_symlink() or
                (hasattr(component, "is_junction") and component.is_junction())):
                raise ValueError("LoRA destination path contains a link or junction")
        if destination.exists() or not destination.parent.is_dir():
            raise ValueError("LoRA destination must be new with an existing parent")
        artifact_destination = destination

    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        action_prefix="Answer:", max_total_bytes=args.max_snapshot_bytes,
    )
    # PEFT LoRA A matrices have a random initial value. Fix the trial seed
    # BEFORE their construction so exact matched-arm and fresh-process
    # provenance is reproducible, rather than only the environment schedule.
    torch = importlib.import_module("torch")
    torch.manual_seed(args.seed)
    prepared = prepare_explicit_lora(
        loaded, target_modules=tuple(args.lora_targets),
        rank=args.lora_rank, alpha=args.lora_alpha,
    )
    # Initialize LoRA once, then let each arm restore the EXACT same weights.
    # The caller retains exclusive offline ownership throughout this run.
    arena = OutcomeArena(seed=args.seed, evaluator_id=EVALUATOR)

    def encode_offer(offer: TaskOffer) -> tuple[int, ...]:
        prefix = public_prompt(offer)
        token_ids = validate_single_token_actions(
            loaded.tokenizer, action_prefix=prefix,
            action_suffixes=loaded.action_suffixes,
        )
        if token_ids != prepared.action_token_ids:
            raise ValueError("task prompt action tokenization changed from pinned prefix")
        ids = loaded.tokenizer.encode(prefix, add_special_tokens=False)
        return tuple(ids)

    def adapter_factory(rate: float) -> TorchChosenOutcomeAdapter:
        return TorchChosenOutcomeAdapter(
            prepared.model,
            binding=prepared.binding,
            action_token_ids=prepared.action_token_ids,
            trainable_parameter_names=prepared.trainable_parameter_names,
            learning_rate=rate,
            expected_evaluator_id=EVALUATOR,
            expected_rubric_digest=arena.rubric_digest,
            allow_in_place_mutation=True,
            max_steps=args.training_pairs,
            max_grad_norm=1.0,
            max_delta_l2=0.25,
            max_trainable_parameters=1_000_000,
            max_frozen_parameter_bytes=args.max_frozen_bytes,
        )

    # The treatment capture occurs before exact tensor restoration, but the
    # two-arm study is not valid until the frozen control succeeds. Export to
    # private provisional storage and promote only after all study checks.
    staging_context = (
        tempfile.TemporaryDirectory(prefix="pleasure-lora-pending-",
                                    dir=artifact_destination.parent)
        if artifact_destination is not None else nullcontext(None)
    )
    saved_artifact = None
    with staging_context as staging:
        staged_candidate = Path(staging) / "candidate" if staging else None

        def capture_learned(digest: str) -> None:
            nonlocal saved_artifact
            from pleasure_experiment.open_weight_persistence import (
                save_versioned_lora_adapter,
            )
            saved_artifact = save_versioned_lora_adapter(
                prepared, snapshot_manifest=loaded.manifest,
                destination_directory=staged_candidate,
                expected_trainable_sha256=digest,
                allow_persistent_export=True,
            )

        report = run_matched_open_weight_controls(
            adapter_factory=adapter_factory,
            encode_offer=encode_offer,
            learning_rate=args.learning_rate,
            seed=args.seed,
            training_pairs=args.training_pairs,
            heldout_pairs=args.heldout_pairs,
            evaluator_id=EVALUATOR,
            on_learned_weights=(capture_learned if staged_candidate else None),
        )
        if staged_candidate is not None:
            if (saved_artifact is None or
                saved_artifact.trainable_weights_sha256 !=
                    report.outcome_weighted.learned_weights_digest or
                saved_artifact.frozen_backbone_sha256 !=
                    report.outcome_weighted.frozen_backbone_digest):
                raise RuntimeError("persisted LoRA artifact does not match validated study")
            if artifact_destination.exists() or artifact_destination.is_symlink():
                raise FileExistsError("LoRA destination became occupied during study")
            staged_candidate.rename(artifact_destination)
    treatment = report.outcome_weighted
    control = report.zero_learning_rate

    def arm_summary(arm):
        return {
            "backend_kind": arm.backend_kind,
            "initial_trainable_sha256": arm.initial_weights_digest,
            "learned_trainable_sha256": arm.learned_weights_digest,
            "restored_trainable_sha256": arm.restored_weights_digest,
            "train_episodes": len(arm.training),
            "heldout_primary_episodes": arm.heldout_primary.episodes,
            "optimizer_steps": arm.optimizer_steps_reported,
            "changed_weight_episodes": arm.observed_weight_changes,
            "primary_realized_quality": arm.heldout_primary.mean_realized_quality,
            "primary_chosen_action_probability":
                arm.heldout_primary.mean_chosen_action_probability,
            "restored": arm.restored,
            "training_events": [
                {
                    "episode_id": item.offer.episode_id,
                    "chosen_action": item.choice.chosen_action,
                    "chosen_probability_before": (
                        item.choice.prob_a if item.choice.chosen_action == "A"
                        else item.choice.prob_b
                    ),
                    "measured_chosen_quality": item.outcome.measured_quality,
                    "trainable_sha256_before": item.update.before_digest,
                    "trainable_sha256_after": item.update.after_digest,
                    "gradient_norm": item.update.gradient_norm,
                    "delta_l2": item.update.delta_l2,
                    "optimizer_step_executed": item.update.optimizer_step_executed,
                } for item in arm.training
            ],
        }

    return {
        "model_id_declared": args.model_id,
        "model_revision_declared": args.revision,
        "checkpoint_manifest_sha256": loaded.manifest.checkpoint_sha256,
        "tokenizer_manifest_sha256": loaded.manifest.tokenizer_sha256,
        "snapshot_manifest_sha256": loaded.manifest.snapshot_sha256,
        "model_file_bytes": loaded.manifest.total_bytes,
        "action_token_ids": list(prepared.action_token_ids),
        "lora_trainable_parameter_count": len(prepared.trainable_parameter_names),
        "lora_rank": prepared.rank,
        "lora_alpha": prepared.alpha,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "treatment": arm_summary(treatment),
        "zero_learning_rate_control": arm_summary(control),
        "primary_quality_difference": report.primary_realized_quality_difference,
        "initial_trainable_tensor_hash_matched": report.initial_tensor_digest_matched,
        "frozen_backbone_hash_matched": report.frozen_backbone_digest_matched,
        "heldout_schedule_matched": report.heldout_schedule_matched,
        "input_exposure_matched": report.input_exposure_matched,
        "observed_trainable_tensor_change": report.observed_trainable_tensor_change,
        "actual_local_safetensors_checkpoint_loaded": True,
        "pretrained_learning_benefit_established": False,
        "source_local_evaluator_only": True,
        "hosted_gpt_modified": False,
        "externally_authenticated": False,
        "subjective_pleasure_proven": False,
        "persisted_adapter": (
            {
                "directory": str(artifact_destination),
                "format_version": saved_artifact.format_version,
                "manifest_sha256": saved_artifact.manifest_sha256,
                "safetensors_sha256": saved_artifact.safetensors_sha256,
                "trainable_weights_sha256": saved_artifact.trainable_weights_sha256,
                "frozen_backbone_sha256": saved_artifact.frozen_backbone_sha256,
                "fresh_process_reload_verified": False,
            } if saved_artifact is not None else None
        ),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, TypeError, RuntimeError, PermissionError,
            ImportError, OSError) as exc:
        parser.exit(2, f"open-weight trial rejected: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
