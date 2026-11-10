#!/usr/bin/env python3
"""Opt-in multi-seed real-LM benchmark with three blinded source-local arms.

Each independent seed loads a NEW local pretrained base/LoRA initialization;
within that seed the frozen, genuine-outcome, and deliberately shuffled-label
arms share exact starting tensors and unseen heldout offers. No artifacts are
saved, model host touched, or subjective experience claimed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.open_weight_adapter import TorchChosenOutcomeAdapter
from pleasure_experiment.open_weight_benchmark import run_multiseed_open_weight_benchmark
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, prepare_explicit_lora,
    validate_single_token_actions,
)
from pleasure_experiment.outcome_arena import OutcomeArena
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION_RE = re.compile(r"[a-f0-9]{40}\Z")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matched multi-seed offline LoRA A/B study")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--runtime-instance-id", default="offline-multiseed-study")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    p.add_argument("--training-pairs", type=int, default=2)
    p.add_argument("--heldout-pairs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=2)
    p.add_argument("--lora-alpha", type=int, default=4)
    p.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    p.add_argument("--max-frozen-bytes", type=int, default=1024**3)
    p.add_argument("--allow-in-place-training", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_in_place_training is not True:
        raise PermissionError("explicit --allow-in-place-training consent required")
    if type(args.revision) is not str or REVISION_RE.fullmatch(args.revision) is None:
        raise ValueError("revision must be an exact 40-hex commit")
    # Validation must finish BEFORE creating any opted-in model adapter.
    seeds = tuple(args.seeds)
    if (not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds) or
        any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("seeds must be 1..32 unique uint32 integers")
    if type(args.training_pairs) is not int or not 2 <= args.training_pairs <= 64:
        raise ValueError("training-pairs must be 2..64")
    if type(args.heldout_pairs) is not int or not 1 <= args.heldout_pairs <= 64:
        raise ValueError("heldout-pairs must be 1..64")

    state: dict[str, object] = {}

    def adapter_factory(rate: float, seed: int, arm: str) -> TorchChosenOutcomeAdapter:
        if state.get("seed") != seed:
            # Load a separately allocated base for EACH seed. No warm-start
            # or cross-seed LoRA contamination is permitted.
            loaded = load_verified_local_causal_model(
                args.snapshot, model_id=args.model_id,
                model_revision=args.revision,
                runtime_instance_id=f"{args.runtime_instance_id}-seed-{seed}",
                max_total_bytes=args.max_snapshot_bytes,
            )
            torch = importlib.import_module("torch")
            torch.manual_seed(seed)
            prepared = prepare_explicit_lora(
                loaded, target_modules=tuple(args.lora_targets),
                rank=args.lora_rank, alpha=args.lora_alpha,
            )
            state.clear()
            state.update(seed=seed, loaded=loaded, prepared=prepared)
        prepared = state["prepared"]
        rubric = OutcomeArena(seed=seed, evaluator_id=EVALUATOR).rubric_digest
        return TorchChosenOutcomeAdapter(
            prepared.model, binding=prepared.binding,
            action_token_ids=prepared.action_token_ids,
            trainable_parameter_names=prepared.trainable_parameter_names,
            learning_rate=rate, expected_evaluator_id=EVALUATOR,
            expected_rubric_digest=rubric,
            allow_in_place_mutation=True, max_steps=args.training_pairs,
            max_grad_norm=1.0, max_delta_l2=0.25,
            max_trainable_parameters=1_000_000,
            max_frozen_parameter_bytes=args.max_frozen_bytes,
        )

    def encode_offer(offer):
        loaded, prepared = state["loaded"], state["prepared"]
        prefix = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prefix,
            action_suffixes=loaded.action_suffixes,
        ) != prepared.action_token_ids:
            raise ValueError("heldout or TRAIN offer breaks pinned A/B tokenization")
        return tuple(loaded.tokenizer.encode(prefix, add_special_tokens=False))

    study = run_multiseed_open_weight_benchmark(
        adapter_factory=adapter_factory, encode_offer=encode_offer,
        seeds=seeds, training_pairs=args.training_pairs,
        heldout_pairs=args.heldout_pairs, learning_rate=args.learning_rate,
        evaluator_id=EVALUATOR,
    )

    def arm_view(arm):
        return {
            "initial_sha256": arm.initial_weights_sha256,
            "learned_sha256": arm.learned_weights_sha256,
            "frozen_sha256": arm.frozen_backbone_sha256,
            "mean_primary_quality": arm.mean_primary_quality,
            "mean_primary_chosen_probability": arm.mean_primary_chosen_probability,
            "primary_heldout_chosen_a": sum(
                item.chosen_action == "A" for item in arm.heldout_primary),
            "primary_heldout_chosen_b": sum(
                item.chosen_action == "B" for item in arm.heldout_primary),
            "optimizer_steps": arm.optimizer_steps,
            "restored": arm.restored,
            "train_feedback": [
                {"actual_quality": ep.actual_measured_quality,
                 "teaching_quality": ep.teaching_quality,
                 "synthetic_feedback_id": (ep.teaching_evidence_id
                   if ep.donor_evidence_id is not None else None),
                 "donor_evidence_id": ep.donor_evidence_id}
                for ep in arm.training
            ],
        }

    def contrast(obj):
        return {
            "n_independent_seeds": obj.number_of_seeds,
            "mean_difference": obj.mean_difference,
            "sample_standard_deviation": obj.sample_standard_deviation,
            "descriptive_normal_95_interval_not_significance": obj.approximate_normal_95_interval,
        }

    return {
        "model_id_declared": args.model_id,
        "model_revision_declared": args.revision,
        "seeds": list(seeds),
        "training_pairs_per_seed": args.training_pairs,
        "independent_primary_heldout_pairs_per_seed": args.heldout_pairs,
        "arms": ["frozen", "outcome_weighted", "shuffled_feedback"],
        "seed_reports": [
            {"seed": row.seed,
             "frozen": arm_view(row.frozen),
             "outcome_weighted": arm_view(row.outcome_weighted),
             "shuffled_feedback": arm_view(row.shuffled_feedback),
             "treatment_minus_frozen_quality": row.treatment_minus_frozen_quality,
             "treatment_minus_shuffled_quality": row.treatment_minus_shuffled_quality,
             "treatment_vs_frozen_primary_action_switches": sum(
                 a.chosen_action != b.chosen_action
                 for a, b in zip(row.outcome_weighted.heldout_primary,
                                 row.frozen.heldout_primary)
             ),
             "shuffled_vs_frozen_primary_action_switches": sum(
                 a.chosen_action != b.chosen_action
                 for a, b in zip(row.shuffled_feedback.heldout_primary,
                                 row.frozen.heldout_primary)
             ),
             "schedule_and_initial_tensors_matched": (
                 row.control_baselines_matched and
                 row.train_and_heldout_schedules_matched and
                 row.input_exposure_matched),
             "heldout_disjoint": row.heldout_family_disjoint}
            for row in study.seed_reports
        ],
        "treatment_vs_frozen_quality": contrast(study.treatment_vs_frozen_quality),
        "treatment_vs_shuffled_quality": contrast(study.treatment_vs_shuffled_quality),
        "treatment_vs_frozen_chosen_probability": contrast(
            study.treatment_vs_frozen_chosen_probability),
        "all_restored": study.all_arms_restored,
        "all_arms_matched": study.all_baselines_and_schedules_matched,
        "heldout_tasks_share_training_payoff_rule": True,
        "external_heldout_validation": False,
        "hosted_gpt_modified": False,
        "subjective_pleasure_proven": False,
    }


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError,
            ImportError, OSError) as exc:
        parser.exit(2, f"multi-seed study rejected: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
