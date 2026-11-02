#!/usr/bin/env python3
"""Opt-in frozen-protocol joint LoRA + appraisal outcome experiment (v3).

Loads only an already-local, safetensors verified SmolLM2 snapshot. The
benchmark uses separate freshly loaded pretrained model objects for matched
LoRA arms, source-local chosen-only feedback and an isolated EVAL decision
barrier. No model downloads, hosted ChatGPT writes, or paid compute occur.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, validate_single_token_actions,
)
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_SAFETENSORS_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
SEEDS = (163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predeclared chosen-feedback joint LoRA+appraisal study")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--save-artifact-root", required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--runtime-instance-id", default="offline-joint-appraisal-v3-20261102")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--training-pairs-per-family", type=int, default=8)
    p.add_argument("--heldout-pairs-per-generator", type=int, default=4)
    p.add_argument("--exploration-rate", type=float, default=0.4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--appraisal-gain", type=float, default=12.0)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--max-snapshot-bytes", type=int, default=1024 ** 3)
    p.add_argument("--allow-joint-training", action="store_true")
    p.add_argument("--allow-persistent-export", action="store_true")
    p.add_argument("--print-full-report", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_joint_training is not True or args.allow_persistent_export is not True:
        raise PermissionError("explicit joint LoRA training AND artifact export consent required")
    if args.model_id != MODEL_ID or args.revision != REVISION:
        raise ValueError("v3 protocol fixes the exact pretrained model identity/revision")
    if type(args.runtime_instance_id) is not str or not 1 <= len(args.runtime_instance_id) <= 80:
        raise ValueError("bounded runtime identity required")
    if type(args.cpu_threads) is not int or args.cpu_threads != 2:
        raise ValueError("v3 fixed CPU threading budget is two")
    if type(args.max_snapshot_bytes) is not int or not 1 <= args.max_snapshot_bytes <= 1024 ** 3:
        raise ValueError("invalid local snapshot byte budget")
    # No silent hyperparameter variation under a fixed predeclared label.
    if (tuple(args.seeds) != SEEDS or
        type(args.training_pairs_per_family) is not int or
        args.training_pairs_per_family != 8 or
        type(args.heldout_pairs_per_generator) is not int or
        args.heldout_pairs_per_generator != 4 or
        args.exploration_rate != 0.4 or args.learning_rate != 0.05 or
        args.appraisal_gain != 12.0):
        raise ValueError("v3 experiment parameters differ from frozen protocol")
    import torch

    torch.set_num_threads(args.cpu_threads)
    def reload_model(runtime_instance_id: str):
        if type(runtime_instance_id) is not str or not 1 <= len(runtime_instance_id) <= 128:
            raise ValueError("invalid fresh model runtime identity")
        instance = load_verified_local_causal_model(
            args.snapshot, model_id=args.model_id, model_revision=args.revision,
            runtime_instance_id=runtime_instance_id,
            max_total_bytes=args.max_snapshot_bytes,
        )
        if not any(
            row.name == "model.safetensors" and row.sha256 == MODEL_SAFETENSORS_SHA256
            for row in instance.manifest.files
        ):
            raise ValueError("v3 snapshot does not contain the exact pinned safetensors")
        return instance

    loaded = reload_model(args.runtime_instance_id)
    def encode_offer(offer):
        prompt = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prompt,
            action_suffixes=loaded.action_suffixes,
        ) != loaded.action_token_ids:
            raise RuntimeError("A/B action token order changed on v3 offer")
        return tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))

    from pleasure_experiment.open_weight_joint_benchmark import (
        run_joint_lora_appraisal_benchmark,
    )
    return run_joint_lora_appraisal_benchmark(
        loaded, encode_offer, reload_model,
        seeds=tuple(args.seeds), training_pairs_per_family=args.training_pairs_per_family,
        heldout_pairs_per_generator=args.heldout_pairs_per_generator,
        exploration_rate=args.exploration_rate, learning_rate=args.learning_rate,
        appraisal_gain=args.appraisal_gain, artifact_root=args.save_artifact_root,
        evaluator_id=EVALUATOR, allow_joint_training=True,
        allow_persistent_export=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        outcome = run(args)
    except (TypeError, ValueError, RuntimeError, PermissionError, ImportError, OSError) as exc:
        parser.exit(2, f"joint v3 study rejected: {exc}\n")
    if args.print_full_report:
        print(json.dumps(outcome, sort_keys=True, indent=2, allow_nan=False))
    else:
        # The complete run-level evidence report is already saved atomically.
        protocol = outcome.get("predeclared_protocol", {})
        summary = {
            "artifact_run_directory": outcome.get("artifact_run_directory"),
            "evidence_class": outcome.get("evidence_class"),
            "predeclared_protocol": protocol,
            "contrasts": outcome.get("co_primary_contrasts"),
            "exploratory_utility_criterion_met": outcome.get(
                "exploratory_positive_local_criterion_met"
            ),
            "hosted_gpt_modified": False,
            "subjective_pleasure_proven": False,
        }
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
