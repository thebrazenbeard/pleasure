#!/usr/bin/env python3
"""Reload a calibrated appraisal head in a distinct, fresh Python process.

The public CLI starts an isolated Python interpreter. Only that child loads
the exact, locally verified frozen model and safetensors head. It checks
before/after model/head digests and scores fixed v2 EVAL prompts from a
separate source-local outcome worker, without accessing any rewards.

This verifies persisted model-bound tensor behavior, not remote attestation,
generalizable reward utility, modifications to hosted ChatGPT or sensation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal_persistence import (
    load_versioned_appraisal_head,
)
from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, validate_single_token_actions,
)
from pleasure_experiment.isolated_outcome_evaluator import (
    IsolatedOutcomeEvaluator, NOVEL_EVAL_FAMILIES,
)
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION = re.compile(r"[a-f0-9]{40}\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
MAX_OUTPUT_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a calibrated head in a NEW Python interpreter and model",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--head-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--runtime-instance-id", default="offline-calibrated-fresh-reload",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--heldout-pairs", type=int, default=3)
    parser.add_argument(
        "--heldout-family", choices=NOVEL_EVAL_FAMILIES,
        default=NOVEL_EVAL_FAMILIES[0],
    )
    parser.add_argument("--appraisal-gain", type=float, default=12.0)
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--expected-head-sha256")
    parser.add_argument("--allow-appraisal-head-reload", action="store_true")
    # Internal process boundary. A direct invocation still must be parented by
    # the declared process, and neither route counts as external attestation.
    parser.add_argument("--_fresh-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_parent-pid", type=int, help=argparse.SUPPRESS)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.allow_appraisal_head_reload is not True:
        raise PermissionError("pass --allow-appraisal-head-reload for explicit import")
    if type(args.revision) is not str or REVISION.fullmatch(args.revision) is None:
        raise ValueError("--revision must be an exact lowercase 40-hex commit")
    if type(args.runtime_instance_id) is not str or not args.runtime_instance_id:
        raise ValueError("fresh runtime instance ID must be nonempty")
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be an unsigned 32-bit integer")
    if type(args.heldout_pairs) is not int or not 1 <= args.heldout_pairs <= 64:
        raise ValueError("--heldout-pairs must be between 1 and 64")
    if (type(args.appraisal_gain) is not float or
        not 0 < args.appraisal_gain <= 16):
        raise ValueError("--appraisal-gain must be finite in (0,16]")
    if (type(args.heldout_family) is not str or
        args.heldout_family not in NOVEL_EVAL_FAMILIES):
        raise ValueError("heldout family must be a new structurally withheld v2 family")
    if (type(args.max_snapshot_bytes) is not int or
        not 1 <= args.max_snapshot_bytes <= 1024**3):
        raise ValueError("snapshot byte budget outside [1,1 GiB]")
    if args.expected_head_sha256 is not None and (
        type(args.expected_head_sha256) is not str or
        SHA.fullmatch(args.expected_head_sha256) is None
    ):
        raise ValueError("--expected-head-sha256 must be lowercase SHA-256")
    if type(args._fresh_worker) is not bool or (
        args._parent_pid is not None and
        (type(args._parent_pid) is not int or args._parent_pid <= 0)
    ):
        raise ValueError("invalid internal process identity")


def _worker_command(args: argparse.Namespace) -> list[str]:
    result = [
        sys.executable, "-I", "-B", str(Path(__file__).resolve()),
        "--_fresh-worker", "--_parent-pid", str(os.getpid()),
        "--snapshot", args.snapshot, "--head-dir", args.head_dir,
        "--model-id", args.model_id, "--revision", args.revision,
        "--runtime-instance-id", args.runtime_instance_id,
        "--seed", str(args.seed), "--heldout-pairs", str(args.heldout_pairs),
        "--heldout-family", args.heldout_family,
        "--appraisal-gain", str(args.appraisal_gain),
        "--max-snapshot-bytes", str(args.max_snapshot_bytes),
        "--allow-appraisal-head-reload",
    ]
    if args.expected_head_sha256 is not None:
        result.extend(("--expected-head-sha256", args.expected_head_sha256))
    return result


def _fresh_process(args: argparse.Namespace) -> dict[str, object]:
    """Spawn a NEW interpreter; pass only declarative options, never pickle."""
    if args._parent_pid is not None:
        raise ValueError("parent PID is reserved for internally spawned worker")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        _worker_command(args), cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=900, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[:512]
        raise RuntimeError(f"fresh child process failed ({completed.returncode}): {detail}")
    if len(completed.stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError("fresh child report exceeds bounded output")
    try:
        report = json.loads(completed.stdout, parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("nonfinite child JSON"),
        ))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("fresh child did not produce one valid JSON report") from exc
    if (type(report) is not dict or
        type(report.get("fresh_process_pid")) is not int or
        report["fresh_process_pid"] == os.getpid() or
        report.get("parent_process_pid") != os.getpid() or
        report.get("spawned_fresh_model_process") is not True or
        report.get("tensor_import_verified") is not True):
        raise RuntimeError("fresh process identity or reload verification failed")
    return report


def _score(probe: TorchCalibratedAppraisal, offers: tuple[object, ...],
           ids: tuple[tuple[int, ...], ...], bundles: tuple[object, ...],
           gain: float) -> list[dict[str, float | str]]:
    rows = []
    for offer, token_ids, features in zip(offers, ids, bundles):
        decision = probe.score(
            token_ids, gain=gain, partition="EVAL",
            episode_id=offer.episode_id, features=features,
        )
        rows.append({
            "episode_id": offer.episode_id,
            "input_ids_sha256": decision.input_ids_sha256,
            "head_sha256": decision.head_digest,
            "prob_a": decision.prob_a,
            "prob_b": decision.prob_b,
            "predicted_quality_a": decision.predicted_quality_a,
            "predicted_quality_b": decision.predicted_quality_b,
            "base_prob_a": decision.base_prob_a,
            "base_prob_b": decision.base_prob_b,
        })
    return rows


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    if args._parent_pid is None or args._parent_pid != os.getppid():
        raise PermissionError("fresh worker must have its declared live parent process")

    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )
    torch = importlib.import_module("torch")
    model = loaded.model
    model.eval()
    named_model = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, buffer) for name, buffer in model.named_buffers()
    )
    frozen_before = _named_digest(torch, named_model)
    cache = FrozenActionFeatureCache(
        model, action_token_ids=loaded.action_token_ids,
        checkpoint_sha256=loaded.binding.checkpoint_sha256,
        tokenizer_sha256=loaded.binding.tokenizer_sha256,
        expected_frozen_model_sha256=frozen_before,
        max_entries=min(args.heldout_pairs, 64),
    )
    probe = TorchCalibratedAppraisal(
        model, action_token_ids=loaded.action_token_ids,
        hidden_size=model.config.hidden_size, learning_rate=0.0,
        allow_training=False, feature_cache=cache,
    )
    initial_head_sha = probe.head_digest
    with IsolatedOutcomeEvaluator(
        seed=args.seed, evaluator_id=EVALUATOR,
        arms=("trained_on",), training_pairs=1,
        heldout_pairs_per_generator=args.heldout_pairs,
        suite="novel_v2",
    ) as evaluator:
        evaluator_pid = evaluator.worker_pid
        offers = tuple(evaluator.issue(
            partition="EVAL", arm="trained_on",
            family_id=args.heldout_family, pair_index=index, side=0,
        ) for index in range(args.heldout_pairs))
    ids = []
    for offer in offers:
        prompt = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prompt,
            action_suffixes=loaded.action_suffixes,
        ) != loaded.action_token_ids:
            raise RuntimeError("heldout A/B action token order differs from snapshot")
        ids.append(tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False)))
    encoded = tuple(ids)

    with cache.verified_epoch() as epoch:
        features = tuple(epoch.get_features(
            token_ids, partition="EVAL", episode_id=offer.episode_id,
        ) for offer, token_ids in zip(offers, encoded))
        initial_on = _score(probe, offers, encoded, features, args.appraisal_gain)
        initial_off = _score(probe, offers, encoded, features, 0.0)
        artifact = load_versioned_appraisal_head(
            probe, binding=loaded.binding, snapshot_manifest=loaded.manifest,
            source_directory=args.head_dir, allow_in_place_mutation=True,
        )
        if artifact.binding.runtime_instance_id == loaded.binding.runtime_instance_id:
            raise RuntimeError("origin and fresh-process runtime instance IDs must differ")
        imported_on = _score(probe, offers, encoded, features, args.appraisal_gain)
        imported_off = _score(probe, offers, encoded, features, 0.0)
        imported_inverted = _score(probe, offers, encoded, features, -args.appraisal_gain)

    if (probe.head_digest != artifact.head_weights_sha256 or
        frozen_before != artifact.frozen_model_sha256 or
        frozen_before != _named_digest(torch, named_model)):
        raise RuntimeError("head readback or frozen-model persistence mismatch")
    if (args.expected_head_sha256 is not None and
        probe.head_digest != args.expected_head_sha256):
        raise ValueError("persisted head differs from expected exported SHA-256")
    if any(
        (before["prob_a"], before["prob_b"], before["base_prob_a"], before["base_prob_b"])
        != (after["prob_a"], after["prob_b"], after["base_prob_a"], after["base_prob_b"])
        for before, after in zip(initial_off, imported_off)
    ) or any(
        before["prob_a"] != off["prob_a"] or before["prob_b"] != off["prob_b"]
        for before, off in zip(initial_on, initial_off)
    ):
        raise RuntimeError("zero-head or disabled-appraisal controls changed after import")
    changed = any(
        before["prob_a"] != after["prob_a"] or
        before["predicted_quality_a"] != after["predicted_quality_a"] or
        before["predicted_quality_b"] != after["predicted_quality_b"]
        for before, after in zip(initial_on, imported_on)
    )
    return {
        "fresh_process_pid": os.getpid(),
        "parent_process_pid": os.getppid(),
        "spawned_fresh_model_process": True,
        "independent_evaluator_worker_pid": evaluator_pid,
        "new_runtime_instance_id_declared": loaded.binding.runtime_instance_id,
        "origin_runtime_instance_id_declared": artifact.binding.runtime_instance_id,
        "distinct_runtime_instance_ids": True,
        "model_id_declared": loaded.binding.model_id,
        "model_revision_declared": loaded.binding.model_revision,
        "snapshot_manifest_sha256": loaded.manifest.snapshot_sha256,
        "head_manifest_sha256": artifact.manifest_sha256,
        "head_safetensors_sha256": artifact.safetensors_sha256,
        "initial_untrained_head_sha256": initial_head_sha,
        "imported_trained_head_sha256": probe.head_digest,
        "frozen_model_sha256_unchanged": frozen_before,
        "feature_schema_verified_on_import": probe.feature_schema,
        "fixed_heldout_family": args.heldout_family,
        "fixed_heldout_episode_ids": [offer.episode_id for offer in offers],
        "before_import_on": initial_on,
        "before_import_off": initial_off,
        "after_import_on": imported_on,
        "after_import_off": imported_off,
        "after_import_inverted": imported_inverted,
        "head_import_changes_heldout_predictions": changed,
        "frozen_head_and_disabled_head_controls_preserved": True,
        "tensor_import_verified": True,
        "unseen_generator_outcomes_not_read_during_reload": True,
        "separate_process_invocation_source_local_not_authenticated": True,
        "quality_gain_established_by_reload_alone": False,
        "hosted_gpt_modified": False,
        "subjective_pleasure_proven": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate(args)
    if args._fresh_worker:
        return _run_worker(args)
    return _fresh_process(args)


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError, ImportError,
            OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"fresh-calibrated reload rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
