#!/usr/bin/env python3
"""Independently reload a persisted appraisal head into a fresh local model.

Run in a *new Python process* after the training process exits. The frozen
base is loaded again from an already-local, hash-checked safetensors snapshot;
no weights or data are downloaded. Fixed withheld-generator EVAL prompts are
scored before and after importing the head. This verifies local tensor and
prediction persistence, not publisher attestation, utility, or felt pleasure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal
from pleasure_experiment.open_weight_appraisal_persistence import (
    load_versioned_appraisal_head,
)
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, validate_single_token_actions,
)
from pleasure_experiment.withheld_payoffs import EVAL_FAMILIES, WithheldPayoffArena
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION_RE = re.compile(r"[a-f0-9]{40}\Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify persisted model-hidden-state appraisal in a NEW Python process",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--head-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--runtime-instance-id", default="offline-appraisal-fresh-reload")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--heldout-pairs", type=int, default=3)
    parser.add_argument("--heldout-family", choices=EVAL_FAMILIES, default=EVAL_FAMILIES[0])
    parser.add_argument("--appraisal-gain", type=float, default=16.0)
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--allow-appraisal-head-reload", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_appraisal_head_reload is not True:
        raise PermissionError("fresh-model appraisal import requires --allow-appraisal-head-reload")
    if type(args.revision) is not str or REVISION_RE.fullmatch(args.revision) is None:
        raise ValueError("--revision must be an exact 40-hex commit ID")
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be a uint32 integer")
    if type(args.heldout_pairs) is not int or not 1 <= args.heldout_pairs <= 64:
        raise ValueError("--heldout-pairs must be between 1 and 64")
    if type(args.heldout_family) is not str or args.heldout_family not in EVAL_FAMILIES:
        raise ValueError("--heldout-family must be structurally withheld EVAL generator")
    if (type(args.appraisal_gain) is not float or
        not 0 < args.appraisal_gain <= 16):
        raise ValueError("--appraisal-gain must be finite in (0, 16]")
    if type(args.runtime_instance_id) is not str or not args.runtime_instance_id:
        raise ValueError("fresh runtime instance ID must be nonempty")

    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )
    torch = importlib.import_module("torch")
    model = loaded.model
    model.eval()
    named_model_state = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, state) for name, state in model.named_buffers()
    )
    frozen_before = _named_digest(torch, named_model_state)
    probe = TorchHiddenAppraisal(
        model, action_token_ids=loaded.action_token_ids,
        hidden_size=model.config.hidden_size, learning_rate=0.0,
        allow_training=False,
    )
    initial_head = probe.head_digest
    arena = WithheldPayoffArena(seed=args.seed, evaluator_id=EVALUATOR)
    offers = tuple(arena.issue(
        partition="EVAL", family_id=args.heldout_family,
        pair_index=index, side=0,
    ) for index in range(args.heldout_pairs))
    encoded = []
    for offer in offers:
        prompt = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prompt,
            action_suffixes=loaded.action_suffixes,
        ) != loaded.action_token_ids:
            raise RuntimeError("heldout A/B action tokens differ from pinned source")
        encoded.append(tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False)))

    def score(gain: float) -> list[dict[str, float | str]]:
        rows = []
        for ids in encoded:
            decision = probe.score(ids, gain=gain, partition="EVAL")
            rows.append({
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

    initial_on = score(args.appraisal_gain)
    initial_off = score(0.0)
    artifact = load_versioned_appraisal_head(
        probe, binding=loaded.binding,
        snapshot_manifest=loaded.manifest, source_directory=args.head_dir,
        allow_in_place_mutation=True,
    )
    if artifact.binding.runtime_instance_id == loaded.binding.runtime_instance_id:
        raise RuntimeError("origin and fresh-process runtime instance IDs must differ")
    imported_on = score(args.appraisal_gain)
    imported_off = score(0.0)
    imported_inverted = score(-args.appraisal_gain)
    if (probe.head_digest != artifact.head_weights_sha256 or
        frozen_before != artifact.frozen_model_sha256 or
        frozen_before != _named_digest(torch, named_model_state)):
        raise RuntimeError("import changed frozen model or failed head digest readback")
    if any(
        (before["base_prob_a"], before["base_prob_b"]) !=
        (after["base_prob_a"], after["base_prob_b"])
        or before["prob_a"] != after["prob_a"]
        or before["prob_b"] != after["prob_b"]
        for before, after in zip(initial_off, imported_off)
    ):
        raise RuntimeError("disabled appraisal changed after head import")
    changed = any(
        before["prob_a"] != after["prob_a"] or
        before["predicted_quality_a"] != after["predicted_quality_a"] or
        before["predicted_quality_b"] != after["predicted_quality_b"]
        for before, after in zip(initial_on, imported_on)
    )
    return {
        "fresh_process_pid": os.getpid(),
        "new_runtime_instance_id_declared": loaded.binding.runtime_instance_id,
        "origin_runtime_instance_id_declared": artifact.binding.runtime_instance_id,
        "distinct_runtime_instance_ids": True,
        "model_id_declared": loaded.binding.model_id,
        "model_revision_declared": loaded.binding.model_revision,
        "snapshot_manifest_sha256": loaded.manifest.snapshot_sha256,
        "head_manifest_sha256": artifact.manifest_sha256,
        "head_safetensors_sha256": artifact.safetensors_sha256,
        "initial_untrained_head_sha256": initial_head,
        "imported_trained_head_sha256": probe.head_digest,
        "frozen_model_sha256_unchanged": frozen_before,
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
        "separate_process_invocation_required_not_attested": True,
        "withheld_generators_source_visible_not_authenticated": True,
        "quality_gain_established_by_reload_alone": False,
        "hosted_gpt_modified": False,
        "subjective_pleasure_proven": False,
    }


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError,
            ImportError, OSError) as exc:
        parser.exit(2, f"fresh-appraisal reload rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
