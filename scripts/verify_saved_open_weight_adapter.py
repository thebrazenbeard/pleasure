#!/usr/bin/env python3
"""Fresh-process replay of an offline, versioned learned LoRA checkpoint.

Launch this as a SEPARATE Python process after the training CLI exits. The
verification starts by independently loading the frozen local safetensors
base, checks fixed unseen task logits before and after importing the saved
LoRA matrices, and records the actual tensor readback. The check is local,
not an external publisher/host attestation or a phenomenology test.
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

from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model,
    prepare_explicit_lora,
    validate_single_token_actions,
)
from pleasure_experiment.open_weight_persistence import load_versioned_lora_adapter
from pleasure_experiment.open_weight_trial import EVAL_FAMILY
from pleasure_experiment.outcome_arena import OutcomeArena
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION_RE = re.compile(r"[a-f0-9]{40}\Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reload a persisted LoRA into a fresh offline model and check inference"
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--runtime-instance-id", default="offline-llm-fresh-reload")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--heldout-pairs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--allow-in-place-reload", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_in_place_reload is not True:
        raise PermissionError("fresh-model LoRA import requires --allow-in-place-reload")
    if type(args.revision) is not str or REVISION_RE.fullmatch(args.revision) is None:
        raise ValueError("revision must be an exact 40-hex commit ID")
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if type(args.heldout_pairs) is not int or not 1 <= args.heldout_pairs <= 64:
        raise ValueError("heldout-pairs must be between 1 and 64")
    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id,
        model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )
    torch = importlib.import_module("torch")
    torch.manual_seed(args.seed)
    prepared = prepare_explicit_lora(
        loaded, target_modules=tuple(args.lora_targets),
        rank=args.lora_rank, alpha=args.lora_alpha,
    )
    arena = OutcomeArena(seed=args.seed, evaluator_id=EVALUATOR)
    offers = tuple(arena.issue(
        partition="EVAL", family_id=EVAL_FAMILY,
        pair_index=i, side=i % 2,
    ) for i in range(args.heldout_pairs))
    encoded = []
    for offer in offers:
        prefix = public_prompt(offer)
        tokens = validate_single_token_actions(
            loaded.tokenizer, action_prefix=prefix,
            action_suffixes=loaded.action_suffixes,
        )
        if tokens != prepared.action_token_ids:
            raise RuntimeError("heldout A/B tokenizer identity changed")
        encoded.append(tuple(loaded.tokenizer.encode(prefix, add_special_tokens=False)))

    def probabilities() -> list[tuple[float, float]]:
        prepared.model.eval()
        values = []
        with torch.inference_mode():
            for ids in encoded:
                tensor = torch.tensor([ids], dtype=torch.long)
                logits = prepared.model(input_ids=tensor).logits[0, -1]
                a, b = prepared.action_token_ids
                scores = torch.softmax(torch.stack((logits[a], logits[b])), dim=0)
                values.append((float(scores[0]), float(scores[1])))
        return values

    baseline = probabilities()
    artifact = load_versioned_lora_adapter(
        prepared, snapshot_manifest=loaded.manifest,
        source_directory=args.adapter_dir,
        allow_in_place_mutation=True,
    )
    learned = probabilities()
    changed = any(abs(a[0] - b[0]) > 1e-9 for a, b in zip(baseline, learned))
    return {
        "model_id_declared": args.model_id,
        "model_revision_declared": args.revision,
        "new_runtime_instance_id_declared": args.runtime_instance_id,
        "origin_runtime_instance_id_declared": artifact.binding.runtime_instance_id,
        "snapshot_manifest_sha256": loaded.manifest.snapshot_sha256,
        "adapter_manifest_sha256": artifact.manifest_sha256,
        "adapter_safetensors_sha256": artifact.safetensors_sha256,
        "imported_trainable_sha256": artifact.trainable_weights_sha256,
        "frozen_backbone_sha256": artifact.frozen_backbone_sha256,
        "fixed_heldout_episode_ids": [o.episode_id for o in offers],
        "before_import_action_probabilities": baseline,
        "after_import_action_probabilities": learned,
        "inference_change_observed": changed,
        "tensor_import_verified": True,
        "fresh_process_required_but_not_independently_attested": True,
        "study_validated_by_import_alone": False,
        "source_local_evaluator_only": True,
        "hosted_gpt_modified": False,
        "subjective_pleasure_proven": False,
    }


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError,
            ImportError, OSError) as exc:
        parser.exit(2, f"fresh-LoRA reload rejected: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
