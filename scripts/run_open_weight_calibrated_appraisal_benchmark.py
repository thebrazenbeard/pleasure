#!/usr/bin/env python3
"""Run the predeclared calibrated head / isolated payoff evaluator study.

No downloads, hosted model writes, source-control changes or unsafe object
deserialization. Only explicitly authorized local head training and export.
The reproducible 12-seed v2 defaults are frozen in the accompanying protocol.
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


REVISION = re.compile(r"[a-f0-9]{40}\Z")
PROTOCOL_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predeclared 12-seed calibrated appraisal study with isolated evaluator",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--runtime-instance-id", default="offline-calibrated-appraisal-20261102",
    )
    parser.add_argument("--save-head-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--training-pairs", type=int, default=24)
    parser.add_argument("--heldout-pairs-per-generator", type=int, default=6)
    parser.add_argument("--exploration-rate", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--appraisal-gain", type=float, default=12.0)
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--allow-appraisal-training", action="store_true")
    parser.add_argument("--allow-persistent-export", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_appraisal_training is not True or args.allow_persistent_export is not True:
        raise PermissionError("both explicit appraisal-training and persistent-export flags required")
    if type(args.revision) is not str or REVISION.fullmatch(args.revision) is None:
        raise ValueError("revision must be an immutable lowercase 40-hex commit")
    if type(args.runtime_instance_id) is not str or not args.runtime_instance_id:
        raise ValueError("runtime instance ID must be nonempty")
    if (type(args.max_snapshot_bytes) is not int or
        not 1 <= args.max_snapshot_bytes <= 1024**3):
        raise ValueError("snapshot byte budget outside [1,1 GiB]")

    # Lazy import allows argument validation and --help without optional torch.
    from pleasure_experiment.open_weight_calibrated_benchmark import (
        run_calibrated_appraisal_benchmark,
    )

    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )

    def encode_offer(offer):
        prompt = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prompt,
            action_suffixes=loaded.action_suffixes,
        ) != loaded.action_token_ids:
            raise RuntimeError("A/B token order changed on a withheld offer")
        return tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))

    return run_calibrated_appraisal_benchmark(
        loaded, encode_offer,
        seeds=tuple(args.seeds), training_pairs=args.training_pairs,
        heldout_pairs_per_generator=args.heldout_pairs_per_generator,
        exploration_rate=args.exploration_rate,
        learning_rate=args.learning_rate,
        appraisal_gain=args.appraisal_gain,
        artifact_root=args.save_head_root, evaluator_id=EVALUATOR,
        allow_appraisal_training=True, allow_persistent_export=True,
    )


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError,
            ImportError, OSError) as exc:
        parser.exit(2, f"calibrated appraisal study rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
