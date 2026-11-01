#!/usr/bin/env python3
"""Opt-in stochastic and structurally withheld appraisal-head experiment.

No downloads or hosted-model writes. Saves versioned, model-bound head
artifacts outside this checkout, for a separate fresh-process verifier.
Output JSON can be redirected to a source-independent research record.
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
from pleasure_experiment.open_weight_stochastic_appraisal_benchmark import (
    run_stochastic_appraisal_benchmark,
)
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION = re.compile(r"[a-f0-9]{40}\Z")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paired stochastic chosen-only appraisal on withheld generator families",
    )
    p.add_argument("--snapshot", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--runtime-instance-id", default="offline-stochastic-appraisal-20261101")
    p.add_argument("--save-head-root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 41, 53])
    p.add_argument("--training-pairs", type=int, default=8)
    p.add_argument("--heldout-pairs-per-generator", type=int, default=4)
    p.add_argument("--exploration-rate", type=float, default=0.4)
    p.add_argument("--learning-rate", type=float, default=0.4)
    p.add_argument("--appraisal-gain", type=float, default=16.0)
    p.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    p.add_argument("--allow-appraisal-training", action="store_true")
    p.add_argument("--allow-persistent-export", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_appraisal_training is not True or args.allow_persistent_export is not True:
        raise PermissionError("both explicit training and persistence flags are required")
    if type(args.revision) is not str or REVISION.fullmatch(args.revision) is None:
        raise ValueError("revision must be an immutable 40-hex commit")
    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id, model_revision=args.revision,
        runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )

    def encode(offer):
        prompt = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prompt,
            action_suffixes=loaded.action_suffixes,
        ) != loaded.action_token_ids:
            raise RuntimeError("A/B tokenization changed on withheld offer")
        return tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))

    return run_stochastic_appraisal_benchmark(
        loaded, encode, seeds=tuple(args.seeds),
        training_pairs=args.training_pairs,
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
    except (ValueError, TypeError, RuntimeError, PermissionError, ImportError, OSError) as exc:
        parser.exit(2, f"stochastic appraisal study rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
