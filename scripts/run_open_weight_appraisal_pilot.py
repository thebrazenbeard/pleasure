#!/usr/bin/env python3
"""Pretrained LM hidden-state appraisal causal-ablation pilot.

Trains a separately registered, source-local quality head ONLY from chosen
TRAIN outcomes. On disjoint EVAL offers, replay the *same head/model/tokens*
with appraisal gain zero, positive, and inverted. This can test causal action
influence, but not establish subjective experience or independent-world reward.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from math import fsum
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.open_weight_appraisal import (
    TorchHiddenAppraisal, compare_appraisal_ablations,
)
from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, validate_single_token_actions,
)
from pleasure_experiment.outcome_arena import OutcomeArena
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


REVISION_RE = re.compile(r"[a-f0-9]{40}\Z")
TRAIN_FAMILY = "appraisal_hidden_train"
EVAL_FAMILY = "appraisal_hidden_heldout"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Opt-in hidden-state appraisal ON/OFF pilot")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--runtime-instance-id", default="offline-appraisal-pilot")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    p.add_argument("--training-pairs", type=int, default=2)
    p.add_argument("--heldout-pairs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.4)
    p.add_argument("--appraisal-gain", type=float, default=16.0)
    p.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    p.add_argument("--allow-appraisal-training", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_appraisal_training is not True:
        raise PermissionError("pass --allow-appraisal-training for local head SGD")
    if type(args.revision) is not str or REVISION_RE.fullmatch(args.revision) is None:
        raise ValueError("revision must be an exact 40-hex commit")
    seeds = tuple(args.seeds)
    if (not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("seeds must be 1..32 distinct uint32 integers")
    if type(args.training_pairs) is not int or not 1 <= args.training_pairs <= 64:
        raise ValueError("training-pairs must be 1..64")
    if type(args.heldout_pairs) is not int or not 1 <= args.heldout_pairs <= 64:
        raise ValueError("heldout-pairs must be 1..64")
    if type(args.appraisal_gain) not in (int, float) or not 0 < args.appraisal_gain <= 16:
        raise ValueError("appraisal-gain must be finite in (0,16]")

    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=args.model_id,
        model_revision=args.revision, runtime_instance_id=args.runtime_instance_id,
        max_total_bytes=args.max_snapshot_bytes,
    )
    torch = importlib.import_module("torch")
    model = loaded.model
    hidden_size = model.config.hidden_size
    model.eval()
    model_named_state = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, buffer) for name, buffer in model.named_buffers()
    )
    unchanged_model_sha = _named_digest(torch, model_named_state)

    def encode(offer):
        prompt = public_prompt(offer)
        if validate_single_token_actions(
            loaded.tokenizer, action_prefix=prompt,
            action_suffixes=loaded.action_suffixes,
        ) != loaded.action_token_ids:
            raise RuntimeError("appraisal offer changes pinned A/B tokenizer tokens")
        return tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))

    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        probe = TorchHiddenAppraisal(
            model, action_token_ids=loaded.action_token_ids,
            hidden_size=hidden_size, learning_rate=args.learning_rate,
            allow_training=True, max_updates=args.training_pairs,
            max_delta_l2=1.0,
        )
        arena = OutcomeArena(seed=seed, evaluator_id=EVALUATOR)
        initial_head = probe.head_digest
        train_rows = []
        for i in range(args.training_pairs):
            offer = arena.issue(
                partition="TRAIN", family_id=TRAIN_FAMILY,
                pair_index=i, side=i % 2,
            )
            decision = probe.score(encode(offer), partition="TRAIN", gain=0.0)
            ticket = arena.commit(
                offer, decision.chosen_action, decision.decision_id,
                model_id=loaded.binding.model_id,
                runtime_instance_id=loaded.binding.runtime_instance_id,
            )
            actual = arena.resolve(ticket)
            update = probe.observe_chosen(
                decision, measured_quality=actual.measured_quality,
                evidence_id=actual.evidence_id,
            )
            train_rows.append({
                "episode_id": offer.episode_id,
                "observed_chosen_quality": actual.measured_quality,
                "head_sha256_before": update.before_digest,
                "head_sha256_after": update.after_digest,
                "optimizer_step_executed": update.optimizer_step_executed,
            })
        trained_head = probe.head_digest
        # Check that the learned head can be exported and independently
        # reconstructed as an inference-only module without touching the LM.
        restored_head = TorchHiddenAppraisal(
            model, action_token_ids=loaded.action_token_ids,
            hidden_size=hidden_size, learning_rate=0.0, allow_training=False,
        )
        restored_head.import_tensors(
            probe.export_tensors(), expected_digest=trained_head,
        )
        if restored_head.head_digest != trained_head:
            raise RuntimeError("appraisal export/import head readback diverged")
        by_arm = {arm: [] for arm in ("off", "on", "inverted")}
        paired = []
        for i in range(args.heldout_pairs):
            for side in (0, 1):
                offer = arena.issue(
                    partition="EVAL", family_id=EVAL_FAMILY,
                    pair_index=i, side=side,
                )
                ids = encode(offer)
                decisions = compare_appraisal_ablations(
                    probe, ids, gain=args.appraisal_gain,
                )
                outcomes = {}
                for arm, decision in decisions.items():
                    # Each intervention has its own one-use arena with the
                    # SAME fixed offer; no fake double commit is accepted.
                    evaluation = OutcomeArena(seed=seed, evaluator_id=EVALUATOR)
                    independent_offer = evaluation.issue(
                        partition="EVAL", family_id=EVAL_FAMILY,
                        pair_index=i, side=side,
                    )
                    if independent_offer != offer:
                        raise RuntimeError("appraisal heldout arm task mismatch")
                    ticket = evaluation.commit(
                        independent_offer, decision.chosen_action, decision.decision_id,
                        model_id=loaded.binding.model_id,
                        runtime_instance_id=loaded.binding.runtime_instance_id,
                    )
                    outcome = evaluation.resolve(ticket)
                    predicted = (decision.predicted_quality_a
                                 if decision.chosen_action == "A"
                                 else decision.predicted_quality_b)
                    record = {
                        "episode_id": offer.episode_id,
                        "chosen_action": decision.chosen_action,
                        "prob_a": decision.prob_a,
                        "chosen_quality": outcome.measured_quality,
                        "predicted_chosen_quality": predicted,
                        "chosen_quality_squared_error": (
                            predicted - outcome.measured_quality
                        ) ** 2,
                        "head_sha256": decision.head_digest,
                    }
                    by_arm[arm].append(record)
                    outcomes[arm] = record
                if side == i % 2:
                    paired.append({
                        "episode_id": offer.episode_id,
                        "on_minus_off_quality": (
                            outcomes["on"]["chosen_quality"]
                            - outcomes["off"]["chosen_quality"]),
                        "inverted_minus_off_quality": (
                            outcomes["inverted"]["chosen_quality"]
                            - outcomes["off"]["chosen_quality"]),
                        "on_vs_off_action_changed": (
                            outcomes["on"]["chosen_action"] !=
                            outcomes["off"]["chosen_action"]),
                        "on_minus_off_prob_a": (
                            outcomes["on"]["prob_a"] -
                            outcomes["off"]["prob_a"]),
                    })
        if probe.head_digest != trained_head:
            raise RuntimeError("appraisal EVAL unexpectedly mutated learned head")
        per_seed.append({
            "seed": seed,
            "initial_head_sha256": initial_head,
            "trained_head_sha256": trained_head,
            "head_changed": trained_head != initial_head,
            "head_export_import_round_trip_verified": True,
            "training": train_rows,
            "primary_paired": paired,
            "arms": {name: {
                "mean_realized_quality_primary": fsum(
                    row["chosen_quality"] for j, row in enumerate(rows)
                    if j // 2 % 2 == j % 2
                ) / args.heldout_pairs,
                "mean_chosen_reward_mse_all_mirrors": fsum(
                    row["chosen_quality_squared_error"] for row in rows
                ) / len(rows),
                "evaluated_mirror_count": len(rows),
            } for name, rows in by_arm.items()},
        })

    differences = [
        p["on_minus_off_quality"]
        for row in per_seed for p in row["primary_paired"]
    ]
    changed_actions = sum(
        p["on_vs_off_action_changed"]
        for row in per_seed for p in row["primary_paired"]
    )
    prob_deltas = [
        abs(p["on_minus_off_prob_a"])
        for row in per_seed for p in row["primary_paired"]
    ]
    if _named_digest(torch, model_named_state) != unchanged_model_sha:
        raise RuntimeError("appraisal training modified pretrained LM tensors")
    return {
        "model_id_declared": args.model_id,
        "model_revision_declared": args.revision,
        "seeds": list(seeds),
        "train_outcomes_per_seed": args.training_pairs,
        "independent_primary_heldout_pairs_per_seed": args.heldout_pairs,
        "gain": args.appraisal_gain,
        "seed_reports": per_seed,
        "mean_on_minus_off_quality_pooled_primary_exploratory": (
            fsum(differences) / len(differences)),
        "on_off_primary_action_switches": changed_actions,
        "mean_absolute_on_off_prob_a_difference_primary": (
            fsum(prob_deltas) / len(prob_deltas)),
        "pretrained_lm_tensor_sha256_unchanged": unchanged_model_sha,
        "appraisal_head_source": "MODEL_HIDDEN_STATES_SEPARATE_TRAINABLE_HEAD",
        "appraisal_inference_causally_intervened": changed_actions > 0,
        "quality_gain_established": False,
        "external_evaluator_authenticated": False,
        "hosted_gpt_modified": False,
        "subjective_pleasure_proven": False,
    }


def main() -> None:
    parser = build_parser()
    try:
        result = run(parser.parse_args())
    except (ValueError, TypeError, RuntimeError, PermissionError,
            ImportError, OSError) as exc:
        parser.exit(2, f"appraisal pilot rejected: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
