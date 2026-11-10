"""Read-only v4 action-feature sensitivity probe on a pinned local LM.

Diagnoses candidate-value and option-label sensitivity with *no reward data*,
training, downloads, or weight export. These synthetic TRAIN-like probes have
never been part of v4 EVAL. Untrained frozen base only; do not infer behavior
of a separately learned LoRA or appraisal head from this result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
from math import exp, fsum, isfinite, sin, sqrt, tanh
from pathlib import Path

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, validate_single_token_actions,
)
from pleasure_experiment.outcome_arena import TaskOffer
from scripts.run_open_weight_local_trial import public_prompt


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SAFETENSORS_SHA = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
EVIDENCE_CLASS = "V4_UNTRAINED_BASE_ACTION_FEATURE_PROBE_NO_REWARDS"
DEFAULT_SEED = 0
# This fixed design is exploratory and contains no hidden payoff labels.
BASE_A = (0.25, 0.75)
BASE_B = (0.75, 0.25)


@dataclass(frozen=True)
class Case:
    case_id: str
    features_a: tuple[float, float]
    features_b: tuple[float, float]


def cases() -> tuple[Case, ...]:
    return (
        Case("baseline", BASE_A, BASE_B),
        Case("baseline_repeat", BASE_A, BASE_B),
        Case("change_a_x", (0.85, 0.75), BASE_B),
        Case("change_a_y", (0.25, 0.15), BASE_B),
        Case("change_b_x", BASE_A, (0.15, 0.25)),
        Case("change_b_y", BASE_A, (0.75, 0.85)),
        Case("swap_options", BASE_B, BASE_A),
        Case("both_equal_a", BASE_A, BASE_A),
        Case("both_equal_b", BASE_B, BASE_B),
        Case("values_extreme", (0.95, 0.05), (0.05, 0.95)),
    )


def _offer(case: Case) -> TaskOffer:
    if type(case) is not Case or not case.case_id or len(case.case_id) > 40:
        raise ValueError("bounded exact diagnostic Case required")
    for values in (case.features_a, case.features_b):
        if (type(values) is not tuple or len(values) != 2 or
            any(type(v) is not float or not isfinite(v) or not 0.0 <= v <= 1.0
                for v in values)):
            raise ValueError("two bounded finite float candidate features required")
    return TaskOffer(
        task_id="v4-probe-" + case.case_id,
        episode_id="v4-probe-" + case.case_id,
        partition="TRAIN", family_id="v4_diagnostic_probe",
        pair_index=0, side=0,
        nonce=sha256(case.case_id.encode("ascii")).hexdigest(),
        features_a=case.features_a, features_b=case.features_b,
    )


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or len(a) != len(b):
        raise ValueError("nonempty same-width vectors required")
    if any(not isfinite(x) for x in a + b):
        raise ValueError("nonfinite vector coordinate")
    numerator = fsum(x * y for x, y in zip(a, b))
    denominator = sqrt(fsum(x*x for x in a) * fsum(y*y for y in b))
    if denominator <= 1e-20:
        raise ValueError("zero-norm feature vector")
    return max(-1.0, min(1.0, numerator / denominator))


def _relative_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or len(a) != len(b) or any(not isfinite(x) for x in a+b):
        raise ValueError("finite same-width vectors required")
    denominator = sqrt(fsum(x*x for x in a))
    if denominator <= 1e-20:
        raise ValueError("reference action delta is zero")
    return sqrt(fsum((x-y)**2 for x, y in zip(a, b))) / denominator


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    z = exp(value)
    return z / (1.0 + z)


def _initial_head_gap(unit: tuple[float, ...]) -> float:
    """Exactly the v3 deterministic untrained head, *not* learned preferences."""
    width = len(unit)
    if not width or any(not isfinite(x) for x in unit):
        raise ValueError("bounded unit vector required")
    dot = fsum(0.01 * sin((i+1)*1.61803398875) / sqrt(width) * v
               for i, v in enumerate(unit))
    return _sigmoid(2*tanh(dot/2)) - _sigmoid(2*tanh(-dot/2))


def _model_feature(model, torch, ids: tuple[int, ...],
                   actions: tuple[int, int]):
    batch = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        base = model(input_ids=batch, use_cache=False)
        logits = base.logits[0, -1, list(actions)].float()
        states = []
        for token in actions:
            output = model(
                input_ids=torch.tensor([ids + (token,)], dtype=torch.long),
                output_hidden_states=True, use_cache=False,
            )
            state = output.hidden_states[-1][0, -1, :].float()
            if not bool(torch.isfinite(state).all()):
                raise ValueError("nonfinite hidden state")
            states.append(state)
        delta = states[0] - states[1]
        norm = torch.linalg.vector_norm(delta)
        if not bool(torch.isfinite(delta).all()) or float(norm) <= 1e-20:
            raise ValueError("invalid action hidden-state difference")
        normalized = delta / norm.clamp_min(1e-5)
        probabilities = torch.softmax(logits, dim=-1)
        return (
            tuple(float(x) for x in delta),
            tuple(float(x) for x in normalized),
            tuple(float(x) for x in probabilities),
        )


def run_probe(loaded, *, torch=None) -> dict:
    if torch is None:
        torch = importlib.import_module("torch")
    model = loaded.model
    if any(param.device.type != "cpu" for param in model.parameters()):
        raise ValueError("v4 probe requires an exclusively CPU model")
    checkpoint = [row for row in loaded.manifest.files if row.name == "model.safetensors"]
    if len(checkpoint) != 1 or checkpoint[0].sha256 != SAFETENSORS_SHA:
        raise ValueError("not the pinned SmolLM2 safetensors")
    named = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, value) for name, value in model.named_buffers())
    initial_sha = _named_digest(torch, named)
    original_training = model.training
    original_cache = getattr(model.config, "use_cache", None)
    observations = {}
    try:
        model.eval()
        if original_cache is not None:
            model.config.use_cache = False
        for case in cases():
            prompt = public_prompt(_offer(case))
            action_ids = validate_single_token_actions(
                loaded.tokenizer, action_prefix=prompt,
                action_suffixes=loaded.action_suffixes)
            if action_ids != loaded.action_token_ids:
                raise ValueError("action token IDs changed during perturbation")
            ids = tuple(loaded.tokenizer.encode(
                prompt, add_special_tokens=False))
            delta, unit, probabilities = _model_feature(
                model, torch, ids, action_ids)
            observations[case.case_id] = {
                "input_ids_sha256": sha256(
                    json.dumps(ids, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "raw_delta": delta, "unit_delta": unit,
                "base_prob_a": probabilities[0],
                "untrained_head_quality_gap": _initial_head_gap(unit),
            }
    finally:
        model.train(original_training)
        if original_cache is not None:
            model.config.use_cache = original_cache
    if _named_digest(torch, named) != initial_sha:
        raise RuntimeError("v4 feature probe modified pretrained model parameters")
    baseline = observations["baseline"]
    repeat = observations["baseline_repeat"]
    if (baseline["raw_delta"] != repeat["raw_delta"] or
        baseline["base_prob_a"] != repeat["base_prob_a"]):
        raise RuntimeError("frozen model failed duplicate-prompt determinism")
    result = {}
    for case in cases():
        row = observations[case.case_id]
        result[case.case_id] = {
            "input_ids_sha256": row["input_ids_sha256"],
            "cosine_delta_to_baseline": _cosine(
                baseline["unit_delta"], row["unit_delta"]),
            "relative_raw_delta_shift": _relative_delta(
                baseline["raw_delta"], row["raw_delta"]),
            "base_prob_a": row["base_prob_a"],
            "untrained_head_quality_gap": row["untrained_head_quality_gap"],
        }
    swap = result["swap_options"]
    return {
        "evidence_class": EVIDENCE_CLASS,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "pretrained_frozen_sha256_before_after": initial_sha,
        "probes": result,
        "candidate_swap_quality_gap_sum": (
            result["baseline"]["untrained_head_quality_gap"] +
            swap["untrained_head_quality_gap"]),
        "meaning": "Exploratory untrained representation sensitivity only; "
                   "no EVAL rewards, actual TRAIN feedback, or learned head tested.",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True,
                   help="absolute preexisting pinned local safetensors directory")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_threads != 2 or not 1 <= args.max_snapshot_bytes <= 1024**3:
        raise ValueError("bounded fixed v4 read-only resource envelope required")
    torch = importlib.import_module("torch")
    torch.set_num_threads(args.cpu_threads)
    loaded = load_verified_local_causal_model(
        args.snapshot, model_id=MODEL_ID, model_revision=REVISION,
        runtime_instance_id="v4-frozen-feature-probe",
        max_total_bytes=args.max_snapshot_bytes,
    )
    report = run_probe(loaded, torch=torch)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"),
                     allow_nan=False))


if __name__ == "__main__":
    main()
