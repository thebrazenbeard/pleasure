#!/usr/bin/env python3
"""Compare sealed v2 study predictions to a *new-process* appraisal reload.

Input: finished study-report.json and the standalone verifier's JSON stdout.
No model or payoff rules are executed; the comparator ignores the reward
values present in the already finalized study record. This is purely
source-local trace consistency. In particular, this
does not establish that a persistent appraisal improved future outcomes.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any


SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_STUDY_BYTES = 16 * 1024 * 1024
MAX_RELOAD_BYTES = 1024 * 1024
PROTOCOL_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157)
FAMILIES = ("sealed_eval_00_v2", "sealed_eval_01_v2",
            "sealed_eval_02_v2", "sealed_eval_03_v2")
ARM_PAIRS = (
    ("after_import_on", "trained_on"),
    ("after_import_off", "trained_off"),
    ("after_import_inverted", "trained_inverted"),
    ("before_import_on", "frozen_on"),
)


def _reject_duplicates(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in rows:
        if key in output:
            raise ValueError("duplicate JSON key in evidence record")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r} rejected")


def _read_json(path: str | Path, *, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if (not source.is_absolute() or ".." in source.parts or
        source.is_symlink() or not source.is_file()):
        raise ValueError("evidence must be an existing absolute regular file")
    size = source.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise ValueError("evidence file exceeds its strict size bound")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"),
                           object_pairs_hook=_reject_duplicates,
                           parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError("JSON evidence top level must be an object")
    return value, raw


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: Any, name: str, *, unit: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} outside finite float range") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if unit and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} outside [0, 1]")
    return number


def _matches(a: Any, b: Any, tolerance: float, name: str) -> float:
    difference = abs(_finite(a, name, unit=True) -
                     _finite(b, name, unit=True))
    if difference > tolerance:
        raise ValueError(f"independent-process {name} differs by {difference:.9g}")
    return difference


def _behavior(raw_a: Any, raw_b: Any, epsilon: float) -> float:
    a = _finite(raw_a, "fresh P(A)", unit=True)
    b = _finite(raw_b, "fresh P(B)", unit=True)
    total = a + b
    if not 0.999999 <= total <= 1.000001:
        raise ValueError("fresh model probabilities are not normalized")
    return (1.0 - epsilon) * (a / total) + epsilon / 2.0


def compare_reports(study: dict[str, Any], reload: dict[str, Any], *,
                    seed: int, family: str, tolerance: float = 2e-6) -> dict[str, object]:
    """Fail closed on missing, mismatched, or reordered fresh-process offers."""
    if type(study) is not dict or type(reload) is not dict:
        raise ValueError("both reports must be JSON objects")
    if type(seed) is not int or seed not in PROTOCOL_SEEDS or family not in FAMILIES:
        raise ValueError("seed/family outside predeclared novel_v2 study")
    if type(tolerance) is not float or not 0 < tolerance <= 2e-6:
        raise ValueError("tolerance exceeds strict fresh-process comparison bound")
    if (study.get("evidence_class") !=
        "SOURCE_LOCAL_CALIBRATED_SEALED_PROCESS_APPRAISAL_UNAUTHENTICATED" or
        study.get("pretrained_model_unchanged") is not True or
        study.get("in_process_saved_head_import_verified") is not True or
        study.get("subjective_pleasure_proven") is not False):
        raise ValueError("study is not an intact completed calibrated v2 report")
    protocol = study.get("predeclared_protocol")
    if (type(protocol) is not dict or
        protocol.get("seeds") != list(PROTOCOL_SEEDS) or
        protocol.get("training_pairs_per_seed") != 24 or
        protocol.get("heldout_pairs_per_family_per_seed") != 6 or
        protocol.get("suite") != "novel_v2" or
        protocol.get("primary_ood_generators") != list(FAMILIES[1:]) or
        protocol.get("positive_control_generator") != FAMILIES[0] or
        protocol.get("appraisal_gain") != 12.0 or
        protocol.get("global_all_arm_seal_before_any_eval_reward") is not True):
        raise ValueError("study protocol differs from frozen v2 plan")
    epsilon = _finite(protocol.get("exploration_rate"), "exploration", unit=True)
    if epsilon != 0.4:
        raise ValueError("exploration rate differs from predeclared protocol")
    rows = study.get("seed_reports")
    if (type(rows) is not list or len(rows) != len(PROTOCOL_SEEDS) or
        [item.get("seed") for item in rows if type(item) is dict]
        != list(PROTOCOL_SEEDS)):
        raise ValueError("study is missing one or more seed reports")
    origin = next(item for item in rows if item["seed"] == seed)

    if (reload.get("tensor_import_verified") is not True or
        reload.get("spawned_fresh_model_process") is not True or
        reload.get("unseen_generator_outcomes_not_read_during_reload") is not True or
        reload.get("frozen_head_and_disabled_head_controls_preserved") is not True or
        reload.get("quality_gain_established_by_reload_alone") is not False or
        reload.get("subjective_pleasure_proven") is not False or
        reload.get("fixed_heldout_family") != family or
        reload.get("model_id_declared") != study.get("model_id") or
        reload.get("model_revision_declared") != study.get("model_revision")):
        raise ValueError("independent verifier identity, gate or claims mismatch")
    for field in ("fresh_process_pid", "parent_process_pid", "independent_evaluator_worker_pid"):
        if type(reload.get(field)) is not int or reload[field] <= 0:
            raise ValueError("fresh process identities are invalid")
    if len({reload["fresh_process_pid"], reload["parent_process_pid"],
            reload["independent_evaluator_worker_pid"]}) != 3:
        raise ValueError("fresh process is not distinct from parent and evaluator")
    if (reload.get("distinct_runtime_instance_ids") is not True or
        not reload.get("origin_runtime_instance_id_declared") or
        not reload.get("new_runtime_instance_id_declared") or
        reload["origin_runtime_instance_id_declared"] ==
        reload["new_runtime_instance_id_declared"]):
        raise ValueError("reloaded model uses the original runtime identity")

    sha_fields = (
        (reload.get("imported_trained_head_sha256"), origin.get("trained_head_sha256"), "head"),
        (reload.get("head_manifest_sha256"), origin.get("saved_head_manifest_sha256"), "head manifest"),
        (reload.get("head_safetensors_sha256"), origin.get("saved_head_safetensors_sha256"), "safetensors"),
        (reload.get("frozen_model_sha256_unchanged"),
         study.get("frozen_pretrained_model_sha256"), "frozen model"),
    )
    for fresh, recorded, label in sha_fields:
        if _digest(fresh, label) != _digest(recorded, label):
            raise ValueError(f"fresh-process {label} digest mismatch")
    if reload.get("feature_schema_verified_on_import") != study.get("head_feature_schema"):
        raise ValueError("reloaded feature schema differs from study")
    if reload.get("initial_untrained_head_sha256") != origin.get("initial_head_sha256"):
        raise ValueError("new process did not start from exact zero head")

    episodes = reload.get("fixed_heldout_episode_ids")
    if (type(episodes) is not list or len(episodes) != 6 or
        any(type(x) is not str or not x for x in episodes) or
        len(set(episodes)) != 6):
        raise ValueError("fresh verifier must cover six unique heldout offers")
    eval_arms = origin.get("eval_arms")
    if type(eval_arms) is not dict:
        raise ValueError("study evaluation arm records absent")
    maxima = {"probability": 0.0, "quality": 0.0, "baseline_probability": 0.0}
    for fresh_key, arm in ARM_PAIRS:
        benchmark = eval_arms.get(arm)
        original_rows = benchmark.get("rows") if type(benchmark) is dict else None
        if type(original_rows) is not list or len(original_rows) != 24:
            raise ValueError(f"{arm} must include 24 EVAL rows")
        selected = [entry for entry in original_rows
                    if type(entry) is dict and entry.get("family_id") == family]
        if (len(selected) != 6 or
            [row.get("episode_id") for row in selected] != episodes or
            len({row["episode_id"] for row in selected}) != 6):
            raise ValueError(f"{arm} misses or reorders heldout episodes")
        fresh_rows = reload.get(fresh_key)
        if type(fresh_rows) is not list or len(fresh_rows) != 6:
            raise ValueError(f"fresh verifier {fresh_key} must contain exactly six rows")
        for original, new in zip(selected, fresh_rows):
            if type(new) is not dict or new.get("episode_id") != original["episode_id"]:
                raise ValueError("fresh and original episode IDs differ")
            observed = original.get("sampled_pre_feedback")
            if type(observed) is not dict:
                raise ValueError("benchmark sampled probability trace missing")
            if _digest(new.get("input_ids_sha256"), "input") != _digest(
                observed.get("input_ids_sha256"), "input"
            ):
                raise ValueError("fresh and original token sequences differ")
            expected_head = (origin["initial_head_sha256"] if arm == "frozen_on"
                             else origin["trained_head_sha256"])
            if _digest(new.get("head_sha256"), "head") != expected_head:
                raise ValueError("fresh and original head digest mismatch")
            base_p = _behavior(new.get("base_prob_a"), new.get("base_prob_b"), epsilon)
            # For trained ON and inverted, base P(A) must agree with the
            # disabled-head trace for this exact episode, not ON's shifted P.
            if arm in ("trained_on", "trained_inverted"):
                disabled = eval_arms["trained_off"]["rows"]
                matching_disabled = next(
                    (row for row in disabled if row.get("episode_id") == original["episode_id"]),
                    None,
                )
                if matching_disabled is None:
                    raise ValueError("disabled-head baseline episode not found")
                if type(matching_disabled.get("sampled_pre_feedback")) is not dict:
                    raise ValueError("disabled-head baseline trace absent")
                delta_base = _matches(
                    base_p, matching_disabled["sampled_pre_feedback"]["behavior_prob_a"],
                    tolerance, "disabled baseline behavior P(A)",
                )
            else:
                delta_base = _matches(
                    base_p, observed.get("behavior_prob_a"),
                    tolerance, "base behavior P(A)",
                )
            maxima["baseline_probability"] = max(maxima["baseline_probability"], delta_base)
            quality_a = _matches(new.get("predicted_quality_a"),
                                 original.get("prediction_a"), tolerance,
                                 "predicted quality A")
            quality_b = _matches(new.get("predicted_quality_b"),
                                 original.get("prediction_b"), tolerance,
                                 "predicted quality B")
            maxima["quality"] = max(maxima["quality"], quality_a, quality_b)
            behavior_a = _behavior(new.get("prob_a"), new.get("prob_b"), epsilon)
            difference = _matches(behavior_a, observed.get("behavior_prob_a"),
                                  tolerance, "behavior policy P(A)")
            maxima["probability"] = max(maxima["probability"], difference)
    # Difference in model probabilities must be the measured head intervention,
    # not a change in the pretrained model or an imported wrong tensor schema.
    return {
        "source_local_cross_process_behavior_verified": True,
        "fresh_reload_does_not_establish_reward_gain": True,
        "fresh_reload_does_not_establish_subjective_experience": True,
        "model_and_head_digests_matched": True,
        "new_process_pid": reload["fresh_process_pid"],
        "original_runtime_instance_id": reload["origin_runtime_instance_id_declared"],
        "fresh_runtime_instance_id": reload["new_runtime_instance_id_declared"],
        "seed": seed, "family": family, "episodes_verified": len(episodes),
        "treatment_control_pairs_verified": len(ARM_PAIRS),
        "max_abs_behavior_prob_a_error": maxima["probability"],
        "max_abs_base_prob_a_error": maxima["baseline_probability"],
        "max_abs_quality_prediction_error": maxima["quality"],
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-local independent fresh-model prediction cross-check",
    )
    parser.add_argument("--study-report", required=True)
    parser.add_argument("--fresh-report", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    args = parser.parse_args()
    try:
        study, raw = _read_json(args.study_report, maximum_bytes=MAX_STUDY_BYTES)
        if Path(args.study_report).name != "study-report.json":
            raise ValueError("expected the canonical completed study-report.json")
        complete = Path(args.study_report).parent / "COMPLETE.sha256"
        if complete.is_symlink() or not complete.is_file():
            raise ValueError("the study is missing its complete-run marker")
        if complete.read_text(encoding="ascii") != (
            sha256(raw).hexdigest() + "  study-report.json\n"
        ):
            raise ValueError("completed study-report checksum mismatch")
        if study.get("artifact_run_directory") != str(Path(args.study_report).parent):
            raise ValueError("study report is not in its declared completed run directory")
        fresh, _ = _read_json(args.fresh_report, maximum_bytes=MAX_RELOAD_BYTES)
        outcome = compare_reports(study, fresh, seed=args.seed, family=args.family)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"fresh-process comparison rejected: {exc}\n")
    print(json.dumps(outcome, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
