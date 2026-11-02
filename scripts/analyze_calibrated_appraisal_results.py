#!/usr/bin/env python3
"""Read-only, posthoc diagnostics for the frozen calibrated v2 study.

This module does not train or select models, choose seeds/families, adjust the
predeclared success criterion, or contact the reward evaluator. It measures
calibration against *already disclosed posthoc* qA/qB in study-report.json.
Pooled diagnostic rows are descriptive; inferential comparisons are paired
over twelve independent seed schedules, never over individual offers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
from statistics import stdev
from typing import Any


ARMS = (
    "trained_on", "trained_off", "trained_inverted", "shuffled_on", "frozen_on",
)
EXPECTED_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157)
OOD_FAMILIES = (
    "sealed_eval_01_v2", "sealed_eval_02_v2", "sealed_eval_03_v2",
)
IN_FAMILY = "sealed_eval_00_v2"
T_CRITICAL_95_DF11 = 2.200985160082949  # two-sided Student t, df=11
MAX_REPORT_BYTES = 20 * 1024 * 1024


def _number(value: object, name: str) -> float:
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be a probability or [0,1] quality")
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("missing diagnostic rows")
    errors, chosen_errors, correct, ties, spreads = [], [], [], [], []
    for row in rows:
        a = _number(row["prediction_a"], "prediction_a")
        b = _number(row["prediction_b"], "prediction_b")
        qa = _number(row["quality_a_posthoc"], "quality_a_posthoc")
        qb = _number(row["quality_b_posthoc"], "quality_b_posthoc")
        choice = row["sampled_pre_feedback"]["chosen_action"]
        if choice not in ("A", "B"):
            raise ValueError("unknown sampled action")
        observed = _number(row["chosen_quality"], "chosen_quality")
        if abs(observed - (qa if choice == "A" else qb)) > 1e-10:
            raise ValueError("chosen receipt disagrees with sealed oracle")
        errors.extend((a - qa, b - qb))
        chosen_errors.append((a - qa) if choice == "A" else (b - qb))
        # The source generator excludes near-equal true action payoffs; treat
        # *predicted* ties as misses and disclose their prevalence separately.
        if abs(qa - qb) <= 0.02 - 1e-10:
            raise ValueError("source-local gap-filtered offer violated")
        correct.append(int((a - b) * (qa - qb) > 0))
        ties.append(int(a == b))
        spreads.append(abs(a - b))
    n = len(rows)
    return {
        "n_offers": n,
        "n_candidate_actions": 2 * n,
        "mean_candidate_mae": fsum(abs(e) for e in errors) / (2 * n),
        "mean_candidate_brier": fsum(e * e for e in errors) / (2 * n),
        "mean_chosen_mae": fsum(abs(e) for e in chosen_errors) / n,
        "mean_chosen_brier": fsum(e * e for e in chosen_errors) / n,
        "strict_preference_sign_accuracy": fsum(correct) / n,
        "predicted_tie_fraction": fsum(ties) / n,
        "mean_absolute_predicted_ab_spread": fsum(spreads) / n,
    }


def _t_interval(values: list[float]) -> dict[str, object]:
    if len(values) != len(EXPECTED_SEEDS):
        raise ValueError("t11 sensitivity requires exactly twelve paired seeds")
    mean = fsum(values) / len(values)
    sd = stdev(values)
    margin = T_CRITICAL_95_DF11 * sd / sqrt(len(values))
    return {
        "n_seed_clusters": len(values),
        "paired_seed_differences": values,
        "mean": mean,
        "sample_sd": sd,
        "t_df": 11,
        "t_critical_two_sided_95": T_CRITICAL_95_DF11,
        "paired_student_t_95_sensitivity_interval": [mean - margin, mean + margin],
        "interpretation": "descriptive sensitivity only; fixed seed set, not an externally preregistered randomized study",
    }


def analyze_study_report(study: dict[str, Any]) -> dict[str, object]:
    """Calculate fixed diagnostics without accessing any evaluator or model."""
    if type(study) is not dict:
        raise ValueError("study report must be an object")
    protocol = study.get("predeclared_protocol")
    if type(protocol) is not dict or (
        tuple(protocol.get("seeds", ())) != EXPECTED_SEEDS
        or tuple(protocol.get("primary_ood_generators", ())) != OOD_FAMILIES
        or protocol.get("positive_control_generator") != IN_FAMILY
        or protocol.get("suite") != "novel_v2"
        or protocol.get("training_pairs_per_seed") != 24
        or protocol.get("heldout_pairs_per_family_per_seed") != 6
    ):
        raise ValueError("report differs from frozen calibrated v2 protocol")
    seeds = study.get("seed_reports")
    if type(seeds) is not list or len(seeds) != len(EXPECTED_SEEDS) or (
        tuple(item.get("seed") for item in seeds) != EXPECTED_SEEDS
    ):
        raise ValueError("twelve paired seed reports must be complete and in protocol order")
    families = (IN_FAMILY,) + OOD_FAMILIES
    pooled: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {family: [] for family in families} for arm in ARMS
    }
    per_seed: list[dict[str, object]] = []
    contrasts: dict[str, list[float]] = {"trained_on_minus_off": [],
                                         "trained_on_minus_shuffled": []}
    for seed_report in seeds:
        arm_data = seed_report.get("eval_arms")
        if type(arm_data) is not dict or set(arm_data) != set(ARMS):
            raise ValueError("missing or unexpected EVAL arms")
        this_seed: dict[str, object] = {"seed": seed_report["seed"], "arms": {}}
        aligned: list[tuple[str, str]] | None = None
        family_expected: dict[str, dict[str, float]] = defaultdict(dict)
        for arm in ARMS:
            rows = arm_data[arm].get("rows")
            if type(rows) is not list or len(rows) != 24:
                raise ValueError("each EVAL arm must contain exactly 24 offers")
            identities = [(row["episode_id"], row["family_id"]) for row in rows]
            if aligned is None:
                aligned = identities
            elif identities != aligned:
                raise ValueError("paired EVAL episode order differs across arms")
            if any(sum(row["family_id"] == family for row in rows) != 6
                   for family in families):
                raise ValueError("each EVAL family must have exactly six offers")
            arm_by_family: dict[str, object] = {}
            for family in families:
                relevant = [row for row in rows if row["family_id"] == family]
                # Independently reconstruct the behavioral policy's expected
                # payoff from the *sampled* epsilon-mixture probabilities.
                values = []
                for row in relevant:
                    p = _number(row["sampled_pre_feedback"]["behavior_prob_a"],
                                "behavior_prob_a")
                    p_b = _number(row["sampled_pre_feedback"]["behavior_prob_b"],
                                  "behavior_prob_b")
                    if abs(p + p_b - 1) > 1e-8:
                        raise ValueError("logged behavioral probabilities do not sum to one")
                    expected = (p * _number(row["quality_a_posthoc"], "quality_a_posthoc") +
                                p_b * _number(row["quality_b_posthoc"], "quality_b_posthoc"))
                    if abs(expected - row["expected_policy_quality_posthoc"]) > 1e-9:
                        raise ValueError("posthoc expected quality arithmetic mismatch")
                    values.append(expected)
                family_expected[arm][family] = fsum(values) / len(values)
                pooled[arm][family].extend(relevant)
                arm_by_family[family] = _summary(relevant)
            this_seed["arms"][arm] = {"by_family": arm_by_family,
                                      "all_families": _summary(rows)}
        def primary(arm: str) -> float:
            return fsum(family_expected[arm][family] for family in OOD_FAMILIES) / 3
        contrasts["trained_on_minus_off"].append(primary("trained_on") - primary("trained_off"))
        contrasts["trained_on_minus_shuffled"].append(primary("trained_on") - primary("shuffled_on"))
        per_seed.append(this_seed)
    for label, report_label in (
        ("trained_on_minus_off", "primary_ood_trained_on_minus_off"),
        ("trained_on_minus_shuffled", "primary_ood_trained_on_minus_shuffled"),
    ):
        recorded = study.get(report_label)
        if type(recorded) is not dict or len(recorded.get("seed_level_differences", ())) != 12:
            raise ValueError("original primary paired result is absent")
        if any(abs(a - b) > 1e-9 for a, b in zip(
            contrasts[label], recorded["seed_level_differences"],
        )):
            raise ValueError("independently recomputed seed effects disagree with report")
    return {
        "analysis_class": "POSTHOC_READ_ONLY_DESCRIPTIVE_NO_SELECTION_OR_TUNING",
        "source_local_oracle_not_external_validation": True,
        "fixed_primary_ood_families": list(OOD_FAMILIES),
        "calibration_definition": "deterministic quality regression; Brier denotes squared error against q in [0,1], not Bernoulli outcome Brier",
        "preference_definition": "strict sign agreement between predicted and true A-minus-B quality; predicted ties count as misses",
        "pooled_descriptive_by_arm": {
            arm: {
                "all_families": _summary([r for fam in families for r in pooled[arm][fam]]),
                "by_family": {family: _summary(pooled[arm][family]) for family in families},
            }
            for arm in ARMS
        },
        "per_seed_diagnostics": per_seed,
        "paired_primary_t11_sensitivity": {
            label: _t_interval(values) for label, values in contrasts.items()
        },
        "caveat": "T intervals assume approximately independent, representative, normal seed effects; multiplicity, source-visible rules, deterministic seed schedule, and gap filtering limit inference.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path,
                        help="finished calibrated-v2 study-report.json (read-only)")
    parser.add_argument("--save-json", type=Path,
                        help="create a new posthoc diagnostics JSON file without overwriting")
    arguments = parser.parse_args(argv)
    path = arguments.report
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        parser.error("report must be an existing JSON file below 20 MiB")
    try:
        body = path.read_bytes()
        report = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("JSON contains a nonfinite constant"),
        ))
        analysis = analyze_study_report(report)
        analysis["input_study_report_sha256"] = sha256(body).hexdigest()
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        parser.error(f"invalid study-report.json: {exc}")
    serialized = json.dumps(analysis, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.save_json is not None:
        destination = arguments.save_json
        if (not destination.is_absolute() or ".." in destination.parts or
            not destination.parent.is_dir() or destination.is_symlink()):
            parser.error("posthoc diagnostics destination must be a new absolute file")
        try:
            with destination.open("x", encoding="utf-8", newline="\n") as output:
                output.write(serialized)
        except FileExistsError:
            parser.error("posthoc diagnostics destination already exists")
        print(json.dumps({"saved_posthoc_analysis": str(destination)}, sort_keys=True))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
