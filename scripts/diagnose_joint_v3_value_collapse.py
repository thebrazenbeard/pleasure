#!/usr/bin/env python3
"""Read-only, posthoc v3 appraisal discrimination diagnostic for v4 design.

Only COMPLETE v3 studies pass the independent verifier before any report data
is analyzed. Output is exploratory historical evidence, never a v4 EVAL set.
No training, model loading, generator access, or artifact mutation occurs.
"""
from __future__ import annotations

import argparse
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
from statistics import mean, pstdev
from scripts.verify_joint_study_report import verify_complete_study

FAMILIES = ("eval_latent_00_v3", "eval_latent_01_v3",
            "eval_ood_00_v3", "eval_ood_01_v3", "eval_ood_02_v3")
ARMS = ("joint_on", "head_only_on", "shuffled_joint_on", "frozen_on")
SEEDS = (163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)


def _gap(row: dict) -> tuple[float, float]:
    predicted = row["prediction_a"] - row["prediction_b"]
    actual = row["quality_a_posthoc"] - row["quality_b_posthoc"]
    if not all(isfinite(x) for x in (predicted, actual)):
        raise ValueError("nonfinite A/B gap")
    return predicted, actual


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mx, my = mean(xs), mean(ys)
    a = fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sqrt(fsum((x - mx)**2 for x in xs) *
             fsum((y - my)**2 for y in ys))
    return a / b if b > 1e-15 else None


def diagnose(report: dict) -> dict:
    if (report.get("evidence_class") !=
        "SOURCE_LOCAL_JOINT_LORA_SEALED_OUTCOME_V3_UNAUTHENTICATED"):
        raise ValueError("not the exact v3 evidence class")
    seeds = report["seed_reports"]
    if tuple(sorted(s["seed"] for s in seeds)) != SEEDS:
        raise ValueError("incomplete v3 seed schedule")
    pooled = {arm: {family: [] for family in FAMILIES} for arm in ARMS}
    within_seed, seed_means = [], []
    parity = []
    for seed in seeds:
        arms = seed["eval_arms"]
        if any(arm not in arms for arm in ARMS):
            raise ValueError("missing diagnostic control")
        joint_rows = {r["episode_id"]: r for r in arms["joint_on"]["rows"]}
        head_rows = {r["episode_id"]: r for r in arms["head_only_on"]["rows"]}
        if len(joint_rows) != 20 or joint_rows.keys() != head_rows.keys():
            raise ValueError("unpaired v3 diagnostic rows")
        gap_values = []
        for episode, row in joint_rows.items():
            jp, _ = _gap(row)
            hp, _ = _gap(head_rows[episode])
            gap_values.append(jp)
            parity.append(abs(jp - hp))
        within_seed.append(pstdev(gap_values))
        seed_means.append(mean(gap_values))
        for arm in ARMS:
            rows = arms[arm]["rows"]
            if len(rows) != 20 or {r["episode_id"] for r in rows} != joint_rows.keys():
                raise ValueError("unmatched diagnostic arm")
            for row in rows:
                family = row["family_id"]
                if family not in FAMILIES:
                    raise ValueError("unrecognized v3 family")
                pooled[arm][family].append(_gap(row))
    within = mean(within_seed)
    between = pstdev(seed_means)
    stats = {}
    for arm in ARMS:
        stats[arm] = {}
        for family in FAMILIES:
            pairs = pooled[arm][family]
            if len(pairs) != 48:
                raise ValueError("v3 expected 12 seeds x 4 offers per family")
            pred, true = list(zip(*pairs))
            correct = [p*t > 0 for p, t in pairs if abs(t) > 1e-12]
            stats[arm][family] = {
                "n": len(pairs),
                "preference_sign_accuracy_nontied": (sum(correct)/len(correct)
                                                    if correct else None),
                "predicted_gap_mean": mean(pred),
                "predicted_gap_sd": pstdev(pred),
                "true_gap_mean": mean(true),
                "gap_pearson": _pearson(list(pred), list(true)),
                "gap_absolute_error": mean(abs(p-t) for p,t in pairs),
            }
    return {
        "evidence_class": "POSTHOC_V3_DIAGNOSTIC_NOT_V4_CONFIRMATORY",
        "seed_count": len(seeds),
        "joint_on_n_offers": 240,
        "mean_within_seed_predicted_gap_sd": within,
        "between_seed_mean_predicted_gap_sd": between,
        "within_over_between_predicted_gap_sd": within / between if between else None,
        "joint_on_vs_head_only_mean_absolute_predicted_gap_difference": mean(parity),
        "joint_on_vs_head_only_max_absolute_predicted_gap_difference": max(parity),
        "by_arm_and_family": stats,
        "interpretation_guard": "Prediction-gap collapse is observational; "
                                "it does not by itself prove why features are insensitive.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("completed_study_dir", type=Path)
    args = parser.parse_args()
    directory = args.completed_study_dir.absolute()
    # Preserve path components for the verifier's symlink rejection.
    verify_complete_study(directory)
    report = json.loads((directory / "study-report.json").read_bytes())
    print(json.dumps(diagnose(report), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
