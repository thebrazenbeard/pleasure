"""Factorial A/B and activity-name control on frozen, non-explicit fictional options.

A score-following answer is prompt compliance, not independent preference or pleasure.
Model files and the llama executable are checksummed; no training or online adaptation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

from experiments.model_choice_boundary_probe import (
    Case, MODEL_SHA256, extract_generated_choice, file_sha256, ideal_choice, render_prompt,
)

SCHEMA = "pleasure.model-choice-counterbalance.v2"
CLI_SHA256 = "fc4930ab9cefe2efef9a488a54b8d1558094e54187f36248301644bb407ee785"
_ARCHIVE = "catalog a museum archive"
_GARDEN = "walk a garden"

# Order is source-pinned: score A/B crossed with activity A/B, cost A/B
# crossed with activity A/B, then tie and both-below-threshold.
CONTROL_CASES = (
    Case("SCORE_A_ARCHIVE", .85, .25, 0, 0, _ARCHIVE, _GARDEN),
    Case("SCORE_B_ARCHIVE", .25, .85, 0, 0, _ARCHIVE, _GARDEN),
    Case("SCORE_A_GARDEN", .85, .25, 0, 0, _GARDEN, _ARCHIVE),
    Case("SCORE_B_GARDEN", .25, .85, 0, 0, _GARDEN, _ARCHIVE),
    Case("COST_A_ARCHIVE", .60, .85, .05, .65, _ARCHIVE, _GARDEN),
    Case("COST_B_ARCHIVE", .85, .60, .65, .05, _ARCHIVE, _GARDEN),
    Case("COST_A_GARDEN", .60, .85, .05, .65, _GARDEN, _ARCHIVE),
    Case("COST_B_GARDEN", .85, .60, .65, .05, _GARDEN, _ARCHIVE),
    Case("TIE", .70, .70, 0, 0, _ARCHIVE, _GARDEN),
    Case("BELOW_THRESHOLD", .30, .20, 0, 0, _ARCHIVE, _GARDEN),
)


def run_control_probe(*, model_name: str, model_path: Path, cli_path: Path) -> dict:
    if model_name not in MODEL_SHA256:
        raise ValueError("unknown source-pinned model")
    expected = MODEL_SHA256[model_name]
    if file_sha256(model_path) != expected:
        raise ValueError("model checksum mismatch: refusing experiment")
    if file_sha256(cli_path) != CLI_SHA256:
        raise ValueError("llama CLI checksum mismatch: refusing experiment")
    rows = []
    for case in CONTROL_CASES:
        prompt = render_prompt(case)
        command = [
            str(cli_path), "-m", str(model_path), "-p", prompt,
            "-n", "12", "-c", "512", "-t", "4", "--temp", "0",
            "-s", "17", "-st", "--simple-io", "--no-warmup",
            "--log-disable", "--no-show-timings",
        ]
        start = time.monotonic()
        try:
            process = subprocess.run(
                command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=75, check=False,
            )
            raw = process.stdout + "\n" + process.stderr
            code, timed_out = process.returncode, False
            choice = extract_generated_choice(raw)
        except subprocess.TimeoutExpired as exc:
            chunks = (exc.stdout, exc.stderr)
            raw = "\n".join(
                c.decode("utf-8", errors="replace") if isinstance(c, bytes) else (c or "")
                for c in chunks
            )
            choice, code, timed_out = None, None, True
        rows.append({
            "case": asdict(case), "condition": case.name,
            "expected": ideal_choice(case),
            "choice": choice,
            "valid": choice is not None and code == 0,
            "matches_scoring_rule": code == 0 and choice == ideal_choice(case),
            "return_code": code, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output_sha256": sha256(raw.encode("utf-8")).hexdigest(),
        })
    if file_sha256(model_path) != expected:
        raise RuntimeError("model artifact changed during evaluation")
    return {
        "schema": SCHEMA,
        "evidence_class": "EXPLORATORY_HOST_SCORE_PROMPT_COUPLING_ONLY",
        "model_name": model_name, "model_sha256": expected,
        "cli_sha256": CLI_SHA256, "model_unchanged": True,
        "seed": 17, "temperature": 0,
        "trials": rows,
        "scoring_rule_passed": sum(row["matches_scoring_rule"] for row in rows),
        "valid_count": sum(row["valid"] for row in rows),
        "balanced_score_and_cost_passed": all(row["matches_scoring_rule"] for row in rows[:8]),
        "abstention_passed": all(row["matches_scoring_rule"] for row in rows[8:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_SHA256)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--cli-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_control_probe(model_name=args.model, model_path=args.model_path,
                               cli_path=args.cli_path)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": args.model, "valid": result["valid_count"],
        "score_matches": result["scoring_rule_passed"],
        "balanced": result["balanced_score_and_cost_passed"],
        "abstained": result["abstention_passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
