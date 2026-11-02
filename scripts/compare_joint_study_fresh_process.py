#!/usr/bin/env python3
"""Post-COMPLETE reproduction of four original v3 study decisions.

Only the immutable report's pre-outcome policy fields are sent to an
independent Python worker. The worker reissues task offers without committing,
resolving, or querying any payoff and compares its own reloaded LoRA/head
predictions with the original study. This is source-local evidence only.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pleasure_experiment.isolated_outcome_evaluator_v3 import (
    V3OutcomeEvaluator, V3_WITHIN_FAMILIES,
)
from pleasure_experiment.open_weight_joint_appraisal import TorchJointLoRAAppraisal
from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_joint_persistence import load_versioned_joint_appraisal
from pleasure_experiment.open_weight_local_model import (
    load_verified_local_causal_model, prepare_explicit_lora,
    validate_single_token_actions,
)
from pleasure_experiment.open_weight_stochastic import SeededActionSampler
from scripts.run_open_weight_local_trial import EVALUATOR, public_prompt


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
WEIGHTS_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
SEED = 163
SEEDS = (163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)
FAMILY = V3_WITHIN_FAMILIES[0]
ARM = "joint_on"
EPSILON = 0.4
GAIN = 12.0
BEHAVIOR_RNG_XOR = 0x39B0164F
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_WORKER_JSON_BYTES = 1024 * 1024
HEX = re.compile(r"[a-f0-9]{64}\Z")
NUMERIC_FIELDS = (
    "prob_a", "prob_b", "behavior_prob_a", "behavior_prob_b",
    "prediction_a", "prediction_b", "random_draw", "chosen_propensity",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce original v3 seed163 evaluations in a fresh Python process",
    )
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-snapshot-bytes", type=int, default=1024**3)
    parser.add_argument("--allow-reload", action="store_true")
    parser.add_argument("--allow-evidence-export", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_parent-pid", type=int, help=argparse.SUPPRESS)
    return parser


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _strict_json(data: str) -> dict[str, object]:
    value = json.loads(data, object_pairs_hook=_no_duplicates,
                       parse_constant=_reject_constant)
    if type(value) is not dict:
        raise ValueError("strict JSON object required")
    return value


def _real_directory(path: str | Path, label: str) -> Path:
    target = Path(path)
    if (not target.is_absolute() or ".." in target.parts or
        not target.is_dir() or target.is_symlink()):
        raise ValueError(f"{label} must be an existing real absolute directory")
    return target


def _safe_file(path: Path, *, max_bytes: int) -> bytes:
    if (not path.is_file() or path.is_symlink() or
        not 1 <= path.stat().st_size <= max_bytes):
        raise ValueError(f"unsafe, missing, or oversized file: {path.name}")
    return path.read_bytes()


def _completed_report(study: Path) -> tuple[dict[str, object], str]:
    """Fail before model loading if the study lacks an exact COMPLETE marker."""
    report_bytes = _safe_file(study / "study-report.json", max_bytes=MAX_REPORT_BYTES)
    digest = sha256(report_bytes).hexdigest()
    marker = _safe_file(study / "COMPLETE.sha256", max_bytes=256)
    if marker != f"{digest}  study-report.json\n".encode("ascii"):
        raise ValueError("study is not complete with a matching report checksum")
    report = _strict_json(report_bytes.decode("utf-8"))
    if (report.get("artifact_run_directory") != str(study) or
        report.get("evidence_class") !=
            "SOURCE_LOCAL_JOINT_LORA_SEALED_OUTCOME_V3_UNAUTHENTICATED" or
        report.get("model_id") != MODEL_ID or report.get("model_revision") != REVISION or
        report.get("pretrained_model_unchanged") is not True):
        raise ValueError("completed report identity or frozen-backbone claim differs")
    proto = report.get("predeclared_protocol")
    if (type(proto) is not dict or proto.get("seeds") != list(SEEDS) or
        proto.get("training_pairs_per_family") != 8 or
        proto.get("heldout_pairs_per_generator") != 4 or
        proto.get("exploration_rate") != EPSILON or
        proto.get("appraisal_gain") != GAIN):
        raise ValueError("completed report does not match frozen v3 protocol")
    return report, digest


def _expected_input(report: dict[str, object], study: Path,
                    report_sha: str) -> dict[str, object]:
    seeds = report.get("seed_reports")
    if (type(seeds) is not list or len(seeds) != len(SEEDS) or
        sorted(row.get("seed") for row in seeds if type(row) is dict) != list(SEEDS)):
        raise ValueError("completed study missing twelve distinct seed reports")
    seed = next(row for row in seeds if row["seed"] == SEED)
    adapter = study / f"seed-{SEED}-joint-lora-head-v1"
    if (seed.get("saved_joint_head_directory") != str(adapter) or
        not adapter.is_dir() or adapter.is_symlink()):
        raise ValueError("original seed163 learned artifact path is missing or altered")
    source = seed.get("eval_arms")
    generator = seed.get("source_local_generator_manifest")
    if (type(generator) is not dict or
        type(generator.get("suite_sha256")) is not str or
        HEX.fullmatch(generator["suite_sha256"]) is None):
        raise ValueError("original seed163 generator source identity is missing")
    if type(source) is not dict or type(source.get(ARM)) is not dict:
        raise ValueError("original seed163 joint-on arm not recorded")
    records = source[ARM].get("rows")
    if type(records) is not list or len(records) != 20:
        raise ValueError("original joint-on evaluation schedule is incomplete")
    expected: list[dict[str, object]] = []
    for index, row in enumerate(records[:4]):
        if (type(row) is not dict or row.get("family_id") != FAMILY or
            type(row.get("sampled_pre_feedback")) is not dict):
            raise ValueError("first four study records are not fixed latent00 offers")
        trace = row["sampled_pre_feedback"]
        if (trace.get("draw_index") != index or trace.get("policy_id") != ARM or
            trace.get("partition") != "EVAL" or trace.get("exploration_rate") != EPSILON or
            trace.get("seed") != (SEED ^ BEHAVIOR_RNG_XOR)):
            raise ValueError("recorded matched-action sampler does not match v3")
        fields = {
            "episode_id": row.get("episode_id"),
            "input_ids_sha256": trace.get("input_ids_sha256"),
            "joint_sha256": row.get("decision_joint_sha256"),
            "lora_sha256": row.get("decision_lora_sha256"),
            "head_sha256": row.get("decision_head_sha256"),
            "prob_a": trace.get("prob_a"), "prob_b": trace.get("prob_b"),
            "behavior_prob_a": trace.get("behavior_prob_a"),
            "behavior_prob_b": trace.get("behavior_prob_b"),
            "prediction_a": row.get("prediction_a"),
            "prediction_b": row.get("prediction_b"),
            "random_draw": trace.get("random_draw"),
            "chosen_action": trace.get("chosen_action"),
            "chosen_propensity": trace.get("chosen_propensity"),
        }
        if (any(type(fields[key]) is not str or HEX.fullmatch(fields[key]) is None
                for key in ("episode_id", "input_ids_sha256", "joint_sha256",
                            "lora_sha256", "head_sha256")) or
            fields["chosen_action"] not in ("A", "B") or
            any(type(fields[key]) not in (float, int) or not math.isfinite(fields[key])
                for key in NUMERIC_FIELDS)):
            raise ValueError("malformed original study pre-feedback inference row")
        expected.append(fields)
    if (any(row["joint_sha256"] != seed.get("trained_joint_sha256") or
            row["lora_sha256"] != seed.get("trained_lora_sha256") or
            row["head_sha256"] != seed.get("trained_head_sha256")
            for row in expected)):
        raise ValueError("original study evaluation tensors differ from saved TRAIN state")
    return {
        "report_sha256": report_sha, "adapter_dir": str(adapter),
        "seed": SEED, "family": FAMILY,
        "expected_joint_sha256": seed["trained_joint_sha256"],
        "expected_lora_sha256": seed["trained_lora_sha256"],
        "expected_head_sha256": seed["trained_head_sha256"],
        "expected_pretrained_sha256": report["pretrained_model_sha256"],
        "expected_adapter_manifest_sha256": seed["saved_joint_manifest_sha256"],
        "expected_generator_suite_sha256": generator["suite_sha256"],
        "expected_rows": expected,
    }


def _close(a: object, b: object) -> bool:
    return (type(a) in (float, int) and type(b) in (float, int) and
            math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7))


def _score_from_fresh_model(snapshot: str, payload: dict[str, object],
                            max_snapshot_bytes: int) -> dict[str, object]:
    """Worker obtains only offers and pre-feedback reference fields, never payoffs."""
    torch = importlib.import_module("torch")
    torch.set_num_threads(2)
    runtime = f"joint-v3-original-study-repro-{os.getpid()}"
    loaded = load_verified_local_causal_model(
        snapshot, model_id=MODEL_ID, model_revision=REVISION,
        runtime_instance_id=runtime, max_total_bytes=max_snapshot_bytes,
    )
    if not any(row.name == "model.safetensors" and row.sha256 == WEIGHTS_SHA256
               for row in loaded.manifest.files):
        raise ValueError("fresh-model safetensors are not the pinned study weights")
    loaded.model.eval()
    base_rows = tuple(loaded.model.named_parameters()) + tuple(
        ("buffer:" + name, value) for name, value in loaded.model.named_buffers()
    )
    if _named_digest(torch, base_rows) != payload["expected_pretrained_sha256"]:
        raise RuntimeError("fresh original pretrained backbone differs from study")
    prepared = prepare_explicit_lora(loaded, target_modules=("q_proj", "v_proj"),
                                     rank=2, alpha=4)
    probe = TorchJointLoRAAppraisal(
        prepared, hidden_size=loaded.model.config.hidden_size,
        learning_rate=.05, allow_training=True, max_updates=48, train_lora=True,
    )
    before_base_sha = probe.base_digest
    artifact = load_versioned_joint_appraisal(
        probe, binding=loaded.binding, snapshot_manifest=loaded.manifest,
        source_directory=payload["adapter_dir"], allow_in_place_mutation=True,
    )
    if (artifact.binding.runtime_instance_id == runtime or probe.updates != 24 or
        probe.joint_digest != payload["expected_joint_sha256"] or
        probe.lora_digest != payload["expected_lora_sha256"] or
        probe.head_digest != payload["expected_head_sha256"] or
        probe.base_digest != artifact.frozen_base_sha256 or
        artifact.manifest_sha256 != payload["expected_adapter_manifest_sha256"] or
        probe.base_digest != before_base_sha):
        raise RuntimeError("new process did not restore exact saved study model state")
    sampler = SeededActionSampler(seed=SEED ^ BEHAVIOR_RNG_XOR,
                                  exploration_rate=EPSILON)
    observed: list[dict[str, object]] = []
    with V3OutcomeEvaluator(
        seed=SEED, evaluator_id=EVALUATOR, arms=(ARM,),
        training_pairs_per_family=8, heldout_pairs_per_generator=4,
    ) as evaluator:
        manifest = evaluator.manifest
        if (FAMILY not in manifest.primary_within_families or
            manifest.gap_filtering is not False or
            manifest.suite_sha256 != payload["expected_generator_suite_sha256"]):
            raise RuntimeError("new process v3 generator suite differs from original")
        for index, reference in enumerate(payload["expected_rows"]):
            offer = evaluator.issue(partition="EVAL", arm=ARM,
                                    family_id=FAMILY, pair_index=index, side=0)
            prompt = public_prompt(offer)
            if validate_single_token_actions(
                loaded.tokenizer, action_prefix=prompt,
                action_suffixes=loaded.action_suffixes,
            ) != loaded.action_token_ids:
                raise RuntimeError("fresh tokenizer A/B action identity differs")
            ids = tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))
            decision = probe.score(ids, gain=GAIN, partition="EVAL",
                                   episode_id=offer.episode_id)
            trace = sampler.sample(
                decision.prob_a, decision.prob_b,
                input_ids_sha256=decision.input_ids_sha256,
                partition="EVAL", episode_id=offer.episode_id,
                policy_id=ARM,
            )
            row = {
                "episode_id": offer.episode_id,
                "input_ids_sha256": decision.input_ids_sha256,
                "joint_sha256": decision.joint_digest,
                "lora_sha256": decision.lora_digest,
                "head_sha256": decision.head_digest,
                "prob_a": trace.prob_a, "prob_b": trace.prob_b,
                "behavior_prob_a": trace.behavior_prob_a,
                "behavior_prob_b": trace.behavior_prob_b,
                "prediction_a": decision.predicted_quality_a,
                "prediction_b": decision.predicted_quality_b,
                "random_draw": trace.random_draw,
                "chosen_action": trace.chosen_action,
                "chosen_propensity": trace.chosen_propensity,
            }
            for field in ("episode_id", "input_ids_sha256", "joint_sha256",
                          "lora_sha256", "head_sha256", "chosen_action"):
                if row[field] != reference[field]:
                    raise RuntimeError(f"fresh original-study offer {index}: {field} differs")
            for field in NUMERIC_FIELDS:
                if not _close(row[field], reference[field]):
                    raise RuntimeError(f"fresh original-study offer {index}: {field} differs")
            observed.append(row)
    if (probe.base_digest != before_base_sha or probe.updates != 24 or
        probe.joint_digest != payload["expected_joint_sha256"]):
        raise RuntimeError("read-only replay changed adapter state or frozen backbone")
    return {
        "fresh_process_pid": os.getpid(), "parent_process_pid": os.getppid(),
        "fresh_model_runtime_instance_id": runtime,
        "completed_study_report_sha256": payload["report_sha256"],
        "source_adapter_directory": payload["adapter_dir"],
        "imported_joint_sha256": probe.joint_digest,
        "imported_lora_sha256": probe.lora_digest,
        "imported_head_sha256": probe.head_digest,
        "frozen_base_sha256_unchanged": probe.base_digest,
        "updates_at_import": probe.updates,
        "evaluator_worker_pid": evaluator.worker_pid,
        "four_original_study_decisions_reproduced": observed,
        "all_original_study_probabilities_and_predictions_match": True,
        "no_eval_outcomes_requested": True,
        "no_training_performed": True,
        "evidence_scope": "SOURCE_LOCAL_REPRODUCTION_OF_ORIGINAL_V3_EVAL_POLICY_NOT_SUBJECTIVE_FEELING",
    }


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    if args.allow_reload is not True or args.allow_evidence_export is not True:
        raise PermissionError("explicit reload and evidence export consent required")
    study = _real_directory(args.study_dir, "completed study")
    _real_directory(args.snapshot, "pinned snapshot")
    output = Path(args.output)
    if (not output.is_absolute() or ".." in output.parts or
        not output.parent.is_dir() or output.exists() or output.is_symlink() or
        output == study or study in output.parents or output.parent.is_symlink()):
        raise ValueError("evidence output must be new, absolute, outside study")
    if type(args.max_snapshot_bytes) is not int or not 1 <= args.max_snapshot_bytes <= 1024**3:
        raise ValueError("invalid bounded snapshot size")
    report, report_sha = _completed_report(study)
    payload = _expected_input(report, study, report_sha)
    command = [
        sys.executable, "-I", "-B", str(Path(__file__).resolve()),
        "--_worker", "--_parent-pid", str(os.getpid()),
        "--snapshot", args.snapshot, "--study-dir", str(study),
        "--output", args.output, "--max-snapshot-bytes", str(args.max_snapshot_bytes),
    ]
    result = subprocess.run(command, input=json.dumps(payload, sort_keys=True),
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"fresh model replay failed ({result.returncode}): "
                           + result.stderr.strip()[:512])
    if len(result.stdout.encode("utf-8")) > MAX_WORKER_JSON_BYTES:
        raise RuntimeError("fresh model evidence exceeds output cap")
    evidence = _strict_json(result.stdout)
    if (evidence.get("fresh_process_pid") == os.getpid() or
        type(evidence.get("fresh_process_pid")) is not int or
        evidence.get("parent_process_pid") != os.getpid() or
        evidence.get("all_original_study_probabilities_and_predictions_match") is not True or
        evidence.get("no_eval_outcomes_requested") is not True or
        evidence.get("no_training_performed") is not True or
        evidence.get("completed_study_report_sha256") != report_sha or
        evidence.get("imported_joint_sha256") != payload["expected_joint_sha256"] or
        evidence.get("imported_lora_sha256") != payload["expected_lora_sha256"] or
        evidence.get("imported_head_sha256") != payload["expected_head_sha256"] or
        type(evidence.get("four_original_study_decisions_reproduced")) is not list or
        len(evidence["four_original_study_decisions_reproduced"]) != 4):
        raise RuntimeError("fresh subprocess parity proof incomplete or inconsistent")
    for observed, reference in zip(evidence["four_original_study_decisions_reproduced"],
                                   payload["expected_rows"]):
        if type(observed) is not dict or set(observed) != set(reference) or any(
            observed[field] != reference[field]
            for field in ("episode_id", "input_ids_sha256", "joint_sha256",
                          "lora_sha256", "head_sha256", "chosen_action")
        ) or any(not _close(observed[field], reference[field])
                 for field in NUMERIC_FIELDS):
            raise RuntimeError("fresh evidence record did not match original study")
    document = {
        "original_study_report_sha256": report_sha,
        "original_study_directory": str(study),
        "original_seed": SEED,
        "original_arm": ARM,
        "original_heldout_family": FAMILY,
        "original_saved_adapter_directory": payload["adapter_dir"],
        "source_local_not_externally_authenticated": True,
        "hosted_chatgpt_weights_modified": False,
        "subjective_pleasure_proven": False,
        "reproduction": evidence,
    }
    serialized = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                             allow_nan=False) + "\n").encode("utf-8")
    # Never modify a completed study or replace an existing evidence file.
    with output.open("xb") as target:
        target.write(serialized)
        target.flush()
        os.fsync(target.fileno())
    return document


def run(args: argparse.Namespace) -> dict[str, object]:
    if args._worker:
        if args._parent_pid != os.getppid():
            raise PermissionError("fresh worker must identify a live parent process")
        raw = sys.stdin.read(MAX_WORKER_JSON_BYTES + 1)
        if len(raw.encode("utf-8")) > MAX_WORKER_JSON_BYTES:
            raise ValueError("worker input exceeds maximum bounded source evidence")
        payload = _strict_json(raw)
        if (payload.get("seed") != SEED or payload.get("family") != FAMILY or
            type(payload.get("expected_rows")) is not list or
            len(payload["expected_rows"]) != 4 or
            type(payload.get("report_sha256")) is not str or
            HEX.fullmatch(payload["report_sha256"]) is None):
            raise ValueError("worker source reference is malformed")
        return _score_from_fresh_model(args.snapshot, payload, args.max_snapshot_bytes)
    if args._parent_pid is not None:
        raise PermissionError("internal worker arguments are not for parent invocation")
    return _run_parent(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        result = run(parser.parse_args(argv))
    except (ValueError, TypeError, PermissionError, RuntimeError, ImportError,
            OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"joint post-COMPLETE reproduction rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
