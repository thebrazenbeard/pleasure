#!/usr/bin/env python3
"""Independently recompute a COMPLETE v3 joint-appraisal study from its receipts.

Read-only and deliberately usable without PyTorch, PEFT, or an evaluator.
It never opens pending study directories or obtains outcome-generator access.
Hashes and deterministic trace checks establish local consistency, not
independent authentication of the synthetic payoff process or phenomenology.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import exp, fsum, isclose, isfinite, log, sqrt
from pathlib import Path
import random
import re
from statistics import stdev
import struct


SEEDS = (163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)
ARMS = ("joint_on", "joint_off", "joint_inverted", "head_only_on",
        "shuffled_joint_on", "frozen_on", "frozen_off")
TRAIN_FAMILIES = ("train_latent_00_v3", "train_latent_01_v3", "train_latent_02_v3")
PRIMARY_FAMILIES = ("eval_latent_00_v3", "eval_latent_01_v3")
OOD_FAMILIES = ("eval_ood_00_v3", "eval_ood_01_v3", "eval_ood_02_v3")
EVAL_FAMILIES = PRIMARY_FAMILIES + OOD_FAMILIES
CONTRASTS = {"joint_on_minus_off": "joint_off",
             "joint_on_minus_shuffled": "shuffled_joint_on",
             "joint_on_minus_head_only": "head_only_on"}
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
EVIDENCE_CLASS = "SOURCE_LOCAL_JOINT_LORA_SEALED_OUTCOME_V3_UNAUTHENTICATED"
T_CRITICAL_DF11 = 2.200985160082949
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_WEIGHTS_BYTES = 64 * 1024 * 1024
LORA_SEED_SALT = 0x4A0175C3
SHA = re.compile(r"[0-9a-f]{64}\Z")
RUN = re.compile(r"joint-lora-v3-[0-9a-f]{20}\Z")
ARTIFACT_FIELDS = frozenset((
    "format", "format_version", "capture_stage", "binding", "snapshot_sha256",
    "frozen_base_sha256", "action_token_ids", "hidden_size", "feature_schema",
    "lora_config", "optimizer_spec", "updates_at_export", "seen_evidence_ids",
    "seen_episode_ids", "replay_ledger_sha256", "joint_weights_sha256",
    "head_weights_sha256", "lora_weights_sha256", "tensors",
    "weights_safetensors_sha256", "optimizer_state_format",
))


def _require(ok: object, explanation: str) -> None:
    if not ok:
        raise ValueError("invalid COMPLETE v3 study: " + explanation)


def _close(actual: object, expected: object, explanation: str,
           tolerance: float = 1e-11) -> None:
    _require(type(actual) in (float, int) and type(expected) in (float, int)
             and isfinite(actual) and isfinite(expected)
             and isclose(float(actual), float(expected), rel_tol=0.0,
                         abs_tol=tolerance), explanation)


def _digest_object(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        _require(name not in result, "duplicate JSON key: " + name)
        result[name] = value
    return result


def _no_constant(value: str) -> object:
    raise ValueError("invalid COMPLETE v3 study: nonfinite JSON value " + value)


def _unlinked(path: Path, *, directory: bool = False) -> None:
    _require(path.is_absolute(), "path must be absolute")
    for component in (path,) + tuple(path.parents):
        _require(not component.is_symlink(), "symbolic-link path forbidden")
    _require(path.is_dir() if directory else path.is_file(),
             "required directory/file missing: " + path.name)


def _read(path: Path, budget: int) -> bytes:
    _unlinked(path)
    size = path.stat().st_size
    _require(0 < size <= budget, "file byte budget exceeded: " + path.name)
    value = path.read_bytes()
    _require(len(value) == size and path.stat().st_size == size,
             "file changed during read")
    return value


def _load_json(raw: bytes, name: str) -> dict[str, object]:
    try:
        result = json.loads(raw, object_pairs_hook=_no_duplicates,
                            parse_constant=_no_constant)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid COMPLETE v3 study: unreadable JSON: " + name) from exc
    _require(type(result) is dict, name + " must be JSON object")
    return result


def _hex(value: object, label: str) -> str:
    _require(type(value) is str and SHA.fullmatch(value) is not None,
             label + " is not SHA256")
    return value


def _trace(row: dict[str, object], *, seed: int, index: int,
           arm: str, partition: str, rng: random.Random) -> dict[str, object]:
    trace = row["sampled_pre_feedback"]
    _require(type(trace) is dict, "missing action trace")
    sampled_seed = seed ^ (0x72AC2911 if partition == "TRAIN" else 0x39B0164F)
    _require(trace["seed"] == sampled_seed and trace["draw_index"] == index and
             trace["partition"] == partition and trace["episode_id"] == row["episode_id"] and
             trace["policy_id"] == arm and trace["exploration_rate"] == 0.4,
             "stochastic action trace identity or order")
    _hex(trace["input_ids_sha256"], "trace token digest")
    _hex(trace["decision_digest"], "trace decision digest")
    a, b = trace["prob_a"], trace["prob_b"]
    _require(type(a) in (float, int) and type(b) in (float, int) and
             isfinite(a) and isfinite(b) and 0 <= a <= 1 and 0 <= b <= 1,
             "invalid pre-exploration probabilities")
    _close(a + b, 1.0, "pre-exploration policy not normalized", 1e-6)
    normalized_a = a / (a + b)
    normalized_b = 1.0 - normalized_a
    behavior_a = 0.6 * normalized_a + 0.2
    behavior_b = 1.0 - behavior_a
    _close(trace["behavior_prob_a"], behavior_a, "epsilon mixture A changed")
    _close(trace["behavior_prob_b"], behavior_b, "epsilon mixture B changed")
    actual_draw = rng.random()
    _close(trace["random_draw"], actual_draw, "seeded choice draw changed", 0.0)
    chosen = "A" if actual_draw < behavior_a else "B"
    _require(trace["chosen_action"] == chosen, "chosen action inconsistent with draw")
    propensity = behavior_a if chosen == "A" else behavior_b
    _close(trace["chosen_propensity"], propensity, "chosen propensity changed")
    payload = ("PLEASURE_STOCHASTIC_CHOICE_V1", sampled_seed, index, partition,
               row["episode_id"], trace["input_ids_sha256"], arm,
               normalized_a.hex(), normalized_b.hex(), (0.4).hex(),
               behavior_a.hex(), behavior_b.hex(), actual_draw.hex(),
               chosen, propensity.hex())
    decision_sha = sha256(json.dumps(payload, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()
    _require(trace["decision_digest"] == decision_sha, "choice commitment digest changed")
    return trace


def _paired(effects: list[float]) -> dict[str, object]:
    _require(len(effects) == 12, "paired statistic requires exactly 12 seeds")
    mean = fsum(effects) / 12
    sd = stdev(effects)
    margin = T_CRITICAL_DF11 * sd / sqrt(12)
    observed = abs(fsum(effects))
    extreme = 0
    for mask in range(1 << 12):
        alternate = fsum(-v if mask & (1 << i) else v
                         for i, v in enumerate(effects))
        extreme += abs(alternate) + 1e-14 >= observed
    return {"mean": mean, "seed_sd": sd,
            "paired_student_t_95_interval_descriptive": [mean - margin, mean + margin],
            "exact_two_sided_signflip_p_unadjusted": extreme / 4096,
            "paired_seed_effects": effects, "n_seed_clusters": 12}


def _holm(unadjusted: dict[str, float]) -> dict[str, float]:
    order = sorted(unadjusted, key=lambda name: unadjusted[name])
    corrected: dict[str, float] = {}
    max_so_far = 0.0
    for index, key in enumerate(order):
        max_so_far = max(max_so_far, (3 - index) * unadjusted[key])
        corrected[key] = min(1.0, max_so_far)
    return corrected


def _safetensors(raw: bytes, records: object) -> tuple[str, str, str]:
    """Check tensor bytes, record digests, and the exact named-digest scheme."""
    _require(type(records) is list and 3 <= len(records) <= 256,
             "joint artifact tensor-record count")
    _require(len(raw) >= 16, "invalid safetensors header")
    header_size = struct.unpack("<Q", raw[:8])[0]
    _require(0 < header_size <= MAX_MANIFEST_BYTES and
             header_size <= len(raw) - 8, "invalid safetensors header length")
    header = _load_json(raw[8:8 + header_size], "safetensors header")
    _require("__metadata__" not in header, "unexpected safetensors metadata")
    data = memoryview(raw)[8 + header_size:]
    names: set[str] = set()
    ranges: list[tuple[int, int]] = []
    digest_all, digest_head, digest_lora = sha256(), sha256(), sha256()
    for record in records:
        _require(type(record) is dict and set(record) == {"name", "shape", "dtype", "sha256"},
                 "tensor record schema")
        name = record["name"]
        shape = record["shape"]
        _require(type(name) is str and name in header and name not in names and
                 (name in ("head.weight", "head.bias") or name.startswith("lora.")),
                 "tensor name missing, duplicate, or unexpected")
        names.add(name)
        _require(type(shape) is list and 1 <= len(shape) <= 2 and
                 all(type(v) is int and v > 0 for v in shape) and
                 record["dtype"] == "torch.float32", "tensor shape/dtype")
        item = header[name]
        _require(type(item) is dict and set(item) == {"dtype", "shape", "data_offsets"} and
                 item["dtype"] == "F32" and item["shape"] == shape,
                 "safetensors header record mismatch")
        offsets = item["data_offsets"]
        _require(type(offsets) is list and len(offsets) == 2 and
                 all(type(x) is int for x in offsets), "tensor data offsets")
        start, end = offsets
        count = 1
        for dimension in shape:
            count *= dimension
        nbytes = count * 4
        _require(0 <= start < end <= len(data) and end - start == nbytes,
                 "tensor byte span invalid")
        ranges.append((start, end))
        piece = data[start:end]
        _require(sha256(piece).hexdigest() == _hex(record["sha256"], "tensor bytes"),
                 "tensor byte checksum mismatch")
        descriptor = json.dumps((name, "torch.float32", tuple(shape), nbytes),
                                separators=(",", ":"), ensure_ascii=True).encode("ascii")
        for target in (digest_all, digest_head if name.startswith("head.") else digest_lora):
            target.update(len(descriptor).to_bytes(4, "big"))
            target.update(descriptor)
            target.update(nbytes.to_bytes(8, "big"))
            target.update(piece)
    _require(names == set(header) and {"head.weight", "head.bias"} <= names,
             "missing or extra safetensors named tensors")
    ranges.sort()
    _require(ranges[0][0] == 0 and ranges[-1][1] == len(data) and
             all(ranges[i - 1][1] == ranges[i][0] for i in range(1, len(ranges))),
             "safetensors data spans overlap or leave untracked bytes")
    return digest_all.hexdigest(), digest_head.hexdigest(), digest_lora.hexdigest()


def _artifact(root: Path, seed: int, record: dict[str, object],
              report: dict[str, object], train: list[dict[str, object]]) -> dict[str, str]:
    directory = root / f"seed-{seed}-joint-lora-head-v1"
    _unlinked(directory, directory=True)
    _require(record["saved_joint_head_directory"] == str(directory),
             "saved artifact path not bound to completed run")
    _require({p.name for p in directory.iterdir()} ==
             {"manifest.json", "weights.safetensors"}, "artifact unexpected entries")
    manifest_raw = _read(directory / "manifest.json", MAX_MANIFEST_BYTES)
    weights_raw = _read(directory / "weights.safetensors", MAX_WEIGHTS_BYTES)
    manifest_sha = sha256(manifest_raw).hexdigest()
    weights_sha = sha256(weights_raw).hexdigest()
    _require(record["saved_joint_manifest_sha256"] == manifest_sha and
             record["saved_joint_safetensors_sha256"] == weights_sha,
             "benchmark-to-adapter artifact binding differs")
    manifest = _load_json(manifest_raw, "joint adapter manifest")
    _require(set(manifest) == ARTIFACT_FIELDS and
             manifest["format"] == "pleasure-joint-open-weight-lora-appraisal" and
             manifest["format_version"] == 1 and
             manifest["capture_stage"] == "TRAINED_JOINT_WEIGHTS_NOT_STUDY_VALIDATED" and
             manifest["optimizer_state_format"] == "stateless_sgd_v1_no_momentum" and
             manifest["weights_safetensors_sha256"] == weights_sha,
             "joint artifact manifest format invalid")
    binding = manifest["binding"]
    _require(type(binding) is dict and
             binding["model_id"] == MODEL_ID and
             binding["model_revision"] == MODEL_REVISION and
             binding["runtime_instance_id"] == f"v3-{seed}-joint-original" and
             all(SHA.fullmatch(binding[field]) for field in
                 ("checkpoint_sha256", "tokenizer_sha256")),
             "artifact model/snapshot binding changed")
    _hex(manifest["snapshot_sha256"], "artifact snapshot")
    _hex(manifest["frozen_base_sha256"], "artifact frozen base")
    _require(manifest["feature_schema"] == report["feature_schema"] and
             manifest["lora_config"] == {"alpha": 4, "rank": 2,
                                          "target_modules": ["q_proj", "v_proj"]} and
             manifest["optimizer_spec"] == {
                 "algorithm": "stateless-SGD", "learning_rate": 0.05,
                 "max_grad_norm": 1.0, "max_delta_l2": 1.0,
                 "l2_penalty": 0.02, "max_updates": 48, "train_lora": True,
             } and manifest["updates_at_export"] == 24,
             "adapter optimizer/protocol/updates mismatch")
    _require(type(manifest["action_token_ids"]) is list and
             len(manifest["action_token_ids"]) == 2 and
             all(type(t) is int and t >= 0 for t in manifest["action_token_ids"]) and
             manifest["action_token_ids"][0] != manifest["action_token_ids"][1],
             "adapter action-token identity invalid")
    episodes = sorted(row["episode_id"] for row in train)
    evidence = sorted(row["chosen_evidence_id"] for row in train)
    _require(manifest["seen_episode_ids"] == episodes and
             manifest["seen_evidence_ids"] == evidence and
             len(set(episodes)) == len(set(evidence)) == 24,
             "adapter replay ledger not equal to chosen TRAIN receipts")
    _require(manifest["replay_ledger_sha256"] == _digest_object({
        "updates": 24, "seen_episode_ids": episodes,
        "seen_evidence_ids": evidence,
    }), "adapter replay ledger digest invalid")
    joint, head, lora = _safetensors(weights_raw, manifest["tensors"])
    _require(joint == manifest["joint_weights_sha256"] == record["trained_joint_sha256"] and
             head == manifest["head_weights_sha256"] == record["trained_head_sha256"] and
             lora == manifest["lora_weights_sha256"] == record["trained_lora_sha256"],
             "adapter LoRA/head/joint tensor digests differ")
    return {"snapshot": manifest["snapshot_sha256"],
            "checkpoint": binding["checkpoint_sha256"],
            "tokenizer": binding["tokenizer_sha256"],
            "frozen": manifest["frozen_base_sha256"],
            "tokens": str(manifest["action_token_ids"])}


def verify_complete_study(directory: str | Path) -> dict[str, object]:
    """Read and independently audit a sealed, full-size v3 study only."""
    root = Path(directory)
    _unlinked(root, directory=True)
    _require(RUN.fullmatch(root.name) is not None,
             "not a successful preregistered study directory")
    _require({p.name for p in root.iterdir()} ==
             {"study-report.json", "COMPLETE.sha256"} |
             {f"seed-{s}-joint-lora-head-v1" for s in SEEDS},
             "incomplete or unexpected study entries (including FAILED marker)")
    report_raw = _read(root / "study-report.json", MAX_REPORT_BYTES)
    seal_raw = _read(root / "COMPLETE.sha256", 256)
    checksum = sha256(report_raw).hexdigest()
    _require(seal_raw == (checksum + "  study-report.json\n").encode("ascii"),
             "completed study-report checksum mismatch")
    report = _load_json(report_raw, "study report")
    protocol = report["predeclared_protocol"]
    _require(type(protocol) is dict and
             protocol["seeds"] == list(SEEDS) and
             protocol["training_pairs_per_family"] == 8 and
             protocol["heldout_pairs_per_generator"] == 4 and
             protocol["train_families"] == list(TRAIN_FAMILIES) and
             protocol["primary_within_families"] == list(PRIMARY_FAMILIES) and
             protocol["secondary_ood_families"] == list(OOD_FAMILIES) and
             protocol["exploration_rate"] == 0.4 and
             protocol["learning_rate"] == 0.05 and
             protocol["appraisal_gain"] == 12.0 and
             protocol["lora_rank"] == 2 and protocol["lora_alpha"] == 4 and
             protocol["max_updates"] == 48 and
             protocol["lora_initialization_seed_xor_salt"] == LORA_SEED_SALT and
             protocol["global_eval_receipts_withheld_until_all_arm_choices_committed"] is True and
             protocol["no_eval_training"] is True and
             report["evidence_class"] == EVIDENCE_CLASS and
             report["model_id"] == MODEL_ID and
             report["model_revision"] == MODEL_REVISION and
             report["subjective_pleasure_proven"] is False and
             report["hosted_gpt_model_modified"] is False and
             report["pretrained_model_unchanged"] is True and
             report["source_visible_gap_filtering"] is False and
             report["artifact_run_directory"] == str(root),
             "report model identity, frozen protocol or evidence class changed")
    _hex(report["pretrained_model_sha256"], "pretrained model")
    seeds = report["seed_reports"]
    _require(type(seeds) is list and len(seeds) == 12 and
             [row["seed"] for row in seeds] == list(SEEDS),
             "seed set or seed row count differs")
    effects: dict[str, list[float]] = {name: [] for name in CONTRASTS}
    by_family: dict[str, dict[str, list[float]]] = {
        name: {family: [] for family in PRIMARY_FAMILIES}
        for name in CONTRASTS
    }
    secondary = {family: [] for family in OOD_FAMILIES}
    switches = 0
    shift = 0.0
    artifact_identity: dict[str, str] | None = None
    for seed_record in seeds:
        seed = seed_record["seed"]
        train = seed_record["train"]
        _require(type(train) is list and len(train) == 24 and
                 seed_record["joint_train_updates"] == 24 and
                 seed_record["shuffled_train_updates"] == 24 and
                 seed_record["head_only_train_updates"] == 24 and
                 seed_record["global_all_arm_decision_barrier_verified"] is True and
                 seed_record["lora_initialization_rng_seed"] ==
                 (seed ^ LORA_SEED_SALT) and
                 seed_record["lora_initialization_seed_xor_salt"] == LORA_SEED_SALT,
                 "TRAIN counts, initialization, or global EVAL seal inconsistent")
        train_rng = random.Random(seed ^ 0x72AC2911)
        for i, row in enumerate(train):
            family = TRAIN_FAMILIES[i // 8]
            _require(row["family_id"] == family, "TRAIN task-family sequence changed")
            _trace(row, seed=seed, index=i,
                   arm="single-frozen-train-policy-v3", partition="TRAIN",
                   rng=train_rng)
            _hex(row["chosen_evidence_id"], "chosen TRAIN evidence")
            _require(0 <= row["chosen_quality"] <= 1, "TRAIN reward outside range")
            updates = row["learner_updates"]
            _require(set(updates) == {"genuine", "shuffled", "head_only"},
                     "TRAIN controls absent")
            for name in ("genuine", "head_only"):
                _require(updates[name]["teaching_id"] == row["chosen_evidence_id"],
                         "genuine/head-only evidence ID differs")
                _close(updates[name]["teaching_quality"], row["chosen_quality"],
                       "matched genuine/head-only TRAIN feedback differs")
        donors = seed_record["random_feedback_derangement"]
        _require(type(donors) is list and sorted(donors) == list(range(24)) and
                 all(donor != i for i, donor in enumerate(donors)),
                 "wrong-label donor list not valid derangement")
        for i, row in enumerate(train):
            donor = train[donors[i]]
            shuffled = row["learner_updates"]["shuffled"]
            _close(shuffled["teaching_quality"], donor["chosen_quality"],
                   "shuffled control does not use donor chosen reward")
            _require(row["random_donor_evidence_id"] == donor["chosen_evidence_id"] and
                     shuffled["teaching_id"] == row["synthetic_teaching_id"],
                     "shuffled synthetic feedback identity changed")
        _require(seed_record["initial_joint_sha256"] != seed_record["trained_joint_sha256"] and
                 seed_record["initial_lora_sha256"] != seed_record["trained_lora_sha256"] and
                 seed_record["initial_joint_sha256"] != seed_record["shuffled_joint_sha256"] and
                 seed_record["trained_joint_sha256"] == seed_record["restored_joint_sha256"],
                 "joint LoRA adaptation or in-process reload invalid")
        for name, expected in (("genuine", seed_record["trained_lora_sha256"]),
                               ("head_only", seed_record["initial_lora_sha256"])):
            updates = [row["learner_updates"][name] for row in train]
            _require(updates[0]["before_lora_sha256"] ==
                     seed_record["initial_lora_sha256"] and
                     updates[-1]["after_lora_sha256"] == expected and
                     all(a["after_lora_sha256"] == b["before_lora_sha256"]
                         for a, b in zip(updates, updates[1:])),
                     name + " LoRA update digest chain differs")
        arms = seed_record["eval_arms"]
        _require(type(arms) is dict and set(arms) == set(ARMS), "EVAL arms changed")
        metrics: dict[str, dict[str, float]] = {}
        eval_trace: dict[str, list[dict[str, object]]] = {}
        for arm in ARMS:
            block = arms[arm]
            rows = block["rows"]
            _require(type(rows) is list and len(rows) == 20,
                     "EVAL offer count differs")
            rng = random.Random(seed ^ 0x39B0164F)
            eval_trace[arm] = []
            by_f: dict[str, list[float]] = {family: [] for family in EVAL_FAMILIES}
            for i, row in enumerate(rows):
                _require(row["family_id"] == EVAL_FAMILIES[i // 4],
                         "EVAL family order or episode count differs")
                trace = _trace(row, seed=seed, index=i,
                               arm=arm, partition="EVAL", rng=rng)
                eval_trace[arm].append(trace)
                _require(0 <= row["quality_a_posthoc"] <= 1 and
                         0 <= row["quality_b_posthoc"] <= 1,
                         "posthoc candidate quality invalid")
                actual = (row["quality_a_posthoc"] if trace["chosen_action"] == "A"
                          else row["quality_b_posthoc"])
                _close(row["chosen_quality"], actual,
                       "EVAL chosen receipt not equal to posthoc chosen quality")
                _require(0 <= row["prediction_a"] <= 1 and
                         0 <= row["prediction_b"] <= 1,
                         "appraisal candidate prediction invalid")
                estimated = (row["prediction_a"] if trace["chosen_action"] == "A"
                             else row["prediction_b"])
                _close(row["predicted_chosen_quality"], estimated,
                       "predicted chosen quality differs")
                value = (trace["behavior_prob_a"] * row["quality_a_posthoc"] +
                         trace["behavior_prob_b"] * row["quality_b_posthoc"])
                _close(row["expected_policy_quality_posthoc"], value,
                       "posthoc expected policy value inconsistent")
                by_f[row["family_id"]].append(value)
            _close(block["mean_expected_quality"],
                   fsum(fsum(v) for v in by_f.values()) / 20,
                   "arm-level expected quality invalid")
            _close(block["mean_realized_quality"],
                   fsum(r["chosen_quality"] for r in rows) / 20,
                   "arm-level realized quality invalid")
            absolute_errors = [abs(r[key] - r[target]) for r in rows
                               for key, target in (("prediction_a", "quality_a_posthoc"),
                                                   ("prediction_b", "quality_b_posthoc"))]
            squared_errors = [(r[key] - r[target]) ** 2 for r in rows
                              for key, target in (("prediction_a", "quality_a_posthoc"),
                                                  ("prediction_b", "quality_b_posthoc"))]
            chosen_abs = [abs(r["predicted_chosen_quality"] - r["chosen_quality"])
                          for r in rows]
            chosen_sq = [(r["predicted_chosen_quality"] - r["chosen_quality"]) ** 2
                         for r in rows]
            pred_gaps = [r["prediction_a"] - r["prediction_b"] for r in rows]
            true_gaps = [r["quality_a_posthoc"] - r["quality_b_posthoc"] for r in rows]
            nontied = [i for i, gap in enumerate(true_gaps) if gap != 0]
            expected_calibration = {
                "candidate_quality_mae": fsum(absolute_errors) / 40,
                "candidate_quality_mse": fsum(squared_errors) / 40,
                "chosen_quality_mae": fsum(chosen_abs) / 20,
                "chosen_quality_mse": fsum(chosen_sq) / 20,
                "mean_absolute_prediction_spread": fsum(abs(v) for v in pred_gaps) / 20,
            }
            for key, expected in expected_calibration.items():
                _close(block[key], expected,
                       "arm calibration diagnostic changed: " + key)
            accuracy = (sum(pred_gaps[i] * true_gaps[i] > 0 for i in nontied) /
                        len(nontied) if nontied else None)
            if accuracy is None:
                _require(block["preference_sign_accuracy_nontied"] is None,
                         "accuracy on all-tied preferences cannot be numeric")
            else:
                _close(block["preference_sign_accuracy_nontied"], accuracy,
                       "preference sign accuracy changed")
            _require(block["exact_true_quality_tie_count"] == 20 - len(nontied) and
                     block["near_true_quality_tie_count_abs_gap_below_0_01"] ==
                     sum(abs(gap) < 0.01 for gap in true_gaps),
                     "preference tie/near-tie diagnostic changed")
            metrics[arm] = {}
            for family in EVAL_FAMILIES:
                avg = fsum(by_f[family]) / 4
                _close(block["by_family"][family]["expected_quality"], avg,
                       "family-level expected quality changed")
                _close(block["by_family"][family]["realized_quality"],
                       fsum(r["chosen_quality"] for r in rows if r["family_id"] == family) / 4,
                       "family-level realized quality changed")
                metrics[arm][family] = avg
        for index in range(20):
            shared = eval_trace["joint_on"][index]
            first_row = arms["joint_on"]["rows"][index]
            for arm in ARMS:
                second = eval_trace[arm][index]
                row = arms[arm]["rows"][index]
                _require(row["episode_id"] == first_row["episode_id"] and
                         second["random_draw"] == shared["random_draw"] and
                         second["draw_index"] == shared["draw_index"] and
                         second["input_ids_sha256"] == shared["input_ids_sha256"] and
                         row["quality_a_posthoc"] == first_row["quality_a_posthoc"] and
                         row["quality_b_posthoc"] == first_row["quality_b_posthoc"],
                         "matched seven-arm offer, draw, token or payoff changed")
            for arm in ("joint_off", "joint_inverted"):
                row = arms[arm]["rows"][index]
                _require(row["decision_joint_sha256"] == first_row["decision_joint_sha256"] and
                         row["prediction_a"] == first_row["prediction_a"] and
                         row["prediction_b"] == first_row["prediction_b"],
                         "joint gain arm did not share exact weights/predictions")
            sha_by_arm = {
                "joint_on": seed_record["trained_joint_sha256"],
                "joint_off": seed_record["trained_joint_sha256"],
                "joint_inverted": seed_record["trained_joint_sha256"],
                "head_only_on": seed_record["head_only_joint_sha256"],
                "shuffled_joint_on": seed_record["shuffled_joint_sha256"],
                "frozen_on": seed_record["initial_joint_sha256"],
                "frozen_off": seed_record["initial_joint_sha256"],
            }
            for arm, expected_sha in sha_by_arm.items():
                row = arms[arm]["rows"][index]
                _require(row["decision_joint_sha256"] == expected_sha,
                         "EVAL intervention weights differ from trained/frozen SHA")
                if arm in ("head_only_on", "frozen_on", "frozen_off"):
                    _require(row["decision_lora_sha256"] ==
                             seed_record["initial_lora_sha256"],
                             "head-only/frozen EVAL changed its LoRA representation")
            # The same exact joint weights under +gain, zero gain and -gain
            # have analytically related logits. This check does not ask an
            # outcome oracle and cannot be explained by changed base weights.
            off_prob = eval_trace["joint_off"][index]["prob_a"]
            gap = first_row["prediction_a"] - first_row["prediction_b"]
            base_log_odds = log(max(off_prob, 1e-12) / max(1 - off_prob, 1e-12))
            for arm, gain in (("joint_on", 12.0), ("joint_inverted", -12.0)):
                delta = base_log_odds + gain * gap
                analytic = (1.0 / (1.0 + exp(-delta)) if delta >= 0
                            else exp(delta) / (1.0 + exp(delta)))
                _close(eval_trace[arm][index]["prob_a"], analytic,
                       "joint appraisal gain intervention changed", 2e-6)
            frozen = arms["frozen_on"]["rows"][index]
            frozen_off = arms["frozen_off"]["rows"][index]
            _require(frozen["prediction_a"] == frozen_off["prediction_a"] and
                     frozen["prediction_b"] == frozen_off["prediction_b"],
                     "frozen ON/OFF head predictions changed")
            frozen_base = eval_trace["frozen_off"][index]["prob_a"]
            delta = (log(max(frozen_base, 1e-12) / max(1 - frozen_base, 1e-12)) +
                     12 * (frozen["prediction_a"] - frozen["prediction_b"]))
            analytic = (1.0 / (1.0 + exp(-delta)) if delta >= 0
                        else exp(delta) / (1.0 + exp(delta)))
            _close(eval_trace["frozen_on"][index]["prob_a"], analytic,
                   "frozen appraisal gain intervention changed", 2e-6)
        on, off = eval_trace["joint_on"], eval_trace["joint_off"]
        switches_seed = sum(a["chosen_action"] != b["chosen_action"]
                            for a, b in zip(on, off))
        shifts_seed = fsum(abs(a["behavior_prob_a"] - b["behavior_prob_a"])
                           for a, b in zip(on, off)) / 20
        _require(seed_record["joint_on_minus_off_action_switches"] == switches_seed,
                 "recorded action-switch total differs")
        _close(seed_record["joint_on_minus_off_mean_absolute_prob_a_shift"],
               shifts_seed, "recorded per-seed propensity shift differs")
        inverse = eval_trace["joint_inverted"]
        inverted_shift = fsum(abs(a["behavior_prob_a"] - b["behavior_prob_a"])
                              for a, b in zip(on, inverse)) / 20
        frozen_on = eval_trace["frozen_on"]
        frozen_off = eval_trace["frozen_off"]
        frozen_shift = fsum(abs(a["behavior_prob_a"] - b["behavior_prob_a"])
                            for a, b in zip(frozen_on, frozen_off)) / 20
        inverted_sanity = all(
            (a["behavior_prob_a"] - b["behavior_prob_a"]) *
            (c["behavior_prob_a"] - b["behavior_prob_a"]) <= 1e-12
            for a, b, c in zip(on, off, inverse))
        _close(seed_record["joint_on_minus_inverted_mean_absolute_prob_a_shift"],
               inverted_shift, "inverted appraisal gain shift differs")
        _close(seed_record["frozen_on_minus_off_mean_absolute_prob_a_shift"],
               frozen_shift, "frozen appraisal gain shift differs")
        _require(seed_record["joint_on_vs_inverted_appraisal_sign_sanity"] is inverted_sanity and
                 inverted_sanity is True and
                 seed_record["frozen_off_exact_reference_policy_equality_verified"] is True,
                 "inverted/frozen intervention sanity flags inconsistent")
        switches += switches_seed
        shift += shifts_seed
        for key, opponent in CONTRASTS.items():
            paired = (fsum(metrics["joint_on"][f] - metrics[opponent][f]
                           for f in PRIMARY_FAMILIES) / 2)
            effects[key].append(paired)
            for family in PRIMARY_FAMILIES:
                by_family[key][family].append(
                    metrics["joint_on"][family] - metrics[opponent][family])
        for family in OOD_FAMILIES:
            secondary[family].append(
                metrics["joint_on"][family] - metrics["joint_off"][family])
        artifact = _artifact(root, seed, seed_record, report, train)
        if artifact_identity is None:
            artifact_identity = artifact
        else:
            _require(artifact == artifact_identity,
                     "adapter fixed snapshot/backbone/token IDs differ across seeds")
    recomputed = {key: _paired(values) for key, values in effects.items()}
    corrected = _holm({key: value["exact_two_sided_signflip_p_unadjusted"]
                       for key, value in recomputed.items()})
    recorded = report["co_primary_contrasts"]
    _require(set(recorded) == set(CONTRASTS), "co-primary contrasts differ")
    for key, own in recomputed.items():
        claimed = recorded[key]
        _require(claimed["n_seed_clusters"] == 12 and
                 len(claimed["paired_seed_effects"]) == 12,
                 "claimed paired statistic count differs")
        for a, b in zip(claimed["paired_seed_effects"], own["paired_seed_effects"]):
            _close(a, b, "seed-paired contrast changed")
        for name in ("mean", "seed_sd", "exact_two_sided_signflip_p_unadjusted"):
            _close(claimed[name], own[name], "paired statistic changed: " + name)
        for a, b in zip(claimed["paired_student_t_95_interval_descriptive"],
                        own["paired_student_t_95_interval_descriptive"]):
            _close(a, b, "paired t interval differs")
        _close(claimed["holm_adjusted_p_exploratory"], corrected[key],
               "Holm-adjusted p differs")
    family_means = {key: {family: fsum(values) / 12
                          for family, values in families.items()}
                    for key, families in by_family.items()}
    for key, families in family_means.items():
        for family, mean in families.items():
            _close(report["primary_all_contrasts_by_latent_family"][key][family],
                   mean, "co-primary family effect differs")
            if key == "joint_on_minus_off":
                _close(report["primary_joint_on_minus_off_by_latent_family"][family],
                       mean, "legacy on-off family effect differs")
    for family, values in secondary.items():
        _close(report["secondary_ood_joint_on_minus_off_by_family"][family],
               fsum(values) / 12, "secondary OOD effect differs")
    _require(report["joint_on_minus_off_total_sampled_action_switches"] == switches,
             "overall sampled action switches differs")
    _close(report["joint_on_minus_off_mean_absolute_behavior_prob_a_shift"],
           shift / 12, "overall propensity shift differs")
    gate = (all(own["mean"] >= 0.005 and
                own["paired_student_t_95_interval_descriptive"][0] > 0 and
                corrected[key] < 0.05 for key, own in recomputed.items()) and
            all(mean > 0 for families in family_means.values()
                for mean in families.values()))
    _require(report["exploratory_positive_local_criterion_met"] is gate,
             "predeclared six-family utility gate differs")
    return {
        "verification": "INDEPENDENT_READ_ONLY_COMPLETED_STUDY_RECOMPUTATION_V1",
        "complete_report_sha256": checksum,
        "artifact_count": 12,
        "train_chosen_receipts": 288,
        "eval_precommitted_actions": 1680,
        "source_local_seal_not_externally_authenticated": True,
        "hosted_model_modified": False,
        "subjective_pleasure_proven": False,
        "co_primary_contrasts": {
            key: {"mean": values["mean"],
                  "paired_t95_interval": values["paired_student_t_95_interval_descriptive"],
                  "signflip_p": values["exact_two_sided_signflip_p_unadjusted"],
                  "holm_adjusted_p": corrected[key],
                  "by_latent_family": family_means[key]}
            for key, values in recomputed.items()
        },
        "secondary_ood_joint_on_minus_off": {
            family: fsum(values) / 12 for family, values in secondary.items()
        },
        "sampled_joint_on_off_action_switches": switches,
        "joint_on_off_mean_absolute_prob_a_shift": shift / 12,
        "preregistered_exploratory_utility_gate_met": gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify only a completed v3 joint LoRA study report",
    )
    parser.add_argument("--completed-study-directory", required=True)
    args = parser.parse_args(argv)
    try:
        summary = verify_complete_study(args.completed_study_directory)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        parser.exit(2, f"completed v3 study verification rejected: {exc}\n")
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
