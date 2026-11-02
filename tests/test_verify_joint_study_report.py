"""Strict independent v3 report verifier: complete synthetic fixtures only.

All fixtures are fabricated in memory. This test never reads any real,
pending, completed, or withheld payoff study artifact.
"""

from __future__ import annotations

from hashlib import sha256
import json
from math import exp, fsum
from pathlib import Path
import random
import struct
from tempfile import TemporaryDirectory
import unittest

from scripts import verify_joint_study_report as verify


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value):
    return sha256(str(value).encode("ascii")).hexdigest()


def _choice(seed, index, partition, episode, arm, rng, prob_a):
    raw_a = float(prob_a)
    raw_b = 1.0 - raw_a
    mixed = 0.6 * raw_a + 0.2
    draw = rng.random()
    action = "A" if draw < mixed else "B"
    propensity = mixed if action == "A" else 1.0 - mixed
    token_sha = _sha(f"tokens-{episode}")
    payload = ("PLEASURE_STOCHASTIC_CHOICE_V1", seed, index, partition,
               episode, token_sha, arm, raw_a.hex(), raw_b.hex(), (0.4).hex(),
               mixed.hex(), (1.0 - mixed).hex(), draw.hex(), action,
               propensity.hex())
    digest = sha256(json.dumps(payload, separators=(",", ":"),
                               ensure_ascii=True).encode("ascii")).hexdigest()
    return {
        "seed": seed, "draw_index": index, "partition": partition,
        "episode_id": episode, "input_ids_sha256": token_sha,
        "policy_id": arm, "prob_a": raw_a, "prob_b": raw_b,
        "exploration_rate": 0.4, "behavior_prob_a": mixed,
        "behavior_prob_b": 1.0 - mixed, "random_draw": draw,
        "chosen_action": action, "chosen_propensity": propensity,
        "decision_digest": digest,
    }


def _data_artifact(root, seed, train, feature_schema):
    directory = root / f"seed-{seed}-joint-lora-head-v1"
    directory.mkdir()
    tensors = (
        ("head.weight", [1, 4], struct.pack("<4f", 0.2, 0.1, -0.1, 0.05)),
        ("head.bias", [1], struct.pack("<f", 0.02)),
        ("lora.model.q_proj.lora_A.default.weight", [2, 2],
         struct.pack("<4f", 0.01, 0.02, -0.03, 0.04)),
    )
    header = {}
    records = []
    data = b""
    digests = {"joint": sha256(), "head": sha256(), "lora": sha256()}
    for name, shape, body in tensors:
        offset = len(data)
        data += body
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [offset, len(data)]}
        records.append({"name": name, "shape": shape,
                        "dtype": "torch.float32", "sha256": sha256(body).hexdigest()})
        descriptor = json.dumps((name, "torch.float32", tuple(shape), len(body)),
                                separators=(",", ":"), ensure_ascii=True).encode("ascii")
        for key in ("joint", "head" if name.startswith("head.") else "lora"):
            target = digests[key]
            target.update(len(descriptor).to_bytes(4, "big"))
            target.update(descriptor)
            target.update(len(body).to_bytes(8, "big"))
            target.update(body)
    header_bytes = _json(header)
    safetensors = struct.pack("<Q", len(header_bytes)) + header_bytes + data
    (directory / "weights.safetensors").write_bytes(safetensors)
    evidence = sorted(row["chosen_evidence_id"] for row in train)
    episodes = sorted(row["episode_id"] for row in train)
    payload = {
        "format": "pleasure-joint-open-weight-lora-appraisal",
        "format_version": 1,
        "capture_stage": "TRAINED_JOINT_WEIGHTS_NOT_STUDY_VALIDATED",
        "optimizer_state_format": "stateless_sgd_v1_no_momentum",
        "binding": {"model_id": verify.MODEL_ID,
                    "model_revision": verify.MODEL_REVISION,
                    "runtime_instance_id": f"v3-{seed}-joint-original",
                    "checkpoint_sha256": "a" * 64,
                    "tokenizer_sha256": "b" * 64},
        "snapshot_sha256": "c" * 64,
        "frozen_base_sha256": "d" * 64,
        "action_token_ids": [12, 13], "hidden_size": 4,
        "feature_schema": feature_schema,
        "lora_config": {"alpha": 4, "rank": 2,
                        "target_modules": ["q_proj", "v_proj"]},
        "optimizer_spec": {"algorithm": "stateless-SGD",
                           "learning_rate": 0.05, "max_grad_norm": 1.0,
                           "max_delta_l2": 1.0, "l2_penalty": 0.02,
                           "max_updates": 48, "train_lora": True},
        "updates_at_export": 24,
        "seen_evidence_ids": evidence,
        "seen_episode_ids": episodes,
        "replay_ledger_sha256": verify._digest_object({
            "updates": 24, "seen_evidence_ids": evidence,
            "seen_episode_ids": episodes,
        }),
        "joint_weights_sha256": digests["joint"].hexdigest(),
        "head_weights_sha256": digests["head"].hexdigest(),
        "lora_weights_sha256": digests["lora"].hexdigest(),
        "tensors": records,
        "weights_safetensors_sha256": sha256(safetensors).hexdigest(),
    }
    manifest_raw = _json(payload) + b"\n"
    (directory / "manifest.json").write_bytes(manifest_raw)
    return directory, payload, sha256(manifest_raw).hexdigest(), sha256(safetensors).hexdigest()


def _seed(root, seed):
    train = []
    train_rng = random.Random(seed ^ 0x72AC2911)
    for i in range(24):
        episode = f"train-{seed}-{i}"
        chosen_evidence = _sha(f"train-receipt-{seed}-{i}")
        chosen_quality = (1 + (i * 7) % 19) / 21
        trace = _choice(seed ^ 0x72AC2911, i, "TRAIN", episode,
                        "single-frozen-train-policy-v3", train_rng, 0.45)
        train.append({"episode_id": episode, "family_id": verify.TRAIN_FAMILIES[i // 8],
                      "sampled_pre_feedback": trace,
                      "chosen_evidence_id": chosen_evidence,
                      "chosen_quality": chosen_quality,
                      "learner_updates": {
                          name: {"teaching_id": chosen_evidence,
                                 "teaching_quality": chosen_quality}
                          for name in ("genuine", "head_only")
                      }})
    donors = [(i + 1) % 24 for i in range(24)]
    for i, row in enumerate(train):
        donor = train[donors[i]]
        evidence = _sha(f"shuffled-{seed}-{i}")
        row["random_donor_evidence_id"] = donor["chosen_evidence_id"]
        row["synthetic_teaching_id"] = evidence
        row["learner_updates"]["shuffled"] = {
            "teaching_id": evidence, "teaching_quality": donor["chosen_quality"]}

    trained_marker = _sha(f"trained-marker-{seed}")
    # The artifact SHA is produced below. Fill the final genuine digest once
    # those exact tensor bytes are known; intermediate digests form a chain.
    for i, row in enumerate(train):
        genuine = row["learner_updates"]["genuine"]
        genuine["before_lora_sha256"] = (
            "1" * 64 if i == 0 else _sha(f"genuine-stage-{seed}-{i - 1}"))
        genuine["after_lora_sha256"] = (
            trained_marker if i == 23 else _sha(f"genuine-stage-{seed}-{i}"))
        headonly = row["learner_updates"]["head_only"]
        headonly["before_lora_sha256"] = "1" * 64
        headonly["after_lora_sha256"] = "1" * 64

    artifact_dir, artifact, manifest_sha, safetensors_sha = _data_artifact(
        root, seed, train, "fixture-feature-v1")
    train[-1]["learner_updates"]["genuine"]["after_lora_sha256"] = artifact["lora_weights_sha256"]
    joint_on_prob = 1.0 / (1.0 + exp(-12 * 0.3))
    probs = {"joint_on": joint_on_prob, "joint_off": 0.50,
             "joint_inverted": 1 - joint_on_prob, "head_only_on": 0.49,
             "shuffled_joint_on": 0.48, "frozen_on": 0.51,
             "frozen_off": 0.50}
    probs["frozen_on"] = 0.50
    eval_arms = {}
    for arm in verify.ARMS:
        rng = random.Random(seed ^ 0x39B0164F)
        rows = []
        for i in range(20):
            family = verify.EVAL_FAMILIES[i // 4]
            episode = f"eval-{seed}-{i}"
            trace = _choice(seed ^ 0x39B0164F, i, "EVAL", episode,
                            arm, rng, probs[arm])
            qa, qb = ((0.9, 0.1) if family in verify.PRIMARY_FAMILIES
                      else (0.2 + (i % 4) * .02, 0.8 - (i % 4) * .02))
            expected = trace["behavior_prob_a"] * qa + trace["behavior_prob_b"] * qb
            chosen = qa if trace["chosen_action"] == "A" else qb
            pred_a = 0.7 if arm in ("joint_on", "joint_off", "joint_inverted") else 0.5
            pred_b = 0.4 if arm in ("joint_on", "joint_off", "joint_inverted") else 0.5
            predicted = pred_a if trace["chosen_action"] == "A" else pred_b
            sha = (artifact["joint_weights_sha256"] if arm in (
                "joint_on", "joint_off", "joint_inverted") else
                "0" * 64 if arm in ("frozen_on", "frozen_off") else
                "3" * 64 if arm == "head_only_on" else "2" * 64)
            lora_sha = (artifact["lora_weights_sha256"] if arm in (
                "joint_on", "joint_off", "joint_inverted") else "1" * 64)
            rows.append({"episode_id": episode, "family_id": family,
                         "sampled_pre_feedback": trace,
                         "prediction_a": pred_a, "prediction_b": pred_b,
                         "predicted_chosen_quality": predicted,
                         "chosen_quality": chosen,
                         "quality_a_posthoc": qa, "quality_b_posthoc": qb,
                         "expected_policy_quality_posthoc": expected,
                         "decision_joint_sha256": sha,
                         "decision_head_sha256": _sha(arm + "head"),
                         "decision_lora_sha256": lora_sha})
        families = {}
        for family in verify.EVAL_FAMILIES:
            relevant = [row for row in rows if row["family_id"] == family]
            families[family] = {
                "expected_quality": fsum(
                    row["expected_policy_quality_posthoc"] for row in relevant) / 4,
                "realized_quality": fsum(row["chosen_quality"] for row in relevant) / 4,
            }
        eval_arms[arm] = {
            "rows": rows, "by_family": families,
            "mean_expected_quality": fsum(row["expected_policy_quality_posthoc"]
                                           for row in rows) / 20,
            "mean_realized_quality": fsum(row["chosen_quality"] for row in rows) / 20,
        }
        gaps = [r["quality_a_posthoc"] - r["quality_b_posthoc"] for r in rows]
        pred_gaps = [r["prediction_a"] - r["prediction_b"] for r in rows]
        indices = [i for i, gap in enumerate(gaps) if gap != 0]
        eval_arms[arm].update({
            "candidate_quality_mae": fsum(abs(r[key] - r[target]) for r in rows
                                           for key, target in (("prediction_a", "quality_a_posthoc"),
                                                               ("prediction_b", "quality_b_posthoc"))) / 40,
            "candidate_quality_mse": fsum((r[key] - r[target]) ** 2 for r in rows
                                           for key, target in (("prediction_a", "quality_a_posthoc"),
                                                               ("prediction_b", "quality_b_posthoc"))) / 40,
            "chosen_quality_mae": fsum(abs(r["predicted_chosen_quality"] - r["chosen_quality"])
                                        for r in rows) / 20,
            "chosen_quality_mse": fsum((r["predicted_chosen_quality"] - r["chosen_quality"]) ** 2
                                        for r in rows) / 20,
            "mean_absolute_prediction_spread": fsum(abs(v) for v in pred_gaps) / 20,
            "preference_sign_accuracy_nontied": (
                sum(pred_gaps[i] * gaps[i] > 0 for i in indices) / len(indices)
                if indices else None),
            "exact_true_quality_tie_count": 20 - len(indices),
            "near_true_quality_tie_count_abs_gap_below_0_01": sum(
                abs(v) < 0.01 for v in gaps),
        })
    on = eval_arms["joint_on"]["rows"]
    off = eval_arms["joint_off"]["rows"]
    switches = sum(a["sampled_pre_feedback"]["chosen_action"] !=
                   b["sampled_pre_feedback"]["chosen_action"]
                   for a, b in zip(on, off))
    shift = fsum(abs(a["sampled_pre_feedback"]["behavior_prob_a"] -
                     b["sampled_pre_feedback"]["behavior_prob_a"])
                 for a, b in zip(on, off)) / 20
    inverse = eval_arms["joint_inverted"]["rows"]
    frozen_on = eval_arms["frozen_on"]["rows"]
    frozen_off = eval_arms["frozen_off"]["rows"]
    inv_shift = fsum(abs(a["sampled_pre_feedback"]["behavior_prob_a"] -
                         b["sampled_pre_feedback"]["behavior_prob_a"])
                     for a, b in zip(on, inverse)) / 20
    frozen_shift = fsum(abs(a["sampled_pre_feedback"]["behavior_prob_a"] -
                            b["sampled_pre_feedback"]["behavior_prob_a"])
                        for a, b in zip(frozen_on, frozen_off)) / 20
    return {
        "seed": seed, "lora_initialization_rng_seed": seed ^ verify.LORA_SEED_SALT,
        "lora_initialization_seed_xor_salt": verify.LORA_SEED_SALT,
        "joint_train_updates": 24, "shuffled_train_updates": 24,
        "head_only_train_updates": 24,
        "global_all_arm_decision_barrier_verified": True,
        "initial_joint_sha256": "0" * 64, "initial_lora_sha256": "1" * 64,
        "trained_joint_sha256": artifact["joint_weights_sha256"],
        "trained_lora_sha256": artifact["lora_weights_sha256"],
        "trained_head_sha256": artifact["head_weights_sha256"],
        "restored_joint_sha256": artifact["joint_weights_sha256"],
        "shuffled_joint_sha256": "2" * 64,
        "head_only_joint_sha256": "3" * 64,
        "saved_joint_head_directory": str(artifact_dir),
        "saved_joint_manifest_sha256": manifest_sha,
        "saved_joint_safetensors_sha256": safetensors_sha,
        "random_feedback_derangement": donors, "train": train,
        "eval_arms": eval_arms,
        "joint_on_minus_off_action_switches": switches,
        "joint_on_minus_off_mean_absolute_prob_a_shift": shift,
        "joint_on_minus_inverted_mean_absolute_prob_a_shift": inv_shift,
        "frozen_on_minus_off_mean_absolute_prob_a_shift": frozen_shift,
        "joint_on_vs_inverted_appraisal_sign_sanity": True,
        "frozen_off_exact_reference_policy_equality_verified": True,
    }


def _fixture(root):
    seed_rows = [_seed(root, seed) for seed in verify.SEEDS]
    effects = {name: [] for name in verify.CONTRASTS}
    families = {name: {family: [] for family in verify.PRIMARY_FAMILIES}
                for name in verify.CONTRASTS}
    ood = {family: [] for family in verify.OOD_FAMILIES}
    for seed_record in seed_rows:
        arms = seed_record["eval_arms"]
        for name, opponent in verify.CONTRASTS.items():
            per_f = [arms["joint_on"]["by_family"][family]["expected_quality"] -
                     arms[opponent]["by_family"][family]["expected_quality"]
                     for family in verify.PRIMARY_FAMILIES]
            effects[name].append(fsum(per_f) / 2)
            for family, value in zip(verify.PRIMARY_FAMILIES, per_f):
                families[name][family].append(value)
        for family in verify.OOD_FAMILIES:
            ood[family].append(
                arms["joint_on"]["by_family"][family]["expected_quality"] -
                arms["joint_off"]["by_family"][family]["expected_quality"])
    contrasts = {name: verify._paired(vals) for name, vals in effects.items()}
    holm = verify._holm({name: value["exact_two_sided_signflip_p_unadjusted"]
                         for name, value in contrasts.items()})
    for name, corrected in holm.items():
        contrasts[name]["holm_adjusted_p_exploratory"] = corrected
    primary_means = {name: {family: fsum(vals) / 12 for family, vals in fams.items()}
                     for name, fams in families.items()}
    report = {
        "evidence_class": verify.EVIDENCE_CLASS,
        "pretrained_model_sha256": "e" * 64,
        "pretrained_model_unchanged": True,
        "model_id": verify.MODEL_ID,
        "model_revision": verify.MODEL_REVISION,
        "feature_schema": "fixture-feature-v1",
        "source_visible_gap_filtering": False,
        "subjective_pleasure_proven": False,
        "hosted_gpt_model_modified": False,
        "artifact_run_directory": str(root),
        "predeclared_protocol": {
            "seeds": list(verify.SEEDS),
            "training_pairs_per_family": 8,
            "heldout_pairs_per_generator": 4,
            "train_families": list(verify.TRAIN_FAMILIES),
            "primary_within_families": list(verify.PRIMARY_FAMILIES),
            "secondary_ood_families": list(verify.OOD_FAMILIES),
            "exploration_rate": 0.4, "learning_rate": 0.05,
            "appraisal_gain": 12.0, "lora_rank": 2, "lora_alpha": 4,
            "max_updates": 48,
            "lora_initialization_seed_xor_salt": verify.LORA_SEED_SALT,
            "global_eval_receipts_withheld_until_all_arm_choices_committed": True,
            "no_eval_training": True,
        },
        "seed_reports": seed_rows,
        "co_primary_contrasts": contrasts,
        "primary_all_contrasts_by_latent_family": primary_means,
        "primary_joint_on_minus_off_by_latent_family": primary_means["joint_on_minus_off"],
        "secondary_ood_joint_on_minus_off_by_family": {
            family: fsum(vals) / 12 for family, vals in ood.items()},
        "joint_on_minus_off_total_sampled_action_switches": sum(
            row["joint_on_minus_off_action_switches"] for row in seed_rows),
        "joint_on_minus_off_mean_absolute_behavior_prob_a_shift": fsum(
            row["joint_on_minus_off_mean_absolute_prob_a_shift"]
            for row in seed_rows) / 12,
        "exploratory_positive_local_criterion_met": True,
    }
    raw = _json(report) + b"\n"
    (root / "study-report.json").write_bytes(raw)
    (root / "COMPLETE.sha256").write_bytes(
        (sha256(raw).hexdigest() + "  study-report.json\n").encode("ascii"))
    return report


def _reseal(root, report):
    raw = _json(report) + b"\n"
    (root / "study-report.json").write_bytes(raw)
    (root / "COMPLETE.sha256").write_bytes(
        (sha256(raw).hexdigest() + "  study-report.json\n").encode("ascii"))


class VerifyJointStudyReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "joint-lora-v3-0123456789abcdef0123"
        self.root.mkdir()
        self.report = _fixture(self.root)

    def test_complete_synthetic_12_seed_study_verifies(self):
        result = verify.verify_complete_study(self.root)
        self.assertTrue(result["preregistered_exploratory_utility_gate_met"])
        self.assertEqual(result["train_chosen_receipts"], 288)
        self.assertEqual(result["eval_precommitted_actions"], 1680)
        self.assertEqual(result["artifact_count"], 12)
        self.assertEqual(len(result["co_primary_contrasts"]), 3)
        self.assertTrue(all(value["holm_adjusted_p"] < .05
                            for value in result["co_primary_contrasts"].values()))

    def test_incomplete_directory_or_failed_marker_rejected(self):
        (self.root / "COMPLETE.sha256").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete|unexpected"):
            verify.verify_complete_study(self.root)

    def test_failed_marker_rejected_even_if_complete_marker_exists(self):
        (self.root / "FAILED.json").write_bytes(b'{"status":"FAILED"}\n')
        with self.assertRaisesRegex(ValueError, "incomplete|unexpected"):
            verify.verify_complete_study(self.root)

    def test_symlinked_complete_marker_rejected(self):
        seal = self.root / "COMPLETE.sha256"
        target = self.root / "temporary-complete-target"
        target.write_bytes(seal.read_bytes())
        seal.unlink()
        seal.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic-link|unexpected"):
            verify.verify_complete_study(self.root)

    def test_unresealed_tamper_rejected_before_any_outcome_read(self):
        report_file = self.root / "study-report.json"
        report_file.write_bytes(report_file.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify.verify_complete_study(self.root)

    def test_resealed_false_claimed_primary_gain_rejected(self):
        self.report["co_primary_contrasts"]["joint_on_minus_off"]["mean"] += .2
        _reseal(self.root, self.report)
        with self.assertRaisesRegex(ValueError, "paired statistic"):
            verify.verify_complete_study(self.root)

    def test_resealed_seven_arm_draw_tamper_rejected(self):
        trace = self.report["seed_reports"][0]["eval_arms"]["head_only_on"]["rows"][0]["sampled_pre_feedback"]
        trace["random_draw"] += 0.01
        _reseal(self.root, self.report)
        with self.assertRaisesRegex(ValueError, "seeded choice draw"):
            verify.verify_complete_study(self.root)

    def test_resealed_shuffled_donor_training_label_tamper_rejected(self):
        bad = self.report["seed_reports"][0]["train"][0]["learner_updates"]["shuffled"]
        bad["teaching_quality"] += 0.1
        _reseal(self.root, self.report)
        with self.assertRaisesRegex(ValueError, "shuffled control"):
            verify.verify_complete_study(self.root)

    def test_resealed_head_only_lora_digest_tamper_rejected(self):
        row = self.report["seed_reports"][1]["eval_arms"]["head_only_on"]["rows"][0]
        row["decision_lora_sha256"] = "f" * 64
        _reseal(self.root, self.report)
        with self.assertRaisesRegex(ValueError, "head-only/frozen"):
            verify.verify_complete_study(self.root)

    def test_resealed_false_positive_gate_rejected(self):
        self.report["exploratory_positive_local_criterion_met"] = False
        _reseal(self.root, self.report)
        with self.assertRaisesRegex(ValueError, "six-family utility gate"):
            verify.verify_complete_study(self.root)

    def test_resealed_calibration_claim_tamper_rejected(self):
        self.report["seed_reports"][2]["eval_arms"]["joint_on"]["candidate_quality_mae"] = .99
        _reseal(self.root, self.report)
        with self.assertRaisesRegex(ValueError, "calibration"):
            verify.verify_complete_study(self.root)

    def test_corrupted_adapter_tensor_rejected_even_if_study_report_untouched(self):
        weights = self.root / f"seed-{verify.SEEDS[0]}-joint-lora-head-v1" / "weights.safetensors"
        raw = weights.read_bytes()
        weights.write_bytes(raw[:-1] + bytes((raw[-1] ^ 1,)))
        with self.assertRaisesRegex(ValueError, "artifact binding"):
            verify.verify_complete_study(self.root)

    def test_paired_statistics_positive_and_balanced(self):
        positive = verify._paired([.01] * 12)
        self.assertEqual(positive["exact_two_sided_signflip_p_unadjusted"], 2 / 4096)
        balanced = verify._paired([.01] * 6 + [-.01] * 6)
        self.assertEqual(balanced["exact_two_sided_signflip_p_unadjusted"], 1.0)
        self.assertEqual(verify._holm({"a": .01, "b": .03, "c": .2}),
                         {"a": .03, "b": .06, "c": .2})


if __name__ == "__main__":
    unittest.main()
