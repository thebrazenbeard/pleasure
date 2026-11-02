"""Independent posthoc comparison of actual-vs-fresh process predictions."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts import compare_calibrated_fresh_reload as comparator


class CalibratedCrossProcessComparisonTests(unittest.TestCase):
    def setUp(self):
        self.seed = 101
        self.family = "sealed_eval_01_v2"
        self.initial = "1" * 64
        self.head = "a" * 64
        self.frozen = "f" * 64
        self.artifact = "c" * 64
        self.safetensors = "d" * 64
        self.epsilon = 0.4

        self.episodes = [f"episode-{i}" for i in range(6)]
        self.hashes = [sha256(f"tokenized-{i}".encode()).hexdigest() for i in range(6)]
        family_rows = {}
        for family in comparator.FAMILIES:
            rows = {}
            for arm in ("trained_on", "trained_off", "trained_inverted", "frozen_on"):
                probs = {"trained_on": .7, "trained_off": .6,
                         "trained_inverted": .5, "frozen_on": .6}
                a = probs[arm]
                q_a, q_b = ((.6, .4) if arm != "frozen_on" else (.5, .5))
                rows[arm] = [{
                    "episode_id": f"episode-{index}" if family == self.family
                    else f"{family}-episode-{index}",
                    "family_id": family,
                    "prediction_a": q_a,
                    "prediction_b": q_b,
                    "sampled_pre_feedback": {
                        "behavior_prob_a": (1 - self.epsilon) * a + self.epsilon / 2,
                        "input_ids_sha256": self.hashes[index],
                    },
                } for index in range(6)]
            family_rows[family] = rows
        eval_arms = {}
        for arm in ("trained_on", "trained_off", "trained_inverted", "frozen_on"):
            eval_arms[arm] = {"rows": [row for family in comparator.FAMILIES
                                      for row in family_rows[family][arm]]}
        self.study = {
            "evidence_class": "SOURCE_LOCAL_CALIBRATED_SEALED_PROCESS_APPRAISAL_UNAUTHENTICATED",
            "pretrained_model_unchanged": True,
            "in_process_saved_head_import_verified": True,
            "subjective_pleasure_proven": False,
            "model_id": "tiny/stub", "model_revision": "pinned-revision",
            "frozen_pretrained_model_sha256": self.frozen,
            "head_feature_schema": "schema-v1",
            "predeclared_protocol": {
                "seeds": list(comparator.PROTOCOL_SEEDS),
                "training_pairs_per_seed": 24,
                "heldout_pairs_per_family_per_seed": 6,
                "suite": "novel_v2",
                "primary_ood_generators": list(comparator.FAMILIES[1:]),
                "positive_control_generator": comparator.FAMILIES[0],
                "exploration_rate": .4,
                "appraisal_gain": 12.0,
                "global_all_arm_seal_before_any_eval_reward": True,
            },
            "seed_reports": [{
                "seed": seed,
                "initial_head_sha256": self.initial,
                "trained_head_sha256": self.head,
                "saved_head_manifest_sha256": self.artifact,
                "saved_head_safetensors_sha256": self.safetensors,
                "eval_arms": eval_arms,
            } for seed in comparator.PROTOCOL_SEEDS],
        }

        def fresh_rows(prob: float, head: str, q_a: float, q_b: float):
            return [{
                "episode_id": episode,
                "input_ids_sha256": digest,
                "head_sha256": head,
                "prob_a": prob,
                "prob_b": 1 - prob,
                "predicted_quality_a": q_a,
                "predicted_quality_b": q_b,
                "base_prob_a": .6,
                "base_prob_b": .4,
            } for episode, digest in zip(self.episodes, self.hashes)]

        self.reload = {
            "tensor_import_verified": True,
            "spawned_fresh_model_process": True,
            "unseen_generator_outcomes_not_read_during_reload": True,
            "frozen_head_and_disabled_head_controls_preserved": True,
            "quality_gain_established_by_reload_alone": False,
            "subjective_pleasure_proven": False,
            "fixed_heldout_family": self.family,
            "model_id_declared": "tiny/stub", "model_revision_declared": "pinned-revision",
            "fresh_process_pid": 2, "parent_process_pid": 1,
            "independent_evaluator_worker_pid": 3,
            "distinct_runtime_instance_ids": True,
            "origin_runtime_instance_id_declared": "train-runtime",
            "new_runtime_instance_id_declared": "fresh-runtime",
            "imported_trained_head_sha256": self.head,
            "head_manifest_sha256": self.artifact,
            "head_safetensors_sha256": self.safetensors,
            "frozen_model_sha256_unchanged": self.frozen,
            "feature_schema_verified_on_import": "schema-v1",
            "initial_untrained_head_sha256": self.initial,
            "fixed_heldout_episode_ids": self.episodes,
            "after_import_on": fresh_rows(.7, self.head, .6, .4),
            "after_import_off": fresh_rows(.6, self.head, .6, .4),
            "after_import_inverted": fresh_rows(.5, self.head, .6, .4),
            "before_import_on": fresh_rows(.6, self.initial, .5, .5),
        }

    def _check(self):
        return comparator.compare_reports(self.study, self.reload, seed=self.seed,
                                          family=self.family)

    def test_six_fresh_process_episodes_match_four_study_arms(self):
        result = self._check()
        self.assertTrue(result["source_local_cross_process_behavior_verified"])
        self.assertEqual(result["episodes_verified"], 6)
        self.assertEqual(result["treatment_control_pairs_verified"], 4)
        self.assertEqual(result["max_abs_quality_prediction_error"], 0.0)
        self.assertEqual(result["max_abs_behavior_prob_a_error"], 0.0)
        self.assertTrue(result["fresh_reload_does_not_establish_reward_gain"])

    def test_rejects_subtle_changed_action_probability(self):
        self.reload["after_import_on"][1]["prob_a"] += .0001
        self.reload["after_import_on"][1]["prob_b"] -= .0001
        with self.assertRaisesRegex(ValueError, "behavior policy"):
            self._check()

    def test_rejects_changed_quality_prediction(self):
        self.reload["after_import_on"][0]["predicted_quality_a"] += .0001
        with self.assertRaisesRegex(ValueError, "predicted quality"):
            self._check()

    def test_rejects_duplicate_missing_reordered_or_wrong_episode(self):
        for value in (
            self.episodes[:5],
            self.episodes[:5] + [self.episodes[0]],
            list(reversed(self.episodes)),
            self.episodes[:5] + ["wrong-episode"],
        ):
            with self.subTest(episodes=value):
                self.reload["fixed_heldout_episode_ids"] = value
                with self.assertRaises(ValueError):
                    self._check()
        self.reload["fixed_heldout_episode_ids"] = self.episodes

    def test_rejects_schema_digest_parent_and_incomplete_study(self):
        for field, value in (
            ("imported_trained_head_sha256", "0" * 64),
            ("head_manifest_sha256", "0" * 64),
            ("fresh_process_pid", 1),
            ("distinct_runtime_instance_ids", False),
        ):
            with self.subTest(field=field):
                original = self.reload[field]
                self.reload[field] = value
                with self.assertRaises(ValueError):
                    self._check()
                self.reload[field] = original
        self.study["seed_reports"].pop()
        with self.assertRaisesRegex(ValueError, "seed reports"):
            self._check()

    def test_cli_rejects_tampered_complete_marker(self):
        from scripts import compare_calibrated_fresh_reload as module
        with TemporaryDirectory() as temp:
            run = Path(temp) / "completed-run"
            run.mkdir()
            study_path = run / "study-report.json"
            fresh_path = Path(temp) / "fresh.json"
            self.study["artifact_run_directory"] = str(run)
            raw = (json.dumps(self.study, sort_keys=True) + "\n").encode()
            study_path.write_bytes(raw)
            fresh_path.write_text(json.dumps(self.reload), encoding="utf-8")
            marker = run / "COMPLETE.sha256"
            marker.write_text(sha256(raw).hexdigest() + "  study-report.json\n")
            argv = ["compare", "--study-report", str(study_path),
                    "--fresh-report", str(fresh_path),
                    "--seed", str(self.seed), "--family", self.family]
            with mock.patch("sys.argv", argv):
                with mock.patch("builtins.print") as printed:
                    module.main()
                self.assertTrue(printed.called)
            marker.write_text("0" * 64 + "  study-report.json\n")
            with mock.patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as exit_event:
                    module.main()
                self.assertEqual(exit_event.exception.code, 2)

    def test_duplicate_json_keys_and_nan_rejected(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "evidence.json"
            source.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                comparator._read_json(source, maximum_bytes=1024)
            source.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                comparator._read_json(source, maximum_bytes=1024)
