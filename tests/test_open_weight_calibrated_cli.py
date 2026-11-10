"""Public calibrated benchmark CLI consent and experiment wiring."""

from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from unittest import mock

from scripts.run_open_weight_calibrated_appraisal_benchmark import (
    PROTOCOL_SEEDS, build_parser, run,
)


class CalibratedBenchmarkCLITests(unittest.TestCase):
    def args(self, *additional):
        return build_parser().parse_args([
            "--snapshot", "/tmp/pinned-snapshot",
            "--model-id", "offline-test-model",
            "--revision", "a" * 40,
            "--save-head-root", "/tmp/new-heads",
            *additional,
        ])

    def test_protocol_defaults_exact_and_pinned(self):
        args = self.args()
        self.assertEqual(tuple(args.seeds), PROTOCOL_SEEDS)
        self.assertEqual(len(args.seeds), 12)
        self.assertEqual(args.training_pairs, 24)
        self.assertEqual(args.heldout_pairs_per_generator, 6)
        self.assertEqual(args.exploration_rate, 0.4)
        self.assertEqual(args.learning_rate, 0.1)
        self.assertEqual(args.appraisal_gain, 12.0)
        self.assertFalse(args.allow_appraisal_training)
        self.assertFalse(args.allow_persistent_export)

    def test_reject_without_both_explicit_consents_preload(self):
        with mock.patch(
            "scripts.run_open_weight_calibrated_appraisal_benchmark.load_verified_local_causal_model",
        ) as loader:
            for flags in (
                (), ("--allow-appraisal-training",),
                ("--allow-persistent-export",),
            ):
                with self.assertRaisesRegex(PermissionError, "both explicit"):
                    run(self.args(*flags))
            loader.assert_not_called()

    def test_invalid_revision_rejected_before_model_load(self):
        with mock.patch(
            "scripts.run_open_weight_calibrated_appraisal_benchmark.load_verified_local_causal_model",
        ) as loader:
            with self.assertRaisesRegex(ValueError, "40-hex"):
                run(self.args(
                    "--revision", "main", "--allow-appraisal-training",
                    "--allow-persistent-export",
                ))
            loader.assert_not_called()

    def test_encode_action_order_and_forward_exact_benchmark_api(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last_prompt = text
                return [1, 2, 4]

        tokenizer = Tokenizer()
        loaded = SimpleNamespace(
            tokenizer=tokenizer, action_suffixes=(" A", " B"),
            action_token_ids=(3, 4),
        )
        fake_result = {"status": "test-fixture", "seeds": 12}
        study = mock.Mock(return_value=fake_result)
        module = SimpleNamespace(run_calibrated_appraisal_benchmark=study)
        path = "pleasure_experiment.open_weight_calibrated_benchmark"
        with (
            mock.patch.dict(sys.modules, {path: module}),
            mock.patch(
                "scripts.run_open_weight_calibrated_appraisal_benchmark.load_verified_local_causal_model",
                return_value=loaded,
            ) as loader,
            mock.patch(
                "scripts.run_open_weight_calibrated_appraisal_benchmark.validate_single_token_actions",
                return_value=(3, 4),
            ) as action_tokens,
        ):
            output = run(self.args(
                "--allow-appraisal-training", "--allow-persistent-export",
            ))
            self.assertEqual(output, fake_result)
            self.assertEqual(study.call_count, 1)
            positional, keyword = study.call_args
            self.assertIs(positional[0], loaded)
            offer = SimpleNamespace(
                features_a=(0.22, 0.7), features_b=(0.92, 0.11),
            )
            self.assertEqual(positional[1](offer), (1, 2, 4))
            self.assertIn("Option A features: [0.220000, 0.700000]", tokenizer.last_prompt)
            self.assertEqual(action_tokens.call_count, 1)
            self.assertEqual(keyword["seeds"], PROTOCOL_SEEDS)
            self.assertEqual(keyword["training_pairs"], 24)
            self.assertEqual(keyword["heldout_pairs_per_generator"], 6)
            self.assertEqual(keyword["exploration_rate"], 0.4)
            self.assertEqual(keyword["learning_rate"], 0.1)
            self.assertEqual(keyword["appraisal_gain"], 12.0)
            self.assertEqual(keyword["artifact_root"], "/tmp/new-heads")
            self.assertTrue(keyword["allow_appraisal_training"])
            self.assertTrue(keyword["allow_persistent_export"])
            self.assertEqual(loader.call_count, 1)

        with (
            mock.patch.dict(sys.modules, {path: module}),
            mock.patch(
                "scripts.run_open_weight_calibrated_appraisal_benchmark.load_verified_local_causal_model",
                return_value=loaded,
            ),
            mock.patch(
                "scripts.run_open_weight_calibrated_appraisal_benchmark.validate_single_token_actions",
                return_value=(4, 3),
            ),
        ):
            run(self.args("--allow-appraisal-training", "--allow-persistent-export"))
            with self.assertRaisesRegex(RuntimeError, "token order"):
                study.call_args[0][1](offer)
