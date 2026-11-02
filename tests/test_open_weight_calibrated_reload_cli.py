"""Fresh subprocess, safe model-bound import and fixed v2 prompt controls."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from io import StringIO
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.verify_saved_open_weight_calibrated_appraisal_head import (
    NOVEL_EVAL_FAMILIES, _run_worker, build_parser, run,
)


class CalibratedReloadGateTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args([
            "--snapshot", "/tmp/verified-snapshot",
            "--head-dir", "/tmp/verified-calibrated-head",
            "--model-id", "offline-test-model",
            "--revision", "a" * 40,
            *extra,
        ])

    def test_explicit_permission_and_identity_rejected_before_spawn(self):
        with mock.patch(
            "scripts.verify_saved_open_weight_calibrated_appraisal_head.subprocess.run",
        ) as spawn:
            with self.assertRaisesRegex(PermissionError, "allow-appraisal-head-reload"):
                run(self.args())
            with self.assertRaisesRegex(ValueError, "40-hex"):
                run(self.args("--allow-appraisal-head-reload", "--revision", "main"))
            with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
                run(self.args("--allow-appraisal-head-reload", "--seed", "-1"))
            with self.assertRaisesRegex(ValueError, "heldout-pairs"):
                run(self.args("--allow-appraisal-head-reload", "--heldout-pairs", "0"))
            with self.assertRaisesRegex(ValueError, "appraisal-gain"):
                run(self.args("--allow-appraisal-head-reload", "--appraisal-gain", "nan"))
            with self.assertRaisesRegex(ValueError, "expected-head-sha256"):
                run(self.args(
                    "--allow-appraisal-head-reload", "--expected-head-sha256", "bad",
                ))
            spawn.assert_not_called()

    def test_unrecognized_holdout_family_rejected_by_parser(self):
        with mock.patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit):
                self.args("--heldout-family", "old_heldout_train_affine")

    def test_parent_spawns_distinct_isolated_python_interpreter(self):
        args = self.args(
            "--allow-appraisal-head-reload", "--expected-head-sha256", "e" * 64,
        )

        def fake_spawn(command, **kwargs):
            from os import getpid
            self.assertEqual(command[1:3], ["-I", "-B"])
            self.assertIn("--_fresh-worker", command)
            self.assertEqual(command[command.index("--_parent-pid") + 1], str(getpid()))
            self.assertIn("--expected-head-sha256", command)
            self.assertTrue(kwargs["capture_output"])
            self.assertFalse(kwargs["check"])
            self.assertNotIn("shell", kwargs)
            return SimpleNamespace(
                returncode=0, stderr="", stdout=json.dumps({
                    "fresh_process_pid": getpid() + 10_000,
                    "parent_process_pid": getpid(),
                    "spawned_fresh_model_process": True,
                    "tensor_import_verified": True,
                }),
            )

        with mock.patch(
            "scripts.verify_saved_open_weight_calibrated_appraisal_head.subprocess.run",
            side_effect=fake_spawn,
        ) as spawn:
            report = run(args)
            self.assertEqual(spawn.call_count, 1)
            self.assertTrue(report["tensor_import_verified"])

    def test_parent_rejects_invalid_child_pid_bad_json_and_subprocess_failure(self):
        args = self.args("--allow-appraisal-head-reload")
        from os import getpid
        for output in (
            "not json",
            json.dumps({"fresh_process_pid": getpid(),
                        "parent_process_pid": getpid(),
                        "spawned_fresh_model_process": True,
                        "tensor_import_verified": True}),
            json.dumps({"fresh_process_pid": getpid() + 5,
                        "parent_process_pid": getpid() + 99,
                        "spawned_fresh_model_process": True,
                        "tensor_import_verified": True}),
            '{"fresh_process_pid":NaN}',
        ):
            with mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stderr="", stdout=output),
            ):
                with self.assertRaises(RuntimeError):
                    run(args)
        with mock.patch(
            "scripts.verify_saved_open_weight_calibrated_appraisal_head.subprocess.run",
            return_value=SimpleNamespace(returncode=2, stderr="unsafe head rejected", stdout=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsafe head rejected"):
                run(args)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "optional PyTorch unavailable")
class CalibratedReloadTinyModelTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args([
            "--snapshot", "/tmp/verified-snapshot",
            "--head-dir", "/tmp/verified-calibrated-head",
            "--model-id", "offline-test-model",
            "--revision", "a" * 40,
            "--heldout-pairs", "2", "--_fresh-worker", "--_parent-pid", "777",
            "--allow-appraisal-head-reload",
            *extra,
        ])

    def loaded(self):
        import torch

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=2)
                self.embedding = torch.nn.Embedding(8, 2)
                self.output = torch.nn.Linear(2, 8)
                self.calls = 0
                with torch.no_grad():
                    self.embedding.weight.zero_()
                    self.embedding.weight[3] = torch.tensor([1.0, 0.0])
                    self.embedding.weight[4] = torch.tensor([0.0, 1.0])
                    self.output.weight.zero_()
                    self.output.bias.zero_()
                    self.output.bias[3] = 0.2

            def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
                self.calls += 1
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.output(hidden),
                    hidden_states=(hidden,) if output_hidden_states else None,
                )

        class Tokenizer:
            def encode(self, prompt, add_special_tokens=False):
                return [1, 2]

        return SimpleNamespace(
            model=TinyLM(), tokenizer=Tokenizer(),
            action_token_ids=(3, 4), action_suffixes=(" A", " B"),
            binding=SimpleNamespace(
                model_id="offline-test-model", model_revision="a" * 40,
                runtime_instance_id="offline-calibrated-fresh-reload",
                checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
            ),
            manifest=SimpleNamespace(snapshot_sha256="c" * 64),
        )

    def fake_artifact(self, probe, **kwargs):
        import torch
        from pleasure_experiment.open_weight_adapter import _named_digest

        tensors = probe.export_tensors()
        tensors["weight"][0, 0] = 3.0
        tensors["weight"][0, 1] = -3.0
        trained = _named_digest(torch, tuple(tensors.items()))
        probe.import_tensors(tensors, expected_digest=trained)
        named = tuple(probe._model.named_parameters()) + tuple(
            ("buffer:" + name, state) for name, state in probe._model.named_buffers()
        )
        return SimpleNamespace(
            binding=SimpleNamespace(runtime_instance_id="original-training-process"),
            head_weights_sha256=trained,
            frozen_model_sha256=_named_digest(torch, named),
            manifest_sha256=sha256(b"head manifest").hexdigest(),
            safetensors_sha256=sha256(b"head tensors").hexdigest(),
        )

    def test_worker_reloads_head_and_scores_fixed_eval_without_rewards(self):
        loaded = self.loaded()
        evaluator = mock.MagicMock()
        evaluator.__enter__.return_value = evaluator
        evaluator.worker_pid = 733

        def issue(*, partition, arm, family_id, pair_index, side):
            self.assertEqual(partition, "EVAL")
            self.assertEqual(arm, "trained_on")
            self.assertIn(family_id, NOVEL_EVAL_FAMILIES)
            return SimpleNamespace(
                episode_id=sha256(f"heldout-{pair_index}".encode()).hexdigest(),
                features_a=(0.15, 0.85), features_b=(0.75, 0.35),
            )

        evaluator.issue.side_effect = issue
        with (
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.os.getppid",
                return_value=777,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.load_verified_local_causal_model",
                return_value=loaded,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.IsolatedOutcomeEvaluator",
                return_value=evaluator,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.validate_single_token_actions",
                return_value=(3, 4),
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.load_versioned_appraisal_head",
                side_effect=self.fake_artifact,
            ) as import_head,
        ):
            report = _run_worker(self.args())
        self.assertEqual(import_head.call_count, 1)
        self.assertTrue(report["tensor_import_verified"])
        self.assertTrue(report["spawned_fresh_model_process"])
        self.assertTrue(report["unseen_generator_outcomes_not_read_during_reload"])
        self.assertEqual(report["parent_process_pid"], 777)
        self.assertEqual(loaded.model.calls, 6)  # 3 forwards per offer, not per arm/gain
        self.assertEqual(len(report["fixed_heldout_episode_ids"]), 2)
        self.assertNotEqual(
            report["initial_untrained_head_sha256"],
            report["imported_trained_head_sha256"],
        )
        self.assertTrue(report["head_import_changes_heldout_predictions"])
        evaluator.commit.assert_not_called()
        evaluator.resolve.assert_not_called()
        evaluator.posthoc_evaluate.assert_not_called()
        for before_on, before_off, after_on, after_off in zip(
            report["before_import_on"], report["before_import_off"],
            report["after_import_on"], report["after_import_off"],
        ):
            self.assertEqual(before_on["prob_a"], before_off["prob_a"])
            self.assertEqual(before_off["prob_a"], after_off["prob_a"])
            self.assertNotEqual(before_on["predicted_quality_a"],
                                after_on["predicted_quality_a"])

    def test_worker_does_not_accept_same_parent_pid_or_train_runtime(self):
        loaded = self.loaded()
        args = self.args()
        with mock.patch(
            "scripts.verify_saved_open_weight_calibrated_appraisal_head.os.getppid",
            return_value=888,
        ):
            with self.assertRaisesRegex(PermissionError, "declared live parent"):
                _run_worker(args)

        class Evaluator:
            worker_pid = 733

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def issue(self, **kwargs):
                return SimpleNamespace(
                    episode_id="x", features_a=(0.1, 0.3), features_b=(0.5, 0.8),
                )

        def same_origin(probe, **kwargs):
            artifact = self.fake_artifact(probe, **kwargs)
            artifact.binding.runtime_instance_id = loaded.binding.runtime_instance_id
            return artifact

        with (
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.os.getppid",
                return_value=777,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.load_verified_local_causal_model",
                return_value=loaded,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.IsolatedOutcomeEvaluator",
                return_value=Evaluator(),
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.validate_single_token_actions",
                return_value=(3, 4),
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_calibrated_appraisal_head.load_versioned_appraisal_head",
                side_effect=same_origin,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "origin and fresh-process"):
                _run_worker(args)
