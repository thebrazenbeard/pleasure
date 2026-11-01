"""Validation and deterministic fixture tests for fresh-process head verifier."""

from __future__ import annotations

from contextlib import redirect_stderr
from hashlib import sha256
import importlib.util
from io import StringIO
from types import SimpleNamespace
from unittest import TestCase, mock, skipUnless

from scripts.verify_saved_open_weight_appraisal_head import build_parser, run


class AppraisalReloadGateTests(TestCase):
    def _args(self, *extra):
        return build_parser().parse_args([
            "--snapshot", "/tmp/fake-snapshot",
            "--head-dir", "/tmp/fake-persisted-head",
            "--model-id", "offline-test-model",
            "--revision", "a" * 40,
            *extra,
        ])

    def test_explicit_reload_permission_required_before_model_load(self):
        with mock.patch(
            "scripts.verify_saved_open_weight_appraisal_head.load_verified_local_causal_model",
        ) as load:
            with self.assertRaisesRegex(PermissionError, "allow-appraisal-head-reload"):
                run(self._args())
            load.assert_not_called()

    def test_invalid_revision_and_seed_rejected_before_model_load(self):
        with mock.patch(
            "scripts.verify_saved_open_weight_appraisal_head.load_verified_local_causal_model",
        ) as load:
            with self.assertRaisesRegex(ValueError, "40-hex"):
                run(self._args("--allow-appraisal-head-reload", "--revision", "main"))
            with self.assertRaisesRegex(ValueError, "uint32"):
                run(self._args("--allow-appraisal-head-reload", "--seed", "-1"))
            with self.assertRaisesRegex(ValueError, "heldout-pairs"):
                run(self._args("--allow-appraisal-head-reload", "--heldout-pairs", "0"))
            with self.assertRaisesRegex(ValueError, "appraisal-gain"):
                run(self._args("--allow-appraisal-head-reload", "--appraisal-gain", "nan"))
            load.assert_not_called()

    def test_unknown_holdout_family_refused_by_parser(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                self._args("--heldout-family", "heldout_train_affine")


@skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch unavailable")
class AppraisalReloadFixtureTests(TestCase):
    def _loaded(self):
        import torch

        class TinyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=2)
                self.embedding = torch.nn.Embedding(8, 2)
                self.output = torch.nn.Linear(2, 8)
                with torch.no_grad():
                    self.embedding.weight.zero_()
                    self.embedding.weight[3] = torch.tensor([1.0, 0.0])
                    self.embedding.weight[4] = torch.tensor([0.0, 1.0])
                    self.output.weight.zero_()
                    self.output.bias.zero_()
                    self.output.bias[3] = 0.1

            def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.output(hidden),
                    hidden_states=(hidden,) if output_hidden_states else None,
                )

        class TinyTokenizer:
            def encode(self, prompt, add_special_tokens=False):
                return [1]

        content_sha = sha256(b"pinned snapshot").hexdigest()
        return SimpleNamespace(
            model=TinyCausalLM(), tokenizer=TinyTokenizer(),
            binding=SimpleNamespace(
                model_id="offline-test-model", model_revision="a" * 40,
                runtime_instance_id="offline-appraisal-fresh-reload",
            ),
            manifest=SimpleNamespace(snapshot_sha256=content_sha),
            action_token_ids=(3, 4), action_suffixes=(" A", " B"),
        )

    def _fake_load(self, probe, **kwargs):
        import torch

        from pleasure_experiment.open_weight_adapter import _named_digest

        tensors = probe.export_tensors()
        tensors["weight"][0, 0] = 4.0
        tensors["weight"][0, 1] = -4.0
        trained_sha = _named_digest(torch, tuple(tensors.items()))
        probe.import_tensors(tensors, expected_digest=trained_sha)
        model_named = tuple(probe._model.named_parameters()) + tuple(
            ("buffer:" + name, b) for name, b in probe._model.named_buffers()
        )
        return SimpleNamespace(
            binding=SimpleNamespace(runtime_instance_id="original-training-process"),
            head_weights_sha256=trained_sha,
            frozen_model_sha256=_named_digest(torch, model_named),
            manifest_sha256=sha256(b"head manifest").hexdigest(),
            safetensors_sha256=sha256(b"head safetensors").hexdigest(),
        )

    def test_untrained_to_imported_head_changes_predictions_without_changing_base(self):
        loaded = self._loaded()
        with (
            mock.patch(
                "scripts.verify_saved_open_weight_appraisal_head.load_verified_local_causal_model",
                return_value=loaded,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_appraisal_head.validate_single_token_actions",
                return_value=(3, 4),
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_appraisal_head.load_versioned_appraisal_head",
                side_effect=self._fake_load,
            ) as import_head,
        ):
            report = run(build_parser().parse_args([
                "--snapshot", "/tmp/fake-snapshot",
                "--head-dir", "/tmp/fake-head",
                "--model-id", "offline-test-model",
                "--revision", "a" * 40,
                "--heldout-pairs", "2",
                "--allow-appraisal-head-reload",
            ]))
        self.assertEqual(import_head.call_count, 1)
        self.assertTrue(report["tensor_import_verified"])
        self.assertTrue(report["distinct_runtime_instance_ids"])
        self.assertNotEqual(
            report["initial_untrained_head_sha256"],
            report["imported_trained_head_sha256"],
        )
        self.assertTrue(report["head_import_changes_heldout_predictions"])
        self.assertEqual(len(report["fixed_heldout_episode_ids"]), 2)
        for off_before, off_after, on_before, on_after in zip(
            report["before_import_off"], report["after_import_off"],
            report["before_import_on"], report["after_import_on"],
        ):
            self.assertEqual(off_before["prob_a"], off_after["prob_a"])
            self.assertEqual(off_before["prob_b"], off_after["prob_b"])
            self.assertNotEqual(on_before["predicted_quality_a"], on_after["predicted_quality_a"])
            self.assertEqual(off_after["base_prob_a"], on_after["base_prob_a"])

    def test_same_runtime_identifier_rejected_after_import(self):
        loaded = self._loaded()

        def reused_origin(probe, **kwargs):
            result = self._fake_load(probe, **kwargs)
            result.binding.runtime_instance_id = loaded.binding.runtime_instance_id
            return result

        with (
            mock.patch(
                "scripts.verify_saved_open_weight_appraisal_head.load_verified_local_causal_model",
                return_value=loaded,
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_appraisal_head.validate_single_token_actions",
                return_value=(3, 4),
            ),
            mock.patch(
                "scripts.verify_saved_open_weight_appraisal_head.load_versioned_appraisal_head",
                side_effect=reused_origin,
            ),
        ):
            args = build_parser().parse_args([
                "--snapshot", "/tmp/fake-snapshot", "--head-dir", "/tmp/fake-head",
                "--model-id", "offline-test-model", "--revision", "a" * 40,
                "--allow-appraisal-head-reload",
            ])
            with self.assertRaisesRegex(RuntimeError, "origin and fresh-process"):
                run(args)
