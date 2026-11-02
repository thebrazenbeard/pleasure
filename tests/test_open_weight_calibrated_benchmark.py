"""Small-model protocol integration: cached head, worker seal and full run custody."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFE = importlib.util.find_spec("safetensors") is not None


@unittest.skipUnless(HAS_TORCH and HAS_SAFE, "requires optional torch/safetensors")
class CalibratedBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _toy_loaded():
        import torch
        from pleasure_experiment.open_weight_adapter import OpenWeightBinding
        from pleasure_experiment.open_weight_local_model import LocalModelManifest

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)
                self.output = torch.nn.Linear(4, 32)
                self.config = SimpleNamespace(hidden_size=4)
                with torch.no_grad():
                    self.embedding.weight.zero_()
                    self.embedding.weight[3] = torch.tensor([1., -0.5, 0.3, 0.0])
                    self.embedding.weight[4] = torch.tensor([0.1, 0.8, -0.2, 0.5])
                    self.output.weight.zero_()
                    self.output.bias.zero_()
                    self.output.bias[3] = 0.04

            def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.output(hidden),
                    hidden_states=(hidden,) if output_hidden_states else None,
                )

        loaded = SimpleNamespace(
            model=Toy(), action_token_ids=(3, 4),
            binding=OpenWeightBinding(
                model_id="toy/stub", model_revision="fixed-revision",
                runtime_instance_id="toy-stub-runtime",
                checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
            ),
            manifest=LocalModelManifest(
                snapshot_directory="/tmp/toy-model-fixture", files=(),
                checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
                snapshot_sha256="c" * 64, total_bytes=42,
            ),
        )
        return loaded

    def test_predeclared_exact_parameters_and_opt_in(self):
        from pleasure_experiment.open_weight_calibrated_benchmark import (
            _predeclared, SEEDS_20261102,
        )
        _predeclared(SEEDS_20261102, 24, 6, 0.4, 0.1, 12.0)
        with self.assertRaisesRegex(ValueError, "predeclared"):
            _predeclared(SEEDS_20261102[:-1], 24, 6, 0.4, 0.1, 12.0)
        with self.assertRaisesRegex(ValueError, "predeclared"):
            _predeclared(SEEDS_20261102, 24, 6, 0.4, 0.1, 8.0)

    def test_completed_run_promotion_never_replaces_existing_directory(self):
        from pleasure_experiment.open_weight_calibrated_benchmark import (
            _atomic_no_replace_directory,
        )
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "staged"
            target = Path(tmp) / "already-exists"
            source.mkdir()
            target.mkdir()
            (source / "complete-data").write_text("new", encoding="utf-8")
            (target / "user-data").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _atomic_no_replace_directory(source, target)
            self.assertEqual((target / "user-data").read_text(), "keep")
            self.assertEqual((source / "complete-data").read_text(), "new")

    def test_full_small_model_study_with_complete_atomic_run_directory(self):
        from pleasure_experiment import open_weight_calibrated_benchmark as module
        loaded = self._toy_loaded()
        with TemporaryDirectory() as temp:
            # Test-only short protocol, deliberately not a scientific run.
            with mock.patch.object(module, "_predeclared", return_value=None):
                result = module.run_calibrated_appraisal_benchmark(
                    loaded,
                    lambda offer: (1, 2, int(offer.features_a[0] * 25) + 5,
                                   int(offer.features_b[0] * 25) + 5),
                    seeds=(17, 19), training_pairs=4,
                    heldout_pairs_per_generator=2, exploration_rate=0.4,
                    learning_rate=0.1, appraisal_gain=12.0,
                    artifact_root=temp, evaluator_id="toy-isolated-test",
                    allow_appraisal_training=True, allow_persistent_export=True,
                )
            self.assertEqual(len(result["seed_reports"]), 2)
            self.assertEqual(result["predeclared_protocol"]["suite"], "novel_v2")
            self.assertEqual(len(result["predeclared_protocol"]["primary_ood_generators"]), 3)
            self.assertEqual(result["predeclared_protocol"]["positive_control_generator"],
                             "sealed_eval_00_v2")
            self.assertEqual(result["primary_ood_trained_on_minus_off"]["n_seeds"], 2)
            self.assertTrue(result["pretrained_model_unchanged"])
            self.assertEqual(result["total_on_vs_off_sampled_action_switches"],
                             sum(s["sampled_action_switches_on_vs_off"]
                                 for s in result["seed_reports"]))
            run = Path(result["artifact_run_directory"])
            self.assertTrue(run.is_dir())
            record = (run / "study-report.json").read_bytes()
            self.assertEqual(record[-1:], b"\n")
            self.assertEqual((run / "COMPLETE.sha256").read_text(),
                             sha256(record).hexdigest() + "  study-report.json\n")
            self.assertEqual(json.loads(record)["seed_reports"],
                             json.loads(json.dumps(result["seed_reports"])))
            self.assertEqual(len(list(run.glob("seed-*-calibrated-head-v1"))), 2)
            for row in result["seed_reports"]:
                self.assertEqual(row["train_updates"], 4)
                self.assertEqual(row["shuffled_updates"], 4)
                self.assertTrue(row["globally_sealed_before_eval_feedback"])
                self.assertTrue(Path(row["saved_head_directory"]).is_dir())
                for arm in module.ARMS:
                    self.assertEqual(len(row["eval_arms"][arm]["rows"]), 8)
            self.assertEqual(list(Path(temp).glob(".pending-*")), [])

    def test_run_failure_keeps_no_final_artifacts(self):
        from pleasure_experiment import open_weight_calibrated_benchmark as module
        loaded = self._toy_loaded()
        with TemporaryDirectory() as temp:
            with mock.patch.object(module, "_predeclared", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "injected test tokenizer"):
                    module.run_calibrated_appraisal_benchmark(
                        loaded,
                        lambda offer: (_ for _ in ()).throw(
                            RuntimeError("injected test tokenizer")
                        ),
                        seeds=(17, 19), training_pairs=4,
                        heldout_pairs_per_generator=2, exploration_rate=0.4,
                        learning_rate=0.1, appraisal_gain=12.0,
                        artifact_root=temp, evaluator_id="toy-isolated-failure",
                        allow_appraisal_training=True, allow_persistent_export=True,
                    )
            self.assertEqual(list(Path(temp).iterdir()), [])
