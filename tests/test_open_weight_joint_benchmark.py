"""Matched joint LoRA study: predeclaration, full tiny-model seal, complete custody."""

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
HAS_PEFT = importlib.util.find_spec("peft") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def _toy_model(runtime_instance_id: str):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from pleasure_experiment.open_weight_adapter import OpenWeightBinding
    from pleasure_experiment.open_weight_local_model import (
        LoadedLocalCausalModel, LocalModelManifest,
    )

    config = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, max_position_embeddings=64,
        attention_dropout=0.0, rms_norm_eps=1e-6,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(918273)
        model = LlamaForCausalLM(config)
    return LoadedLocalCausalModel(
        model=model, tokenizer=SimpleNamespace(),
        action_token_ids=(3, 4),
        binding=OpenWeightBinding(
            model_id="tiny/llama-fixture", model_revision="fixed-toy",
            runtime_instance_id=runtime_instance_id,
            checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
        ),
        manifest=LocalModelManifest(
            snapshot_directory="/tmp/fixture-only", files=(),
            checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
            snapshot_sha256="c" * 64, total_bytes=123,
        ),
        action_prefix="Answer:", action_suffixes=(" A", " B"),
    )


@unittest.skipUnless(HAS_TORCH and HAS_PEFT and HAS_TRANSFORMERS and HAS_SAFETENSORS,
                     "optional PyTorch/PEFT/Transformers/safetensors missing")
class JointBenchmarkTests(unittest.TestCase):
    def test_predeclared_parameters_cannot_be_optimized_after_eval(self):
        from pleasure_experiment.open_weight_joint_benchmark import _predeclared, SEEDS
        _predeclared(SEEDS, 8, 4, .4, .05, 12.0)
        with self.assertRaisesRegex(ValueError, "preregistered"):
            _predeclared(SEEDS[:-1], 8, 4, .4, .05, 12.0)
        with self.assertRaisesRegex(ValueError, "preregistered"):
            _predeclared(SEEDS, 8, 4, .4, .05, 16.0)

    def test_complete_two_seed_toy_run_and_global_seal(self):
        from pleasure_experiment import open_weight_joint_benchmark as module

        loaded = _toy_model("tiny-shared-frozen-policy")
        def encode(offer):
            return (1, 2, 5 + int(offer.features_a[0] * 22),
                    5 + int(offer.features_b[1] * 22))

        with TemporaryDirectory() as directory:
            # This deliberately short toy fixture is NOT a scientific v3 run.
            with mock.patch.object(module, "_predeclared", return_value=None), \
                 mock.patch.object(module, "SEEDS", (17, 19)):
                report = module.run_joint_lora_appraisal_benchmark(
                    loaded, encode, _toy_model,
                    seeds=(17, 19), training_pairs_per_family=2,
                    heldout_pairs_per_generator=2,
                    exploration_rate=.4, learning_rate=.05, appraisal_gain=12.0,
                    artifact_root=directory, evaluator_id="tiny-v3-process",
                    allow_joint_training=True, allow_persistent_export=True,
                )
            self.assertEqual(len(report["seed_reports"]), 2)
            self.assertEqual(set(report["co_primary_contrasts"]), {
                "joint_on_minus_off", "joint_on_minus_shuffled",
                "joint_on_minus_head_only",
            })
            self.assertEqual(set(report["primary_all_contrasts_by_latent_family"]),
                             set(report["co_primary_contrasts"]))
            for family_contrasts in report["primary_all_contrasts_by_latent_family"].values():
                self.assertEqual(set(family_contrasts), set(module.V3_WITHIN_FAMILIES))
            self.assertEqual(report["predeclared_protocol"]["train_families"],
                             module.V3_TRAIN_FAMILIES)
            self.assertEqual(report["predeclared_protocol"]["secondary_ood_families"],
                             module.V3_OOD_FAMILIES)
            self.assertTrue(report["pretrained_model_unchanged"])
            run = Path(report["artifact_run_directory"])
            self.assertTrue(run.is_dir())
            full = (run / "study-report.json").read_bytes()
            self.assertEqual((run / "COMPLETE.sha256").read_text(),
                             sha256(full).hexdigest() + "  study-report.json\n")
            self.assertEqual(json.loads(full)["evidence_class"], module.EVIDENCE_CLASS)
            for seed_row in report["seed_reports"]:
                self.assertEqual(seed_row["joint_train_updates"], 6)
                self.assertEqual(seed_row["shuffled_train_updates"], 6)
                self.assertEqual(seed_row["head_only_train_updates"], 6)
                self.assertTrue(seed_row["global_all_arm_decision_barrier_verified"])
                self.assertTrue(seed_row["frozen_off_exact_reference_policy_equality_verified"])
                self.assertTrue(seed_row["joint_on_vs_inverted_appraisal_sign_sanity"])
                self.assertEqual(seed_row["lora_initialization_rng_seed"],
                                 seed_row["seed"] ^ module.LORA_INIT_SEED_XOR_SALT)
                self.assertEqual(len(seed_row["train"]), 6)
                self.assertEqual(len(seed_row["random_feedback_derangement"]), 6)
                self.assertEqual(len(set(seed_row["random_feedback_derangement"])), 6)
                self.assertNotEqual(seed_row["trained_lora_sha256"],
                                    seed_row["initial_lora_sha256"])
                self.assertEqual(seed_row["trained_joint_sha256"],
                                 seed_row["restored_joint_sha256"])
                self.assertTrue(Path(seed_row["saved_joint_head_directory"]).is_dir())
                for arm in module.ARMS:
                    self.assertEqual(len(seed_row["eval_arms"][arm]["rows"]), 10)
                    metrics = seed_row["eval_arms"][arm]
                    self.assertGreaterEqual(metrics["candidate_quality_mae"], 0)
                    self.assertGreaterEqual(metrics["candidate_quality_mse"], 0)
                    self.assertGreaterEqual(metrics["chosen_quality_mae"], 0)
                    self.assertGreaterEqual(metrics["chosen_quality_mse"], 0)
                    self.assertGreaterEqual(metrics["mean_absolute_prediction_spread"], 0)
            self.assertEqual(list(Path(directory).glob(".pending-*")), [])

    def test_uncommitted_failed_study_leaves_no_artifact(self):
        from pleasure_experiment import open_weight_joint_benchmark as module
        loaded = _toy_model("tiny-original-frozen")
        with TemporaryDirectory() as directory:
            with mock.patch.object(module, "_predeclared", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "bad tokenizer"):
                    module.run_joint_lora_appraisal_benchmark(
                        loaded,
                        lambda _offer: (_ for _ in ()).throw(RuntimeError("bad tokenizer")),
                        _toy_model, seeds=(17, 19), training_pairs_per_family=2,
                        heldout_pairs_per_generator=2,
                        exploration_rate=.4, learning_rate=.05, appraisal_gain=12.0,
                        artifact_root=directory, evaluator_id="toy-v3-failure",
                        allow_joint_training=True, allow_persistent_export=True,
                    )
            failures = list(Path(directory).iterdir())
            self.assertEqual(len(failures), 1)
            self.assertIn(".FAILED-", failures[0].name)
            self.assertEqual(json.loads((failures[0] / "FAILED.json").read_text())["stage"],
                             "shared_frozen_policy_train_and_receipts")
            self.assertFalse((failures[0] / "COMPLETE.sha256").exists())
            self.assertFalse((failures[0] / "study-report.json").exists())
