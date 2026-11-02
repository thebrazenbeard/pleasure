"""Fixed-parameter, explicit-consent wrapper for source-local joint training."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest import mock

from scripts import run_open_weight_joint_appraisal_benchmark as cli


def _args(*extra: str):
    return cli.build_parser().parse_args([
        "--snapshot", "/tmp/pinned-source-only",
        "--save-artifact-root", "/tmp/private-new-artifacts", *extra,
    ])


class JointBenchmarkCLITests(unittest.TestCase):
    def test_frozen_defaults_and_help(self):
        args = _args()
        self.assertEqual(tuple(args.seeds), cli.SEEDS)
        self.assertEqual(args.model_id, cli.MODEL_ID)
        self.assertEqual(args.revision, cli.REVISION)
        self.assertEqual(args.training_pairs_per_family, 8)
        self.assertEqual(args.heldout_pairs_per_generator, 4)
        self.assertEqual(args.learning_rate, 0.05)
        self.assertEqual(args.cpu_threads, 2)
        self.assertFalse(args.allow_joint_training)
        self.assertFalse(args.allow_persistent_export)

    def test_explicit_consent_is_required_before_checkpoint_loading(self):
        with mock.patch.object(cli, "load_verified_local_causal_model") as load:
            with self.assertRaises(PermissionError):
                cli.run(_args())
            with self.assertRaises(PermissionError):
                cli.run(_args("--allow-joint-training"))
            load.assert_not_called()

    def test_changed_hyperparameters_or_model_identity_fail_closed(self):
        for change in (
            ("--seeds", "163", "167"),
            ("--appraisal-gain", "8"),
            ("--training-pairs-per-family", "16"),
            ("--heldout-pairs-per-generator", "8"),
            ("--cpu-threads", "8"),
            ("--model-id", "unverified/model"),
            ("--revision", "0" * 40),
        ):
            with self.subTest(change=change):
                with mock.patch.object(cli, "load_verified_local_causal_model") as load:
                    with self.assertRaises(ValueError):
                        cli.run(_args("--allow-joint-training",
                                      "--allow-persistent-export", *change))
                    load.assert_not_called()

    def test_verified_local_model_factory_and_tokenizer_wiring(self):
        # This is a standard-library wiring test, also run on CI hosts that
        # intentionally do not install the optional PyTorch training stack.
        fake_torch = ModuleType("torch")
        fake_torch.set_num_threads = lambda _count: None

        correct_record = SimpleNamespace(
            name="model.safetensors", sha256=cli.MODEL_SAFETENSORS_SHA256,
        )
        tokenizer = SimpleNamespace(encode=lambda prefix, add_special_tokens=False: [8, 9])
        loaded = SimpleNamespace(
            model=object(), tokenizer=tokenizer,
            action_suffixes=(" A", " B"), action_token_ids=(330, 389),
            manifest=SimpleNamespace(files=(correct_record,)),
        )
        called = []

        def fake_load(path, **kwargs):
            called.append(kwargs["runtime_instance_id"])
            return loaded

        def fake_benchmark(first, encode, reload_model, **kwargs):
            self.assertIs(first, loaded)
            self.assertEqual(kwargs["seeds"], cli.SEEDS)
            self.assertEqual(kwargs["training_pairs_per_family"], 8)
            self.assertTrue(kwargs["allow_joint_training"])
            self.assertEqual(encode(SimpleNamespace()), (8, 9))
            self.assertIs(reload_model("fresh-arm-process"), loaded)
            return {"result": "source-only-stub"}

        benchmark_module = ModuleType("pleasure_experiment.open_weight_joint_benchmark")
        benchmark_module.run_joint_lora_appraisal_benchmark = fake_benchmark
        with (mock.patch.object(cli, "load_verified_local_causal_model", side_effect=fake_load),
              mock.patch.object(cli, "public_prompt", return_value="prompt:"),
              mock.patch.object(cli, "validate_single_token_actions", return_value=(330, 389)),
              mock.patch.dict(sys.modules, {
                  benchmark_module.__name__: benchmark_module,
                  "torch": fake_torch,
              })):
            outcome = cli.run(_args("--allow-joint-training", "--allow-persistent-export"))
        self.assertEqual(outcome, {"result": "source-only-stub"})
        self.assertEqual(called, ["offline-joint-appraisal-v3-20261102", "fresh-arm-process"])

    def test_wrong_snapshot_digest_rejected_before_runner(self):
        wrong = SimpleNamespace(
            manifest=SimpleNamespace(files=(SimpleNamespace(
                name="model.safetensors", sha256="0" * 64,
            ),)),
        )
        fake_torch = ModuleType("torch")
        fake_torch.set_num_threads = lambda _count: None
        with (mock.patch.dict(sys.modules, {"torch": fake_torch}),
              mock.patch.object(cli, "load_verified_local_causal_model", return_value=wrong)):
            with self.assertRaisesRegex(ValueError, "exact pinned safetensors"):
                cli.run(_args("--allow-joint-training", "--allow-persistent-export"))

    def test_cli_compact_result_does_not_dump_full_episode_evidence(self):
        with mock.patch.object(cli, "run", return_value={
            "artifact_run_directory": "/tmp/stub-complete",
            "evidence_class": "source-only",
            "predeclared_protocol": {"seeds": cli.SEEDS},
            "co_primary_contrasts": {"joint_on_minus_off": {"mean": -0.01}},
            "exploratory_positive_local_criterion_met": False,
            "seed_reports": [{"large": "not printed"}],
        }):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli.main([
                    "--snapshot", "/tmp/a", "--save-artifact-root", "/tmp/b",
                ]), 0)
        result = json.loads(output.getvalue())
        self.assertNotIn("seed_reports", result)
        self.assertEqual(result["artifact_run_directory"], "/tmp/stub-complete")
        self.assertEqual(result["contrasts"]["joint_on_minus_off"]["mean"], -0.01)
        self.assertIs(result["exploratory_utility_criterion_met"], False)
        self.assertFalse(result["hosted_gpt_modified"])
        self.assertFalse(result["subjective_pleasure_proven"])


if __name__ == "__main__":
    unittest.main()
