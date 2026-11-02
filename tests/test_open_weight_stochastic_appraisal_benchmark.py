"""End-to-end stochastic appraisal study on a tiny, read-only Torch language model."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_OPTIONALS = (importlib.util.find_spec("torch") is not None and
                 importlib.util.find_spec("safetensors") is not None)


def fixture():
    import torch
    from pleasure_experiment.open_weight_adapter import OpenWeightBinding
    from pleasure_experiment.open_weight_local_model import LocalModelManifest

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)
            self.out = torch.nn.Linear(2, 8)
            with torch.no_grad():
                self.embedding.weight.zero_()
                self.embedding.weight[3] = torch.tensor([1.0, 0.0])
                self.embedding.weight[4] = torch.tensor([0.0, 1.0])
                self.out.weight.zero_()
                self.out.bias.zero_()
                self.out.bias[3] = 0.1
            self.config = SimpleNamespace(hidden_size=2)

        def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
            hidden = self.embedding(input_ids)
            return SimpleNamespace(
                logits=self.out(hidden),
                hidden_states=(hidden,) if output_hidden_states else None,
            )

    binding = OpenWeightBinding(
        model_id="tiny/study", model_revision="test-pinned",
        runtime_instance_id="tiny-runtime", checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
    )
    manifest = LocalModelManifest(
        snapshot_directory="/tmp/verified-toy", files=(),
        checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
        snapshot_sha256="c" * 64, total_bytes=100,
    )
    return SimpleNamespace(
        model=TinyLM(), binding=binding, manifest=manifest,
        action_token_ids=(3, 4),
    )


@unittest.skipUnless(HAS_OPTIONALS, "optional torch/safetensors unavailable")
class StochasticAppraisalStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.loaded = fixture()

    def _run(self, **kwargs):
        from pleasure_experiment.open_weight_stochastic_appraisal_benchmark import (
            run_stochastic_appraisal_benchmark,
        )
        options = dict(
            seeds=(11, 23), training_pairs=4,
            heldout_pairs_per_generator=2, exploration_rate=0.6,
            learning_rate=0.4, appraisal_gain=8.0,
            artifact_root=self.temp.name, evaluator_id="test-source-evaluator",
            allow_appraisal_training=True, allow_persistent_export=True,
        )
        options.update(kwargs)
        return run_stochastic_appraisal_benchmark(
            self.loaded, lambda offer: (1, 2), **options,
        )

    def test_all_five_arms_and_logged_stochastic_choices_persist(self):
        from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal
        from pleasure_experiment.open_weight_appraisal_persistence import (
            load_versioned_appraisal_head,
        )
        from pleasure_experiment.open_weight_stochastic import (
            StochasticAction, validate_stochastic_trace,
        )
        from pleasure_experiment.withheld_payoffs import EVAL_FAMILIES

        baseline = tuple(p.detach().clone() for p in self.loaded.model.parameters())
        report = self._run()
        self.assertEqual(len(report["seed_reports"]), 2)
        self.assertEqual(sorted(p.name for p in Path(self.temp.name).iterdir()),
                         ["seed-11-trained-head-v1", "seed-23-trained-head-v1"])
        self.assertEqual(set(report["seed_reports"][0]["eval_arms"]), {
            "trained_on", "trained_off", "trained_inverted", "shuffled_on", "frozen_on",
        })
        for seed_report in report["seed_reports"]:
            self.assertNotEqual(seed_report["initial_head_sha256"],
                                seed_report["learned_head_sha256"])
            self.assertEqual(seed_report["learned_head_sha256"],
                             seed_report["restored_head_sha256"])
            source = Path(seed_report["saved_head_directory"])
            self.assertTrue((source / "head.safetensors").is_file())
            self.assertTrue((source / "manifest.json").is_file())
            self.assertEqual(len(seed_report["train"]), 4)
            traces = tuple(StochasticAction(**row["stochastic_pre_feedback"])
                           for row in seed_report["train"])
            validate_stochastic_trace(traces)
            for row in seed_report["train"]:
                sampled = row["stochastic_pre_feedback"]
                self.assertIn(sampled["chosen_action"], ("A", "B"))
                self.assertGreater(sampled["chosen_propensity"], 0)
                self.assertEqual(row["updates"]["genuine"]["teaching_quality"],
                                 row["actual_chosen_quality"])
                self.assertNotEqual(row["updates"]["shuffled"]["teaching_evidence_id"],
                                    row["committed_chosen_evidence_id"])
            self.assertEqual(set(seed_report["eval_arms"]["trained_on"]["by_family"]),
                             set(EVAL_FAMILIES))
            for arm in seed_report["eval_arms"].values():
                self.assertEqual(len(arm["rows"]), 6)
                self.assertTrue(all(0 <= row["expected_policy_quality_posthoc"] <= 1
                                    for row in arm["rows"]))
                for row in arm["rows"]:
                    trace = row["stochastic_pre_feedback"]
                    recomputed = (trace["behavior_prob_a"] * row["quality_a_posthoc"] +
                                  trace["behavior_prob_b"] * row["quality_b_posthoc"])
                    self.assertAlmostEqual(row["expected_policy_quality_posthoc"], recomputed,
                                           places=12)
            primary = seed_report["eval_arms"]["trained_on"]["rows"]
            for arm in seed_report["eval_arms"].values():
                self.assertEqual([r["stochastic_pre_feedback"]["random_draw"]
                                  for r in primary],
                                 [r["stochastic_pre_feedback"]["random_draw"]
                                  for r in arm["rows"]])
            off = seed_report["eval_arms"]["trained_off"]["rows"]
            frozen = seed_report["eval_arms"]["frozen_on"]["rows"]
            self.assertEqual([r["stochastic_pre_feedback"]["behavior_prob_a"] for r in off],
                             [r["stochastic_pre_feedback"]["behavior_prob_a"] for r in frozen])
            self.assertEqual(
                [row["episode_id"] for row in
                 seed_report["eval_arms"]["trained_on"]["rows"]],
                [row["episode_id"] for row in
                 seed_report["eval_arms"]["shuffled_on"]["rows"]],
            )
            import_probe = TorchHiddenAppraisal(
                self.loaded.model, action_token_ids=(3, 4), hidden_size=2,
                learning_rate=0.0, allow_training=False,
            )
            imported = load_versioned_appraisal_head(
                import_probe, binding=self.loaded.binding,
                snapshot_manifest=self.loaded.manifest,
                source_directory=source, allow_in_place_mutation=True,
            )
            self.assertEqual(imported.head_weights_sha256,
                             seed_report["learned_head_sha256"])
        self.assertEqual(report["predeclared_protocol"]["primary_metric"],
                         "seed-paired_posthoc_exact_expected_behavior_policy_quality")
        import torch
        for original, after in zip(baseline, self.loaded.model.parameters()):
            self.assertTrue(torch.equal(original, after))

    def test_training_or_export_must_be_explicitly_authorized(self):
        with self.assertRaises(PermissionError):
            self._run(allow_appraisal_training=False)
        with self.assertRaises(PermissionError):
            self._run(allow_persistent_export=False)

    def test_tampered_receipt_action_fails_before_training_or_persistence(self):
        from pleasure_experiment.withheld_payoffs import WithheldPayoffArena
        original = WithheldPayoffArena.resolve

        def mutated(arena, ticket):
            receipt = original(arena, ticket)
            if receipt.partition == "TRAIN":
                return replace(receipt, action="B" if receipt.action == "A" else "A")
            return receipt

        with mock.patch.object(WithheldPayoffArena, "resolve", mutated):
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                self._run()
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_no_counterfactual_before_every_arm_has_committed_every_offer(self):
        from pleasure_experiment.withheld_payoffs import WithheldPayoffArena

        commits = []
        original_commit = WithheldPayoffArena.commit
        original_resolve = WithheldPayoffArena.resolve
        original_posthoc = WithheldPayoffArena.posthoc_evaluate

        def observe_commit(arena, offer, *args, **kwargs):
            ticket = original_commit(arena, offer, *args, **kwargs)
            if offer.partition == "EVAL":
                commits.append((offer.episode_id, ticket.action))
            return ticket

        def guarded_posthoc(arena, offer):
            # Two seeds total, but the first seed must finish all five arms
            # and all three generators (two offers each) before any reveal.
            self.assertIn(len(commits), (30, 60))
            return original_posthoc(arena, offer)

        def guarded_resolve(arena, ticket):
            # An EVAL chosen receipt is as sensitive as a counterfactual:
            # different matched arms can expose both actions' outcomes.
            if ticket.episode_id in {id for id, _ in commits}:
                self.assertIn(len(commits), (30, 60))
            return original_resolve(arena, ticket)

        with mock.patch.object(WithheldPayoffArena, "commit", observe_commit), \
                mock.patch.object(WithheldPayoffArena, "resolve", guarded_resolve), \
                mock.patch.object(WithheldPayoffArena, "posthoc_evaluate", guarded_posthoc):
            report = self._run()
        self.assertEqual(len(commits), 60)
        self.assertEqual(len(report["seed_reports"]), 2)

    def test_failed_posthoc_does_not_publish_partial_trained_heads(self):
        from pleasure_experiment.withheld_payoffs import WithheldPayoffArena

        with mock.patch.object(WithheldPayoffArena, "posthoc_evaluate",
                               side_effect=RuntimeError("injected evaluator failure")):
            with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
                self._run()
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])


class CLIUnitTests(unittest.TestCase):
    def test_missing_flags_rejected_before_local_checkpoint_loading(self):
        from scripts.run_open_weight_stochastic_appraisal_benchmark import (
            build_parser, run,
        )
        options = build_parser().parse_args([
            "--snapshot", "/invalid", "--model-id", "tiny/study",
            "--revision", "a" * 40, "--save-head-root", "/tmp",
        ])
        with self.assertRaises(PermissionError):
            run(options)
