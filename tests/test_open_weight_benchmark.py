"""Matched multi-seed adverse-control benchmark contract tests."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from unittest import TestCase, skipUnless
import json

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, OpenWeightChoice, OpenWeightRestore, OpenWeightUpdate,
    TorchChosenOutcomeAdapter,
)
from pleasure_experiment.open_weight_benchmark import (
    EVIDENCE_CLASS, run_multiseed_open_weight_benchmark,
)
from pleasure_experiment.outcome_arena import OutcomeArena


def _sha(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


_BINDING = OpenWeightBinding(
    "benchmark-test-double", "test-r1", "benchmark-001",
    _sha("checkpoint"), _sha("tokenizer"),
)


class _FakePolicy:
    """A hash-changing fake policy, NOT evidence of real LLM optimization."""

    def __init__(self, rate: float, *, baseline: str = "a" * 64):
        self.rate = rate
        self.original = baseline
        self.digest = baseline
        self._step = 0
        self.pending_choice = None
        self.closed = False
        self.infer_partitions = []
        self.received_outcomes = []

    @property
    def binding(self):
        return _BINDING

    @property
    def weights_digest(self):
        return self.digest

    @property
    def frozen_backbone_digest(self):
        return "b" * 64

    @property
    def step_index(self):
        return self._step

    @property
    def pending(self):
        return self.pending_choice is not None

    def infer(self, *, input_ids, offer_nonce, episode_id,
              source_partition, score_bias=(0.0, 0.0)):
        if self.closed or self.pending:
            raise RuntimeError("closed or pending")
        a = 0.6 if self.digest == self.original else 0.73
        b = 1 - a
        action = "A"
        evidence = _sha(f"{self.digest}:{self._step}:{offer_nonce}:{episode_id}")
        choice = OpenWeightChoice(
            binding=self.binding, episode_id=episode_id,
            source_partition=source_partition, chosen_action=action,
            decision_digest=evidence, weights_digest=self.digest,
            frozen_backbone_digest=self.frozen_backbone_digest,
            input_ids_digest=_sha(json.dumps(list(input_ids), separators=(",", ":"))),
            offer_nonce_digest=_sha(offer_nonce), session_nonce_digest="c" * 64,
            prob_a=a, prob_b=b, score_a=a, score_b=b, step_index=self._step,
        )
        if source_partition == "TRAIN":
            self.pending_choice = choice
        self.infer_partitions.append(source_partition)
        return choice

    def observe(self, outcome):
        choice = self.pending_choice
        if choice is None or outcome.decision_digest != choice.decision_digest:
            raise RuntimeError("bad pending outcome")
        before = self.digest
        changed = self.rate > 0 and outcome.measured_quality != 0.5
        if changed:
            self.digest = _sha(f"{self.digest}:{outcome.measured_quality}:{self._step}")
        self._step += 1
        self.pending_choice = None
        self.received_outcomes.append(outcome)
        return OpenWeightUpdate(
            decision_digest=choice.decision_digest, evidence_id=outcome.evidence_id,
            chosen_action=choice.chosen_action,
            measured_quality=outcome.measured_quality,
            prediction_before=choice.prob_a,
            before_digest=before, after_digest=self.digest,
            frozen_backbone_digest=self.frozen_backbone_digest,
            step_index=self._step,
            gradient_norm=0.1 if changed else 0.0,
            delta_l2=0.1 if changed else 0.0,
            optimizer_applied=changed, optimizer_step_executed=changed,
        )

    def restore(self):
        before = self.digest
        self.digest = self.original
        self.pending_choice = None
        self.closed = True
        return OpenWeightRestore(
            before_digest=before, restored_digest=self.original,
            frozen_backbone_digest=self.frozen_backbone_digest,
            optimizer_state_cleared=True, cancelled_pending=False, closed=True,
        )


def _encode(offer):
    return (11, 4 + offer.side, 32 + offer.pair_index)


class MultiSeedOpenWeightBenchmarkTests(TestCase):
    def test_three_matched_arms_per_seed_and_explicit_synthetic_feedback(self):
        created = []

        def factory(rate, seed, arm):
            value = _FakePolicy(rate)
            created.append((seed, arm, value))
            return value

        results = run_multiseed_open_weight_benchmark(
            adapter_factory=factory, encode_offer=_encode,
            seeds=(7, 19, 42), training_pairs=5, heldout_pairs=4,
            learning_rate=0.2,
        )
        self.assertEqual(len(created), 9)
        self.assertTrue(all(item.closed for _, _, item in created))
        self.assertEqual(results.evidence_class, EVIDENCE_CLASS)
        self.assertFalse(results.externally_authenticated)
        self.assertFalse(results.hosted_model_modified)
        self.assertEqual(results.phenomenology, "UNRESOLVED")
        self.assertTrue(results.all_arms_restored)
        self.assertTrue(results.all_baselines_and_schedules_matched)
        self.assertEqual(results.treatment_vs_frozen_quality.number_of_seeds, 3)
        self.assertIsNotNone(results.treatment_vs_frozen_quality.approximate_normal_95_interval)
        self.assertFalse(results.treatment_vs_frozen_quality.statistically_significant_claimed)
        for seed_report in results.seed_reports:
            arms = (seed_report.frozen, seed_report.outcome_weighted,
                    seed_report.shuffled_feedback)
            self.assertTrue(seed_report.heldout_family_disjoint)
            self.assertTrue(seed_report.input_exposure_matched)
            self.assertTrue(seed_report.train_and_heldout_schedules_matched)
            self.assertTrue(seed_report.control_baselines_matched)
            self.assertEqual({arm.initial_weights_sha256 for arm in arms}, {"a" * 64})
            self.assertEqual({arm.restored_weights_sha256 for arm in arms}, {"a" * 64})
            self.assertEqual({arm.frozen_backbone_sha256 for arm in arms}, {"b" * 64})
            self.assertEqual(seed_report.frozen.optimizer_steps, 0)
            self.assertEqual(seed_report.outcome_weighted.optimizer_steps, 5)
            self.assertEqual(seed_report.shuffled_feedback.optimizer_steps, 5)
            self.assertEqual(len(seed_report.frozen.heldout_primary), 4)
            self.assertEqual(len(seed_report.frozen.heldout_mirrored_diagnostic), 8)
            self.assertEqual(
                {x.episode_id for x in seed_report.frozen.training} &
                {x.episode_id for x in seed_report.frozen.heldout_mirrored_diagnostic}, set()
            )
            control_donors = {x.committed_evidence_id: x for x in seed_report.frozen.training}
            for event in seed_report.shuffled_feedback.training:
                self.assertIsNotNone(event.donor_evidence_id)
                self.assertIn(event.donor_evidence_id, control_donors)
                self.assertNotEqual(event.episode_id, control_donors[event.donor_evidence_id].episode_id)
                self.assertEqual(event.teaching_quality,
                                 control_donors[event.donor_evidence_id].actual_measured_quality)
                self.assertNotEqual(event.teaching_evidence_id, event.committed_evidence_id)
            self.assertTrue(all(x.donor_evidence_id is None
                                for x in seed_report.outcome_weighted.training))

    def test_learned_capture_runs_before_treatment_restoration(self):
        created = []
        captured = []

        def factory(rate, seed, arm):
            value = _FakePolicy(rate)
            created.append((seed, arm, value))
            return value

        def capture(seed, arm, learned):
            value = next(v for s, a, v in created if s == seed and a == arm)
            self.assertFalse(value.closed)
            self.assertEqual(value.weights_digest, learned)
            self.assertNotEqual(value.original, learned)
            captured.append((seed, arm))

        result = run_multiseed_open_weight_benchmark(
            adapter_factory=factory, encode_offer=_encode,
            seeds=(4, 8), training_pairs=2, heldout_pairs=2,
            learning_rate=0.4, on_learned_weights=capture,
        )
        self.assertEqual(captured, [(4, "outcome_weighted"), (8, "outcome_weighted")])
        self.assertTrue(all(value.closed for _, _, value in created))
        self.assertTrue(all(seed.outcome_weighted.learned_callback_invoked
                            for seed in result.seed_reports))

    def test_invalid_parameters_rejected_before_constructing_mutating_model(self):
        def forbidden(*args):
            self.fail("invalid request must not construct an adapter")

        cases = (
            {"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (True,)},
            {"seeds": (-1,)}, {"seeds": (2**32,)},
            {"training_pairs": 1}, {"heldout_pairs": 0},
            {"learning_rate": 0}, {"learning_rate": float("nan")},
            {"evaluator_id": ""},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                options = dict(seeds=(12,), training_pairs=3,
                               heldout_pairs=2, learning_rate=0.3)
                options.update(case)
                run_multiseed_open_weight_benchmark(
                    adapter_factory=forbidden, encode_offer=_encode, **options,
                )

    def test_baseline_mismatch_aborts_and_closes_new_adapter(self):
        created = []

        def factory(rate, seed, arm):
            value = _FakePolicy(rate, baseline="a" * 64 if arm == "frozen" else "f" * 64)
            created.append(value)
            return value

        with self.assertRaisesRegex(RuntimeError, "matched binding/tensors"):
            run_multiseed_open_weight_benchmark(
                adapter_factory=factory, encode_offer=_encode,
                seeds=(23,), training_pairs=3, heldout_pairs=1, learning_rate=0.2,
            )
        self.assertEqual(len(created), 2)
        self.assertTrue(all(x.closed for x in created))

    def test_capture_failure_aborts_and_restores_trainable_state(self):
        created = []

        def factory(rate, seed, arm):
            value = _FakePolicy(rate)
            created.append(value)
            return value

        def broken_capture(seed, arm, digest):
            raise OSError("disk is full")

        with self.assertRaisesRegex(RuntimeError, "disk is full"):
            run_multiseed_open_weight_benchmark(
                adapter_factory=factory, encode_offer=_encode,
                seeds=(31,), training_pairs=3, heldout_pairs=1,
                learning_rate=0.2, on_learned_weights=broken_capture,
            )
        self.assertTrue(all(x.closed and x.weights_digest == x.original for x in created))

    def test_capture_cannot_modify_weights_during_persistence(self):
        created = []

        def factory(rate, seed, arm):
            value = _FakePolicy(rate)
            created.append((arm, value))
            return value

        def corrupt_capture(seed, arm, learned):
            target = next(item for label, item in created if label == arm)
            target.digest = _sha("malicious captured tensor edit")

        with self.assertRaisesRegex(RuntimeError, "capture mutated live model state"):
            run_multiseed_open_weight_benchmark(
                adapter_factory=factory, encode_offer=_encode,
                seeds=(41,), training_pairs=3, heldout_pairs=1,
                learning_rate=0.2, on_learned_weights=corrupt_capture,
            )
        self.assertTrue(all(item.closed and item.weights_digest == item.original
                            for _, item in created))


@skipUnless(importlib.util.find_spec("torch") is not None,
            "optional PyTorch unavailable")
class RealTorchMultiSeedBenchmarkTests(TestCase):
    def test_three_arm_real_autograd_updates_unseen_tasks_and_restores(self):
        import torch

        class TinyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Embedding(8, 2)
                self.policy = torch.nn.Linear(2, 2)
                with torch.no_grad():
                    self.backbone.weight.fill_(0.25)
                    self.policy.weight.zero_()
                    self.policy.bias.zero_()

            def forward(self, *, input_ids):
                return self.policy(self.backbone(input_ids))

        created = []

        def factory(rate, seed, arm):
            model = TinyCausalLM()
            rubric = OutcomeArena(seed=seed, evaluator_id="source-local-open-weight-oracle").rubric_digest
            adapter = TorchChosenOutcomeAdapter(
                model, binding=OpenWeightBinding(
                    "tiny-multiseed-model", "fake-model-revision", "offline-001",
                    _sha("tiny-checkpoint"), _sha("tiny-tokenizer"),
                ),
                action_token_ids=(0, 1),
                trainable_parameter_names=("policy.weight", "policy.bias"),
                learning_rate=rate,
                expected_evaluator_id="source-local-open-weight-oracle",
                expected_rubric_digest=rubric,
                allow_in_place_mutation=True,
                max_steps=3, max_delta_l2=1.0,
            )
            created.append((seed, arm, adapter))
            return adapter

        report = run_multiseed_open_weight_benchmark(
            adapter_factory=factory,
            encode_offer=lambda offer: (1 + offer.side, 2 + offer.pair_index % 4),
            seeds=(11, 22, 33), training_pairs=3, heldout_pairs=2,
            learning_rate=0.2,
        )
        self.assertEqual(len(created), 9)
        for entry in report.seed_reports:
            self.assertEqual(entry.frozen.model_backend,
                             "SOURCE_LOCAL_TORCH_ADAPTER_UNAUTHENTICATED")
            self.assertEqual(entry.outcome_weighted.optimizer_steps, 3)
            self.assertEqual(entry.shuffled_feedback.optimizer_steps, 3)
            self.assertNotEqual(entry.outcome_weighted.learned_weights_sha256,
                                entry.outcome_weighted.initial_weights_sha256)
            self.assertEqual(entry.frozen.learned_weights_sha256,
                             entry.frozen.initial_weights_sha256)
            self.assertEqual(entry.outcome_weighted.restored_weights_sha256,
                             entry.outcome_weighted.initial_weights_sha256)
        self.assertTrue(report.all_arms_restored)
        self.assertTrue(all(adapter.weights_digest == entry.frozen.initial_weights_sha256
                            for entry in report.seed_reports
                            for _, _, adapter in created if adapter.binding == entry.frozen.binding))
