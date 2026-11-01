"""Source-local matched open-weight treatment/control orchestration tests."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from unittest import TestCase, mock, skipUnless
import json

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, OpenWeightChoice, OpenWeightRestore, OpenWeightUpdate,
    TorchChosenOutcomeAdapter,
)
from pleasure_experiment.open_weight_controls import (
    run_matched_open_weight_controls,
)
from pleasure_experiment.outcome_arena import OutcomeArena


def _sha(text: str) -> str:
    return sha256(text.encode("ascii")).hexdigest()


_BINDING = OpenWeightBinding(
    "fake-matched-tokens", "source-r1", "test-runtime",
    _sha("declared-checkpoint"), _sha("declared-tokenizer"),
)


class _FakePolicy:
    """Hash-changing test double; intentionally never real neural evidence."""

    def __init__(
        self, learning_rate: float, *, base: str = "b" * 64,
        wrong_zero: bool = False,
    ) -> None:
        self._binding = _BINDING
        self.base = base
        self.digest = base
        self.learning_rate = learning_rate
        self.wrong_zero = wrong_zero
        self.count = 0
        self.pending_choice = None
        self.closed = False
        self.infer_partitions: list[str] = []

    @property
    def binding(self):
        return self._binding

    @property
    def weights_digest(self):
        return self.digest

    @property
    def frozen_backbone_digest(self):
        return "c" * 64

    @property
    def step_index(self):
        return self.count

    @property
    def pending(self):
        return self.pending_choice is not None

    def infer(self, *, input_ids, offer_nonce, episode_id, source_partition,
              score_bias=(0.0, 0.0)):
        if self.closed or self.pending:
            raise RuntimeError("fake not ready")
        a = 0.6 if input_ids[-1] % 2 == 0 else 0.4
        b = 1.0 - a
        action = "A" if a > b else "B"
        digest = _sha(json.dumps(
            [self.digest, self.count, offer_nonce, episode_id, action],
            separators=(",", ":"),
        ))
        choice = OpenWeightChoice(
            binding=self.binding, episode_id=episode_id,
            source_partition=source_partition, chosen_action=action,
            decision_digest=digest, weights_digest=self.digest,
            frozen_backbone_digest=self.frozen_backbone_digest,
            input_ids_digest=_sha(json.dumps(list(input_ids), separators=(",", ":"))),
            offer_nonce_digest=sha256(offer_nonce.encode()).hexdigest(),
            session_nonce_digest="d" * 64,
            prob_a=a, prob_b=b, score_a=a, score_b=b, step_index=self.count,
        )
        if source_partition == "TRAIN":
            self.pending_choice = choice
        self.infer_partitions.append(source_partition)
        return choice

    def observe(self, outcome):
        choice = self.pending_choice
        if choice is None or outcome.decision_digest != choice.decision_digest:
            raise RuntimeError("missing matching pending fake action")
        before = self.digest
        changed = self.learning_rate > 0 or self.wrong_zero
        if changed:
            self.digest = _sha(f"{before}:{outcome.measured_quality}:{self.count}")
        self.count += 1
        self.pending_choice = None
        return OpenWeightUpdate(
            decision_digest=choice.decision_digest, evidence_id=outcome.evidence_id,
            chosen_action=choice.chosen_action, measured_quality=outcome.measured_quality,
            prediction_before=(
                choice.prob_a if choice.chosen_action == "A" else choice.prob_b
            ),
            before_digest=before, after_digest=self.digest,
            frozen_backbone_digest=self.frozen_backbone_digest,
            step_index=self.count, gradient_norm=0.1 if changed else 0.0,
            delta_l2=0.1 if changed else 0.0,
            optimizer_applied=changed, optimizer_step_executed=changed,
        )

    def restore(self):
        before = self.digest
        self.digest = self.base
        self.pending_choice = None
        self.closed = True
        return OpenWeightRestore(
            before_digest=before, restored_digest=self.base,
            frozen_backbone_digest=self.frozen_backbone_digest,
            optimizer_state_cleared=True, cancelled_pending=False, closed=True,
        )


def _encode(offer):
    return (17, 3 + offer.side, 4 + offer.pair_index % 3)


class MatchedOpenWeightControlsTests(TestCase):
    def test_capture_callback_observes_treatment_only_before_both_restores(self):
        adapters = []
        captures = []

        def factory(lr):
            adapter = _FakePolicy(lr)
            adapters.append(adapter)
            return adapter

        def capture(digest):
            captures.append((digest, adapters[0].weights_digest,
                             adapters[0].closed, len(adapters)))

        report = run_matched_open_weight_controls(
            adapter_factory=factory, encode_offer=_encode,
            learning_rate=0.5, seed=11, training_pairs=2, heldout_pairs=2,
            on_learned_weights=capture,
        )
        self.assertEqual(captures, [(report.outcome_weighted.learned_weights_digest,
                                     report.outcome_weighted.learned_weights_digest,
                                     False, 1)])
        self.assertEqual(len(adapters), 2)
        self.assertTrue(all(adapter.closed for adapter in adapters))

    def test_matched_two_arm_source_local_contrast(self):
        adapters = []

        def factory(lr):
            adapter = _FakePolicy(lr)
            adapters.append(adapter)
            return adapter

        report = run_matched_open_weight_controls(
            adapter_factory=factory, encode_offer=_encode,
            learning_rate=0.5, seed=11, training_pairs=3, heldout_pairs=4,
        )
        self.assertEqual(len(adapters), 2)
        self.assertEqual([adapter.learning_rate for adapter in adapters], [0.5, 0.0])
        self.assertTrue(all(adapter.closed for adapter in adapters))
        self.assertTrue(report.train_schedule_matched)
        self.assertTrue(report.heldout_schedule_matched)
        self.assertTrue(report.input_exposure_matched)
        self.assertTrue(report.initial_tensor_digest_matched)
        self.assertTrue(report.frozen_backbone_digest_matched)
        self.assertEqual(report.outcome_weighted.prequential.episodes, 3)
        self.assertEqual(report.zero_learning_rate.heldout_primary.episodes, 4)
        self.assertEqual(report.outcome_weighted.optimizer_steps_reported, 3)
        self.assertEqual(report.zero_learning_rate.optimizer_steps_reported, 0)
        self.assertGreater(report.outcome_weighted.observed_weight_changes, 0)
        self.assertEqual(report.zero_learning_rate.observed_weight_changes, 0)
        self.assertEqual(report.backend_kind, "SOURCE_ONLY_TEST_DOUBLE_OR_UNKNOWN")
        self.assertFalse(report.observed_trainable_tensor_change)
        self.assertFalse(report.confirmed_pretrained_llm_training)
        self.assertFalse(report.hosted_model_modified)
        self.assertFalse(report.externally_authenticated)
        self.assertEqual(report.phenomenology, "UNRESOLVED")
        self.assertEqual(adapters[0].infer_partitions.count("TRAIN"), 3)
        self.assertEqual(adapters[1].infer_partitions.count("EVAL"), 8)

    def test_fails_closed_on_different_control_baseline_and_restores(self):
        adapters = []

        def factory(lr):
            adapter = _FakePolicy(lr, base="b" * 64 if lr else "f" * 64)
            adapters.append(adapter)
            return adapter

        with self.assertRaisesRegex(ValueError, "matched model baseline"):
            run_matched_open_weight_controls(
                adapter_factory=factory, encode_offer=_encode,
                learning_rate=0.5, training_pairs=1, heldout_pairs=1,
            )
        self.assertEqual(len(adapters), 2)
        self.assertTrue(all(adapter.closed for adapter in adapters))
        self.assertEqual(adapters[1].infer_partitions, [])

    def test_rejects_zero_lr_control_that_performs_optimizer_step(self):
        adapters = []

        def factory(lr):
            adapter = _FakePolicy(lr, wrong_zero=(lr == 0.0))
            adapters.append(adapter)
            return adapter

        with self.assertRaisesRegex(ValueError, "zero-learning-rate control"):
            run_matched_open_weight_controls(
                adapter_factory=factory, encode_offer=_encode,
                learning_rate=0.2, training_pairs=1, heldout_pairs=1,
            )
        self.assertTrue(all(adapter.closed for adapter in adapters))

    def test_rejects_different_pre_feedback_prompts_with_same_arena_schedule(self):
        count = 0

        def stateful_encoder(offer):
            nonlocal count
            count += 1
            return (1, 2 + count)

        with self.assertRaisesRegex(ValueError, "non-matching pre-outcome prompts"):
            run_matched_open_weight_controls(
                adapter_factory=lambda lr: _FakePolicy(lr),
                encode_offer=stateful_encoder,
                learning_rate=0.2, training_pairs=1, heldout_pairs=1,
            )

    def test_invalid_learning_rate_rejected_before_adapter_creation(self):
        for invalid in (0, -1, float("nan"), float("inf"), True, 10 ** 1000):
            with self.subTest(invalid=str(invalid)[:20]):
                with self.assertRaises(ValueError):
                    run_matched_open_weight_controls(
                        adapter_factory=lambda lr: self.fail("factory must not run"),
                        encode_offer=_encode, learning_rate=invalid,
                    )

    def test_invalid_arenas_and_counts_never_invoke_mutating_factory(self):
        configurations = (
            {"training_pairs": 0},
            {"heldout_pairs": 0},
            {"training_pairs": 65},
            {"heldout_pairs": 65},
            {"seed": -1},
            {"seed": True},
            {"seed": 2**32},
            {"evaluator_id": ""},
        )
        for options in configurations:
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_matched_open_weight_controls(
                    adapter_factory=lambda lr: self.fail(
                        "invalid public input must not construct a model adapter"
                    ),
                    encode_offer=_encode,
                    learning_rate=0.2,
                    **options,
                )

    def test_invalid_treatment_readback_is_restored_before_failure(self):
        created = []

        class InvalidDigestFake(_FakePolicy):
            @property
            def weights_digest(self):
                return "not-a-valid-digest"

        def factory(rate):
            adapter = InvalidDigestFake(rate)
            created.append(adapter)
            return adapter

        with self.assertRaisesRegex(ValueError, "digest missing or invalid"):
            run_matched_open_weight_controls(
                adapter_factory=factory, encode_offer=_encode,
                learning_rate=0.2, training_pairs=1, heldout_pairs=1,
            )
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)

    def test_runner_construction_failure_restores_treatment_adapter(self):
        created = []

        def factory(rate):
            adapter = _FakePolicy(rate)
            created.append(adapter)
            return adapter

        with mock.patch(
            "pleasure_experiment.open_weight_controls.OpenWeightTrialRunner",
            side_effect=RuntimeError("failed before runner acquired ownership"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed before runner"):
                run_matched_open_weight_controls(
                    adapter_factory=factory, encode_offer=_encode,
                    learning_rate=0.2, training_pairs=1, heldout_pairs=1,
                )
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)


@skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch unavailable")
class RealTensorMatchedControlTests(TestCase):
    def test_actual_torch_policy_tensor_vs_zero_learning_rate(self):
        import torch

        class TinyCausalPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Embedding(8, 2)
                self.policy = torch.nn.Linear(2, 2)
                with torch.no_grad():
                    self.backbone.weight.fill_(0.5)
                    self.policy.weight.zero_()
                    self.policy.bias.zero_()

            def forward(self, *, input_ids):
                return self.policy(self.backbone(input_ids))

        source = TinyCausalPolicy()
        evaluator = "source-local-open-weight-oracle"
        rubric = OutcomeArena(seed=12345, evaluator_id=evaluator).rubric_digest
        observed_rates = []

        def factory(rate):
            observed_rates.append(rate)
            return TorchChosenOutcomeAdapter(
                source,
                binding=OpenWeightBinding(
                    "tiny-local-torch", "generated-fixture-v1", "test-runtime",
                    _sha("generated-fixture-model"), _sha("generated-fixture-tokenizer"),
                ),
                action_token_ids=(0, 1),
                trainable_parameter_names=("policy.weight", "policy.bias"),
                learning_rate=rate,
                expected_evaluator_id=evaluator,
                expected_rubric_digest=rubric,
                allow_in_place_mutation=True,
                max_delta_l2=0.5,
            )

        result = run_matched_open_weight_controls(
            adapter_factory=factory,
            encode_offer=lambda offer: (1 + offer.side, 2 + offer.pair_index % 3),
            learning_rate=0.2, training_pairs=2, heldout_pairs=2,
        )
        self.assertEqual(observed_rates, [0.2, 0.0])
        self.assertEqual(
            result.backend_kind, "SOURCE_LOCAL_TORCH_ADAPTER_UNAUTHENTICATED",
        )
        self.assertTrue(result.observed_trainable_tensor_change)
        self.assertFalse(result.confirmed_pretrained_llm_training)
        self.assertNotEqual(
            result.outcome_weighted.learned_weights_digest,
            result.outcome_weighted.initial_weights_digest,
        )
        self.assertEqual(result.zero_learning_rate.observed_weight_changes, 0)
        self.assertEqual(result.zero_learning_rate.optimizer_steps_reported, 0)
        self.assertTrue(source.training)
        self.assertTrue(source.backbone.weight.requires_grad)
