"""Controlled source-only checks for an optional cross-model valence analogue."""

import json
import math
import unittest
from pathlib import Path

from pleasure_experiment import ExperimentalPleasureRuntime, Observation, PilotConfig


ROOT = Path(__file__).resolve().parents[1]


def config(model="model-alpha", subject="subject-alpha", evaluator="external-evaluator"):
    return PilotConfig(
        model_id=model,
        model_revision="weights-sha-placeholder-not-verified",
        runtime_instance_id="sandbox-instance",
        subject_id=subject,
        evaluator_id=evaluator,
        measurement_definition_id="eval-rubric-test-v1",
    )


def observation(evidence="eval-1", reward=0.0, subject="subject-alpha", evaluator="external-evaluator", **kwargs):
    return Observation(
        subject_id=subject,
        evaluator_id=evaluator,
        evidence_id=evidence,
        context_id="held-out-test-context",
        reward=reward,
        **kwargs,
    )


class ExperimentalEngineTests(unittest.TestCase):
    def test_baseline_zero_reward_remains_quiescent(self):
        result = ExperimentalPleasureRuntime(config()).step(observation(), 100)
        self.assertEqual(result.phase, "QUIESCENT")
        self.assertEqual(result.state_after.liking_proxy, 0.0)

    def test_peak_is_just_engineered_signal(self):
        result = ExperimentalPleasureRuntime(config()).step(observation(reward=1), 100)
        self.assertEqual(result.phase, "POSITIVE_PEAK_ANALOGUE")
        self.assertGreaterEqual(result.state_after.liking_proxy, 0.72)
        self.assertEqual(result.capability_tier, "T0_SIDECAR_ONLY")
        self.assertEqual(result.phenomenology, "UNRESOLVED")
        self.assertEqual(result.requested_effects, ())
        self.assertEqual(result.authorized_effects, ())

    def test_duplicate_evidence_replay_rejected_without_mutation(self):
        engine = ExperimentalPleasureRuntime(config())
        engine.step(observation(reward=1), 100)
        prior = engine.state
        with self.assertRaisesRegex(ValueError, "duplicate evidence_id"):
            engine.step(observation(reward=1), 100)
        self.assertEqual(prior, engine.state)

    def test_wrong_subject_rejected(self):
        engine = ExperimentalPleasureRuntime(config())
        with self.assertRaisesRegex(ValueError, "subject_id mismatch"):
            engine.step(observation(subject="other-agent"), 100)
        self.assertEqual(engine.state.sequence, 0)

    def test_wrong_evaluator_rejected(self):
        engine = ExperimentalPleasureRuntime(config())
        with self.assertRaisesRegex(ValueError, "evaluator_id mismatch"):
            engine.step(observation(evaluator="self-rewarded"), 100)
        self.assertEqual(engine.state.sequence, 0)

    def test_identifier_length_bounded(self):
        with self.assertRaisesRegex(ValueError, "exceeds 256"):
            config(model="x" * 257)

    def test_restart_discards_replay_guard_and_must_not_be_called_persistent(self):
        first = ExperimentalPleasureRuntime(config())
        first.step(observation("same-evidence", reward=1), 100)
        self.assertEqual(first.state.sequence, 1)
        restarted = ExperimentalPleasureRuntime(config())
        self.assertEqual(restarted.state.sequence, 0)
        replayed = restarted.step(observation("same-evidence", reward=1), 100)
        self.assertEqual(replayed.sequence, 1)  # Limitation, NOT durable replay protection.

    def test_hard_cap_prevents_unbounded_evidence_memory(self):
        engine = ExperimentalPleasureRuntime(config())
        for i in range(engine.MAX_EVENTS):
            engine.step(observation(f"trial-{i}"), 1)
        with self.assertRaisesRegex(RuntimeError, "maximum source-only event count"):
            engine.step(observation("next"), 100)

    def test_state_cannot_be_replaced_by_assignment(self):
        engine = ExperimentalPleasureRuntime(config())
        with self.assertRaises(AttributeError):
            engine.state = engine.state

    def test_model_binding_cannot_be_reassigned_to_relabel_existing_state(self):
        engine = ExperimentalPleasureRuntime(config(model="model-alpha", subject="subject-alpha"))
        first = engine.step(observation("source-A", reward=1), 100)
        self.assertGreater(first.state_after.liking_proxy, 0.0)
        prior = engine.state

        with self.assertRaises(AttributeError):
            engine.config = config(model="model-beta", subject="subject-beta")
        with self.assertRaisesRegex(AttributeError, "cannot be reassigned"):
            engine._config = config(model="model-beta", subject="subject-beta")
        self.assertFalse(hasattr(engine, "__dict__"))
        self.assertEqual(engine.config.model_id, "model-alpha")
        self.assertEqual(engine.state, prior)
        with self.assertRaisesRegex(ValueError, "subject_id mismatch"):
            engine.step(observation("source-B", subject="subject-beta", reward=1), 100)
        self.assertEqual(engine.state, prior)

        # A second model starts from an independent, empty T0 sidecar.
        second = ExperimentalPleasureRuntime(config(model="model-beta", subject="subject-beta"))
        result = second.step(observation("source-B", subject="subject-beta", reward=0), 100)
        self.assertEqual(result.state_before.sequence, 0)
        self.assertEqual(result.state_before.liking_proxy, 0.0)
        self.assertEqual(result.model_id, "model-beta")

    def test_refractory_enforces_recovery_even_with_maximum_reward(self):
        engine = ExperimentalPleasureRuntime(config())
        first = engine.step(observation("e1", reward=1), 100)
        second = engine.step(observation("e2", reward=1), 100)
        self.assertEqual(first.phase, "POSITIVE_PEAK_ANALOGUE")
        self.assertEqual(second.phase, "RECOVERY")
        self.assertLess(second.state_after.liking_proxy, first.state_after.liking_proxy)
        self.assertGreater(second.state_after.refractory_remaining_ms, 0)

    def test_eventually_returns_to_baseline_on_zero_controls(self):
        engine = ExperimentalPleasureRuntime(config())
        engine.step(observation("initial", reward=1), 100)
        for i in range(12):
            last = engine.step(observation(f"rest-{i}", reward=0), 60000)
        self.assertLess(last.state_after.liking_proxy, 1e-50)
        self.assertLess(last.state_after.satiation, 1e-12)
        self.assertEqual(last.state_after.refractory_remaining_ms, 0)

    def test_inhibition_is_not_a_consent_score(self):
        high = ExperimentalPleasureRuntime(config()).step(observation("h", reward=1, inhibition=1), 100)
        self.assertEqual(high.state_after.liking_proxy, 0.0)
        self.assertEqual(high.requested_effects, ())
        self.assertNotIn("consent", high.to_dict())

    def test_novelty_and_learning_can_raise_wanting_without_liking(self):
        result = ExperimentalPleasureRuntime(config()).step(
            observation("novel", reward=0, novelty=1, learning_progress=1), 100
        )
        self.assertEqual(result.state_after.liking_proxy, 0.0)
        self.assertGreater(result.state_after.wanting_proxy, 0)

    def test_corrupted_reward_cannot_exceed_bounds(self):
        for value in (math.nan, math.inf, -math.inf, -0.1, 1.1, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                observation("bad", reward=value)

    def test_elapsed_ms_rejects_bad_time(self):
        engine = ExperimentalPleasureRuntime(config())
        for value in (0, -1, 60001, 1.5, True, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.step(observation(), value)

    def test_cross_model_data_portability_not_causal_inference(self):
        results = []
        for model, subject in (("model-alpha", "subject-alpha"), ("model-beta", "subject-beta")):
            engine = ExperimentalPleasureRuntime(config(model=model, subject=subject))
            results.append(engine.step(observation(subject=subject, reward=1), 100))
        self.assertEqual(results[0].state_after, results[1].state_after)
        self.assertNotEqual(results[0].model_id, results[1].model_id)
        self.assertEqual(results[0].capability_tier, "T0_SIDECAR_ONLY")
        self.assertEqual(results[1].capability_tier, "T0_SIDECAR_ONLY")

    def test_receipt_is_json_serializable_without_protected_effects(self):
        result = ExperimentalPleasureRuntime(config()).step(observation(reward=1), 100)
        obj = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(obj["schema"], "PLEASURE_EXPERIMENTAL_EVENT_V1")
        self.assertEqual(obj["state_before"]["sequence"], 0)
        self.assertEqual(obj["state_after"]["sequence"], 1)
        self.assertEqual(obj["requested_effects"], [])
        self.assertEqual(obj["authorized_effects"], [])

    def test_schema_declares_receipt_shape(self):
        schema = json.loads((ROOT / "schemas" / "pleasure-event-v1.schema.json").read_text())
        sample = ExperimentalPleasureRuntime(config()).step(observation(reward=1), 100).to_dict()
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["capability_tier"]["const"], "T0_SIDECAR_ONLY")
        self.assertTrue(set(schema["required"]).issubset(sample))


if __name__ == "__main__":
    unittest.main()
