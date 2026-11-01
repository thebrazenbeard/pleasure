"""Falsifiers for the deliberately synthetic, non-LLM appraisal loop."""

import math
import unittest

from pleasure_experiment.model_valence_lab import (
    CAPABILITY,
    PHENOMENOLOGY,
    CalibrationExample,
    LabBinding,
    LabChoiceModel,
    LinearAppraiser,
    MeasuredOutcome,
    ModelProbe,
)


GOOD = (
    CalibrationExample((1.0, 0.0), 1.0),
    CalibrationExample((0.9, 0.1), 0.9),
    CalibrationExample((0.0, 1.0), 0.0),
    CalibrationExample((0.1, 0.9), 0.1),
)
SHUFFLED = (
    CalibrationExample((1.0, 0.0), 0.0),
    CalibrationExample((0.9, 0.1), 0.1),
    CalibrationExample((0.0, 1.0), 1.0),
    CalibrationExample((0.1, 0.9), 0.9),
)


def engine(samples=GOOD, *, instance="sandbox-one"):
    model, _ = LinearAppraiser.fit(samples)
    return LabChoiceModel(LabBinding("tiny-model", "frozen-r1", instance, "lab-evaluator"), model)


def probe(*, features=(1.0, 0.0), progress=0.5, instance="sandbox-one", goal="task-one"):
    return ModelProbe("tiny-model", instance, goal, features, progress)


def outcome(*, quality=1.0, instance="sandbox-one", goal="task-one", evidence="result-one", evaluator="lab-evaluator"):
    return MeasuredOutcome("tiny-model", instance, goal, evaluator, evidence, quality)


class ModelValenceLabTests(unittest.TestCase):
    def test_actual_tiny_appraiser_parameter_update_and_held_out_prediction(self):
        fitted, receipt = LinearAppraiser.fit(GOOD)
        self.assertTrue(receipt.changed)
        self.assertEqual(receipt.weights_before, (0.0, 0.0))
        self.assertNotEqual(receipt.weights_after, receipt.weights_before)
        # Held-out interpolation follows the independently supplied labels.
        self.assertAlmostEqual(fitted.predict((0.8, 0.2)), 0.8, delta=0.05)
        self.assertAlmostEqual(fitted.predict((0.2, 0.8)), 0.2, delta=0.05)
        self.assertEqual(receipt.capability, CAPABILITY)

    def test_positive_internal_appraisal_changes_next_opaque_choice(self):
        model = engine()
        self.assertEqual(model.decide().action, "B")
        transition = model.observe(probe(), outcome())
        self.assertGreater(transition.predicted_quality, 0.95)
        self.assertGreater(transition.after.liking_proxy, 0.5)
        self.assertEqual(model.decide(condition="active").action, "A")
        self.assertEqual(model.decide(condition="zero").action, "B")
        self.assertEqual(model.decide(condition="sham").action, "B")
        self.assertEqual(transition.capability, "SYNTHETIC_CHOICE_MODEL_ONLY")
        self.assertEqual(transition.phenomenology, PHENOMENOLOGY)
        self.assertEqual(transition.authorized_effects, ())

    def test_no_independent_positive_outcome_means_no_liking(self):
        model = engine()
        transition = model.observe(probe(progress=1.0), outcome(quality=0))
        self.assertEqual(transition.positive_impact, 0)
        self.assertEqual(transition.after.liking_proxy, 0)
        self.assertGreater(transition.after.wanting_proxy, 0)
        self.assertEqual(model.decide().action, "B")

    def test_model_appraisal_is_not_identical_to_outcome_label(self):
        model = engine()
        transition = model.observe(probe(features=(0.0, 1.0)), outcome(quality=1))
        self.assertLess(transition.predicted_quality, 0.01)
        self.assertLess(transition.after.liking_proxy, 0.01)
        self.assertEqual(model.decide().action, "B")

    def test_shuffled_calibration_fails_original_direction(self):
        correct = engine()
        shuffled = engine(SHUFFLED)
        correct.observe(probe(), outcome())
        shuffled.observe(probe(), outcome())
        self.assertEqual(correct.decide().action, "A")
        self.assertEqual(shuffled.decide().action, "B")

    def test_withdrawal_and_fresh_instance_restore_neutral_choice(self):
        model = engine()
        model.observe(probe(), outcome())
        self.assertEqual(model.decide().action, "A")
        after = model.withdraw(12)
        self.assertLess(after.liking_proxy, 0.01)
        self.assertEqual(model.decide().action, "B")
        self.assertEqual(engine().decide().action, "B")

    def test_identity_and_evaluator_mismatches_do_not_mutate(self):
        model = engine()
        cases = [
            (probe(instance="wrong"), outcome(), "probe model instance"),
            (probe(), outcome(instance="wrong"), "outcome model instance"),
            (probe(), outcome(goal="other"), "outcome goal"),
            (probe(), outcome(evaluator="self-appointed"), "evaluator ID mismatch"),
        ]
        for p, o, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model.observe(p, o)
            self.assertEqual(model.state.sequence, 0)

    def test_replay_and_cross_instance_contamination_rejected(self):
        first = engine()
        first.observe(probe(), outcome())
        prior = first.state
        with self.assertRaisesRegex(ValueError, "duplicate outcome evidence"):
            first.observe(probe(), outcome())
        self.assertEqual(prior, first.state)
        other = engine(instance="sandbox-two")
        self.assertEqual(other.state.sequence, 0)
        with self.assertRaisesRegex(ValueError, "probe model instance"):
            other.observe(probe(), outcome())

    def test_accumulated_state_cannot_be_relabelled_or_appraiser_swapped(self):
        model = engine()
        model.observe(probe(), outcome())
        original_binding = model.binding
        original_appraiser = model.appraiser
        original_state = model.state
        other_binding = LabBinding("other-model", "other-revision", "other-instance", "other-evaluator")
        other_appraiser, _ = LinearAppraiser.fit(SHUFFLED)

        with self.assertRaises(AttributeError):
            model.binding = other_binding
        with self.assertRaises(AttributeError):
            model.appraiser = other_appraiser
        with self.assertRaisesRegex(AttributeError, "cannot be rebound"):
            model._binding = other_binding
        with self.assertRaisesRegex(AttributeError, "cannot be rebound"):
            model._appraiser = other_appraiser
        with self.assertRaisesRegex(AttributeError, "cannot be deleted"):
            del model._binding
        with self.assertRaisesRegex(AttributeError, "cannot be deleted"):
            del model._appraiser

        self.assertFalse(hasattr(model, "__dict__"))
        self.assertIs(model.binding, original_binding)
        self.assertIs(model.appraiser, original_appraiser)
        self.assertEqual(model.state, original_state)
        self.assertEqual(model.decide().action, "A")
        with self.assertRaisesRegex(ValueError, "probe model instance mismatch"):
            model.observe(probe(instance="other-instance"), outcome(instance="other-instance", evidence="other"))
        self.assertEqual(model.state, original_state)

    def test_post_construction_mutation_revalidated(self):
        model = engine()
        invalid = outcome(quality=0)
        object.__setattr__(invalid, "observed_quality", math.inf)
        with self.assertRaisesRegex(ValueError, "observed_quality must be a finite number"):
            model.observe(probe(), invalid)
        self.assertEqual(model.state.sequence, 0)
        bad = probe()
        object.__setattr__(bad, "features", (math.nan, 0.0))
        with self.assertRaisesRegex(ValueError, "feature must be a finite number"):
            model.observe(bad, outcome())
        self.assertEqual(model.state.sequence, 0)

    def test_invalid_training_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least four"):
            LinearAppraiser.fit(GOOD[:3])
        with self.assertRaisesRegex(ValueError, "meaningfully different"):
            LinearAppraiser.fit(tuple(CalibrationExample((0.5, 0.5), 0.5) for _ in range(4)))
        with self.assertRaisesRegex(ValueError, "quality must be a finite number"):
            LinearAppraiser.fit(GOOD[:3] + (CalibrationExample((0.0, 1.0), math.nan),))

    def test_huge_integer_inputs_fail_with_bounded_value_errors(self):
        huge = 10 ** 1000  # float(huge) raises OverflowError in CPython.
        with self.assertRaisesRegex(ValueError, "appraiser weight must be a finite number"):
            LinearAppraiser((huge, 0), 0.5)
        with self.assertRaisesRegex(ValueError, "appraiser bias must be a finite number"):
            LinearAppraiser((0, 0), huge)
        with self.assertRaisesRegex(ValueError, "learning_rate must be a finite number"):
            LinearAppraiser.fit(GOOD, learning_rate=huge)

        model = engine()
        with self.assertRaisesRegex(ValueError, "feature must be a finite number"):
            model.observe(probe(features=(huge, 0)), outcome())
        with self.assertRaisesRegex(ValueError, "observed_quality must be a finite number"):
            model.observe(probe(), outcome(quality=huge))
        with self.assertRaisesRegex(ValueError, "predicted_progress must be a finite number"):
            model.observe(probe(progress=huge), outcome())
        self.assertEqual(model.state.sequence, 0)

    def test_unknown_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "condition must be"):
            engine().decide(condition="self-approved-override")


if __name__ == "__main__":
    unittest.main()
