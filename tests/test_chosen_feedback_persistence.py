"""Adversarial contracts for a mechanism-neutral chosen-only external learner."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.chosen_feedback_persistence import (
    ChosenFeedbackPolicy, load_checkpoint, save_checkpoint, run_study,
)


class PersistenceContractTests(unittest.TestCase):
    def test_two_phase_seals_choice_before_reward_and_blocks_replay(self):
        policy = ChosenFeedbackPolicy()
        decision = policy.propose("TRAIN-000", ("archive", "garden"), forced_position=0)
        self.assertEqual(decision["selected"], "archive")
        self.assertNotIn("reward", str(decision).lower())
        with self.assertRaises(ValueError):
            policy.propose("TRAIN-001", ("garden", "archive"))
        with self.assertRaises(ValueError):
            policy.settle("TRAIN-000", "wrong-seal", 1)
        with self.assertRaises(ValueError):
            policy.settle("TRAIN-000", decision["decision_sha256"], True)
        receipt = policy.settle("TRAIN-000", decision["decision_sha256"], 1)
        self.assertEqual(receipt["reward"], 1)
        with self.assertRaises(ValueError):
            policy.settle("TRAIN-000", decision["decision_sha256"], 1)
        with self.assertRaises(ValueError):
            policy.propose("TRAIN-000", ("garden", "archive"))

    def test_restart_middecision_and_after_feedback_is_exact(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            policy = ChosenFeedbackPolicy()
            decision = policy.propose("TRAIN-000", ("garden", "archive"), forced_position=1)
            save_checkpoint(policy, path)
            recovered = load_checkpoint(path)
            self.assertEqual(recovered.pending, policy.pending)
            recovered.settle("TRAIN-000", decision["decision_sha256"], 1)
            save_checkpoint(recovered, path)
            second = load_checkpoint(path)
            self.assertEqual(second.to_dict(), recovered.to_dict())
            self.assertEqual(second.predict(("garden", "archive")), "archive")

    def test_tampering_or_json_duplicate_fields_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = ChosenFeedbackPolicy()
            decision = state.propose("TRAIN-000", ("archive", "garden"))
            state.settle("TRAIN-000", decision["decision_sha256"], 1)
            save_checkpoint(state, path)
            payload = json.loads(path.read_text())
            payload["events"][0]["reward"] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            path.write_text('{"schema":"x","schema":"x"}')
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_bad_options_rewards_and_uncommitted_decisions_rejected(self):
        state = ChosenFeedbackPolicy()
        with self.assertRaises(ValueError):
            state.propose("TRAIN-000", ("archive", "archive"))
        with self.assertRaises(ValueError):
            state.propose("TRAIN-000", ("archive", "garden"), forced_position=2)
        with self.assertRaises(ValueError):
            state.settle("TRAIN-000", "none", 1)
        d = state.propose("TRAIN-000", ("archive", "garden"))
        with self.assertRaises(ValueError):
            state.settle("TRAIN-001", d["decision_sha256"], 1)
        with self.assertRaises(ValueError):
            state.settle("TRAIN-000", d["decision_sha256"], -1)

    def test_restart_cannot_change_assigned_arm_or_seed(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "study.json"
            from experiments.chosen_feedback_persistence import train_chunk
            policy = ChosenFeedbackPolicy()
            train_chunk(policy, "ON", 17, 0, 3)
            save_checkpoint(policy, path)
            resumed = load_checkpoint(path)
            self.assertEqual(resumed.study_binding, {"mode": "ON", "seed": 17})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_chunk(resumed, "SHAM", 17, 3, 4)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_chunk(resumed, "ON", 73, 3, 4)
            self.assertEqual(len(resumed.events), 3)
            train_chunk(resumed, "ON", 17, 3, 5)
            self.assertEqual(len(resumed.events), 5)

    def test_read_only_evaluation_cannot_change_learned_state(self):
        state = ChosenFeedbackPolicy()
        d = state.propose("TRAIN-000", ("garden", "archive"), forced_position=1)
        state.settle("TRAIN-000", d["decision_sha256"], 1)
        before = state.to_dict()
        for _ in range(12):
            state.predict(("archive", "garden"))
            state.predict(("garden", "archive"))
        self.assertEqual(before, state.to_dict())

    def test_synthetic_study_is_repeatable_and_gates_do_not_claim_model_training(self):
        one = run_study(train_rounds=12, eval_rounds=16, seed=19)
        two = run_study(train_rounds=12, eval_rounds=16, seed=19)
        self.assertEqual(one, two)
        self.assertEqual(set(one["arms"]), {"ON", "SHAM", "INVERTED", "OFF"})
        self.assertEqual(one["evidence_class"], "EXTERNAL_CHOSEN_ONLY_SYNTHETIC_POLICY")
        for arm in one["arms"].values():
            self.assertEqual(arm["eval_trials"], 16)
            self.assertLessEqual(0, arm["eval_target_accuracy"])
            self.assertLessEqual(arm["eval_target_accuracy"], 1)
            self.assertTrue(arm["state_stable_during_eval"])


if __name__ == "__main__":
    unittest.main()
