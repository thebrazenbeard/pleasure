"""TDD contract for v4 explicit candidate-value head, independent of EVAL."""
from hashlib import sha256
import unittest

from pleasure_experiment.candidate_value_head import CandidateValueHead


def eid(i):
    return sha256(f"chosen-feedback-{i}".encode()).hexdigest()


class CandidateValueHeadTests(unittest.TestCase):
    def setUp(self):
        self.a = (.9, .1)
        self.b = (.1, .9)

    def test_swap_equivariance_and_equal_options(self):
        h = CandidateValueHead(allow_training=True, learning_rate=1.0)
        p = h.predict(self.a, self.b, gain=12.0)
        s = h.predict(self.b, self.a, gain=12.0)
        self.assertAlmostEqual(p.quality_a, s.quality_b, places=14)
        self.assertAlmostEqual(p.quality_b, s.quality_a, places=14)
        self.assertAlmostEqual(p.prob_a, s.prob_b, places=14)
        self.assertAlmostEqual(h.predict(self.a, self.a).prob_a, .5, places=14)

    def test_only_chosen_reward_updates_and_persists(self):
        h = CandidateValueHead(allow_training=True, learning_rate=1.0)
        before = h.predict(self.a, self.b)
        offered = h.prepare_train("train:1", self.a, self.b)
        committed = h.commit_train(offered, "A", .35)
        update = h.learn_chosen(committed, eid(1), .95)
        after = h.predict(self.a, self.b)
        self.assertEqual(h.updates, 1)
        self.assertNotEqual(before.head_digest, after.head_digest)
        self.assertGreater(after.quality_a, before.quality_a)
        self.assertEqual(update.update_index, 1)
        state = h.export_state()
        loaded = CandidateValueHead.from_state(state, allow_training=True)
        self.assertEqual(loaded.updates, 1)
        self.assertEqual(loaded.predict(self.a, self.b), after)
        with self.assertRaisesRegex(ValueError, "replay"):
            loaded.prepare_train("train:1", self.a, self.b)

    def test_receipt_only_after_precommitted_choice_and_no_eval_learning(self):
        h = CandidateValueHead(allow_training=True)
        p = h.prepare_train("train:2", self.a, self.b)
        with self.assertRaises((TypeError, ValueError)):
            h.learn_chosen(p, eid(2), .9)
        with self.assertRaises(ValueError):
            h.commit_train(p, "A", 0.0)
        c = h.commit_train(p, "B", .25)
        with self.assertRaisesRegex(ValueError, "pending"):
            h.commit_train(p, "A", .3)
        h.learn_chosen(c, eid(2), .1)
        with self.assertRaisesRegex(ValueError, "replay"):
            h.prepare_train("train:2", self.a, self.b)
        frozen = CandidateValueHead(allow_training=False)
        with self.assertRaises(PermissionError):
            frozen.prepare_train("train:3", self.a, self.b)

    def test_import_rejects_tampering_rollback_and_pending_export(self):
        h = CandidateValueHead(allow_training=True)
        p = h.prepare_train("train:4", self.a, self.b)
        with self.assertRaisesRegex(ValueError, "pending"):
            h.export_state()
        c = h.commit_train(p, "A", .5)
        h.learn_chosen(c, eid(4), .75)
        state = h.export_state()
        copy = dict(state)
        copy["weights"] = [9, 9, 9]
        with self.assertRaisesRegex(ValueError, "digest"):
            CandidateValueHead.from_state(copy, allow_training=True)
        with self.assertRaisesRegex(ValueError, "rollback"):
            CandidateValueHead.from_state(state, allow_training=True, min_updates=2)

    def test_reject_nonfinite_invalid_features_wrong_receipt(self):
        h = CandidateValueHead(allow_training=True)
        for bad in ((float("nan"), .2), (True, .2), (.4,), (-.1, .3)):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                h.predict(bad, self.b)
        p = h.prepare_train("train:5", self.a, self.b)
        c = h.commit_train(p, "A", .4)
        for bad in (float("nan"), True, -1, 1.1):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                h.learn_chosen(c, eid(5), bad)
        self.assertEqual(h.updates, 0)
        h.learn_chosen(c, eid(5), .7)
        self.assertEqual(h.updates, 1)


if __name__ == "__main__":
    unittest.main()
