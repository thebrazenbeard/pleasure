"""Contract tests for source-local structurally held-out outcome generators."""

from dataclasses import asdict, replace
import unittest

from pleasure_experiment.outcome_arena import TaskOffer
from pleasure_experiment.withheld_payoffs import (
    ALL_FAMILIES, EVAL_FAMILIES, TRAIN_FAMILY, WithheldPayoffArena,
    _generator_digest, _rule,
)


_DECISION = "d" * 64


def _arena(seed=712):
    return WithheldPayoffArena(seed=seed, evaluator_id="local-scoring-only")


def _choose(arena, offer, action="A", *, instance="model-instance-one"):
    ticket = arena.commit(offer, action, _DECISION, model_id="open-weight-test", runtime_instance_id=instance)
    return arena.resolve(ticket)


class WithheldPayoffTests(unittest.TestCase):
    def test_manifest_declares_separate_structural_holdouts(self):
        arena = _arena()
        manifest = arena.manifest
        self.assertEqual(manifest.train_family, TRAIN_FAMILY)
        self.assertEqual(manifest.heldout_families, EVAL_FAMILIES)
        self.assertEqual(len(EVAL_FAMILIES), 3)
        self.assertEqual(len(set(manifest.heldout_generator_sha256)), 3)
        self.assertNotIn(manifest.train_generator_sha256, manifest.heldout_generator_sha256)
        self.assertEqual(arena.rubric_digest, manifest.suite_sha256)
        self.assertFalse(manifest.externally_authenticated)

    def test_generator_function_classes_are_not_same_linear_oracle(self):
        # Same *exact* feature vector, not just different episode IDs: the
        # rules are provably distinct (interaction sign, threshold, basin).
        points = ((0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9))
        outcomes = {
            family: tuple(_rule(family, point) for point in points)
            for family in ALL_FAMILIES
        }
        self.assertEqual(len(set(outcomes.values())), len(ALL_FAMILIES))
        # The smooth affine generator has zero cross-interaction, whereas
        # the held-out interaction generator does not.
        def interaction(family):
            score = outcomes[family]
            return score[0] - score[1] - score[2] + score[3]

        self.assertAlmostEqual(interaction(TRAIN_FAMILY), 0.0, places=12)
        self.assertGreater(abs(interaction(EVAL_FAMILIES[0])), 0.1)
        # A discontinuity at x=0.55 is present only in threshold generator.
        below, above = (0.55 - 1e-8, 0.5), (0.55 + 1e-8, 0.5)
        self.assertGreater(_rule(EVAL_FAMILIES[1], above) - _rule(EVAL_FAMILIES[1], below), 0.5)
        self.assertLess(abs(_rule(TRAIN_FAMILY, above) - _rule(TRAIN_FAMILY, below)), 1e-6)
        # Interior optimum defeats a monotonically increasing affine rule.
        self.assertGreater(_rule(EVAL_FAMILIES[2], (0.72, 0.24)), _rule(EVAL_FAMILIES[2], (0.95, 0.95)))

    def test_issue_split_is_strict_and_offers_contain_no_oracle_data(self):
        arena = _arena()
        train = arena.issue(partition="TRAIN", family_id=TRAIN_FAMILY, pair_index=0, side=0)
        held = arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[0], pair_index=0, side=0)
        self.assertIsInstance(train, TaskOffer)
        self.assertNotEqual(train.episode_id, held.episode_id)
        self.assertNotEqual((train.features_a, train.features_b), (held.features_a, held.features_b))
        self.assertTrue(set(asdict(train)).isdisjoint({"reward", "quality", "winner", "oracle_action", "payoff", "regret"}))
        with self.assertRaisesRegex(ValueError, "boundary"):
            arena.issue(partition="TRAIN", family_id=EVAL_FAMILIES[0], pair_index=0, side=0)
        with self.assertRaisesRegex(ValueError, "boundary"):
            arena.issue(partition="EVAL", family_id=TRAIN_FAMILY, pair_index=0, side=0)
        with self.assertRaisesRegex(ValueError, "pair_index"):
            arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[0], pair_index=True, side=0)
        with self.assertRaisesRegex(ValueError, "side"):
            arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[0], pair_index=0, side=True)
        self.assertEqual(arena.issue(partition="TRAIN", family_id=TRAIN_FAMILY, pair_index=0, side=0), train)

    def test_seed_replay_matched_arm_and_mirror_invariance(self):
        first, second = _arena(77), _arena(77)
        for family in EVAL_FAMILIES:
            for pair_index in range(8):
                a = first.issue(partition="EVAL", family_id=family, pair_index=pair_index, side=0)
                mirrored = first.issue(partition="EVAL", family_id=family, pair_index=pair_index, side=1)
                b = second.issue(partition="EVAL", family_id=family, pair_index=pair_index, side=0)
                self.assertEqual(a, b)
                self.assertEqual(a.features_a, mirrored.features_b)
                self.assertEqual(a.features_b, mirrored.features_a)
                self.assertGreater(abs(_rule(family, a.features_a) - _rule(family, a.features_b)), 0.02)
                self.assertEqual(_choose(first, a, "A"), _choose(second, b, "A"))
                _choose(first, mirrored, "B")
        third = _arena(78)
        other = third.issue(partition="EVAL", family_id=EVAL_FAMILIES[0], pair_index=0, side=0)
        self.assertNotEqual(a, other)

    def test_only_committed_chosen_feedback_and_one_use(self):
        arena = _arena()
        offer = arena.issue(partition="TRAIN", family_id=TRAIN_FAMILY, pair_index=1, side=0)
        ticket = arena.commit(offer, "B", _DECISION, model_id="model", runtime_instance_id="one")
        receipt = arena.resolve(ticket)
        self.assertEqual(receipt.action, "B")
        self.assertEqual(receipt.measured_quality, _rule(TRAIN_FAMILY, offer.features_b))
        self.assertEqual(receipt.rubric_digest, _generator_digest(TRAIN_FAMILY))
        self.assertEqual(set(asdict(receipt)).isdisjoint({"quality_a", "quality_b", "counterfactual", "oracle_quality"}), True)
        with self.assertRaisesRegex(ValueError, "consumed"):
            arena.resolve(ticket)
        with self.assertRaisesRegex(ValueError, "already committed"):
            arena.commit(offer, "A", _DECISION, model_id="model", runtime_instance_id="one")
        modified = arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[0], pair_index=3, side=0)
        object.__setattr__(modified, "features_a", (0.1, 0.9))
        with self.assertRaisesRegex(ValueError, "altered"):
            arena.commit(modified, "A", _DECISION, model_id="model", runtime_instance_id="one")

    def test_counterfactuals_locked_until_full_eval_seal_and_never_train(self):
        arena = _arena()
        train = arena.issue(partition="TRAIN", family_id=TRAIN_FAMILY, pair_index=1, side=0)
        eval_a = arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[0], pair_index=1, side=0)
        eval_b = arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[1], pair_index=1, side=0)
        _choose(arena, train, "B")
        with self.assertRaisesRegex(PermissionError, "sealed"):
            arena.posthoc_evaluate(eval_a)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            arena.finalize_evaluation()
        _choose(arena, eval_a, "A")
        with self.assertRaisesRegex(ValueError, "unresolved"):
            arena.finalize_evaluation()
        _choose(arena, eval_b, "B")
        arena.finalize_evaluation()
        with self.assertRaisesRegex(PermissionError, "TRAIN"):
            arena.posthoc_evaluate(train)
        post = arena.posthoc_evaluate(eval_a)
        self.assertEqual(post.chosen_action, "A")
        self.assertEqual(post.chosen_quality, post.quality_a)
        self.assertEqual(post.unchosen_quality, post.quality_b)
        self.assertEqual(post.oracle_quality, max(post.quality_a, post.quality_b))
        self.assertAlmostEqual(post.regret, post.oracle_quality - post.chosen_quality)
        self.assertEqual(post.generator_sha256, _generator_digest(eval_a.family_id))
        self.assertEqual(post, arena.posthoc_evaluate(eval_a))
        for operation in (
            lambda: arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[2], pair_index=5, side=0),
            lambda: arena.commit(eval_a, "B", _DECISION, model_id="model", runtime_instance_id="one"),
        ):
            with self.assertRaisesRegex(ValueError, "sealed"):
                operation()

    def test_invalid_replay_or_unissued_rejected(self):
        arena = _arena()
        unknown = _arena(333).issue(partition="TRAIN", family_id=TRAIN_FAMILY, pair_index=0, side=0)
        with self.assertRaisesRegex(ValueError, "altered"):
            arena.commit(unknown, "A", _DECISION, model_id="model", runtime_instance_id="run")
        offer = arena.issue(partition="EVAL", family_id=EVAL_FAMILIES[2], pair_index=3, side=0)
        with self.assertRaisesRegex(ValueError, "decision_digest"):
            arena.commit(offer, "A", "not sha", model_id="model", runtime_instance_id="run")
        valid = arena.commit(offer, "A", _DECISION, model_id="model", runtime_instance_id="run")
        tampered = replace(valid, action="B")
        with self.assertRaisesRegex(ValueError, "matching"):
            arena.resolve(tampered)
        arena.resolve(valid)
        arena.finalize_evaluation()
        mutated = replace(offer, nonce="b" * 64)
        with self.assertRaisesRegex(ValueError, "altered"):
            arena.posthoc_evaluate(mutated)


if __name__ == "__main__":
    unittest.main()
