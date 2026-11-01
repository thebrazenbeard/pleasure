"""Pure-Python propensity and deterministic replay regression tests."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import replace
import math
import random
from unittest import TestCase

from pleasure_experiment.open_weight_stochastic import (
    SeededActionSampler, StochasticAction, validate_stochastic_trace,
)


_DIGEST = sha256(b"pre-outcome-context").hexdigest()


def _draw(sampler: SeededActionSampler, index: int, a: float = 0.8):
    return sampler.sample(
        a, 1 - a, input_ids_sha256=_DIGEST, partition="TRAIN",
        episode_id=f"offer-{index}", policy_id="appraisal-on",
    )


class SeededSamplerTests(TestCase):
    def test_exact_replay_and_different_seeds(self):
        rows = [_draw(SeededActionSampler(seed=11), 0)]
        first = SeededActionSampler(seed=11, exploration_rate=0.2)
        second = SeededActionSampler(seed=11, exploration_rate=0.2)
        a = [_draw(first, i) for i in range(100)]
        b = [_draw(second, i) for i in range(100)]
        self.assertEqual(a, b)
        self.assertEqual([item.draw_index for item in a], list(range(100)))
        self.assertEqual(first.draws_used, 100)
        self.assertEqual(second.draws_used, 100)
        self.assertNotEqual(a, [_draw(SeededActionSampler(seed=23), 0)])
        self.assertIsInstance(rows[0], StochasticAction)
        self.assertEqual(len(a[0].decision_digest), 64)
        validate_stochastic_trace(tuple(a))
        validate_stochastic_trace((rows[0],))

    def test_trace_rejects_tampered_chosen_propensity_action_and_replay(self):
        sampler = SeededActionSampler(seed=11, exploration_rate=0.2)
        rows = tuple(_draw(sampler, i) for i in range(8))
        for mutation in (
            replace(rows[2], chosen_propensity=0.4),
            replace(rows[2], chosen_action="B" if rows[2].chosen_action == "A" else "A"),
            replace(rows[2], random_draw=0.111),
            replace(rows[2], decision_digest="f" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "replay"):
                validate_stochastic_trace(rows[:2] + (mutation,) + rows[3:])
        with self.assertRaisesRegex(ValueError, "more than once"):
            validate_stochastic_trace(rows + (rows[2],))
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_stochastic_trace(rows[1:])

    def test_non_argmax_action_is_actually_sampled(self):
        sampler = SeededActionSampler(seed=11, exploration_rate=0.1)
        rows = tuple(_draw(sampler, i, a=0.8) for i in range(50))
        alternatives = [row for row in rows if row.chosen_action == "B"]
        self.assertTrue(alternatives, "exploration must sometimes choose below-argmax action")
        self.assertTrue(all(row.chosen_propensity == row.behavior_prob_b
                            for row in alternatives))
        validate_stochastic_trace(rows)

    def test_propensity_is_the_sampling_distribution_not_raw_model_prob(self):
        sampler = SeededActionSampler(seed=101, exploration_rate=0.2)
        item = _draw(sampler, 0, a=1.0)
        self.assertEqual((item.prob_a, item.prob_b), (1.0, 0.0))
        self.assertAlmostEqual(item.behavior_prob_a, 0.9)
        self.assertAlmostEqual(item.behavior_prob_b, 0.1)
        self.assertEqual(item.chosen_propensity,
                         item.behavior_prob_a if item.chosen_action == "A" else item.behavior_prob_b)
        self.assertGreater(item.chosen_propensity, 0)
        self.assertLess(item.chosen_propensity, 1)

    def test_no_global_rng_mutation(self):
        before = random.getstate()
        sampler = SeededActionSampler(seed=123)
        for index in range(16):
            _draw(sampler, index)
        self.assertEqual(random.getstate(), before)

    def test_bad_input_never_consumes_rng(self):
        sampler = SeededActionSampler(seed=11)
        bad = (float("nan"), float("inf"), -0.1, 1.01, True, "0.5")
        for value in bad:
            with self.assertRaises(ValueError):
                sampler.sample(value, 0.5, input_ids_sha256=_DIGEST,
                               partition="TRAIN", episode_id="x", policy_id="frozen")
        for a, b in ((0.1, 0.1), (0.1, 0.8)):
            with self.assertRaises(ValueError):
                sampler.sample(a, b, input_ids_sha256=_DIGEST, partition="TRAIN",
                               episode_id="x", policy_id="frozen")
        for keyword, value in (
            ("partition", "UNKNOWN"), ("episode_id", " "),
            ("input_ids_sha256", "not-digest"), ("policy_id", "../unsafe"),
        ):
            values = dict(input_ids_sha256=_DIGEST, partition="TRAIN",
                          episode_id="x", policy_id="frozen")
            values[keyword] = value
            with self.assertRaises(ValueError):
                sampler.sample(0.5, 0.5, **values)
        self.assertEqual(sampler.draws_used, 0)
        self.assertEqual(_draw(sampler, 0), _draw(SeededActionSampler(seed=11), 0))

    def test_positive_support_and_empirical_exploration(self):
        sampler = SeededActionSampler(seed=71, exploration_rate=0.25)
        rows = [_draw(sampler, i, a=0.0) for i in range(4000)]
        for item in rows:
            self.assertEqual(item.behavior_prob_a, 0.125)
            self.assertEqual(item.behavior_prob_b, 0.875)
            self.assertGreater(item.chosen_propensity, 0.0)
            self.assertTrue(0 <= item.random_draw < 1)
        empirical_a = sum(item.chosen_action == "A" for item in rows) / len(rows)
        self.assertLess(abs(empirical_a - 0.125), 0.025)

    def test_epsilon_endpoints(self):
        direct = _draw(SeededActionSampler(seed=10, exploration_rate=0), 0, a=0.8)
        self.assertAlmostEqual(direct.behavior_prob_a, 0.8)
        self.assertAlmostEqual(direct.behavior_prob_b, 0.2)
        full = _draw(SeededActionSampler(seed=10, exploration_rate=1), 0, a=0.8)
        self.assertEqual((full.behavior_prob_a, full.behavior_prob_b), (0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "positive support"):
            _draw(SeededActionSampler(seed=10, exploration_rate=0), 0, a=0.0)

    def test_eval_partition_and_no_outcome_argument(self):
        sampler = SeededActionSampler(seed=31)
        row = sampler.sample(0.4, 0.6, input_ids_sha256=_DIGEST,
                             partition="EVAL", episode_id="e1", policy_id="off")
        self.assertEqual(row.partition, "EVAL")
        self.assertFalse(hasattr(row, "reward"))
        with self.assertRaises(TypeError):
            sampler.sample(0.4, 0.6, input_ids_sha256=_DIGEST,
                           partition="EVAL", episode_id="e2", policy_id="off",
                           measured_quality=1.0)
        self.assertEqual(sampler.draws_used, 1)

    def test_bounded_seed_and_exploration(self):
        for seed in (-1, 2**32, True, 1.5):
            with self.assertRaises(ValueError):
                SeededActionSampler(seed=seed)
        for epsilon in (-0.1, 1.1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                SeededActionSampler(seed=1, exploration_rate=epsilon)
