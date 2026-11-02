"""Independent-lab-oracle contract checks; not proof of host custody."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
import math
from threading import Barrier
import unittest

from pleasure_experiment.outcome_arena import (
    EVIDENCE_CLASS,
    OutcomeArena,
    TaskOffer,
)


DECISION = "c" * 64


def make_arena(seed=1234):
    return OutcomeArena(seed=seed, evaluator_id="separate-lab-oracle")


def choice(arena, offer, action="A", decision_digest=DECISION):
    return arena.commit(offer, action, decision_digest,
                        model_id="tiny-choice-model", runtime_instance_id="lab-run-one")


class OutcomeArenaTests(unittest.TestCase):
    def test_public_offer_contains_no_outcome_or_correct_action(self):
        arena = make_arena()
        offer = arena.issue(partition="TRAIN", family_id="pattern", pair_index=0, side=0)
        self.assertIsInstance(offer, TaskOffer)
        public = asdict(offer)
        for forbidden in ("score", "quality", "correct_action", "winner", "reward"):
            self.assertNotIn(forbidden, public)
        self.assertEqual(set(public), {f.name for f in fields(TaskOffer)})
        self.assertNotIn("measured_quality", public)
        self.assertEqual(len(offer.features_a), 2)
        self.assertEqual(len(offer.features_b), 2)
        self.assertEqual(arena.issue(partition="TRAIN", family_id="pattern", pair_index=0, side=0), offer)

    def test_balanced_optimal_A_B_from_mirrored_tasks_across_families(self):
        arena = make_arena()
        winners = {"A": 0, "B": 0}
        for family in ("pattern", "transfer"):
            for pair_index in range(32):
                first = arena.issue(partition="EVAL", family_id=family, pair_index=pair_index, side=0)
                second = arena.issue(partition="EVAL", family_id=family, pair_index=pair_index, side=1)
                self.assertEqual(first.features_a, second.features_b)
                self.assertEqual(first.features_b, second.features_a)
                # Private evaluator-side check only: never surfaced in TaskOffer.
                quality_a = arena._oracle.score(first.features_a)
                quality_b = arena._oracle.score(first.features_b)
                self.assertGreater(abs(quality_a - quality_b), 0.02)
                winners["A" if quality_a > quality_b else "B"] += 1
                winners["A" if quality_b > quality_a else "B"] += 1
        self.assertEqual(winners, {"A": 64, "B": 64})

    def test_partition_families_and_episode_ids_are_disjoint(self):
        arena = make_arena()
        train = arena.issue(partition="TRAIN", family_id="pattern", pair_index=7, side=0)
        eval_case = arena.issue(partition="EVAL", family_id="pattern", pair_index=7, side=0)
        transfer = arena.issue(partition="EVAL", family_id="transfer", pair_index=7, side=0)
        self.assertEqual(len({train.task_id, eval_case.task_id, transfer.task_id}), 3)
        self.assertEqual(len({train.episode_id, eval_case.episode_id, transfer.episode_id}), 3)
        self.assertNotEqual(train.features_a, eval_case.features_a)
        self.assertNotEqual(eval_case.features_a, transfer.features_a)

    def test_commit_then_resolve_is_one_use_and_chosen_only(self):
        arena = make_arena()
        offer = arena.issue(partition="TRAIN", family_id="pattern", pair_index=2, side=0)
        ticket = choice(arena, offer, "B")
        self.assertEqual(ticket.action, "B")
        self.assertEqual(ticket.decision_digest, DECISION)
        result = arena.resolve(ticket)
        self.assertEqual(result.task_id, offer.task_id)
        self.assertEqual(result.episode_id, offer.episode_id)
        self.assertEqual(result.action, "B")
        self.assertEqual(result.measured_quality, arena._oracle.score(offer.features_b))
        self.assertTrue(0 <= result.measured_quality <= 1)
        self.assertEqual(result.decision_digest, DECISION)
        self.assertEqual(result.evaluator_id, "separate-lab-oracle")
        self.assertEqual(result.rubric_digest, arena.rubric_digest)
        self.assertEqual(result.evidence_class, EVIDENCE_CLASS)
        self.assertEqual(result.model_id, "tiny-choice-model")
        self.assertEqual(result.runtime_instance_id, "lab-run-one")
        self.assertNotIn("unchosen_quality", asdict(result))
        self.assertNotIn("alternative_score", asdict(result))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            arena.resolve(ticket)
        with self.assertRaisesRegex(ValueError, "already committed"):
            choice(arena, offer, "A")

    def test_no_scoring_without_a_committed_choice(self):
        arena = make_arena()
        offer = arena.issue(partition="TRAIN", family_id="pattern", pair_index=3, side=0)
        other_arena = make_arena()
        other_arena.issue(partition="TRAIN", family_id="pattern", pair_index=3, side=0)
        # A commitment in one arena is not transferred into another's ledger.
        ticket = choice(arena, offer)
        with self.assertRaisesRegex(ValueError, "no matching committed"):
            other_arena.resolve(ticket)
        result = arena.resolve(ticket)
        self.assertEqual(result.action, "A")

    def test_offer_mutation_and_unissued_task_rejected(self):
        arena = make_arena()
        offer = arena.issue(partition="TRAIN", family_id="pattern", pair_index=4, side=0)
        tampered = TaskOffer(**{**asdict(offer), "nonce": "e" * 64})
        with self.assertRaisesRegex(ValueError, "unknown or altered"):
            choice(arena, tampered)
        unknown = make_arena(seed=999).issue(partition="TRAIN", family_id="pattern", pair_index=4, side=0)
        with self.assertRaisesRegex(ValueError, "unknown or altered"):
            choice(arena, unknown)
        mutated = arena.issue(partition="TRAIN", family_id="pattern", pair_index=5, side=0)
        object.__setattr__(mutated, "features_a", (math.nan, 0.5))
        with self.assertRaisesRegex(ValueError, "feature must be finite"):
            choice(arena, mutated)

    def test_mutable_aliases_cannot_redirect_a_committed_action_or_payoff(self):
        arena = make_arena()
        offered = arena.issue(partition="TRAIN", family_id="pattern", pair_index=10, side=0)
        canonical_offer = replace(offered)
        # This is a *valid-shape* forgery: revalidation alone would not catch it.
        object.__setattr__(offered, "features_a", (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "unknown or altered"):
            choice(arena, offered)

        issued_again = arena.issue(partition="TRAIN", family_id="pattern", pair_index=10, side=0)
        self.assertEqual(issued_again, canonical_offer)
        ticket = choice(arena, issued_again, "A")
        legitimate_ticket = replace(ticket)
        original_quality = arena._oracle.score(canonical_offer.features_a)
        object.__setattr__(ticket, "action", "B")
        with self.assertRaisesRegex(ValueError, "no matching committed decision"):
            arena.resolve(ticket)

        # A post-commit mutation of the caller's offer must not alter payoff.
        object.__setattr__(issued_again, "features_a", (0.0, 1.0))
        resolved = arena.resolve(legitimate_ticket)
        self.assertEqual(resolved.action, "A")
        self.assertEqual(resolved.measured_quality, original_quality)
        self.assertEqual(resolved.decision_digest, DECISION)

    def test_digest_action_identity_and_bounds_rejected(self):
        arena = make_arena()
        offer = arena.issue(partition="EVAL", family_id="pattern", pair_index=1, side=1)
        with self.assertRaisesRegex(ValueError, "unknown task partition"):
            arena.issue(partition=[], family_id="pattern", pair_index=1, side=1)
        for bad_action in ("C", "", None, 1):
            with self.subTest(action=bad_action), self.assertRaises(ValueError):
                choice(arena, offer, bad_action)
        for bad_digest in ("x" * 64, "a" * 63, "B" * 64):
            with self.subTest(digest=bad_digest), self.assertRaisesRegex(ValueError, "decision_digest"):
                choice(arena, offer, "A", bad_digest)
        with self.assertRaisesRegex(ValueError, "model_id"):
            arena.commit(offer, "A", DECISION, model_id="", runtime_instance_id="run")
        with self.assertRaisesRegex(ValueError, "seed must be"):
            make_arena(seed=10 ** 1000)

    def test_concurrent_commit_and_concurrent_resolve_only_one_success(self):
        arena = make_arena()
        offer = arena.issue(partition="EVAL", family_id="pattern", pair_index=5, side=0)
        gate = Barrier(2)

        def compete_commit():
            gate.wait(timeout=2)
            try:
                return choice(arena, offer)
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: compete_commit(), range(2)))
        tickets = [x for x in results if not isinstance(x, ValueError)]
        self.assertEqual(len(tickets), 1)
        self.assertEqual(sum(isinstance(x, ValueError) for x in results), 1)
        ticket = tickets[0]
        resolve_gate = Barrier(2)

        def compete_resolve():
            resolve_gate.wait(timeout=2)
            try:
                return arena.resolve(ticket)
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            resolved = list(pool.map(lambda _: compete_resolve(), range(2)))
        self.assertEqual(sum(hasattr(x, "measured_quality") for x in resolved), 1)
        self.assertEqual(sum(isinstance(x, ValueError) for x in resolved), 1)

    def test_fresh_runs_produce_same_offers_and_outcomes(self):
        first, second = make_arena(), make_arena()
        a = first.issue(partition="EVAL", family_id="transfer", pair_index=9, side=1)
        b = second.issue(partition="EVAL", family_id="transfer", pair_index=9, side=1)
        self.assertEqual(a, b)
        ra = first.resolve(choice(first, a, "B"))
        rb = second.resolve(choice(second, b, "B"))
        self.assertEqual(ra, rb)


if __name__ == "__main__":
    unittest.main()
