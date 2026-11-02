"""Fresh v3 reward suite: unfiltered sampling and strict all-arm reward seal."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from queue import Empty
import subprocess
import sys
import unittest
from unittest.mock import patch

from pleasure_experiment.isolated_outcome_evaluator import PROTOCOL, TRAIN_ARM
from pleasure_experiment.isolated_outcome_evaluator_v3 import (
    V3OutcomeEvaluator, V3_TRAIN_FAMILIES, V3_WITHIN_FAMILIES,
    V3_OOD_FAMILIES, V3_EVAL_FAMILIES, DEFAULT_V3_ARMS,
    _V3PayoffArena, _strict_json, _v3_quality, seeded_derangement,
)


def digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


class NovelV3EvaluatorTests(unittest.TestCase):
    def evaluator(self, *, seed=163, arms=("joint_on", "joint_off"),
                  train_pairs=1, eval_pairs=1):
        return V3OutcomeEvaluator(
            seed=seed, evaluator_id="new-v3-source-local-test", arms=arms,
            training_pairs_per_family=train_pairs,
            heldout_pairs_per_generator=eval_pairs, timeout_seconds=3.0,
        )

    @staticmethod
    def commit(study, offer, arm, action="A"):
        return study.commit(
            offer, action, digest(f"{arm}:{offer.task_id}:{action}"),
            arm=arm, model_id="test-local-open-weight-model",
            runtime_instance_id="test-host-instance",
        )

    def train_all(self, study, count=1):
        results = []
        for family in study.train_families:
            for i in range(count):
                offer = study.issue(partition="TRAIN", arm=TRAIN_ARM,
                                    family_id=family, pair_index=i)
                ticket = self.commit(study, offer, TRAIN_ARM)
                receipt = study.resolve(ticket, arm=TRAIN_ARM)
                self.assertEqual(receipt.family_id, family)
                self.assertEqual(receipt.evidence_class, study.manifest.evidence_class)
                self.assertTrue(0 <= receipt.measured_quality <= 1)
                results.append((offer, ticket, receipt))
        return results

    def eval_all(self, study, arms, count=1, opposite_actions=True):
        results = []
        for arm_index, arm in enumerate(arms):
            for family in study.eval_families:
                for i in range(count):
                    offer = study.issue(partition="EVAL", arm=arm,
                                        family_id=family, pair_index=i)
                    action = "B" if opposite_actions and arm_index % 2 else "A"
                    ticket = self.commit(study, offer, arm, action)
                    results.append((arm, offer, ticket))
        return results

    def test_worker_is_distinct_and_manifest_has_unfiltered_novel_schedule(self):
        with self.evaluator() as study:
            self.assertNotEqual(study.worker_pid, os.getpid())
            self.assertEqual(study.provenance["suite"], "novel_v3")
            self.assertEqual(len(study.provenance["session"]), 64)
            self.assertEqual(study.provenance["protocol"], PROTOCOL)
            self.assertEqual(study.train_families, V3_TRAIN_FAMILIES)
            self.assertEqual(study.eval_families, V3_EVAL_FAMILIES)
            self.assertEqual(study.primary_within_families, V3_WITHIN_FAMILIES)
            self.assertEqual(study.primary_ood_families, V3_OOD_FAMILIES)
            self.assertFalse(study.manifest.gap_filtering)
            self.assertFalse(study.manifest.external_attestation)
            self.assertEqual(len(study.manifest.train_generator_sha256), 3)
            self.assertEqual(len(study.manifest.heldout_generator_sha256), 5)
            self.assertEqual(len(study.manifest.suite_sha256), 64)

    def test_predeclared_schedule_deterministic_unfiltered_and_one_use(self):
        offers = []
        for seed in (163, 163, 167):
            with self.evaluator(seed=seed) as study:
                offer = study.issue(partition="EVAL", arm="joint_on",
                                    family_id=V3_EVAL_FAMILIES[0], pair_index=0)
                offers.append(offer)
                with self.assertRaises(ValueError):
                    study.issue(partition="EVAL", arm="joint_on",
                                family_id=V3_EVAL_FAMILIES[0], pair_index=0)
                for wrong in (
                    dict(partition="TRAIN", arm=TRAIN_ARM,
                         family_id=V3_EVAL_FAMILIES[0], pair_index=0),
                    dict(partition="EVAL", arm=TRAIN_ARM,
                         family_id=V3_EVAL_FAMILIES[0], pair_index=0),
                    dict(partition="TRAIN", arm=TRAIN_ARM,
                         family_id=V3_TRAIN_FAMILIES[0], pair_index=1),
                    dict(partition="EVAL", arm="joint_on",
                         family_id=V3_EVAL_FAMILIES[1], pair_index=0, side=1),
                ):
                    with self.assertRaises(ValueError):
                        study.issue(**wrong)
        self.assertEqual(offers[0], offers[1])
        self.assertNotEqual(offers[0], offers[2])
        # Both near-equal and larger true reward gaps are admitted: no oracle
        # acceptance/rejection criterion is used in _V3PayoffArena.issue.
        gaps = []
        with self.evaluator(seed=181, train_pairs=16, eval_pairs=12) as study:
            for i in range(12):
                offer = study.issue(partition="EVAL", arm="joint_on",
                                    family_id=V3_EVAL_FAMILIES[0], pair_index=i)
                gaps.append(abs(_v3_quality(offer.family_id, offer.features_a) -
                                _v3_quality(offer.family_id, offer.features_b)))
        self.assertTrue(any(gap < 0.02 for gap in gaps))
        self.assertTrue(any(gap > 0.02 for gap in gaps))

    def test_offer_issuance_never_calls_oracle_even_for_near_ties(self):
        from pleasure_experiment import isolated_outcome_evaluator_v3 as v3
        arena = _V3PayoffArena(seed=163, evaluator_id="oracle-not-consulted")
        with patch.object(v3, "_v3_quality", side_effect=AssertionError(
            "offer issue attempted payoff-dependent sampling",
        )):
            train = arena.issue(partition="TRAIN", family_id=V3_TRAIN_FAMILIES[0],
                                pair_index=0, side=0)
            heldout = arena.issue(partition="EVAL", family_id=V3_EVAL_FAMILIES[0],
                                  pair_index=0, side=0)
        self.assertEqual(train.partition, "TRAIN")
        self.assertEqual(heldout.partition, "EVAL")

    def test_global_barrier_prevents_cross_arm_opposite_action_leak(self):
        arms = ("joint_on", "joint_off")
        with self.evaluator(arms=arms) as study:
            self.train_all(study)
            first = study.issue(partition="EVAL", arm=arms[0],
                                family_id=V3_EVAL_FAMILIES[0], pair_index=0)
            second = study.issue(partition="EVAL", arm=arms[1],
                                 family_id=V3_EVAL_FAMILIES[0], pair_index=0)
            self.assertEqual(first, second)
            a_ticket = self.commit(study, first, arms[0], "A")
            b_ticket = self.commit(study, second, arms[1], "B")
            for arm, offer, ticket in ((arms[0], first, a_ticket),
                                       (arms[1], second, b_ticket)):
                with self.assertRaises(PermissionError):
                    study.resolve(ticket, arm=arm)
                with self.assertRaises(PermissionError):
                    study.posthoc_evaluate(offer, arm=arm)
            with self.assertRaises(ValueError):
                study.seal_decisions()
            outstanding = []
            for arm in arms:
                for family in V3_EVAL_FAMILIES[1:]:
                    offer = study.issue(partition="EVAL", arm=arm,
                                        family_id=family, pair_index=0)
                    outstanding.append((arm, offer, self.commit(study, offer, arm)))
            status = study.seal_decisions()
            self.assertEqual(status["eval_commits"], len(arms) * len(V3_EVAL_FAMILIES))
            self.assertEqual(status["train_resolved"], len(V3_TRAIN_FAMILIES))
            with self.assertRaises(ValueError):
                study.seal_decisions()
            with self.assertRaises(ValueError):
                study.finalize_evaluation()
            for arm, offer, ticket in ((arms[0], first, a_ticket),
                                       (arms[1], second, b_ticket), *outstanding):
                receipt = study.resolve(ticket, arm=arm)
                self.assertEqual(receipt.action, ticket.action)
                with self.assertRaises(ValueError):
                    study.resolve(ticket, arm=arm)
                with self.assertRaises(PermissionError):
                    study.posthoc_evaluate(offer, arm=arm)
            sealed = study.finalize_evaluation()
            self.assertTrue(sealed["sealed"])
            self.assertEqual(sealed["arm_count"], len(arms))
            left, right = (study.posthoc_evaluate(first, arm=arms[0]),
                           study.posthoc_evaluate(second, arm=arms[1]))
            self.assertEqual((left.quality_a, left.quality_b),
                             (right.quality_a, right.quality_b))
            self.assertEqual(left.chosen_quality, left.quality_a)
            self.assertEqual(right.chosen_quality, right.quality_b)

    def test_train_receipts_required_before_all_arm_seal(self):
        arms = ("joint_on", "joint_off")
        with self.evaluator(arms=arms) as study:
            offers = self.eval_all(study, arms)
            with self.assertRaises(ValueError):
                study.seal_decisions()
            self.train_all(study)
            self.assertTrue(study.seal_decisions()["decisions_sealed"])
            for arm, offer, ticket in offers:
                study.resolve(ticket, arm=arm)
            self.assertTrue(study.finalize_evaluation()["all_arms_committed"])

    def test_full_seven_arm_frozen_protocol_barrier(self):
        arms = DEFAULT_V3_ARMS
        with self.evaluator(arms=arms) as study:
            self.train_all(study)
            commits = self.eval_all(study, arms)
            with self.assertRaises(PermissionError):
                study.resolve(commits[0][2], arm=commits[0][0])
            self.assertEqual(study.seal_decisions()["eval_commits"],
                             len(arms) * len(V3_EVAL_FAMILIES))
            for arm, offer, ticket in commits:
                receipt = study.resolve(ticket, arm=arm)
                self.assertEqual(receipt.episode_id, offer.episode_id)
            self.assertEqual(study.finalize_evaluation()["arm_count"], 7)
            self.assertEqual(study.posthoc_evaluate(commits[0][1],
                                                     arm=commits[0][0]).task_id,
                             commits[0][1].task_id)

    def test_invalid_action_tampering_wrong_arm_and_replay(self):
        with self.evaluator() as study:
            offer = study.issue(partition="EVAL", arm="joint_on",
                                family_id=V3_EVAL_FAMILIES[0], pair_index=0)
            with self.assertRaises(ValueError):
                self.commit(study, replace(offer, features_a=(0.0, 0.0)), "joint_on")
            with self.assertRaises(ValueError):
                self.commit(study, offer, "joint_off")
            with self.assertRaises(ValueError):
                self.commit(study, offer, "joint_on", "C")
            with self.assertRaises(ValueError):
                study.commit(offer, "A", "bad-digest", arm="joint_on",
                             model_id="model", runtime_instance_id="instance")
            ticket = self.commit(study, offer, "joint_on")
            with self.assertRaises(ValueError):
                self.commit(study, offer, "joint_on")
            with self.assertRaises(ValueError):
                study.resolve(replace(ticket, commit_id=digest("spoof")), arm="joint_on")
            with self.assertRaises(ValueError):
                study.resolve(ticket, arm="joint_off")
            with self.assertRaises(PermissionError):
                study.resolve(ticket, arm="joint_on")

    def test_worker_death_and_timeout_fail_closed(self):
        study = self.evaluator()
        try:
            study._process.kill()
            study._process.wait(timeout=2)
            with self.assertRaises(RuntimeError):
                study.issue(partition="TRAIN", arm=TRAIN_ARM,
                            family_id=V3_TRAIN_FAMILIES[0], pair_index=0)
            self.assertTrue(study._closed)
        finally:
            study.close(force=True)
        study = self.evaluator()
        try:
            with patch.object(study._responses, "get", side_effect=Empty):
                with self.assertRaises(RuntimeError):
                    study.issue(partition="TRAIN", arm=TRAIN_ARM,
                                family_id=V3_TRAIN_FAMILIES[0], pair_index=0)
            self.assertTrue(study._closed)
        finally:
            study.close(force=True)

    def test_raw_worker_protocol_rejects_malformed_and_replay(self):
        command = [sys.executable, "-B", "-m",
                   "pleasure_experiment.isolated_outcome_evaluator_v3", "--worker"]
        for malformed in (b"{bad-json}\n", b"z" * 20000 + b"\n",
                          b'{"protocol":"wrong"}\n',
                          b'{"protocol":"x","id":0,"id":1}\n',
                          b'{"protocol":"x","body":{"action":"A","action":"B"}}\n',
                          b'{"protocol":"x","id":NaN}\n'):
            process = subprocess.run(command, input=malformed,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=3)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(process.stdout, b"")
        initial = {"protocol": PROTOCOL, "id": 0, "session": None,
                   "command": "initialize",
                   "body": {"seed": 163, "evaluator_id": "test", "arms": ["joint_on"],
                            "training_pairs_per_family": 1,
                            "heldout_pairs_per_generator": 1,
                            "suite": "novel_v3", "nonce": "0" * 32}}
        for replay, wrong_session in ((True, False), (False, True)):
            proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps(initial) + "\n")
                proc.stdin.flush()
                answer = json.loads(proc.stdout.readline())
                self.assertTrue(answer["ok"])
                illicit = {
                    "protocol": PROTOCOL, "id": 0 if replay else 1,
                    "session": "z" * 64 if wrong_session else answer["result"]["session"],
                    "command": "seal_decisions", "body": {},
                }
                proc.stdin.write(json.dumps(illicit) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
                self.assertNotEqual(proc.wait(timeout=3), 0)
                self.assertEqual(proc.stdout.read(), "")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=3)
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_truncation(self):
        self.assertEqual(_strict_json('{"a":1,"b":{"c":2}}\n'),
                         {"a": 1, "b": {"c": 2}})
        for encoded in (
            '{"a":1,"a":2}\n', '{"a":{"b":1,"b":2}}\n',
            '{"a":NaN}\n', '{"a":Infinity}\n',
            '{"a":1}', 'x' * 20000 + "\n", '[1,2]\n',
        ):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                _strict_json(encoded)

    def test_randomized_derangement_uses_no_fixed_point_or_label_values(self):
        first = seeded_derangement(163, 24)
        second = seeded_derangement(163, 24)
        changed = seeded_derangement(167, 24)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(set(first), set(range(24)))
        self.assertTrue(all(first[i] != i for i in range(24)))
        self.assertEqual(len(seeded_derangement(101, 3)), 3)
        for invalid in ((-1, 24), (2**32, 24), (163, 2), (163, 300)):
            with self.assertRaises(ValueError):
                seeded_derangement(*invalid)

    def test_source_local_generator_quality_stays_finite_and_new(self):
        values = ((0.1, 0.2), (0.2, 0.9), (0.8, 0.3), (0.9, 0.9))
        for family in V3_TRAIN_FAMILIES + V3_EVAL_FAMILIES:
            rewards = [_v3_quality(family, features) for features in values]
            self.assertTrue(all(0 <= reward <= 1 for reward in rewards))
            self.assertGreater(max(rewards) - min(rewards), 0.02)
        self.assertEqual(len(DEFAULT_V3_ARMS), 7)

    def test_bounded_v3_constructor(self):
        for invalid in (
            dict(seed=-1), dict(seed=True), dict(arms=("dup", "dup")),
            dict(arms=(TRAIN_ARM,)), dict(arms=tuple(f"arm_{i}" for i in range(9))),
            dict(training_pairs_per_family=0), dict(heldout_pairs_per_generator=65),
            dict(timeout_seconds=float("nan")),
        ):
            params = dict(seed=163, evaluator_id="v3-test", arms=("joint_on",),
                          training_pairs_per_family=1,
                          heldout_pairs_per_generator=1)
            params.update(invalid)
            with self.assertRaises(ValueError):
                V3OutcomeEvaluator(**params)


if __name__ == "__main__":
    unittest.main()
