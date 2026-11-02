"""Process-isolation and global-decision-seal adversarial regression tests."""

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

from pleasure_experiment.isolated_outcome_evaluator import (
    DEFAULT_ARMS, IsolatedOutcomeEvaluator, NOVEL_EVAL_FAMILIES,
    NOVEL_IN_DISTRIBUTION_FAMILY, NOVEL_OOD_FAMILIES, PROTOCOL, TRAIN_ARM,
    _novel_quality,
)


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


class IsolatedOutcomeEvaluatorTests(unittest.TestCase):
    def evaluator(self, *, seed: int = 29, arms: tuple[str, ...] = ("arm_one", "arm_two"),
                  suite: str = "novel_v2") -> IsolatedOutcomeEvaluator:
        return IsolatedOutcomeEvaluator(
            seed=seed, evaluator_id="isolated_unit_v3", arms=arms,
            suite=suite, training_pairs=1, heldout_pairs_per_generator=1,
            timeout_seconds=3.0,
        )

    def _commit(self, study: IsolatedOutcomeEvaluator, offer, arm: str,
                action: str = "A"):
        return study.commit(
            offer, action, _digest(f"{arm}:{offer.task_id}:{action}"),
            arm=arm, model_id=f"model-{arm}", runtime_instance_id=f"run-{arm}",
        )

    def _resolve_train(self, study: IsolatedOutcomeEvaluator):
        train = study.issue(
            partition="TRAIN", arm=TRAIN_ARM, family_id=study.manifest.train_family,
            pair_index=0, side=0,
        )
        ticket = self._commit(study, train, TRAIN_ARM)
        receipt = study.resolve(ticket, arm=TRAIN_ARM)
        self.assertEqual(receipt.partition, "TRAIN")
        self.assertTrue(0 <= receipt.measured_quality <= 1)
        return train, ticket, receipt

    def _all_eval_commits(self, study: IsolatedOutcomeEvaluator, arms: tuple[str, ...]):
        issued = []
        for arm_index, arm in enumerate(arms):
            for family in study.eval_families:
                offer = study.issue(partition="EVAL", arm=arm, family_id=family,
                                    pair_index=0, side=0)
                ticket = self._commit(study, offer, arm,
                                      action="A" if arm_index == 0 else "B")
                issued.append((arm, offer, ticket))
        return issued

    def test_distinct_worker_process_and_versioned_manifest(self):
        with self.evaluator() as study:
            self.assertNotEqual(study.worker_pid, os.getpid())
            self.assertEqual(study.provenance["worker_pid"], study.worker_pid)
            self.assertEqual(study.provenance["protocol"], PROTOCOL)
            self.assertEqual(len(study.provenance["session"]), 64)
            self.assertEqual(study.eval_families, NOVEL_EVAL_FAMILIES)
            self.assertEqual(study.primary_ood_families, NOVEL_OOD_FAMILIES)
            self.assertEqual(study.eval_families[0], NOVEL_IN_DISTRIBUTION_FAMILY)
            self.assertIn("v2", study.manifest.train_family)
            self.assertFalse(study.manifest.externally_authenticated)
            self.assertEqual(len(study.manifest.heldout_generator_sha256), 4)
            self.assertTrue(all(len(digest) == 64 for digest in
                                study.manifest.heldout_generator_sha256))

    def test_deterministic_seeds_and_immutable_predeclared_schedule(self):
        first = []
        for seed in (7, 7, 8):
            with self.evaluator(seed=seed) as study:
                offer = study.issue(partition="EVAL", arm="arm_one",
                                    family_id=study.eval_families[-1], pair_index=0)
                first.append(offer)
                with self.assertRaises(ValueError):
                    study.issue(partition="EVAL", arm="arm_one",
                                family_id=study.eval_families[-1], pair_index=0)
                for incorrect in (
                    dict(partition="EVAL", arm="arm_three",
                         family_id=study.eval_families[0], pair_index=0),
                    dict(partition="EVAL", arm="arm_one",
                         family_id=study.manifest.train_family, pair_index=0),
                    dict(partition="TRAIN", arm=TRAIN_ARM,
                         family_id=study.manifest.train_family, pair_index=100),
                    dict(partition="EVAL", arm="arm_one",
                         family_id=study.eval_families[0], pair_index=0, side=1),
                ):
                    with self.assertRaises(ValueError):
                        study.issue(**incorrect)
        self.assertEqual(first[0], first[1])
        self.assertNotEqual(first[0], first[2])

    def test_critical_cross_arm_opposite_actions_cannot_reveal_before_barrier(self):
        """Two matched arms choosing A/B MUST NOT disclose either EVAL payoff early."""
        arms = ("arm_one", "arm_two")
        with self.evaluator(arms=arms) as study:
            train, _, _ = self._resolve_train(study)
            family = study.eval_families[0]
            first_offer = study.issue(partition="EVAL", arm=arms[0],
                                      family_id=family, pair_index=0)
            second_offer = study.issue(partition="EVAL", arm=arms[1],
                                       family_id=family, pair_index=0)
            self.assertEqual(first_offer, second_offer)
            first_ticket = self._commit(study, first_offer, arms[0], "A")
            second_ticket = self._commit(study, second_offer, arms[1], "B")
            for arm, ticket, offer in (
                (arms[0], first_ticket, first_offer),
                (arms[1], second_ticket, second_offer),
            ):
                with self.assertRaises(PermissionError):
                    study.resolve(ticket, arm=arm)
                with self.assertRaises(PermissionError):
                    study.posthoc_evaluate(offer, arm=arm)
            with self.assertRaises(ValueError):
                study.seal_decisions()  # Uncommitted other OOD families.
            with self.assertRaises(ValueError):
                study.finalize_evaluation()
            remaining = []
            for arm in arms:
                for other_family in study.eval_families[1:]:
                    offer = study.issue(partition="EVAL", arm=arm,
                                        family_id=other_family, pair_index=0)
                    remaining.append((arm, offer,
                                      self._commit(study, offer, arm)))
            self.assertEqual(study.seal_decisions()["eval_commits"],
                             2 * len(study.eval_families))
            with self.assertRaises(ValueError):
                study.seal_decisions()
            with self.assertRaises(ValueError):
                study.finalize_evaluation()  # Not all EVAL rewards consumed yet.
            for arm, offer, ticket in (
                (arms[0], first_offer, first_ticket),
                (arms[1], second_offer, second_ticket),
                *remaining,
            ):
                receipt = study.resolve(ticket, arm=arm)
                self.assertTrue(0 <= receipt.measured_quality <= 1)
                self.assertEqual(receipt.action, ticket.action)
                with self.assertRaises(ValueError):
                    study.resolve(ticket, arm=arm)
                with self.assertRaises(PermissionError):
                    study.posthoc_evaluate(offer, arm=arm)
            status = study.finalize_evaluation()
            self.assertTrue(status["all_arms_committed"])
            self.assertEqual(status["arm_count"], len(arms))
            with self.assertRaises(ValueError):
                study.finalize_evaluation()
            left = study.posthoc_evaluate(first_offer, arm=arms[0])
            right = study.posthoc_evaluate(second_offer, arm=arms[1])
            self.assertEqual(left.quality_a, right.quality_a)
            self.assertEqual(left.quality_b, right.quality_b)
            self.assertEqual(left.chosen_quality, left.quality_a)
            self.assertEqual(right.chosen_quality, right.quality_b)
            self.assertAlmostEqual(left.regret, left.oracle_quality - left.chosen_quality)
            with self.assertRaises(PermissionError):
                study.posthoc_evaluate(train, arm=TRAIN_ARM)

    def test_commit_integrity_replay_invalid_actions_and_wrong_arm(self):
        with self.evaluator() as study:
            offer = study.issue(partition="EVAL", arm="arm_one",
                                family_id=study.eval_families[0], pair_index=0)
            with self.assertRaises(ValueError):
                self._commit(study, replace(offer, features_a=(0.1, 0.1)), "arm_one")
            with self.assertRaises(ValueError):
                self._commit(study, offer, "arm_two")
            with self.assertRaises(ValueError):
                self._commit(study, offer, "arm_one", "C")
            with self.assertRaises(ValueError):
                study.commit(offer, "A", "not-sha256", arm="arm_one",
                             model_id="model", runtime_instance_id="instance")
            with self.assertRaises(ValueError):
                study.commit(offer, "A", _digest("d"), arm="arm_one",
                             model_id="x" * 300, runtime_instance_id="instance")
            ticket = self._commit(study, offer, "arm_one")
            with self.assertRaises(ValueError):
                self._commit(study, offer, "arm_one")
            with self.assertRaises(ValueError):
                study.resolve(replace(ticket, commit_id=_digest("spoof")), arm="arm_one")
            with self.assertRaises(ValueError):
                study.resolve(ticket, arm="arm_two")
            with self.assertRaises(PermissionError):
                study.resolve(ticket, arm="arm_one")

    def test_worker_kill_and_timeout_fail_closed(self):
        study = self.evaluator()
        try:
            study._process.kill()  # Deliberate fault injection: privileged test only.
            study._process.wait(timeout=2)
            with self.assertRaises(RuntimeError):
                study.issue(partition="TRAIN", arm=TRAIN_ARM,
                            family_id=study.manifest.train_family, pair_index=0)
            self.assertTrue(study._closed)
        finally:
            study.close(force=True)
        study = self.evaluator()
        try:
            with patch.object(study._responses, "get", side_effect=Empty):
                with self.assertRaises(RuntimeError):
                    study.issue(partition="TRAIN", arm=TRAIN_ARM,
                                family_id=study.manifest.train_family, pair_index=0)
            self.assertTrue(study._closed)
            self.assertIsNotNone(study._process.poll())
        finally:
            study.close(force=True)

    def test_worker_malformed_json_and_oversize_fail_closed(self):
        command = [sys.executable, "-B", "-m",
                   "pleasure_experiment.isolated_outcome_evaluator", "--worker"]
        for malformed in (b"{bad-json}\n", b"z" * 20000 + b"\n",
                          b'{"protocol":"wrong"}\n'):
            run = subprocess.run(command, input=malformed, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=3)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(run.stdout, b"")

    def test_worker_rejects_replayed_request_ids_or_wrong_session(self):
        command = [sys.executable, "-B", "-m",
                   "pleasure_experiment.isolated_outcome_evaluator", "--worker"]
        initialize = {
            "protocol": PROTOCOL, "id": 0, "session": None,
            "command": "initialize",
            "body": {"seed": 8, "evaluator_id": "test", "arms": ["arm_a"],
                     "training_pairs": 1, "heldout_pairs_per_generator": 1,
                     "suite": "novel_v2", "nonce": "0" * 32},
        }
        for replay, wrong_session in ((True, False), (False, True)):
            proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps(initialize) + "\n")
                proc.stdin.flush()
                handshake = json.loads(proc.stdout.readline())
                self.assertTrue(handshake["ok"])
                illicit = {
                    "protocol": PROTOCOL, "id": 0 if replay else 1,
                    "session": "z" * 64 if wrong_session else handshake["result"]["session"],
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

    def test_broken_worker_response_fails_closed(self):
        study = self.evaluator()
        try:
            with patch.object(study._responses, "get", return_value="{not-json}\n"):
                with self.assertRaises(RuntimeError):
                    study.issue(partition="TRAIN", arm=TRAIN_ARM,
                                family_id=study.manifest.train_family,
                                pair_index=0)
            self.assertTrue(study._closed)
            self.assertIsNotNone(study._process.poll())
        finally:
            study.close(force=True)

    def test_legacy_dev_suite_available_but_different_and_v2_nonlinear(self):
        with self.evaluator(suite="legacy_v1") as study:
            self.assertEqual(len(study.eval_families), 3)
            self.assertTrue(all("heldout_eval" in f for f in study.eval_families))
            self._resolve_train(study)
        points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8))
        affine = [_novel_quality(NOVEL_IN_DISTRIBUTION_FAMILY, p) for p in points]
        self.assertEqual(affine, [_novel_quality(
            "sealed_train_affine_v2", p) for p in points])
        for nonlinear in NOVEL_OOD_FAMILIES:
            values = [_novel_quality(nonlinear, p) for p in points]
            self.assertNotEqual(affine, values)

    def test_bounds_on_constructor_and_default_arm_schedule(self):
        self.assertEqual(len(DEFAULT_ARMS), 5)
        for invalid in (
            dict(seed=-1), dict(seed=True), dict(arms=("same", "same")),
            dict(arms=(TRAIN_ARM,)), dict(training_pairs=10000),
            dict(heldout_pairs_per_generator=0), dict(suite="unknown"),
            dict(timeout_seconds=float("nan")),
        ):
            kwargs = dict(seed=1, evaluator_id="e", arms=("one",),
                          training_pairs=1, heldout_pairs_per_generator=1)
            kwargs.update(invalid)
            with self.assertRaises(ValueError):
                IsolatedOutcomeEvaluator(**kwargs)


if __name__ == "__main__":
    unittest.main()
