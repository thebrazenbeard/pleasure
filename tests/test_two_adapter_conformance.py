"""Synthetic two-model contract tests, NOT an installed host-security probe.

The fake host below models a *single process* authority oracle. Its replay
ledger deliberately fails a restart/rollback check, so that a green unit run
cannot be mistaken for durable cross-reader JTI protection.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
import unittest

from pleasure_experiment import ExperimentalPleasureRuntime, Observation, PilotConfig
from pleasure_experiment.host_adapter import (
    HostBoundT0Adapter,
    HostPermission,
    HostPrincipal,
    NotAuthorized,
)


class SyntheticHostAuthority:
    """In-memory currentness oracle; NEVER a production authority provider."""

    def __init__(self, permissions):
        self._issued = {p.principal: p for p in permissions}
        # This is a synthetic expected-transition oracle, not a trusted host.
        self._expected = {p.principal: (observation(p), 100) for p in permissions}
        self._spent = set()
        self._lock = Lock()

    def consume(self, permission, principal, observed, elapsed_ms, now):
        with self._lock:
            expected = self._issued.get(principal)
            if (
                expected != permission
                or self._expected.get(principal) != (observed, elapsed_ms)
                or permission.jti in self._spent
                or not now < permission.expires_at
            ):
                return False
            self._spent.add(permission.jti)
            return True

    def revoke(self, principal):
        with self._lock:
            self._issued.pop(principal, None)


def pair():
    principals = [
        HostPrincipal(
            model_id="synthetic-model-A",
            model_revision="sha256:weights-A",
            subject_id="synthetic-subject-A",
            runtime_instance_id="synthetic-instance-A",
            adapter_id="synthetic-adapter-A",
            adapter_revision="sha256:adapter-A",
            principal_id="synthetic-principal-A",
            host_id="synthetic-host",
        ),
        HostPrincipal(
            model_id="synthetic-model-B",
            model_revision="sha256:weights-B",
            subject_id="synthetic-subject-B",
            runtime_instance_id="synthetic-instance-B",
            adapter_id="synthetic-adapter-B",
            adapter_revision="sha256:adapter-B",
            principal_id="synthetic-principal-B",
            host_id="synthetic-host",
        ),
    ]
    permissions = [
        HostPermission(
            principal=p,
            capability_id="pleasure.t0.step",
            operation="advance",
            context_id=f"synthetic-context-{i}",
            consent_ref=f"synthetic-consent-{i}",
            lease_ref=f"synthetic-lease-{i}",
            effect_scope="T0_SIDECAR_ONLY",
            jti=f"synthetic-jti-{i}",
            expires_at=200.0,
        )
        for i, p in enumerate(principals)
    ]
    oracle = SyntheticHostAuthority(permissions)
    adapters = []
    for p in principals:
        engine = ExperimentalPleasureRuntime(
            PilotConfig(
                model_id=p.model_id,
                model_revision=p.model_revision,
                subject_id=p.subject_id,
                runtime_instance_id=p.runtime_instance_id,
                evaluator_id="synthetic-evaluator",
                measurement_definition_id="synthetic-rubric",
            )
        )
        adapters.append(HostBoundT0Adapter(p, engine, oracle.consume))
    return principals, permissions, oracle, adapters


def observation(permission, evidence="synthetic-evidence"):
    return Observation(
        subject_id=permission.principal.subject_id,
        evaluator_id="synthetic-evaluator",
        context_id=permission.context_id,
        evidence_id=evidence,
        reward=0.5,
    )


class TwoDistinctAdapterConformance(unittest.TestCase):
    def test_two_adapters_share_code_but_neither_state_nor_identity(self):
        _, permissions, _, adapters = pair()
        a = adapters[0].advance(permissions[0], observation(permissions[0]), elapsed_ms=100, now=100.0)
        self.assertEqual(a.model_id, "synthetic-model-A")
        self.assertEqual(adapters[1].state.sequence, 0)
        b = adapters[1].advance(permissions[1], observation(permissions[1]), elapsed_ms=100, now=100.0)
        self.assertEqual(b.model_id, "synthetic-model-B")
        self.assertEqual(b.state_before.sequence, 0)
        self.assertEqual(a.authorized_effects, ())
        self.assertEqual(b.authorized_effects, ())

    def test_adapter_swap_and_permission_wrap_both_denied_without_mutation(self):
        _, permissions, _, adapters = pair()
        with self.assertRaises(NotAuthorized):
            adapters[1].advance(permissions[0], observation(permissions[0]), elapsed_ms=100, now=100.0)
        with self.assertRaises(NotAuthorized):
            adapters[0].advance(permissions[1], observation(permissions[1]), elapsed_ms=100, now=100.0)
        self.assertEqual([a.state.sequence for a in adapters], [0, 0])

    def test_no_transfer_of_identity_consent_lease_scope_or_jti(self):
        _, permissions, _, adapters = pair()
        a, b = permissions
        changes = (
            {"principal": b.principal},
            {"consent_ref": b.consent_ref},
            {"lease_ref": b.lease_ref},
            {"effect_scope": "PROTECTED_EFFECT"},
            {"jti": b.jti},
            {"context_id": b.context_id},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(NotAuthorized):
                adapters[0].advance(replace(a, **change), observation(a), elapsed_ms=100, now=100.0)
            self.assertEqual(adapters[0].state.sequence, 0)
        # No failed attempt may consume A's legitimate one-use permission.
        self.assertEqual(adapters[0].advance(a, observation(a), elapsed_ms=100, now=100.0).sequence, 1)

    def test_revoked_expired_missing_or_forged_host_verifier_denied(self):
        _, permissions, oracle, adapters = pair()
        oracle.revoke(permissions[0].principal)
        with self.assertRaises(NotAuthorized):
            adapters[0].advance(permissions[0], observation(permissions[0]), elapsed_ms=100, now=100.0)
        self.assertEqual(adapters[0].state.sequence, 0)
        with self.assertRaises(NotAuthorized):
            adapters[1].advance(permissions[1], observation(permissions[1]), elapsed_ms=100, now=200.0)
        self.assertEqual(adapters[1].state.sequence, 0)

    def test_jti_replay_and_duplicate_parallel_consume_denied(self):
        _, permissions, _, adapters = pair()
        p = permissions[0]
        def use(_):
            try:
                adapters[0].advance(p, observation(p), elapsed_ms=100, now=100.0)
                return True
            except NotAuthorized:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(use, range(8)))
        self.assertEqual(results.count(True), 1)
        self.assertEqual(adapters[0].state.sequence, 1)

    def test_jti_is_global_to_synthetic_oracle_not_namespaced_per_adapter(self):
        principals, permissions, _, adapters = pair()
        # Both otherwise correctly scoped permissions incorrectly share a JTI.
        collision = replace(permissions[1], jti=permissions[0].jti)
        oracle = SyntheticHostAuthority([permissions[0], collision])
        engines = [a._engine for a in adapters]  # Synthetic test access only.
        adapters = [
            HostBoundT0Adapter(principal, engine, oracle.consume)
            for principal, engine in zip(principals, engines)
        ]
        adapters[0].advance(permissions[0], observation(permissions[0]), elapsed_ms=100, now=100.0)
        with self.assertRaises(NotAuthorized):
            adapters[1].advance(collision, observation(collision), elapsed_ms=100, now=100.0)
        self.assertEqual(adapters[1].state.sequence, 0)

    def test_verifier_exception_and_truthy_nonboolean_never_authorize(self):
        principal, permission = pair()[0][0], pair()[1][0]
        def engine():
            return ExperimentalPleasureRuntime(PilotConfig(
                model_id=principal.model_id,
                model_revision=principal.model_revision,
                subject_id=principal.subject_id,
                runtime_instance_id=principal.runtime_instance_id,
                evaluator_id="synthetic-evaluator",
                measurement_definition_id="synthetic-rubric",
            ))
        for consume in (
            lambda *_: 1,
            lambda *_: None,
            lambda *_: (_ for _ in ()).throw(ConnectionError("host unavailable")),
        ):
            with self.subTest(consume=consume):
                adapter = HostBoundT0Adapter(principal, engine(), consume)
                with self.assertRaises(NotAuthorized):
                    adapter.advance(permission, observation(permission), elapsed_ms=100, now=100.0)
                self.assertEqual(adapter.state.sequence, 0)

    def test_source_oracle_rolls_back_and_reaccepts_jti_on_restart(self):
        # Deliberately demonstrates the missing durable host requirement.
        principals, permissions, _, adapters = pair()
        p = permissions[0]
        adapters[0].advance(p, observation(p), elapsed_ms=100, now=100.0)
        restarted = SyntheticHostAuthority(permissions)
        self.assertTrue(restarted.consume(p, principals[0], observation(p), 100, 100.0))

    def test_exact_observation_and_duration_are_host_callback_inputs(self):
        _, permissions, oracle, adapters = pair()
        p = permissions[0]
        altered = replace(observation(p), reward=1.0)
        with self.assertRaises(NotAuthorized):
            adapters[0].advance(p, altered, elapsed_ms=100, now=100.0)
        with self.assertRaises(NotAuthorized):
            adapters[0].advance(p, observation(p), elapsed_ms=101, now=100.0)
        for invalid in (0, 60001, True):
            with self.subTest(invalid=invalid), self.assertRaises(NotAuthorized):
                adapters[0].advance(p, observation(p), elapsed_ms=invalid, now=100.0)
        self.assertEqual(adapters[0].state.sequence, 0)
        self.assertEqual(adapters[0].advance(p, observation(p), elapsed_ms=100, now=100.0).sequence, 1)

    def test_caller_controlled_verifier_is_not_real_host_authority(self):
        # This deliberately proves why same-process facade tests CANNOT certify
        # a trusted caller, consent, lease or non-forgeable authorization.
        _, permissions, _, adapters = pair()
        principal = permissions[0].principal
        unsafe = HostBoundT0Adapter(principal, adapters[0]._engine, lambda *_: True)
        forged = replace(permissions[0], consent_ref="unverified-consent")
        receipt = unsafe.advance(forged, observation(forged), elapsed_ms=100, now=100.0)
        self.assertEqual(receipt.sequence, 1)
        self.assertEqual(receipt.authorized_effects, ())


if __name__ == "__main__":
    unittest.main()
