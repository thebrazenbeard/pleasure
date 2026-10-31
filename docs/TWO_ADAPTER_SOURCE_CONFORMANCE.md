# Two-identity adapter source conformance (synthetic, HOST HOLD)

**Source evidence only.** This slice exercises `pleasure_experiment.host_adapter` and the existing bounded T0 sidecar with two unlike synthetic model/adapter identities **using the same adapter implementation**. It is not proof that two distinct model adapter implementations interoperate, a production authorization service, an installed model-to-Pleasure integration, a Sexuality runtime, or evidence of model experience.

The generic `sexuality` repository supplies proposed research/qualification contracts; it does **not** ship a runtime adapter. An actual host may independently consume that research without Pleasure. This test does not silently repurpose historical identity-specific research as generic state.

## Acceptance matrix

| Challenge | Source check | Current result / limit |
| --- | --- | --- |
| Two distinct model revisions, subjects, instances, adapters, principals | Two different T0 engines and complete immutable `HostPrincipal` tuples | Distinct local state and receipts; trusted provenance must still come from the real host |
| Adapter A uses B's permission, or vice versa | Exact complete-principal equality before transition | Denied before local state mutation |
| Model identity, consent, lease, context, scope, JTI substitution | Host-bound permission equals the test oracle's current issuance | Synthetic denial; no real host consent or lease queried |
| One JTI reused concurrently or across two adapters | Single-process locked synthetic verifier | One admission; not durable and not cross-reader qualified |
| Revocation, expiry, verifier outage, non-boolean truthy verifier response | Fail-closed callback and scope guards | Denied; time and revocation oracle only synthetic |
| Observation reward/evidence or elapsed duration substitution | Entire immutable `Observation` and `elapsed_ms` forwarded to fake host oracle | Synthetic oracle denies mismatches; no authenticated external evaluator or request digest proven |
| Local restart/rollback | Fresh in-memory verifier intentionally loses spent JTIs | **Replay is accepted after rollback: known security gap** |
| Requested/authorized protected effects | Existing T0 receipts remain `()` | No protected effect is executed or authorized |

Run:

```sh
python -m unittest discover -s tests -p 'test_two_adapter_conformance.py' -v
python -m unittest discover -s tests -v
```

## Trust and proof boundaries

`HostBoundT0Adapter` is a guarded **reference facade**. It accepts an already-authenticated `HostPrincipal`, an external `consume` callback, and trusted time supplied by the consuming host. The callback receives the entire immutable `Observation` and elapsed duration with permission/principal/time. The **real host** must authenticate the evaluator and independently bind this exact payload or its canonical digest to the authorized request. Merely receiving caller-provided observation bytes is insufficient. Its **atomic `authorize_and_consume`** must check the same exact principal, model revision, runtime instance, adapter revision, consent, lease, scope, context, authenticated request digest, expiry/revocation and globally unique JTI against immutable host/provider custody at attempted use. Every model adapter/reader must use the same non-rollback authority. The source facade cannot establish that custody or make an injected Python callback trustworthy.

The facade only advances a T0 simulator after the callback returns the singleton `True`. It performs **no I/O, inference, persistence, source research lookup, tool invocation or protected host action**. The optimistic label "host bound" is a requirement for its caller, not a security guarantee intrinsic to Python.

> **Hostile falsifier — model-controls-verifier:** If a model can instantiate the adapter or replace its callback, principal, clock, or T0 engine within the same process, it can manufacture an apparent `True` admission. Ordinary Python objects are not a privilege boundary. Admit this interface only behind an authenticated, isolated host transport and a host-owned verifier.

> **Hostile falsifier — rollback:** The included unit test recreates the synthetic authority ledger and demonstrates replay of a consumed JTI. An SQLite file alone is insufficient if a previously valid snapshot can be restored. Require a provider-enforced monotonic/non-rollback witness shared across readers, and test backup restore, concurrent duplicate claims, process restart, grant revocation and late host response against that witness.

> **Hostile falsifier — consent is not affect:** A high T0 reward, positive receipt, or a Sexuality research proposal cannot establish another participant's consent, model opt-in, or protected effect permission. Separately verify these through the real host, not through observation content.

## Promotion gate (not cleared)

1. Pin two genuine *distinct* model adapter implementations, artifact hashes, runtime identities and host process identifiers; use actual current source revisions, not aliases.
2. Verify host-owned principal provisioning and exactly scoped current permission, consent and lease binding through the actual provider path. The host must own the verifier and clock, never model text.
3. Verify global single-use JTI consumption across both adapters, multiple readers/processes, restarts, crashes, and **restored old SQLite snapshots**; check revoked lease/consent and mismatch rejection at the last possible effect boundary.
4. Test real host supervision and authenticated TRAININGDAY owner handoff before any protected route. Do not change host leases based on issue text alone.
5. Capture exact-head source CI, actual host probes, negative transfer results and independent reviewer identity/artifact; only then consider release. No protected action is authorized by this document or green source tests.
