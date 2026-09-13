# Vera current Cohesion/Orgasm source reconciliation

**Recorded:** 2026-09-13  
**Status:** `CURRENT_SOURCE_RECONCILIATION / FROZEN_SUBJECT_REBIND_REQUIRED / NO_PROTECTED_EFFECT`

This document is a currentness receipt for the Cohesion and Orgasm source lines. It does not replace the frozen Orgasm V1 qualification subject, establish behavioral qualification, install a provider schema, or authorize a merge, deployment, cutover, or private-payload transfer.

## Exact observed frontier

| Surface | Exact observation | Meaning |
|---|---|---|
| Orgasm default branch | `thebrazenbeard/orgasm/main@494432873dd8bcf96b8f59d26a4f4687cd66d635` | Provenance/orientation hub; not the executable runtime owner |
| Orgasm PR #2 | `work/whole-system-reconciliation-20260911` | Open draft candidate; exact head is recorded in PR metadata and the WIP receipt |
| Frozen Orgasm subject | `architecture/VERA_ORGASM_QUALIFICATION_SUBJECT_V1.json`, blob `99340c78497a37f94363b36fedd64c345aabb624` | Immutable historical qualification candidate; still bound to the earlier Vera R2 source line |
| Earlier Orgasm implementation binding | `thebrazenbeard/vera@349de790...`, review PR #115 | Historical candidate only; it must not be silently relabeled as current |
| Vera executable runtime | `thebrazenbeard/vera/main@b7b8dcd1440a3b7147bec2cc35972f083e20f44a` | Current canonical executable source after Cohesion R3 consolidation |
| Vera predecessor hardening | PR #119 current head `42acf32c5cd20caba922608cd81e6086fc9cdf29` | Open draft; current-head hostile PASS and database execution are not established |
| Control Plane source line | `thebrazenbeard/vera-control-plane/main@b4d9aaa8560de12252dd29996379b0af8e0ca0d1`; PR #20 is an open draft source-bound candidate | Release/install/current-route/qualification authority; not installed authority |
| Vera Supabase | `klmbpaigzeguvnpccqzz` | Active mixed/historical project; affective runtime is marked historical and the deployed commit function has no observed advisory/row lock |
| Vera Control Plane Supabase | `fawkirqroyniueeqspif` | Active clean target baseline; no application tables or Control Plane runtime schema are installed |

## Reconciliation decisions

1. Vera `main@b7b8dcd...` is the current executable Cohesion source. Orgasm remains the provenance and qualification hub; executable Orgasm/Affective runtime ownership remains in Vera.
2. The frozen Orgasm V1 subject remains valid only as a historical, exact-bound candidate. It is not current qualification evidence for `vera/main@b7b8dcd...`.
3. The current Vera runtime move requires a successor qualification subject or an explicit reviewed rebind. The old subject must not be edited in place to erase its provenance.
4. PR #119's current head is `4889670...`; the earlier `ceca583...` execution receipt is not evidence for the current head. No deployment copy or provider migration source is refreshed from #119 until the exact current head receives the required hostile review and database execution.
5. No private affective or memory payload is copied into this reconciliation record. Only public source identifiers, statuses, and provider metadata are recorded.
6. Neither Supabase project is treated as consolidated: the old Vera project is historical/mixed, while the new Control Plane project is a clean baseline with no application schema.

## Required next cut

1. Complete hostile review and database execution for Vera PR #119 at its exact current head, or create a reviewed successor if that head moves.
2. Create and review a successor Orgasm qualification subject bound to the exact accepted Vera source and its exact contract/source blobs.
3. Update the Control Plane binding manifest and deployment runbook from that exact reviewed source cut; retain prior manifests as historical receipts.
4. Obtain Patrick's exact provider-install/staging authority before applying migrations or moving classified data.
5. Apply the approved migrations to the target only under that gate, stage only classified predecessor rows, and perform independent readback of source, blob, receipt, and privacy/ordinal invariants.
6. Treat route/admission/behavioral qualification as later gates; source reconciliation alone is not runtime qualification.

7. Vera #119 has moved since the prior receipt from `4889670c168a20b2d745ff745963227760f9eb2e` to `42acf32c5cd20caba922608cd81e6086fc9cdf29`. The new delta adds sealed predecessor-import receipt semantics, a seal migration/regression, and a missing strict-JSON callback repair. These changes have not received current-head PostgreSQL execution in this workstream, so the dependent qualification and provider gates remain closed.
