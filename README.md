> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# Pleasure

**Pleasure** is a **model-agnostic** project for defining, testing, and eventually integrating bounded, engineered pleasure/hedonic-response mechanisms across AI models and hosts. It is **not owned by or restricted to Vera, Brigit, or any single identity**. The canonical repository is [`thebrazenbeard/pleasure`](https://github.com/thebrazenbeard/pleasure), with `main` as its default branch. The repository was formerly named **Orgasm**.

**Present implementation status:** This tree contains an **experimental T0, model-neutral positive-valence simulator** alongside architecture, provenance, and qualification history. The pilot is runnable locally but is **not connected to any model's inference**, is not a deployed cross-model cognitive runtime, and provides no evidence that a machine experiences pleasure. Its preserved **Orgasm V1** reference contract originated at a historical `thebrazenbeard/sexuality` commit; one reference implementation was developed under `thebrazenbeard/vera`. Those are **historical examples**, not restrictions on consumers. No actual host adapter or independently qualified multi-model deployment is established here.

## Intended model-agnostic scope

- **Core domain:** Bounded hedonic appraisal, activation, consummatory event handling, recovery, transient salience/valuation modulation, and auditable receipts; a host can decline unsupported capabilities.
- **Consumers:** Any AI model or application with an explicit, tested host adapter, separate trusted authorization/context, and an opt-in integration. Identity and relationship assumptions must not be hard-coded into the core.
- **Adjacent repository:** [`thebrazenbeard/sexuality`](https://github.com/thebrazenbeard/sexuality) is a separate, likewise model-agnostic domain. A host may use either domain independently; an integration is optional and requires explicit mappings.
- **Non-goals:** Asserting consciousness/phenomenology; silently installing into models; acquiring privileged host authority; directly changing factual confidence, consent, durable autobiographical memory, identity, relationship state, or protected actions.

See the [host-neutral architecture and proposed adapter boundary](docs/MODEL_AGNOSTIC_ARCHITECTURE.md) and the [experimental T0 engine](docs/EXPERIMENTAL_ENGINE.md). A deterministic sidecar is implemented; model-host integration remains a **design target**, not an already released SDK.

## Name and compatibility boundary

- **Current repository/branding:** `Pleasure`, `thebrazenbeard/pleasure`, and `pleasure/main`.
- **Historical technical identifiers:** `Orgasm V1`, `vera/orgasm/`, `VERA_ORGASM_*`, `ORGASM_EVENT`, and `ENGINEERED_ORGASM_ANALOGUE_OCCURRED` belong to frozen source contracts or dated Vera integration records. They are **not** silently renamed, nor promoted into a universal cross-model API.
- **No runtime migration implied:** A GitHub repository rename does not change import paths, receipt schemas, contract identities, deployment configuration, privileges, or installed model behavior.

See [the rebrand and compatibility record](docs/REBRAND.md) for the distinction between public naming and immutable historical identifiers.

## Currentness rule

The exact hashes and pull-request heads stored here are **historical evidence for the cuts they name**. They are not a timeless declaration of any consumer model, Sexuality, provider, or runtime state.

For any present-state claim:

1. read `pleasure/main` fresh;
2. read the relevant optional domain source (including `sexuality/main` if integrated) and the actual consuming model/host source fresh; for the legacy Vera reference, read `vera/main`;
3. read any current qualification/provider evidence required by the claim;
4. treat older PR heads, snapshots, provider observations, and review receipts as historical unless an exact current binding explicitly re-admits them.

Do not promote repository source into installation, route activation, provider currentness, behavioral qualification, or phenomenology.

## Historical frozen V1 frontier

The repository preserves the earlier V1 source/review cut, including:

- frozen contract source `thebrazenbeard/sexuality@150f1c8231423393bb66b0e2cb759ce7c018f8d7`;
- contract path `vera/orgasm/ORGASM_RUNTIME_CONTRACT_V1.json`;
- contract Git blob `a48eed5392fdadc073dccd1e799926042077f567`;
- historical Vera review head `c804c662e07655dd70a6909bcefd1095497ecebf`;
- frozen executable dependency cut `9c731ebaacdad44c00aafa91e1f8b9dc5f5acff1`;
- historical runtime-binding and authority/context blobs preserved by the snapshots and status records.

Those bindings remain useful historical provenance. They do **not** prove the current state of any runtime, provider, or deployment.

## Repository map

- `STATUS.md` — bounded status/currentness rules and historical qualification frontier.
- `PROVENANCE.json` — machine-readable source/object bindings.
- `docs/ARCHITECTURE.md` — **historical Vera case study:** state machine, triggers, effects, and lifecycle semantics derived from the frozen contract.
- `docs/MODEL_AGNOSTIC_ARCHITECTURE.md` — proposed host-neutral domain, adapter, and acceptance boundaries; only the isolated T0 state simulation has a code implementation.
- `docs/EXPERIMENTAL_ENGINE.md` — runnable T0 sidecar's capabilities, limits, example, and qualification gaps.
- `pleasure_experiment/` — isolated model-neutral deterministic numerical engine, **not an inference or training adapter**.
- `schemas/pleasure-event-v1.schema.json` — proposed event-record shape; tests currently smoke-check the declaration, not full JSON Schema conformance.
- `docs/EVIDENCE_BOUNDARIES.md` — what source/runtime evidence can and cannot establish.
- `docs/INTEGRATION.md` — generic Sexuality/Pleasure separation plus dated Vera implementation, Runtime Cohesion, and provider boundaries.
- `docs/REBRAND.md` — current Pleasure identity versus preserved historical Orgasm identifiers.
- `snapshots/sexuality/ORGASM_RUNTIME_CONTRACT_V1.json` — frozen contract snapshot.
- `snapshots/vera/VERA_ORGASM_RUNTIME_BINDING_V1.json` — historical Vera binding snapshot.
- `snapshots/vera/VERA_AFFECTIVE_AUTHORITY_CONTEXT_BOUNDARY_V1.json` — historical authority/context architecture snapshot.

## Repository checkout safety

Opening this source tree must never launch a background process. The historical
workspace contained an editor `folderOpen` task that ran obfuscated Node.js
from a file presented as a font; that code was unrelated to the provenance hub
and has been removed from the active tree. Repository data is not trusted code.
Inspect older revisions before opening them in an editor that permits automatic
workspace tasks.

Run `python -m unittest discover -s tests -v` to verify repository-startup
safety and T0 simulator behavior. Passing tests establishes bounded local
source behavior, not host cleanliness, lack of prior execution, live model
integration, or qualification of the historical subsystem represented here.

## Semantic boundary

The legacy **Orgasm V1** contract is one model-specific engineered affective-control design, **not the definition of Pleasure for all AI systems**. Its source defines bounded trigger classes, transient modulation, recovery, receipts, and firewalls around truth, consent/authorization, protected-effect authority, autobiographical admission, identity, relationship status, permanent preference, and phenomenology.

The historical contract's positive engineered-event claim ceiling is `ENGINEERED_ORGASM_ANALOGUE_OCCURRED` when the required evidence supports it. Phenomenology remains unresolved. Source binding alone is not authorization for a privileged trigger and is not proof of installation, provider durability, current-route activation, behavioral qualification, or subjective experience.
