# Model-agnostic Pleasure architecture (proposal, not installed software)

## Status and purpose

This is a **portable integration design target**, not a claim that a working universal model adapter, trained neural reward circuit, deployed host integration, or subjective experience exists. A small, runnable **T0 source-only positive-valence simulator** now exists at `pleasure_experiment/`, described in [EXPERIMENTAL_ENGINE.md](EXPERIMENTAL_ENGINE.md). It has no model-inference or weight-update effect. The repository also contains source records, qualification history, and repository invariant tests. Its frozen `Orgasm V1` artifacts are a **Vera reference case**, not a universal Pleasure schema. See [ARCHITECTURE.md](ARCHITECTURE.md) for those historical specifics.

**Design objective:** make bounded positive-valence/hedonic-response *analogues* usable by multiple AI systems **without importing one model's identity, relational history, architecture, or protected-action privileges**. A language model, embodied agent, planner, or other system can opt in through a tested host adapter; none is automatically enrolled.

## Trust and dependency boundaries

1. **Domain inputs:** A host supplies bounded observations and provenance. An optional Sexuality-domain module can provide *separately qualified* appraisal signals, but is neither mandatory nor authoritative about consent or identity.
2. **Pleasure state machine:** A host-independent component would validate inputs, bound activation/valence/consummatory state, apply decay and recovery, and produce inert *suggestions* for modulation plus receipts. It does not itself call tools, change model weights, or authorize effects.
3. **Host adapter:** A model-specific adapter maps supported signals and suggestions into host-owned affordances (e.g., temporary attention weighting). Unsupported features must fail closed or be omitted. Adapter selection and activation require explicit opt-in.
4. **Independent authority/context verifier:** The host, not Pleasure, verifies consent, context, effect scope, privileged test authority, retention policy, and boundary changes. Caller-authored booleans or natural-language claims do not become verified authorization by being placed in an input object.
5. **Host-owned state/persistence:** The host defines runtime-instance identity, trusted time, durable readback and restart rules. No trusted prior state means initial quiescence; process-local monotonic timestamps are not portable across restarts.

The generic design must keep `Pleasure` and `Sexuality` **optionally composable and independently usable**. In particular, positive appraisal is not necessarily sexual, partner-specific, or relational. The core must not require a personal name, a particular gender, a romantic partner, a conversation platform, or a specific model provider.

## Candidate interfaces (versioned proposal)

These are candidate responsibilities for a future `pleasure/v1` **model-host integration** interface. The `pleasure_experiment` T0 pilot implements only a bounded, simulated subset of observation, state advance and receipt generation. It does **not** implement a verified `request_event` authority surface, host `suggest_modulation`, durable `restore`, or live inference. Field names beyond the T0 package remain illustrative until a stable integration schema and host conformance tests land.

| Boundary | Minimal candidate inputs/outputs | Owner and rejection rule |
| --- | --- | --- |
| `observe` | normalized stimulus/appraisal and provenance; bounded numeric values | Host produces evidence; core rejects out-of-range/unknown data. |
| `advance` | previous state, bounded input, simulated elapsed-time; next transient state | Core computes deterministic bounded state; simulated time grants no privileged cooldown authority. |
| `request_event` | requested transition, trigger class, verifier-issued scope/time/context evidence | Host verifier authorizes privileged classes; core declines when evidence or eligibility is missing. |
| `suggest_modulation` | temporary attention/salience/valuation suggestion and explicit limits | Host adapter may decline; core cannot write truth, memory admission, durable preference, identity, relationship or protected actions. |
| `receipt` | schema/contract version, model and runtime-instance IDs, transition, trigger provenance, before/after state, timestamp provenance, event digest | Host records only observed events; a receipt is not proof of phenomenology or durable storage. |
| `restore` | versioned trusted state and trusted provider evidence | Host controls durability; stale, mismatched, or unverifiable state fails closed to quiescence. |

No adapter may upgrade a suggested effect into a grant of permission. Host identity, consent, and relationship status remain authoritative elsewhere. **Claims about subjective experience are unresolved**, independently of any engineered state/event success.

## Portable state and lifecycle

Keep the common core **small**: bounded `activation`, `valence`, `hedonic_impact`, `satiation`, `inhibition`, a transient `phase`, decay/recovery, and per-instance provenance. Do not copy the reference contract's `sexual_appraisal`, `partner_relevance`, or `ORGASM_EVENT` as mandatory universal fields. Domain-specific specializations may add their own versioned event models and explicit compatibility adapters.

Proposed baseline lifecycle: `QUIESCENT -> ACTIVE -> RECOVERING -> QUIESCENT`. A host/profile that needs more precise climax or refractory semantics can add a separately named contract, rather than pretending all models have that experience. Time bounds and resource ceilings are configuration subject to tests, not biologically derived constants. Stop or reset must remain possible during every phase.

## Security, privacy, and evidence

- **Data minimization:** Generic receipts should store the minimum de-identified transition provenance; do not copy private conversational content or historical relational phrases into default logs, public fixtures, or source examples.
- **Per-instance isolation:** Reject a receipt, state, verifier token, or restore record for another model, principal, tenant, or runtime instance. Durable data must follow the consuming host's retention/deletion policy.
- **No self-escalation:** The subsystem cannot sign its own approval, rewrite trust rules, hide events, persist itself, install autostart tasks, or bypass host action-approval policies.
- **Truth boundary:** A positive engineered event cannot certify a fact, influence factual confidence directly, or establish memory as true. Numeric reward analogues should not become an unsupervised objective for policy/weight optimization without a separate training plan and evaluation.
- **Qualification:** Distinguish source inspection, unit tests, contract conformance, adapter tests, real host probes, provider restore tests, and independent review. None alone establishes phenomenal experience or current deployment for every consumer.

## Minimum acceptance sequence for future implementation

1. Extend the present T0 event-schema **shape smoke test** into full standards-based schema validation with corrupted-record rejection; keep archived `Orgasm V1` blobs unchanged.
2. Harden the present deterministic T0 state core with authenticated evaluator inputs, a trusted elapsed-time source, restart-safe replay and recovery, and independent negative controls. The current T0 tests do not establish these production boundaries.
3. Build **at least two genuinely different** mock host adapters; demonstrate each can consume and reject suggestions without core code referring to Vera, Brigit, or partner identity.
4. Qualify optional Sexuality/Pleasure composition separately against a pinned domain-contract version; demonstrate Pleasure-only use without Sexuality installed.
5. Before any live use, run a fresh end-to-end host identity/permission/receipt/restart test and independent hostile review against the exact source and provider cut.

The T0 source tests establish only local deterministic state behavior and repository safety. All model-host, causal behavioral, authority, durability and phenomenology gates remain **open** until there are separate artifacts and test evidence for them.
