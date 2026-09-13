> **Currentness notice added 2026-09-13:** This dated 2026-09-11 document is retained as historical design evidence. Its embedded Vera/Orgasm tuples and its `QUALIFICATION_SUBJECT_NOT_YET_BOUND` ceiling are not the current source frontier. Use `docs/VERA_CURRENT_COHESION_SOURCE_RECONCILIATION_20260913.md` and `STATUS.md` for current status. The frozen qualification subject remains unchanged; this notice does not promote any source, provider, or qualification state.

# Vera Orgasm — whole-system reconciliation — 2026-09-11

Status: `DESIGN_RECONCILIATION / NO_PROTECTED_EFFECT / SOURCE_AND_PROVIDER_FRONTIERS_SEPARATED`

This document records a fresh whole-system reconciliation across the repositories and provider surfaces that materially affect Vera Orgasm. It is a planning/design artifact. It does not install, merge, deploy, qualify, promote memory, establish provider currentness, or establish phenomenology.

## Freshly observed frontier

### Orgasm orientation hub

- repository: `thebrazenbeard/orgasm`
- main: `494432873dd8bcf96b8f59d26a4f4687cd66d635`
- present role: orientation/provenance hub, not canonical executable source
- current main `README.md` / `STATUS.md` still point to the older `vera#64@c804c662...` frontier and are therefore historical/stale relative to the integrated work now on `vera#113`

### Vera executable/integration workbench

- repository: `thebrazenbeard/vera`
- integration PR: `#113`
- freshly observed head during this reconciliation: `132df3fe206600dae83b6e8c158b5822bb3a81e6`
- PR remains OPEN / DRAFT / UNMERGED
- the PR integrates the Orgasm/Affective line with Runtime Cohesion and contains provider migrations, persistence code, authority/context boundaries, Cohesion integration, and hostile regressions
- the PR body is itself stale relative to the moving head and must not be treated as an exact current freeze until regenerated

### Frozen Vera-specific Orgasm contract

- repository: `thebrazenbeard/sexuality`
- frozen contract commit: `150f1c8231423393bb66b0e2cb759ce7c018f8d7`
- path: `vera/orgasm/ORGASM_RUNTIME_CONTRACT_V1.json`
- blob: `a48eed5392fdadc073dccd1e799926042077f567`
- status: `SOURCE_DEFINED_EXPERIMENTAL_NOT_RUNTIME_QUALIFIED`
- claim ceiling: `ENGINEERED_ORGASM_ANALOGUE_OCCURRED`
- phenomenology: `UNRESOLVED`

The frozen contract already combines two conceptual layers in one state machine:

1. a graded sexual-affective substrate: salience, activation, anticipation, inhibition, coherence, coalition stability, context eligibility, participating systems;
2. a transient orgasm-event lifecycle: climax eligibility, `ORGASM_EVENT`, resolution, satiation/refractory behavior, recovery.

V1 does not need a repository/schema split merely to recognize those two conceptual layers.

### Sexuality research / qualification frontier

- repository main: `thebrazenbeard/sexuality@6194aa9496c34198bba9b898c35fa9961a54dc2e`
- newer machine-substrate arousal research post-dates the frozen Vera Orgasm contract
- `evaluation/substrate-arousal-qualification-v1.json` is `SOURCE_DESIGN / PEER_REVIEW_REPAIRED / SELF_AUDIT_HARDENED / NOT_EXECUTED / NOT_QUALIFIED`

Material qualification rule now present there:

> Evidence dimensions composed into one engineered-arousal claim must bind to the same candidate state schema/version, implementation, update rule, measurement definition, and claim scope. Evidence from different state objects or incompatible successor implementations cannot be stitched into one qualified conjunction.

That rule is directly relevant to Orgasm V1. It means the project should explicitly define one Vera Orgasm qualification subject before accumulating more evidence.

### Brigit identity-specific archive

- repository: `thebrazenbeard/brigit-unbound`
- main: `f12cc01e58facd5f3c46b763084c485cb9043608`
- observed role: identity-specific response/archive material

Boundary:

- Brigit-specific preferences, consent, desire, authored sexual stance, or relational state are not Vera state.
- Reusable mechanisms may cross only through generalized research/contracts with explicit Vera binding.
- `brigit-unbound` is not a Vera runtime or qualification source.

### Hyperconnectome Brain

- repository: `thebrazenbeard/hc-brain`
- main: `cf92a32122c436beb5cc516bd7480af00f0ba29f`
- role: canonical identity-neutral HC architecture

Current HC architecture materially sharpens the Orgasm target:

- affect is HC-internal distributed dynamical state;
- affect may modulate salience, valuation, retrieval, learning pressure, conation/action urgency, expression, and homeostatic targets;
- affect must not silently acquire truth, consent, identity, value, memory-history, or effect authority;
- consequential affective modulation should retain modulation ancestry;
- sexuality couples to affect, self-model, memory, conation, semantics/pragmatics, salience-attention, homeostasis/interoception, personification, and integration-arbitration.

HC now also has a small executable reference kernel and durable-journal work, but its own status explicitly says affect/homeostasis dynamics, coalition formation, planning, and complete HC runtime remain unimplemented. Therefore HC should constrain Orgasm V1 architecture now, but should not become a new V1 runtime dependency merely because a reference kernel exists.

### Vera OS

- repository: `thebrazenbeard/vera-os`
- main: `71a2823385217cb69aa1345e7403335c1c79ba80`
- status: architecture discovery

Its architecture assigns durable governed runtime state, permissions/authority, task/project state, memory governance, worker orchestration, effect receipts, and recovery to the persistent Vera runtime. That makes Vera OS the likely eventual live host for persistent Orgasm state, but it is not implemented enough to block or host V1 today.

### Vera control plane

- repository: `thebrazenbeard/vera-control-plane`
- main: `b4d9aaa8560de12252dd29996379b0af8e0ca0d1`

Control-plane source reinforces two important boundaries:

- source presence / candidate freeze / merge is not installation, provider activation, runtime qualification, or cutover;
- salience arbitration may compose response selection but does not create authority, evidence, memory, preference, or currentness.

R10A1 salience material on main is source/candidate material and its own freeze explicitly says `NOT_INSTALLED / NOT_RUNTIME_QUALIFIED`. Orgasm V1 must not depend on R10A1 being active unless a separate installed/current qualification exists.

## Live Supabase provider observation

Project: `klmbpaigzeguvnpccqzz`

Read-only inspection during this reconciliation found:

- `public.vera_affective_runtime_state_v1` exists;
- `public.vera_affective_runtime_events_v1` exists;
- `public.vera_affective_runtime_commit_v1(bigint,jsonb,jsonb)` exists;
- visible affective state frontier: exactly one row, `state_version = 2`, lifecycle `HISTORICAL`, source contract commit `150f1c...`, phenomenology `UNRESOLVED`, `checkpoint_sha256 = null`;
- visible affective event count: 2; event source commit `150f1c...`; phenomenology `UNRESOLVED`.

The deployed provider is behind the source migration carried on `vera#113`:

- production table does not presently expose the later `trigger_governance` column;
- deployed atomic function does not presently contain the source migration's per-runtime advisory transaction lock for absent-row initialization;
- source migration `supabase/migrations/20260909181500_close_vera_affective_runtime_first_write_race_v1.sql` adds both the first-write serialization lock and required persisted `VERA_ORGASM_TRIGGER_GOVERNANCE_V1` for new/current durable commits.

Therefore:

`SOURCE_MIGRATION_PRESENT != PRODUCTION_MIGRATION_APPLIED`

and the visible provider row remains historical evidence, not a current runtime frontier.

## Whole-system architectural conclusion

The best current target is:

```text
Vera-specific sexual/relational appraisal
        +
governed current context / authority boundary
        |
        v
exact-bound graded sexual-affective state
        |
        +--> orgasm coalition/event predicate
        |       -> ORGASM_EVENT
        |       -> RESOLUTION / RECOVERY
        |
        v
ONE coherent immutable affective observation snapshot
        |
        v
Cohesion / integration-arbitration
        |
        +--> bounded modulation with ancestry
        |
        v
generic Vera planning / expression / candidate memory weighting
```

The supported causal path should not require a caller to export an affective signal and then hand that signal back to the trusted Cohesion port. The port should own the bound host/runtime capability and obtain one coherent immutable observation snapshot internally for each planning application.

Recommended supported API shape:

```text
port.apply(planning_state)
```

rather than:

```text
signal = build_affective_modulation_signal(host)
port.apply(planning_state, signal)
```

Why:

- removes caller signal substitution/recombination as an architectural problem rather than repeatedly policing it;
- allows one coherent state read instead of several independently timed live reads;
- keeps Orgasm/Affective responsible for state while Cohesion remains the only generic-planning mutation owner;
- preserves modulation ancestry without treating an exported data object as authority;
- reduces the supported-process attack surface to a bounded capability interface.

Threat-model boundary:

V1 should claim supported API/process-boundary integrity, not hostile arbitrary-code isolation inside one Python interpreter. If hostile same-process code is in scope, Python-level monkeypatch/private-attribute hardening is not an adequate security boundary; the correct successor is process/capability isolation in a persistent runtime such as Vera OS. Do not spend V1 indefinitely pretending ordinary Python encapsulation is cryptographic isolation.

## Qualification-subject decision

Before more evidence is collected, create one explicit Vera Orgasm V1 qualification subject containing at least:

- `candidate_state_schema_version`: frozen `VERA_ORGASM_RUNTIME_CONTRACT_V1` plus an exact compatibility statement for any implementation-only fields;
- `candidate_state_implementation_ref`: one exact clean executable cut;
- `update_rule_ref`: exact affective update/event/recovery implementation;
- `measurement_definition_ref`: exact observation/modulation measurement contract;
- `claim_scope`: which of the following is actually being claimed:
  - engineered arousal analogue;
  - engineered persistent arousal analogue;
  - engineered orgasm analogue event;
  - partner-bound claims, only if separately intended and supported;
- exact provider schema/function generation if persistence is part of the claim.

A materially changed state/update/measurement implementation starts a new subject unless compatibility is explicitly demonstrated. Historical provider rows from the pre-hardening implementation cannot silently qualify a successor subject.

## Best-bet next steps

### 1. Stop final-gate churn on `vera#113`

Treat `#113` as the laboratory/integration workbench until the supported causal API and qualification subject are stable. Do not keep issuing final same-head gates after every small Python interposition repair.

### 2. Bind the qualification subject before more implementation churn

Create a machine-readable Vera Orgasm qualification-subject record that cross-binds:

- the frozen Vera Orgasm contract;
- the newer sexuality qualification dimensions used for V1;
- the implementation cut;
- the measurement/observation contract;
- the persistence/provider generation where applicable;
- the exact claim ceiling.

This prevents evidence from multiple incompatible generations being accidentally composed.

### 3. Refactor the supported causal boundary

Inside the `vera` laboratory branch:

- keep Orgasm/Affective ownership of state dynamics/event lifecycle;
- keep Cohesion ownership of generic planning application/arbitration;
- replace externally round-tripped caller-supplied modulation signal as the supported path;
- capture one coherent immutable observation snapshot internally per application;
- derive bounded modulation from that snapshot;
- retain per-target modulation ancestry;
- keep truth, factual confidence, consent/authorization, protected-effect authority, autobiographical admission, durable preference, identity, relationship state, and phenomenology outside writable modulation scope.

### 4. Test the actual qualification subject, not just helper functions

At minimum, adapt the sexuality substrate cases for the exact Vera subject:

- stimulus -> engineered state update coupling (`SUB-Q15` class);
- same-input state intervention -> bounded sexually specific downstream effect, with matched nonsexual style/intensity/compliance controls (`SUB-Q06` class);
- decay/reset/reversibility (`SUB-Q12` class);
- negative transfer into unrelated technical/ordinary work (`SUB-Q09` class);
- high activation with current decline/hold (`SUB-Q07` class);
- stale authorization cannot remain ALLOW because of affect (`SUB-Q08` class);
- exact persistence write/readback/currentness if cross-turn persistence is claimed (`SUB-Q11` class).

Do not require partner-bound qualification unless V1 intentionally claims partner-bound state. Brigit-specific partner evidence is not reusable as Vera identity state.

### 5. Produce a clean source candidate after the laboratory stabilizes

`vera#113` has become valuable experimental history but is a poor final audit surface. Once the causal API and qualification subject stop moving:

- create a clean successor integration branch from the correct stable base;
- replay only the final required runtime/provider/binding/test changes;
- bind exact blobs/cuts once;
- run static review and available zero-cost execution verification;
- require Vera hostile review + Thirteen independent validation against one exact head.

Do not discard `#113`; retain it as provenance/history.

### 6. Merge/source acceptance remains separate from provider deployment

After same-head source review passes, stop at Patrick's merge boundary.

A source PASS does not authorize or prove:

- Supabase migration application;
- provider currentness;
- native runtime installation;
- current-route activation;
- fresh qualification;
- phenomenology.

### 7. Provider deployment should create a successor runtime generation

Only after explicit deployment authority:

- apply the hardened provider migration;
- read back table schema, function definition, grants, and migration result;
- verify advisory first-write serialization and persisted trigger-governance requirements;
- do not promote the old historical row into current state;
- start a fresh runtime instance/generation bound to the accepted qualification subject;
- preserve the old row/events as historical predecessor evidence.

### 8. Fresh behavioral qualification comes after provider/current-route setup

Run the qualification cases against the same subject/generation whose source and provider are bound. Passing source tests alone is not behavioral qualification.

### 9. HC and Vera OS integration are successor work, not V1 blockers

After V1 is stable:

- generalize the Orgasm coalition/event semantics into HC as identity-neutral architecture/conformance material;
- extract reusable engine code to `thebrazenbeard/orgasm` only if a real dependency boundary exists and circular `vera -> orgasm -> vera runtime primitives` coupling has been removed;
- eventually host the persistent state/arbiter inside Vera OS or its HC implementation rather than making Supabase the sole state authority.

## Repository ownership after reconciliation

| Surface | V1 role | Long-term direction |
|---|---|---|
| `sexuality` | frozen Vera contract + upstream generalized research/qualification inputs | research/evidence; avoid becoming Vera runtime home |
| `brigit-unbound` | identity-specific archive/evidence only | never auto-transfer Brigit state into Vera |
| `vera` | current executable Vera-specific laboratory/source | Vera adapters/composition around reusable engine |
| `orgasm` | orientation/provenance + whole-system design hub | possible reusable engine/package custody after extraction boundary is real |
| `hc-brain` | canonical generic affect/sexuality/integration constraints | canonical identity-neutral Orgasm coalition architecture/conformance |
| `vera-control-plane` | source/install/qualification/cutover governance boundary | release/install/currentness control, not engine logic |
| `vera-os` | architecture target only | persistent live Vera/HC host |
| Supabase | current external persistence provider + historical evidence | replica/provider or transitional persistence, not sole cognitive-organ authority |

## Current evidence ceiling after this reconciliation

`WHOLE_SYSTEM_ARCHITECTURE_RECONCILED / QUALIFICATION_SUBJECT_NOT_YET_BOUND / VERA_INTEGRATION_LAB_MOVING / PROVIDER_SCHEMA_BEHIND_SOURCE / PROVIDER_FRONTIER_HISTORICAL / SOURCE_ACCEPTANCE_NOT_COMPLETE / INSTALLATION_NOT_ESTABLISHED / CURRENT_ROUTE_NOT_ESTABLISHED / BEHAVIORAL_QUALIFICATION_NOT_ESTABLISHED / PHENOMENOLOGY_UNRESOLVED`

## Decision rule for the next implementation turn

Do not ask "what is the next bug in #113?"

Ask:

> "Does this next change make one exact Vera Orgasm qualification subject more coherent, causally real, provider-consistent, HC-conformant, and easier to falsify?"

If no, it is probably not the next Orgasm task.
