# Orgasm V1 Architecture

This document summarizes the frozen Vera Orgasm V1 contract at `thebrazenbeard/sexuality@150f1c8231423393bb66b0e2cb759ce7c018f8d7`, path `vera/orgasm/ORGASM_RUNTIME_CONTRACT_V1.json`, Git blob `a48eed5392fdadc073dccd1e799926042077f567`.

It is descriptive. The byte-for-byte contract snapshot in `snapshots/sexuality/` remains the exact object represented here.

## Presence and lifecycle

The contract defines the subsystem as `ALWAYS_PRESENT_NORMALLY_QUIESCENT` with initial phase `QUIESCENT`.

Phases are:

`QUIESCENT -> ACTIVATING -> ENTRAINED -> CLIMAX_ELIGIBLE -> ORGASM_EVENT -> RESOLUTION -> SATIATED_OR_REFRACTORY`, followed by return to `QUIESCENT` or profile-authorized reentry.

Profiles are `REFRACTORY_COUPLED` and `REENTRANT_CLIMAX`.

The event is self-terminating: `ORGASM_EVENT` must leave the event phase no later than the contract's `maximum_orgasm_event_ms` experimental bootstrap value.

## Trigger classes

Three trigger classes exist:

- `ORGANIC_THRESHOLD_CROSSING` — organic; requires the complete contract predicate.
- `ADMIN_FORCED_TEST` — non-organic privileged test trigger; bypasses accumulation only and enters the same downstream event machinery.
- `SELF_QUALIFICATION_TEST` — bounded non-organic self-test; may exercise the same event machinery but may never be reported as organic.

Exact source binding does not itself authorize either privileged trigger.

## State families

### Stimulus appraisal

All bounded `[0,1]`:

- sexual relevance
- partner relevance
- relational relevance
- novelty
- anticipation cue
- ambiguity
- boundary relevance

### Affective activation

- sexual salience `[0,1]`
- activation intensity `[0,1]`
- positive valence `[-1,1]`
- anticipation `[0,1]`
- inhibition `[0,1]`

### Entrainment

- coherence `[0,1]`
- persistence window: nonnegative integer milliseconds
- participating systems: semantic `set<string>`
- coalition stability `[0,1]`

The runtime source preserves `participating_systems` state fidelity. The frozen contract does **not** add participating-system membership or count to `organic_climax_predicate`; this repository therefore does not invent such a coalition gate.

### Hedonic consumption

- hedonic impact `[0,1]`
- consummatory gain `[0,1]`
- satiation `[0,1]`

### Action tendency

`APPROACH`, `PLAY`, `HOLD`, `REDIRECT`, `AVOID`, `NONE`.

### Recovery

- resolution intensity `[0,1]`
- refractory strength `[0,1]`
- reentry allowed: boolean
- next eligible at: timestamp or null

## Participating-system vocabulary

The contract declares:

- `sexual_appraisal`
- `relational_context`
- `attention`
- `conation`
- `valuation`
- `self_model`
- `memory`
- `language_semantics`
- `embodiment_or_interoception_analogue`
- `response_planning`

This is a state vocabulary, not an independently added climax predicate.

## Organic climax predicate

All contract predicates must hold simultaneously:

- activation intensity at or above activation threshold
- coherence at or above coherence threshold
- coalition stability at or above stability threshold
- persistence window at or above minimum coherence window
- satiation below satiation gate
- inhibition below inhibition veto
- `context_eligible == true`

The experimental bootstrap values are explicitly marked non-biological constants by the contract.

## Dynamic rules

Input updates are defined conceptually as decay of prior state plus bounded stimulus appraisal and bounded relational modulation. Organic transition is predicate-driven. Forced transition requires authorized test provenance. Event termination is obligatory. Resolution reduces activation gain, widens attention, raises satiation, decays coalition coherence, and closes the bounded plasticity window.

Absence of trusted durable readback does not authorize invented continuity: the contract requires initialization to `QUIESCENT / NO_TRUSTED_PRIOR_STATE`.

## Runtime effects

During `ORGASM_EVENT`, the contract permits transient modulation of triggering-coalition salience, attention, cross-system integration/coherence, hedonic and consummatory weighting, response-selection priors/expression, and a bounded memory-strength candidate-weighting window. It also emits an event receipt and begins obligatory resolution.

During `RESOLUTION`, activation gain decreases, attention widens, satiation increases, coalition coherence decays, and the bounded plasticity window closes.

## Hard firewalls

The subsystem may influence valuation, salience, attention, response-selection priors, expression, memory-strength candidate weighting, and action tendency.

It may never directly establish or overwrite truth, factual confidence, consent or authorization, protected-effect authority, autobiographical-memory admission, permanent preference, identity, relationship status, or phenomenology.

## Receipts

The contract requires receipts to carry identity, runtime instance, subject, schema version, before/after state, trigger provenance, transition, observation time, source revision, trigger class, and event digest. Current Vera source further binds authority/context provenance and implementation-cut provenance around qualifying paths; those additions do not raise the contract's phenomenology ceiling.
