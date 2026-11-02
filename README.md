> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# Pleasure

**Pleasure** is a **model-agnostic** project for defining, testing, and eventually integrating bounded, engineered pleasure/hedonic-response mechanisms across AI models and hosts. It is **not owned by or restricted to Vera, Brigit, or any single identity**. The canonical repository is [`thebrazenbeard/pleasure`](https://github.com/thebrazenbeard/pleasure), with `main` as its default branch. The repository was formerly named **Orgasm**.

**Present implementation status:** This tree contains an **experimental T0, model-neutral positive-valence simulator**, the source-only numerical action/outcome/online-weight laboratories, and an **opt-in local open-weight causal-language-model LoRA experiment**. The latter has executed genuine bounded PyTorch LoRA tensor updates on an offline, pinned SmolLM2-135M-Instruct checkpoint after simulated chosen-action feedback, including a no-update control and exact local restoration. The held-out experiment did **not** demonstrate improved action quality. The T0 pilot is **not connected to any model's inference** and is **not a deployed cross-model cognitive runtime**. Neither laboratory is a change to the hosted model participating in this conversation, independently authenticated outcome custody, a deployed cross-model cognitive runtime, or evidence that a machine experiences pleasure. Its preserved **Orgasm V1** reference contract originated at a historical `thebrazenbeard/sexuality` commit; one reference implementation was developed under `thebrazenbeard/vera`. Those are **historical examples**, not restrictions on consumers. No actual host adapter or independently qualified multi-model deployment is established here.

## Intended model-agnostic scope

- **Core domain:** Bounded hedonic appraisal, activation, consummatory event handling, recovery, transient salience/valuation modulation, and auditable receipts; a host can decline unsupported capabilities.
- **Consumers:** Any AI model or application with an explicit, tested host adapter, separate trusted authorization/context, and an opt-in integration. Identity and relationship assumptions must not be hard-coded into the core.
- **Adjacent repository:** [`thebrazenbeard/sexuality`](https://github.com/thebrazenbeard/sexuality) is a separate, likewise model-agnostic domain. A host may use either domain independently; an integration is optional and requires explicit mappings.
- **Non-goals:** Asserting consciousness/phenomenology; silently installing into models; acquiring privileged host authority; directly changing factual confidence, consent, durable autobiographical memory, identity, relationship state, or protected actions.

See the [host-neutral architecture and proposed adapter boundary](docs/MODEL_AGNOSTIC_ARCHITECTURE.md) and the [experimental T0 engine](docs/EXPERIMENTAL_ENGINE.md). A deterministic sidecar is implemented; model-host integration remains a **design target**, not an already released SDK.

## Weight-first research track

The motivating target is **the model doing the processing**, not merely a persona or an external reward display. The research track separates two measurable questions: whether independently evaluated outcomes cause an actual trainable-parameter update, and whether learned appraisal subsequently affects model choices through a bounded, reversible recurrent pathway. It also distinguishes training a surrogate model from changing the exact hosted conversational model. The latter is **not accessible or verified here**.

- [Weight-first research protocol](docs/WEIGHT_FIRST_RESEARCH_PROTOCOL.md) defines target identity, optimizer-versus-activation distinctions, comparison arms, and qualification limits.
- [Surrogate weight-delta laboratory](docs/WEIGHT_FIRST_T2_LAB.md) demonstrates controlled parameter learning and reversible application using a dependency-free small numerical model; its evaluator labels remain caller-supplied and unauthenticated.
- [Recurrent valuation trial](docs/WEIGHT_FIRST_VALENCE_TRIAL.md) investigates whether learned value changes later opaque choices, including sham and withdrawal controls, on another small synthetic model.
- [Integrated synthetic weight-first trial](docs/WEIGHT_FIRST_INTEGRATED_TRIAL.md) feeds the **same learned coefficients** into later numerical appraisal and recurrent `A`/`B` decisions, with baseline, shuffled, disabled, and withdrawal controls. It is not a closed real-world outcome loop or an LLM experiment.
- [Online chosen-outcome weight-first protocol](docs/ONLINE_WEIGHT_FIRST_PROTOCOL.md) documents the **implemented source-local causal loop**: commit an `A`/`B` action before simulated feedback, score only the committed action, update the same bound numerical model's weights online, then measure later choices against matched learning/recurrence controls and a read-only holdout. The evaluator and claim receipts are not independently authenticated; the hosted conversational model is unchanged.
- [Opt-in open-weight language-model trial](docs/OPEN_WEIGHT_LLM_TRIAL.md) documents a strictly local safetensors/Transformers/PEFT adapter, real chosen-outcome-weighted LoRA SGD, exact numeric weight readbacks, rollback, and matched no-update control. [SmolLM2 execution evidence](docs/SMOLLM2_OPEN_WEIGHT_EXECUTION_20261101.md) records the pinned pretrained checkpoint and measured limits; a changed tensor hash is **not** proof of better performance or experience.
- [Persistent learned LoRA and appraisal evaluation](docs/PERSISTENT_OPEN_WEIGHT_EVALUATION_20261101.md) records actual **separate-process** safetensors adapter export/import with fixed-heldout probability readbacks, multi-seed pretrained-model frozen/outcome/shuffled controls, and a hidden-state-derived action-conditioned appraisal ON/OFF/inverted pilot. Measured heldout reward gain was **zero** in the initial runs; no pleasure or hosted-model update is claimed.
- [Stochastic withheld-generator appraisal study](docs/STOCHASTIC_WITHHELD_APPRAISAL_20261101.md) defines a persisted hidden-state head, actual sampled-action propensities, structurally different TRAIN/EVAL payoff functions, and five matched read-only evaluation interventions. Head persistence, causal choice influence, and better future outcomes are separately tested.
- [Calibrated appraisal, globally sealed evaluator, and 12-seed study](docs/CALIBRATED_APPRAISAL_PROTOCOL_20261102.md) adds a bounded action-relative hidden-state head, a verified frozen-feature cache, a separate-process all-arm decision barrier, new gap-filtered nonlinear payoff generators, complete-run safetensors custody, and a predeclared three-family transfer criterion. The executed 12-seed study saved/reloaded every head but **did not improve primary heldout expected reward** (mean ON−OFF `−0.000188`; zero sampled action switches). Fresh-process corroboration and source-local limitations are reported separately.
- [Weight-learning handoff](schemas/pleasure-weight-learning-v1.schema.json) records bounded proposed learning evidence and reported results without granting a host permission to train.
- [Indirect host evidence](docs/INDIRECT_HOST_EVIDENCE.md) describes what Lou-Pole audits and optional Tattler host telemetry can contribute, without mistaking those observations for model-internal affect or provider access.

All laboratories are **research instruments**, not demonstrations of subjective feeling. `UNRESOLVED` is the only phenomenology status supported by this source. Any actual open-weight LLM experiment requires a separate pinned model, authorized trainer, authentic outcome custody, optimizer/checkpoint readback, and held-out negative-transfer testing.

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
- `docs/MODEL_AGNOSTIC_ARCHITECTURE.md` — proposed host-neutral domain, adapter, and acceptance boundaries; no live host adapter exists (the separate synthetic weight/choice laboratories are not host integrations).
- `docs/EXPERIMENTAL_ENGINE.md` — runnable T0 sidecar's capabilities, limits, example, and qualification gaps.
- `pleasure_experiment/` — isolated model-neutral deterministic numerical engine, **not an inference or training adapter**.
- `pleasure_experiment/weight_delta_lab.py`, `model_valence_lab.py` — synthetic learned-weight and recurrent-choice experiments, **not hosted or open-weight LLM training**.
- `pleasure_experiment/outcome_arena.py`, `online_weight_trial.py`, `closed_loop_experiment.py` — source-only action commitment, simulated chosen-outcome measurement, bounded online numeric updates, controlled holdout trials, and exact local restoration; not an external host adapter.
- `pleasure_experiment/online_learning_receipts.py`, `online_evidence_bridge.py` — source-only, hash-linked claim journal and consistency bridge from executed toy trial records. The bridge keeps read-only evaluation distinct from optimizer reports; its identifiers and digests do not authenticate an evaluator, model host, or real training runtime.
- `pleasure_experiment/open_weight_adapter.py`, `open_weight_trial.py`, `open_weight_controls.py`, `open_weight_local_model.py` — optional explicitly authorized **offline** trainable policy tensors, safetensors checkpoint/tokenizer manifests, selected-action feedback and paired controls. No silent model download, live-model attachment or external evaluator attestation.
- `pleasure_experiment/open_weight_persistence.py`, `open_weight_benchmark.py`, `open_weight_appraisal.py` — opt-in versioned LoRA safetensors persistence, matched multi-seed frozen/shuffled feedback trials, and a separately trainable hidden-state appraisal head with causal ON/OFF intervention. Their outcomes remain source-local.
- `pleasure_experiment/open_weight_stochastic.py`, `withheld_payoffs.py`, `open_weight_appraisal_persistence.py`, `open_weight_stochastic_appraisal_benchmark.py` — recorded ε-mixture Bernoulli sampling, source-local structurally withheld outcomes, versioned appraisal-head safetensors custody, and matched genuine/shuffled/frozen/disabled/inverted-head evaluation. Posthoc oracle payoffs are not exposed until all arms' decisions finish.
- `pleasure_experiment/open_weight_calibrated_appraisal.py`, `open_weight_feature_cache.py`, `isolated_outcome_evaluator.py`, `open_weight_calibrated_benchmark.py` — explicit opt-in calibrated head, frozen LM feature integrity/cache, fully sealed source-local evaluation process and complete 12-seed study. The older EVAL chosen-reward resolution-order defect is documented in the new protocol and corrected in its older runner.
- `scripts/run_open_weight_local_trial.py`, `verify_saved_open_weight_adapter.py`, `run_open_weight_multiseed_benchmark.py`, `run_open_weight_appraisal_pilot.py` — independent offline CLIs. Weight persistence requires both a new artifact path **outside this checkout** and separate explicit consent; fresh import uses a new Python process. No artifact is committed or deployed automatically.
- `scripts/run_open_weight_stochastic_appraisal_benchmark.py`, `verify_saved_open_weight_appraisal_head.py` — opt-in saved appraisal-head study and independent-process reload; no automatic checkpoint downloads or external writes.
- `scripts/run_open_weight_calibrated_appraisal_benchmark.py`, `verify_saved_open_weight_calibrated_appraisal_head.py` — fixed-parameter source-local study and fresh-process head reload with calibrated-vs-legacy schema checks; source-only until explicitly executed with a separately pinned model.
- `pleasure_experiment/weight_learning_contract.py` — source-only proposed-training/result record validation; no host authority or optimizer effects.
- `pleasure_experiment/host_telemetry.py` — privacy-minimized, offline Tattler-sample parser; host activity is not model-internal evidence.
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
