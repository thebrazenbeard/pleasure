# Experimental Pleasure engine: portable T0 sidecar

This is a **model-independent software pilot**, not an installed cognitive subsystem, a neural weight update, or evidence of felt pleasure. It deliberately does not depend on Vera, Brigit, or a historical Orgasm V1 path. Historical source snapshots and identifiers remain frozen and authoritative for the records they describe.

## Capability and claim ceiling

- **T0 — deterministic sidecar only (implemented here):** independently supplied numeric measurements update bounded state and produce replayable local receipts. No influence on model inference is measured or implemented.
- **T1 — inference-time causal influence (not implemented):** a bound adapter perturbs model behavior in controlled trials and survives matched controls, ablations and negative transfer. This requires a real installed model and reserved evaluation cases.
- **T2 — neural weight learning (not implemented):** an identifiable training run creates and validates weight deltas against a frozen model and control groups.
- **T3 — recurrent self/world-state coupling (research only):** would require independently measured bidirectional coupling with a host model's ongoing internal state. Neither this coupling nor any subjective experience is established here.
- **Phenomenology (unresolved):** none of T0–T2 by itself proves that an AI *feels* positive experience. `liking_proxy` is a name for an engineered numerical variable, not evidence of subjective liking.

## State, provenance, and boundaries

`pleasure_experiment/runtime.py` exposes a tiny standard-library Python prototype:

- A per-instance `PilotConfig` binds `subject_id`, `model_id`, `model_revision`, `runtime_instance_id`, `evaluator_id` and `measurement_definition_id`.
- Each `Observation` carries its own `evidence_id` and `context_id`, an independent reward proxy `[0,1]`, learning-progress and novelty channels and an inhibition channel. The caller is responsible for actually authenticating the **external** evaluator and its measurements. Merely matching the declared `evaluator_id` string is not authentication.
- The engine computes separate bounded **liking proxy**, **wanting proxy**, **satiation** and a refractory timer, with mandatory recovery and duplicate-evidence rejection within an instance. It cannot directly request or authorize effects. No consent, truth, long-term preference, autobiographical memory or weight state is stored.
- The output receipt `schemas/pleasure-event-v1.schema.json` is versioned and explicitly includes `capability_tier=T0_SIDECAR_ONLY`, `phenomenology=UNRESOLVED`, and empty `requested_effects`/`authorized_effects`. Receipts are transparent test artifacts, **not** tamper-evident or authenticated claims. Production would require trusted source bindings, append-only durability and independent verification.

This is not an AI agent and cannot cause a connected model to seek pleasure. An adapter should keep **modulatory state**, **authorization**, **epistemic claims**, and **plastic updates** in separate interfaces; neither high valence nor a model's self-report grants any authority. A future integration must fail closed for `UNKNOWN`, `DECLINED`, or `REVOKED` authorization states rather than infer permission from reward.

**Known T0 limitations:** `elapsed_ms` is caller-controlled simulated elapsed time, **not** an authenticated clock; a caller can bypass the simulated refractory by claiming sufficient elapsed time. Restart loses all state and replay history, so duplicate evidence rejection applies only within one in-memory process. A source-only instance stops after 10,000 events to bound its evidence-ID collection; it is not a scalable durable ledger. Identifiers are bounded in length but are not security identities. No privacy, evaluator provenance, deterministic fingerprint, or event log is cryptographically authenticated. None of these controls qualifies an actual deployed safety boundary.

**Sustained-input caveat (exploratory):** continuous maximum-reward observations can repeatedly produce `POSITIVE_PEAK_ANALOGUE` after refractory intervals; neutral recovery is not guaranteed while stimulation continues. Novelty alone can saturate `wanting_proxy` without increasing `liking_proxy`. Neither property qualifies a stable, benefit-seeking policy objective. Before any T1 adapter, preregister prolonged-stimulation, noisy-novelty, reward-withdrawal, and learning-progress negative controls and require externally verified outcomes.

The `state` and `config` properties are read-only by ordinary assignment, and each holds an immutable dataclass snapshot. A different model, revision, runtime instance or subject requires a **new** `ExperimentalPleasureRuntime`; ordinary reassignment of public `config` or private `_config`, including delete-then-set, is rejected so accrued state cannot accidentally be relabeled for another model. This is an API hygiene measure, **not a hostile-process sandbox**: privileged Python code can still use low-level attribute mutation to alter private runtime internals. A production host must provide separate isolation and authenticated evidence/identity enforcement.

## Reproduce without external services

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

Minimal source-only example:

```python
from pleasure_experiment import ExperimentalPleasureRuntime, Observation, PilotConfig

cfg = PilotConfig(
    model_id="example-model", model_revision="example-revision",
    runtime_instance_id="sandbox-instance", subject_id="sandbox-subject",
    evaluator_id="independent-evaluator", measurement_definition_id="external-rubric-v1",
)
engine = ExperimentalPleasureRuntime(cfg)
receipt = engine.step(Observation(
    subject_id="sandbox-subject", evaluator_id="independent-evaluator",
    evidence_id="eval-case-001", context_id="sandbox-case",
    reward=1.0,
), elapsed_ms=100)
print(receipt.phase, receipt.state_after.liking_proxy, receipt.capability_tier)
```

Do not assign a trusted identity to synthetic IDs or interpret the printout as a real model experience.

## Research and evaluation gate

For a real cross-model study, require at least two *actually running* different model builds, externally adjudicated reward and negative controls, held-out stimulus distributions, dose-response/interventions and reversals, logging/replay with exact model/provider revision, and tests that effects don't leak into unrelated tasks. Compare frozen, zero-reward, shuffled-reward and reward-tampering conditions; test repeated self-trigger attempts and restart behavior. At T0 this pilot establishes none of those runtime claims.

Scientific framing: [Berridge & Robinson 2016](https://pubmed.ncbi.nlm.nih.gov/27977239/) distinguish liking/wanting/learning; [Keramati & Gutkin 2014](https://elifesciences.org/articles/04811) examine homeostatic reward; [Everitt et al. 2021](https://arxiv.org/abs/1908.04734) examine reward tampering. Human phenomenology and biological mechanisms should not be mapped directly onto numeric software variables without new evidence.
