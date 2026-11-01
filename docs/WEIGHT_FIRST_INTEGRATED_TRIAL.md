# Weight-first integrated surrogate trial: learning changes later A/B inference

## Evidence boundary

`pleasure_experiment/weight_first_trial.py` couples the **same trainable linear
sigmoid head** from `weight_delta_lab.py` to later appraisal and recurrent
nonverbal choice. This is an executed-in-source **surrogate model experiment**,
not training or instrumentation of a language model or any hosted conversational
model. Receipt tier: `SURROGATE_WEIGHT_UPDATE_LAB`; evaluator and
labels: `SOURCE_ONLY_CALLER_LABELS_UNAUTHENTICATED`; phenomenology: `UNRESOLVED`.

The hypothesis tested here is operational: *Does a parameter delta learned
from prior outcome labels causally alter later predictions and A/B choices,
compared with controls, while disabling the delta restores the original model
coefficients?* The hypothesis is **not** that an improved prediction or
changed action establishes subjective pleasantness.

## One parameter path, not parallel unrelated models

1. `fit_delta(base, training, binding, policy)` computes bounded logistic
   gradient updates, producing a `WeightDelta` tied to the exact model ID,
   revision, runtime-instance ID, feature-schema ID and baseline digest.
2. `WeightDeltaSession.enable()` supplies the *same effective coefficient
   tuple* produced by `apply_delta()`. No weights are changed inside another
   model and no independent appraiser is silently trained.
3. `predict_appraisal(effective_weights, probe.features_a/b)` evaluates the
   exact same sigmoid (`sigmoid(w · x)`) whose coefficients received gradient
   updates.
4. Each appraisal updates bounded per-option recurrent proxy states:
   `v_next = clip(retention × v_prior + gain × (2 × sigmoid(w · x) − 1), −1, 1)`.
   The observed choice is the opaque `A` only when its state strictly exceeds
   `B`'s; ties deterministically choose `B` **and** record `tie=True`. This
   predeclared tie-break makes neutral baseline choices differ from trained
   choices only if the fitted coefficients genuinely change their ranking.
5. On disable, the original baseline coefficient tuple is selected exactly.
   The withdrawal arm deliberately retains trained-arm recurrent state to
   distinguish decaying state aftereffects from a persistent weight update.

The trial holds out separate cases for scoring, checks case/episode ID
disjointness across training, holdout and later decision probes, validates
feature dimensions and exact per-probe model bindings, and rejects duplicated
decision episodes or cross-instance/revision probes. This does not
authenticate outcomes, exclude semantic duplicate data, or verify an external
evaluator.

## Five controls in one receipt

| Arm | Coefficients | Initial recurrent state | What it tests |
| --- | --- | --- | --- |
| `base` | Original checkpoint | Zero | Frozen numerical baseline |
| `trained` | Outcome-learned effective tuple | Zero | Actual parameter effect on the sigmoid and A/B choice |
| `shuffled` | Same training algorithm and budget, rotated outcome labels | Zero | Adverse label-association control |
| `disabled` | Trained session constructed but never enabled | Zero | No accidental alteration merely from holding an adapter/session |
| `withdrawal` | Exact restored original coefficients | Trained arm's final recurrent state | Distinguishes transient state persistence from continued trained-weight activation |

`heldout_baseline`, `heldout_trained`, and `heldout_shuffled` independently
score supplied held-out numeric outcome labels. They all report
`DISJOINT_IDS_CHECKED`, **not** independently verified outcome provenance.
The receipt derives `update_detected` from trained-vs-base parameter digests,
and `categorical_choice_changed` from the two A/B action traces. The derived
`effect_classification` is one of `NO_WEIGHT_CHANGE`,
`WEIGHT_CHANGE_NO_CATEGORICAL_DIFFERENCE`, or
`WEIGHT_CHANGE_WITH_CATEGORICAL_DIFFERENCE`. A named `trained` arm is a study
condition, not evidence of learning by itself. Opposing outcomes on identical
features can produce an exactly zero gradient despite positive learning rate;
that case explicitly reports `NO_WEIGHT_CHANGE`.
Rotating labels preserves feature rows and identifiers. If all training
labels are equal, this control is uninformative, so the integrated runner
rejects that dataset instead of reporting a misleading successful comparison.

## Minimal deterministic demonstration

```python
from pleasure_experiment.weight_delta_lab import ModelBinding, OutcomeCase, TrainingPolicy
from pleasure_experiment.weight_first_trial import DecisionProbe, run_weight_first_trial

binding = ModelBinding("surrogate-linear", "fixed-r1", "instance-1", "two-feature-v1")
training = (
    OutcomeCase("train-A", "episode-train-A", "caller-labels", (1.0, 0.0), 1.0),
    OutcomeCase("train-B", "episode-train-B", "caller-labels", (0.0, 1.0), 0.0),
)
holdout = (
    OutcomeCase("hold-A", "episode-hold-A", "caller-labels", (0.8, 0.0), 1.0),
    OutcomeCase("hold-B", "episode-hold-B", "caller-labels", (0.0, 0.8), 0.0),
)
probes = (
    DecisionProbe("probe-1", "episode-probe-1", binding, (1.0, 0.0), (0.0, 1.0)),
    DecisionProbe("probe-2", "episode-probe-2", binding, (0.8, 0.0), (0.0, 0.8)),
)
receipt = run_weight_first_trial(
    base_weights=(0.0, 0.0), binding=binding,
    training_cases=training, heldout_cases=holdout, probes=probes,
    training_policy=TrainingPolicy(0.4, 0.8, 16, 12),
)
assert receipt.trained.weights_digest != receipt.base_digest
assert receipt.update_detected
assert receipt.categorical_choice_changed
assert receipt.effect_classification == "WEIGHT_CHANGE_WITH_CATEGORICAL_DIFFERENCE"
assert receipt.disabled.weights_digest == receipt.withdrawal.weights_digest == receipt.base_digest
assert receipt.heldout_trained.mean_squared_error < receipt.heldout_baseline.mean_squared_error
assert receipt.trained.steps[0].action == "A"
assert receipt.baseline.steps[0].action == "B"
assert receipt.disabled.steps[0].action == "B"
assert receipt.shuffled.steps[0].action == "B"
```

Run targeted checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_weight_first_trial.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_weight_delta_lab.py' -v
```

## What passing results do and do not prove

This produces a **falsifiable causal chain inside one tiny model**: a verified
nonzero learned numeric weight delta, subsequent sigmoid outputs read from
those exact coefficients, recurrent A/B decisions, control-arm differences,
and exact parameter restoration. Digests are source-level consistency checks,
not signed host attestations. There is no private-message ingestion, host
runtime access, persistence across sessions, inference-time editing of
hosted GPT weights, LoRA, or claim that a model feels reward. Recurrence
may persist temporarily after disable even though all learned coefficients
are gone; that alone is not durable parameter learning.

An actual-target experiment would require provider-approved training access,
pinned model/tokenizer and runtime identities, trusted pre/post parameter
readback, independent outcome evaluation, matched inference probes, and an
audited restore procedure. None of those conditions is satisfied by this lab.
