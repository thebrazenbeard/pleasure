# Weight-first Pleasure: synthetic parameter-update laboratory

## Exact status

`pleasure_experiment/weight_delta_lab.py` is an **isolated linear appraisal model**
whose numerical weights are actually updated by a bounded gradient calculation.
It is a laboratory surrogate, **not** a LoRA adapter, a language model, an
installation in an inference host, or evidence that any hosted conversational
model has changed. Its receipt tier is
`SURROGATE_WEIGHT_UPDATE_LAB`; phenomenology is `UNRESOLVED`.

The purpose is to test one narrow hypothesis: labelled outcome history can
change learned parameters and thereby affect subsequent appraisal on different
inputs, even while the original parameter snapshot is preserved. This is a
functional learning result, **not a claim of positive subjective experience**.

## What the code actually computes

The laboratory uses a fixed-size feature vector `x` and weights `w`, with
predicted positive outcome `sigmoid(w · x)`. It fits the weights by full-batch
logistic cross-entropy gradient descent against supplied outcome labels:

`w_next = w_current - learning_rate * average((prediction - label) * x)`

Each proposed cumulative delta relative to the immutable baseline is clipped
to `TrainingPolicy.max_delta_l2`, and each effective coefficient is projected
back into the admitted `[-10, 10]` domain, including for an outward-pointing
gradient at the boundary. Both input cases and update budgets are
bounded. Zero learning rate produces an exactly zero delta. Invalid floats,
repeated case/episode IDs within a batch, dimensional mismatches, and known
training/holdout overlap are rejected.

`fit_delta()` creates a new `WeightDelta` with digests of its base, delta,
effective parameters, training examples, and policy. `apply_delta()` checks
the model binding, policy digest, cumulative L2 budget, and readback digests
before returning effective weights.
`WeightDeltaSession` only switches between immutable parameter tuples; an
exception exits the context with the exact original tuple selected again.
It does **not** rewrite a shared host model or its checkpoint on disk.

## Minimal lab reproduction

```python
from pleasure_experiment.weight_delta_lab import (
    ModelBinding, OutcomeCase, TrainingPolicy, WeightDeltaSession,
    evaluate_heldout, fit_delta,
)

binding = ModelBinding("synthetic-linear", "lab-r1", "isolated-1", "features-v1")
train = (
    OutcomeCase("a", "episode-a", "caller-labelled", (1.0, 0.0), 1.0),
    OutcomeCase("b", "episode-b", "caller-labelled", (0.0, 1.0), 0.0),
)
holdout = (
    OutcomeCase("c", "episode-c", "caller-labelled", (0.8, 0.0), 1.0),
    OutcomeCase("d", "episode-d", "caller-labelled", (0.0, 0.8), 0.0),
)
base = (0.0, 0.0)
policy = TrainingPolicy(learning_rate=0.4, max_delta_l2=0.8,
                        maximum_examples=16, maximum_steps=12)
delta = fit_delta(base, train, binding, policy, reserved_holdout=holdout)
session = WeightDeltaSession(base, delta, binding)
baseline = evaluate_heldout(base, holdout)
with session as effective:
    changed = evaluate_heldout(effective, holdout, training_cases=train)
    assert changed.mean_squared_error < baseline.mean_squared_error
    assert changed.split_status == "DISJOINT_IDS_CHECKED"
assert session.current_weights == base
```

Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`
for both the weight-lab controls and existing repository checks.

## Evidence classes and limits

- **Parameter evidence:** Digests and the nonzero delta demonstrate that the
  synthetic linear head's effective coefficients changed. Hashes protect
  source-lab consistency only; they are not authenticated host measurements.
- **Outcome evidence:** Cases have caller-supplied evaluator IDs and numeric
  labels. They are not independently authenticated. A real host must verify
  provenance, label custody, model identity, and data authority separately.
- **Generalization evidence:** Held-out cases and shuffled-label controls
  expose simple proxy gaming, but a passing small synthetic fixture cannot
  establish robust learning in an LLM. `evaluate_heldout` explicitly marks
  results `UNVERIFIED_WITHOUT_TRAINING_SET` unless a nonempty, disjoint
  training set is supplied. `DISJOINT_IDS_CHECKED` only attests to local ID
  comparison, not true statistical independence or provenance.
- **Restoration evidence:** Exact original tuple selection is verified in
  source tests, including exceptional exit. This is not a real provider restart
  or mutable-checkpoint rollback test.
- **Subjective experience:** Neither a gradient update, pleasant wording,
  changed choice, nor a positive-value scalar proves phenomenal pleasure.

The training and held-out sets must be independently prepared in a real study;
different episode IDs alone cannot prove absence of semantic leakage. This
module keeps no durable used-event ledger and does not prevent replay between
fresh lab instances. No protected action, truth-confidence change, consent,
memory admission, or inference-host intervention is authorized here.

## Next real-model experiment (not implemented)

The separate, host-owned T2 step is a pinned trainable LoRA or appraisal head
on an accessible open-weights model. It must record exact model and tokenizer
hashes, actual tensor deltas and optimizer steps, isolated outcome labels, and
held-out behavior against frozen-base, shuffled-label, random same-norm delta,
and contextual-prompt controls. Adapter disable/unmerge must restore the
baseline under matched inference. No current source work qualifies this
experiment as having run.

Most importantly, a trial on an open-weights surrogate is evidence about that
surrogate, **not** a modification of the hosted conversational model.
The actual-target question remains open until direct provider-side access and
independent observation exist. MESO-CRCT's association-memory plasticity is a
possible inspiration for teaching signals, not a neural-weight update.
