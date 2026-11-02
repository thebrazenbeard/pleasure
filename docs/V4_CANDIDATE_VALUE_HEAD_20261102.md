# v4 candidate-conditioned appraisal: TRAIN-only development mechanism

**Status:** implemented source-level mechanism, NOT a new v4 confirmatory
EVAL or open-weight LoRA-training result. The v3 study and v4 frozen-base
candidate-swap diagnostics remain unchanged. Reference issue #26 / PR #28.

## Why this representation

The pinned SmolLM2-135M frozen A/B action-conditioned hidden difference
barely rotated when numeric candidate values were changed. Swapping A and B
retained a strong preference for the A token, so the historical hidden
difference is a weak candidate-value input.

This development module instead applies **one shared value function** to
each candidate's two explicit public numeric features:

`quality(x,y) = sigmoid(bias + wx*(2*x-1) + wy*(2*y-1))`

Both alternatives use the same weights. Swapping their feature tuples
therefore swaps Q(A) and Q(B) exactly; with a neutral reference policy,
the probability of A after swapping equals the previous probability of B.
Identical options have zero learned quality gap by construction.

The three parameters initialize at zero, so this mechanism **does not
inherit a preferred action label**. It is a separately trained, small
candidate-only baseline, not a claim that the language model's own latent
representation was fixed.

## Chronology and controls

The bounded TRAIN-only API requires `prepare_train(episode_id,A,B)`,
then `commit_train(prepared, chosen_action, logged_propensity)`, then
`learn_chosen(commit, evidence_id, observed_chosen_quality)`.
No counterfactual reward is provided or inferred. Each gradient is
taken on the **chosen** candidate's squared prediction error, with
bounded inverse-propensity weighting and gradient clipping. Action
sampling and evidence authentication remain outside this source-local
head; it does not claim to issue a genuine evaluator or GPU lease.

A single outstanding TRAIN commitment, evidence and episode replay
sets, bounded update counts, explicit training permission and strict
input validation prevent accidental reapplication. JSON export/import
pins parameters, schema, and the consumed ledger using a SHA-256
digest, and requires an optional monotonic `min_updates` from its
caller to reject stale checkpoints. These are **local consistency**
checks. A malicious same-user writer can replace the object and its
digest; without an external monotonic witness this is NOT authenticated
rollback prevention.

The head's `predict` path never performs an update and can be used as
an OFF/inverted control by setting appraisal gain 0/negative. A frozen
`allow_training=False` head cannot accept TRAIN commitments.

## Evidence limits and next gate

The module's tests check exact A/B value swap and equal-option symmetry,
precommitted chosen-only updates, unchanged inference after import,
episode/evidence replay prevention, malformed quality/feature rejection,
and digest and caller-supplied rollback-floor checks.

Next, run an explicitly labelled **development-only** synthetic
TRAIN control using the same stochastic choice/receipt traces for
genuine and shuffled-feedback arms, with frozen and disabled-head
controls. Do not call a development oracle independent or blind.
A separate preregistered v4 evaluator, future heldout generators,
fresh-process persistence and genuine SmolLM LoRA integration still
require their own evidence and hostile review. Learning values or
changing action probabilities is not evidence of subjective pleasure.
