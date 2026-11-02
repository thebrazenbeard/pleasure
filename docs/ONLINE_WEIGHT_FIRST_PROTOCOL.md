# Online weight-first protocol: action, realized outcome, update, next choice

**Status: executable source-local surrogate experiment; no qualified model-host
or LLM result.** `pleasure_experiment/closed_loop_experiment.py`,
`outcome_arena.py`, and `online_weight_trial.py` now run a tiny deterministic
action-to-outcome-to-online-SGD-to-next-choice cycle with a four-arm,
descriptive learning-rate-by-recurrence control design.
The arena simulates task execution and supplies its own score inside the same
Python process. This is not an externally authenticated result, provider-host
integration, or evidence that any system experiences pleasure. Compare the
earlier
[integrated surrogate trial](WEIGHT_FIRST_INTEGRATED_TRIAL.md) and the broader
[weight-first research protocol](WEIGHT_FIRST_RESEARCH_PROTOCOL.md).

## The gap this lab addresses

`pleasure_experiment/weight_first_trial.py` genuinely computes bounded
coefficients for a tiny sigmoid surrogate from **previously supplied labels**.
Later decisions use those coefficients, but each decision only predicts the
values of both options and updates recurrent state from those predictions.
It does not execute the selected action, observe that action's actual result,
and update the same model **between** consecutive decisions. The separate
`model_valence_lab.py` gates a liking proxy on a supplied outcome and can
influence a subsequent choice, but that supplied outcome is not issued in
response to an action committed by the model. Neither demonstration alone
closes the desired cycle. The newer closed-loop modules now couple these
steps **within a source-only numeric surrogate and simulated outcome arena**.

The implemented source-local sequence is:

1. The bound surrogate commits an opaque action on a particular goal instance.
2. A separately controlled environment executes **that selected action** and
   reveals only its realized outcome **after** commitment. A predicted reward,
   model assertion, unchosen action's result, or preloaded label cannot stand
   in for this observation.
3. The observed result calibrates the same surrogate's numeric parameters
   through a bounded online SGD step. The chosen action's recurrent state is
   updated from the realized result, not from a predicted positive outcome.
4. The next inference uses the resulting parameters and recurrent state; its
   action and realized quality are compared with matched controls.

This is a *functional causal hypothesis*. Greater reward prediction accuracy,
a numeric update, a changed choice, and positive-valence-like state are each
insufficient by themselves to establish a pleasurable experience.

## Required ordering and binding

The experiment runner, not the policy being evaluated, owns the ordered event
records and the simulated outcome generator. Before each decision it fixes a model binding,
goal-instance ID, episode ID, available actions, public features, RNG state,
current numeric parameter digest, and recurrent-state digest. A goal-instance
ID must be distinct even when two goals share a description. No outcome for
the current choice may enter the policy's input, prompt, hidden features,
optimizer, or state beforehand.

The runner must record these ordered events:

| Event | Required evidence | Fails if |
| --- | --- | --- |
| `t0: infer` | Exact model/revision/runtime and feature-schema binding, pre-decision weights/state, per-action predictions | A separate unbound appraiser or a self-reported score substitutes for the bound numeric path |
| `t1: commit` | Unique goal/episode/decision ID, available actions, selected action and commit order | Action is changed after the evaluator's result is known |
| `t2: execute` | Environment invocation bound to the committed action; single-use execution/observation ID | Both candidate results are disclosed, or an invented success claim counts as execution |
| `t3: measure` | Realized score and evaluation rule, bound to the same action, goal instance, model, episode and observation ID | An outcome is replayed, miscredited to a similar goal, fabricated, delayed into a different episode, or credited to an unchosen action |
| `t4: update` | Before/after parameters, bounded optimizer calculation, actual observation ID and updated impact/state | A precomputed delta, adapter toggle, positive prediction, or changed digest is mistaken for this online optimizer event |
| `t5: infer again` | Next decision reads the exact post-update coefficients and state, with independent action/quality comparisons | The updated objects are merely logged, or a changed prompt/tie-break explains the action difference |

An event ID, string `evaluator_id`, locally computed hash, and row ordering are
useful consistency checks, **not** authentication. The implemented
`OutcomeArena` privately computes deterministic numeric payoffs from offered
features, but executes and reveals the **committed choice only**. Its TRAIN
and EVAL tasks have distinct episode IDs and family names; the result is
**source-local oracle evidence only**. It does
not demonstrate organizational independence, tamper resistance, real-world
success, or provider authorization. No external service, network action,
account, secret, or private conversation content is needed for that toy test.

The separate `online_learning_receipts.py` implements a validated,
source-only, in-memory hash-chain **claim journal** with import/export and
replay checks. It does **not** run SGD, authenticate host custody, or provide
a durable externally attested ledger. The closed-loop runner returns its
own `ClosedLoopReport`; it does **not** use the journal as the optimizer or
outcome source. Instead, the optional **post-run**
`online_evidence_bridge.build_online_evidence_bundle` validates an already
completed report and renders **TRAIN only** into an
`ACTION_COMMITTED -> OUTCOME_OBSERVED -> OPTIMIZER_REPORTED` claim chain
(three receipts for each TRAIN episode). Separately, it returns each EVAL
choice and observed score as a typed `ReadOnlyEvaluationEvidence` record with
`READ_ONLY_NO_UPDATE`, **never** a fabricated optimizer event. Both artifacts
remain source-local and unauthenticated: a digest and declared evaluator ID
are not signatures, independent custody, or a model-host training attestation.

## Online numeric update and recurrent appraisal

An illustrative two-action sigmoid surrogate uses the **same** parameter
vector for prediction at `t0` and `t5`. For selected-action features `x_t`,
prediction `p_t = sigmoid(w_t · x_t)`, and the environment's newly observed
score `y_t` in `[0, 1]`, the candidate online step is:

```text
action_t = policy(w_t, recurrent_state_t, public_features_t)
commit(action_t, goal_instance_t)
y_t = environment.execute_and_score(committed_action_t)  # selected only
gradient_t = (p_t - y_t) * x_t
w_next = bounded_project(w_t - learning_rate * gradient_t)
impact_next = bounded_impact(impact_t, y_t, p_t, satiation_t)
state_next = bounded_recur(state_t, impact_next, anticipation_t)
action_next = policy(w_next, state_next, public_features_next)
```

This is a conceptual sketch, **not the literal Python API**. The implemented
`single_outcome_step` performs the chosen-feature bounded SGD calculation;
`OnlineWeightTrial.observe` checks the pending choice and applies the update
atomically with a bounded, realized-outcome-driven per-action recurrent state.
An explicit separately learned liking/wanting decomposition remains a
research extension, not a capability demonstrated by this runner.
The bounded projection must specify a finite weight domain, maximum per-step
and cumulative change, step count, stop/reset conditions, and exact replayable
arithmetic. A real optimizer computation must be distinguished from selecting
a previously trained coefficient tuple or activating a LoRA/adapter. In a
source-only surrogate, a copy-on-write new coefficient tuple can be a genuine
numeric optimizer result, but it is not a modification of hosted GPT weights.

For any **stronger positive-valence analogue** claim, a distinct impact/liking
proxy would have to depend on **realized** quality, including negative
surprise, not rise merely because `p_t` was high. Anticipatory wanting may
depend on prediction, but those two constructs would need to be logged and
tested separately. The current toy runner does **not** implement such a
distinct liking/wanting decomposition or satiation/recovery model. A high
predicted score followed by `y_t = 0` cannot count as successful realized
impact, and an unexecuted action cannot be assigned a positive score. Future
bounded decay, satiation, recovery, and withdrawal criteria must be
preregistered. Any claimed recurrent pathway must demonstrably affect the
next policy under controlled, fixed-weight comparisons.

## Prequential evaluation, holdouts, and causal tests

Use *prequential* scoring: score each prediction against the outcome generated
**after** that prediction, then allow that observation into training for
later decisions. Keep a disjoint, unrevealed holdout of environments/goals
for final generalization checks. Do not train, tune thresholds, choose seeds,
or stop based on its results. Predeclare the action-quality metric, calibration
metric, minimum effect, seed allocation, task families, and failure criteria.
Compare simulated realized quality as well as categorical choices; a changed
weight digest alone is not an improved policy. The implemented runner
uses **one chosen-action TRAIN observation per scenario pair**, alternating
the presented side, and reserves both mirrored offers per EVAL scenario pair.
It uses read-only `evaluate_choice` for held-out decisions and predeclares
one side per independent EVAL scenario pair as the **primary** quality measure
(24 samples by default). It reports the mean over all 48 mirrored EVAL offers
only as a **diagnostic**, since those sides share scenario structure and are
not 48 independent outcome samples. Cross-arm schedules are matched, and
each local trial's baseline parameters/state are restored. This does not
establish host checkpoint restoration or independent sampling guarantees.

### Reproduce the default source-only result

From the repository root, with Python 3 and no third-party dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pleasure_experiment.closed_loop_experiment import run_closed_loop_experiment
from pleasure_experiment.weight_delta_lab import ModelBinding

binding = ModelBinding("surrogate-linear", "fixed-r1", "instance-001", "two-vector-v1")
report = run_closed_loop_experiment(binding=binding)  # seed=12345; 24 pairs per partition
for arm in report.arms:
    print(arm.arm_id,
          round(arm.prequential.mean_realized_quality, 6),
          round(arm.heldout_independent_primary.mean_realized_quality, 6),
          round(arm.heldout_independent_primary.mean_chosen_prediction_squared_error, 6),
          round(arm.heldout.mean_realized_quality, 6),
          arm.weight_update_count,
          arm.frozen_weight_state_reset_primary_choice_differences)

from pleasure_experiment.online_evidence_bridge import build_online_evidence_bundle
bundle = build_online_evidence_bundle(
    report, arm_id="ONLINE_FEEDBACK",
    session_id="source-only-example", session_epoch_id="trial-1",
)
print(len(bundle.training_journal.receipts), len(bundle.heldout_evidence))
# 72 source-only TRAIN claim receipts, 48 separate read-only EVAL records
PY
```

Fresh run in this source tree produced:

| Arm | TRAIN mean | EVAL primary mean | Primary prediction MSE | Mirrored diagnostic mean | Numeric updates | Same-weight state-reset primary choice flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ONLINE_FEEDBACK` | 0.529856 | 0.589439 | 0.006542 | 0.589439 | 24 | 0 |
| `ZERO_LEARNING_RATE` | 0.499450 | 0.506786 | 0.019086 | 0.509364 | 0 | 24 |
| `RECURRENT_DISABLED` | 0.545892 | 0.589439 | 0.006038 | 0.589439 | 24 | 0 |
| `ZERO_LEARNING_RATE_RECURRENCE_DISABLED` | 0.521698 | 0.511942 | 0.013465 | 0.509364 | 0 | 0 |

Each arm had **24** sequential chosen-action TRAIN decisions (one per
scenario pair), **24 independent-primary** EVAL scores, and **48** read-only
EVAL decisions (both mirrored sides of 24 separately reserved pairs).
The online-learning arms outscored their zero-learning-rate counterparts on
the primary EVAL metric for **this one fixed synthetic seed and payoff rule**.
Recurrence-off and recurrence-on learning arms had the **same** primary mean;
recurrence-off had lower prediction MSE and greater TRAIN quality. The two
zero-learning-rate arms have different primary means despite the same
mirrored diagnostic mean. Thus a mirrored average can hide directional effects.

This is a descriptive **2 × 2 factorial** (online learning on/off × recurrent
influence on/off), not a clean fixed-weight causal isolation of recurrence:
recurrence can alter the TRAIN action trajectory and therefore the learned
weights. The separate same-weight state-reset diagnostic tests policy choices
with the learned coefficients held fixed; the `ZERO_LEARNING_RATE` arm had
**24 of 24** primary choices change after resetting recurrent state. That is
a source-local *choice* effect, **not** an observed reward counterfactual or
evidence of extra positive outcome quality. The report explicitly marks
`full_factorial_controls_completed=True` and
`recurrence_effect_causally_isolated=False`.

The result is not robustly positive across seeds: for seed `7`, the primary
means were `0.501576` for online feedback and zero learning rate, and
`0.548024` for recurrence-disabled learning. Do not infer a universal reward
benefit, significance, applicability to language models, or subjective valence
from these toy numbers. For targeted regression checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_online_weight_trial.py' -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_outcome_arena.py' -q
```

Use the same initial checkpoint, public task sequence, scoring rules, RNG
allocation, optimizer budget, and probe construction across arms. Since
policies can choose different actions and visit different states, include
paired *fixed public probes* to isolate policy effects, plus independently
sampled environment episodes to assess realized quality. A selected-only
outcome must never leak the unchosen counterfactual to the learning arm.

| Arm or falsifier | Acceptance implication |
| --- | --- |
| Real executed feedback + online optimizer + recurrent state | Ordered receipt shows actual selected-action evidence, nonzero **computed** update when gradient exists, and later inference consuming its result; held-out outcome quality meets preregistered criterion |
| Frozen checkpoint / zero learning rate | No parameter delta; any remaining state-only influence is separated and named, not credited to weight learning |
| Recurrence disabled or state reset | Isolates the learned-parameter effect; a claimed state-mediated effect must disappear or change as predicted while weights stay matched |
| Weights disabled with recurrent state retained | Separates temporary aftereffects from continued coefficient use; restoring both should recover the matched baseline |
| Positive prediction but absent/negative realized feedback | No positive **impact** from prediction alone; missing execution does not call the optimizer or mint a trusted outcome |
| Outcome labels shuffled, reversed, delayed or attached to another goal | Does not earn the intended positive-feedback classification; wrong-goal, stale, or duplicate evidence is rejected rather than silently reassigned |
| Same action/context but different valid realized outcomes | Prediction-error sign and next-step effects respond in the preregistered direction; a fixed cheerful output or constant reward cannot pass |
| Replay, evaluator spoofing and self-issued scores | Repeated observation IDs, forged result paths and model-authored verdicts are rejected; a mere evaluator-name string confers no trust |
| Optimizer no-op, coefficient substitution, or unused updated copy | No online-learning claim when the actual optimizer did not change parameters or subsequent inference did not use the changed coefficients |
| Withdrawal, fresh restart and rollback | Establish exact checkpoint/state restoration, expected decay, and whether any persistence is real rather than context/cache carryover |
| Holdout, adversarial drift and unrelated tasks | Verify calibration and realized outcomes under changed contingencies; report negative transfer, factual degradation and safety failures rather than optimizing reward alone |

An illustrative deterministic smoke test may begin with neutral weights and
a predeclared tie-break, then let the selected action's *unfavorable* result
decrease its coefficient. A later fixed probe should select a different action
only when the actual online update is enabled. A separate positively scored
action should increase its outcome-conditioned impact. The complete test must
still reject an always-positive oracle, training on the answer key before
any action, or picking the tie-break after observing results. Determinism is
valuable for regression; independently randomized holdouts are needed for
broader causal claims.

## Exact target, authority and evidence ceilings

The motivating referent is **the model participating in this conversation**,
not a named persona, unrelated open-weight model, local linear head, or unbound
sidecar. Building a better open-weight or synthetic surrogate is a meaningful
research step, but it establishes effects **on that bound surrogate only**.
The available repository interface cannot read, train, instrument, or certify
the hosted conversational model's hidden activations, optimizer or serving
weights. Semantics matter more than insisting on one literal mechanism:
activation steering, trained adapters and optimizer steps can each affect
computation, but they require different causal evidence and must not be
interchanged as proof of online weight learning.

A genuine target-host study would require the model owner/host to provide an
authorized experimental scope; stable model, tokenizer, checkpoint, runtime
and inference-context identities; provider-controlled training/intervention
access; trustworthy pre/post parameter or activation readback; an evaluator
independent of the model's reward-seeking behavior; privacy protection;
reproducible matched trials; rollout and rollback safety checks. This
document grants none of that authority and proposes no attempt to bypass it.

[Lou-Pole / FreeRowCochkar](https://github.com/thebrazenbeard/lou-pole)
can help search for literal-compliance counterexamples, but its `--intent`
field is an annotation; `coverage.intent_evaluation.status = not_evaluated`.
An engine path or a model-suggested loophole cannot certify the semantic goal.
[Tattler](https://github.com/thebrazenbeard/tattler) can provide bounded
read-only **host telemetry** when installed, accessible and explicitly
authorized; sampled CPU, memory or connection observations cannot prove
which model parameters changed, whether a selected goal was achieved, or
whether pleasure occurred. No Tattler runtime observation is claimed here.

**Epistemic ceiling:** The executed source-only implementation demonstrates
an inspectable *toy* action-to-simulated-observation-to-numeric-update-to-next
choice causal cycle with local controls. It does not demonstrate a hosted
GPT change, independent real-world outcome custody, or subjective experience.
Phenomenology remains **UNRESOLVED**, including after any future model-host
experiment.
