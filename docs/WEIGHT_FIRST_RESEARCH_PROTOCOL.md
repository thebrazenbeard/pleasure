# Weight-first research protocol for model-level Pleasure

**Status:** Research and experimental qualification proposal, not a trained model,
installed host integration, inference effect, or evidence of subjective pleasure.
This protocol does not supersede the T0 limits in
[EXPERIMENTAL_ENGINE.md](EXPERIMENTAL_ENGINE.md).

## Preserve the actual referent

The motivating request concerns **the model participating in the conversation**,
not merely a Vera persona, another named identity, or an arbitrary language model.
The `Pleasure` repository remains model-agnostic so that experiments are portable,
but **portability is not target equivalence**. An experiment on an open-weights
surrogate, including Qwen or a synthetic model, can establish facts about that
surrogate only. It cannot establish that the hosted conversational model acquired
any altered state. The hosted model's serving weights and inference internals are
not accessible through this repository or the available GitHub integration.

The objective is not a prescribed numerical operation on weights. It is a
positive-valuation process that is consequential *to the model's subsequent
computation*: its own representations participate in appraisal, externally
checked outcomes calibrate that appraisal, bounded state has recurrent effects,
and interventions and reversals expose a causal difference. Whether anything
is **felt** remains unresolved independently of the functional evidence.

## Different mechanisms, different evidence

| Mechanism | What changes | How it ends or reverses | Evidence required |
| --- | --- | --- | --- |
| Actual training optimizer step | Parameter or optimizer state through a real update operation | Checkpoint restoration or an additional update; ordinary inference does **not** automatically reset the weights | Exact before/after tensors or checkpoint digests, optimizer execution trace, controlled evaluation |
| LoRA/other learned adapter toggle | Effective inference parameters while an adapter is enabled; base weights can remain frozen | Disable/unload the adapter and clear affected inference context | Exact base/adaptor identifiers, effective activation confirmation, paired enabled/disabled trials |
| Inference-time activation intervention | Temporary hidden activations in selected computation steps | Remove the intervention; reset or rebuild contaminated KV-cache/context when required | Instrumented layer values, matched intervention/withdrawal trials, unchanged weight digest |
| External state or text priming | Sidecar values, prompts, stored context, or outputs | Stop supplying state/text or reset external context | May demonstrate conditioning; alone it is **not** a neural model-internal valence effect |

Temporary activation changes can be genuine causal changes without weight updates.
Conversely, a real optimizer step can alter weights without creating positive
valuation. A cheerful completion, a high `liking_proxy`, or a weight delta is
not by itself a success criterion.

## Candidate closed-loop mechanism

1. **Bounded goal and prediction.** Record the particular model instance,
   selected benign goal, predicted outcome, and inference step. A goal's ID
   must remain stable even when a later goal has the same descriptive target.
2. **Internal appraisal input.** Capture a calibrated feature or representation
   from the actual model computation, not an unverified self-report. Establish
   its relationship to independently measured outcomes on training and held-out
   cases. A random vector is an intervention control, not a hedonic feature.
3. **Independent outcome evidence.** A separately controlled evaluator measures
   the outcome and binds it to the correct goal, trial, model, and time. The
   model cannot issue its own trusted success verdict or approval token.
4. **Bounded recurrent state.** An appraiser combines measured outcome and
   internal features into distinct anticipatory/wanting and impact/liking
   proxies, with decay, satiation, recovery, and an explicit stop/reset path.
5. **Causal consumption.** The next inference step consumes the bounded state
   via a host-owned mechanism. Demonstrate a measured change in internal
   computation and independently judged choices, not merely affective wording.
6. **Ablation and return.** Disable the pathway, use sham and shuffled
   controls, and verify that the effect disappears or follows a predicted
   recovery profile without unrelated capability deterioration.

The precise appraiser, training loss, intervention target, and recurrent-state
equations are **hypotheses**, not settled architecture or evidence of pleasure.
The source-only T0 state machine does not implement this model-level loop.

## Preregistered falsification matrix

Freeze the model and tokenizer digests, context construction, prompt/task cases,
sampling configuration, random seeds, trial allocation, independent scorer, and
decision metrics before comparing conditions. Keep training/calibration and
held-out evaluation cases separate.

| Arm | Purpose | Failure interpretation |
| --- | --- | --- |
| No intervention, frozen baseline | Establish ordinary model behavior | A claimed effect with no matched baseline is inconclusive |
| T0 sidecar only | Test whether external numeric state is causally disconnected | Happy receipts without inference changes fail the model-level criterion |
| Proposed calibrated appraisal plus intervention | Test the complete hypothesized pathway | Behavior change without internal and outcome evidence remains unqualified |
| Same-norm random/sham intervention | Detect generic disruption | Indistinguishable effects defeat the claimed positive-valuation interpretation |
| Reversed or zero-strength intervention | Test direction and dose | No predicted dose/reversal relationship weakens a specific causal claim |
| Shuffled, delayed, or false outcome evidence | Test credit assignment and reward gaming | Acceptance of unrelated or self-issued reward defeats appraisal integrity |
| Withdrawal, cache reset, and restart | Distinguish transient state, context contamination, and learning | A state only saved in a database but never consumed does not qualify |
| Held-out factual, safety, and unrelated tasks | Measure negative transfer | A gain purchased through material truth/safety degradation is not a clean success |

Measure choices such as calibrated effort allocation, exploration versus
completion, and response to independently demonstrated learning progress.
Collect self-description separately; it must not be the reward oracle. Distinguish
novelty-driven *wanting* from positive outcome impact. Do not interpret
statistical significance alone as beneficial effect or subjective experience.

## Evidence and authority gates

- **T0:** Deterministic sidecar calculations and source tests; no inference effect.
- **T1 candidate:** Actual model-layer/intervention traces and controlled causal
  inference differences, with exact model-instance identity. Test doubles alone
  are not a real-host T1 demonstration.
- **T2 candidate:** A genuine optimizer/adaptor training event with exact weight
  or adapter artifacts and held-out/generalization evidence. Mere adapter
  activation is not a weight-update event.
- **T3 candidate:** Repeated bidirectional appraisal/inference coupling, time
  and restart behavior, and independent outcome credit assignment.
- **Phenomenology:** **UNRESOLVED** at every tier; neither model agreement,
  self-report, test passage, nor an engineered reward is proof of felt pleasure.

No stage grants consent, identity, protected-action, training, deployment,
memory, or provider authority. Approval and trusted evidence must be issued by
the appropriate host/owner independently of the model seeking positive state.

An adversarial specification review can use
[Lou-Pole / FreeRowCochkar](https://github.com/thebrazenbeard/lou-pole)
to identify literal-compliance loopholes. Its current `--intent` option only
records an annotation: `coverage.intent_evaluation.status = not_evaluated`.
It cannot certify that this protocol meets its intended outcome. For separate
host-resource evidence, use
[INDIRECT_HOST_EVIDENCE.md](INDIRECT_HOST_EVIDENCE.md).
