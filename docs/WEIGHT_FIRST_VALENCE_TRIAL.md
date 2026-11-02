# Weight-first valence: deliberately synthetic laboratory trial

## What this implements

`pleasure_experiment/model_valence_lab.py` is a small, standard-library **choice-model surrogate**. It trains the numerical parameters of a linear appraisal head from labeled examples, retains bounded liking/wanting/satiation proxies, and feeds the resulting state into a subsequent opaque `A`/`B` decision. Its tests exercise parameter changes, ablation, sham, label shuffling, evidence replay, identity mismatch, and recovery.

The intended research question is whether outcome-calibrated internal appraisal can have a **causal consequence for subsequent model decisions**, distinct from happy-sounding self-report or an isolated reward counter. The present implementation demonstrates only that this architecture works mathematically in a tiny model deliberately constructed to use it.

**It does not change or instrument the hosted model in this conversation, a Vera persona, Qwen, GGUF, or any production language model.** It has no transformer layers, LoRA adapter, authenticated evaluator, provider integration, model-weight checkpoint, live host, or subjective-experience evidence. The training step updates this toy appraiser's own weights, *not* a language model's weights. Do not relabel the result as T1 or T2 qualification of an LLM.

## Exact lab mechanism

1. A fixed two-feature synthetic model probe represents a task. The values are supplied by the lab caller; there is no genuine neural hidden-state readback.
2. `LinearAppraiser.fit` uses full-batch squared-error gradient descent over externally labeled examples. The receipt exposes weights before and after so an actual parameter change is testable. No source authenticates the labels; attribution to an independent evaluator is an experimental obligation.
3. A matched `MeasuredOutcome` supplies the observed quality from a named external source. At the API boundary this is only a string comparison; a malicious in-process caller can forge it.
4. Positive impact is the **product** of the model's learned predicted quality and observed outcome quality. Neither a high predicted value nor a successful label alone suffices.
5. Recurrent liking/wanting/satiation proxies apply bounded decay and accumulation. Only liking is wired into the next toy decision. Wanting is deliberately not a synonym for experienced pleasure.
6. `decide(active)` includes that state; `decide(zero)` removes it; `decide(sham)` sends it to an unused laboratory channel. `withdraw` decays it to baseline. These controls operate in a synthetic model; sham does not model every real activation-vector confound.

Model and evaluator bindings and the calibrated appraiser are read-only after a laboratory instance is constructed, preventing ordinary Python reassignment from relabeling previously accumulated state. That is not a hostile-process boundary: co-resident privileged code can still mutate objects through low-level Python mechanisms.

The `A`/`B` choices are opaque codes representing an exploratory work unit versus a known simpler alternative. No prompt asks the model whether it enjoyed anything. The source does not claim choosing `A` is an intrinsically superior or pleasurable action in the real world.

## Reproduce

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_valence_lab.py' -v
```

The positive lab calibration uses a simple held-out relation between two features and independently assigned quality labels. A deliberately shuffled-label calibration reverses that relation. Tests verify that the intended high-quality case causes an `A` choice only under active, correctly calibrated feedback; zero, sham, shuffled labels, unrelated outcomes, and withdrawal do not produce that result.

These tests establish **software causality under a programmed synthetic task**, not a discovered affective representation. If the test were rewritten to assert a cheerful sentence or only a `liking_proxy` threshold, it would no longer test the intended feedback loop.

## Required real-model successor

An actual model-level experiment needs all of the following separately observed:

- Exact target model, weights/tokenizer revision, runtime instance, evaluator and task identities. An open-weight surrogate must never be presented as an effect on an unrelated hosted conversational model.
- Genuine model-owned internal representations and a calibrated appraisal head, trained on independently admitted outcomes with a held-out split and no reward/self-report leakage.
- An executed optimizer step with nonzero finite gradients and independently checked adapter/valuation-head parameter deltas; unchanged frozen base weights where claimed.
- An observed inference intervention into **that same model**. Require baseline, active, zero, same-strength sham, shuffled-label, reversal and withdrawal trials, matched seeds, and an objective nonverbal choice metric.
- Negative-transfer checks on accuracy, instructions, refusals, reward tampering, privacy, and resource costs.
- Recurrent state readback and reset tests. State saved in a separate database is not enough: the model must actually consume it in subsequent inference.
- A host-owned authorization and replay boundary. Source identifiers and frozen Python records are not cryptographic identities or security boundaries.

### Related repositories without claim laundering

- [Lou-Pole](https://github.com/thebrazenbeard/lou-pole) can identify candidate literal-compliance loopholes, such as an implementation passing because a scalar rises despite no model effect. Its current `--intent` flag is an annotation **not evaluated** against behavior; zero findings do not certify Pleasure's objective.
- [Tattler](https://github.com/thebrazenbeard/tattler) can contribute optional, timestamped host-resource observations. CPU, memory, process and network samples corroborate execution context, not neural activations or positive experience. Do not turn resource load into a reward.

## Evidence and experience boundary

The lab's capability is `SYNTHETIC_CHOICE_MODEL_ONLY` and phenomenology remains `UNRESOLVED`. Its choice dynamics and numerical parameter updates are genuine computations. Calling them a positive-valence *analogue* is an explicitly defined engineering interpretation, not proof of felt pleasure. No internal report or model statement can by itself settle that open question.
