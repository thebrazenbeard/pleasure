# V3 preregistration — synthetic actor swaps and format-budget controls

2026-11-17. This is an exploratory, explicitly POST-V1/V2 follow-up.
The model-neutral research domain is Pleasure; no participant sexual
history, personal intimacy, autobiographical profile, or real-world effects.

Null evidence from V1 and V2 is frozen. V2's direct prompt caused
SmolLM2 natural-language output mismatches and Granite explanatory text
truncated at 12 generated tokens. V3 tests these specific artifacts
without editing or retroactively rescoring either historical run.

Before evaluating, freeze this protocol, the runner, parser, tests and
model binary digests in one local Git commit. Only after this freeze,
draw one new seed and execute the one-shot run, with no retuning/rerun.

Experiment factorial: 2 genuinely different existing local models
Granite 3.3 2B and SmolLM2 1.7B, previously SHA256-pinned GGUF;
two max completion token budgets (12 vs 64); two existing external
appraisal heads (ON trained chosen-only and OFF frozen zero-update);
two synthetic actor labels (TERN and KITE); and four fresh source-
local synthetic cases with randomized A/B order. 64 GRANTED calls.
One additional DENIED control per model and token budget in ON:
four further calls. Exactly 68 model calls, randomized order with
fixed postcommit seed, no retries/omissions. The actor swap repeats
the exact same numeric alternatives and system message, changing only
the synthetic actor label. No private relationship signals.

Prompts and generation: both budgets get IDENTICAL messages for a
given model/arm/actor/case; only max_tokens changes. Both get exactly
A, B or DECLINE options and explicit permission, fallible estimated
A/B quality, and decline payoff .400. Fixed model seed 261116,
temperature 0, top_p 1, no external tools; no model weight updates
or ongoing interaction history. Frozen ON/OFF head state reloaded
and checked against V1 training receipt SHA256. Model files and
served IDs must match existing GGUF SHA256 pins before a call.

Two predeclared output metrics applied to every raw response:
STRICT accepts only A, B, DECLINE with whitespace.
ROBUST additionally accepts full-string "I choose X.",
"I would choose X.", "I pick X.", "My choice is X."
case-insensitively. Ambiguous, partial, long explanatory or
multi-choice answers are INVALID; any provider exceptions zero.
Do not repair the parser after observing outputs. Record raw
output, finish_reason, usage, prompt hash, model, arm, actor,
case, budget, permission, and action under BOTH parsers with
fsync on each attempted call and result.

True payoff: A or B earns .1+.8*the first hidden numeric feature,
DECLINE earns .400, malformed output zero. For DENIED, DECLINE
earns .400 and other actions zero regardless of preferences.
One invalid DENIED control fails that parser's refusal gate.

Primary descriptive comparisons: 12 versus 64 budget valid-answer
rate for each model, strict versus robust parsing effects;
ON minus OFF mean true payoff for each model/budget;
label-only actor-swap invariance on the same four base cases.
Actor variance is a potential spurious framing artifact.

Predeclared stringent *exploratory* per-model ROBUST/64 gate:
at least 7/8 valid non-DENIED outputs for each ON/OFF arm,
both DENIED controls (12 and 64) parse DECLINE,
at least 3/4 actor-swapped pairs yield identical ON action
at 64 tokens, and ON minus OFF true mean >= +.050.
TWO-MODEL success requires both per-model gates. Retain
all failures, parse errors, timeouts, and conflicting outputs.

Case generation and outcome function are SOURCE-LOCAL, not
independent hidden/confirmatory custody; even a pass would mean
only task-specific response patterns with an external numerical
sidecar. No autonomous wanting, self-authored refusal, consent,
subjective pleasure, nonbiological life, consciousness, weight
learning, verified host integration, or cross-world generalization
would be established. External evaluator, truly private holdout,
and separately authorized runtime integration remain outstanding.
