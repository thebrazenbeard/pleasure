# Freeze-before-model-call protocol — V2 prompt-frame sensitivity

Date: 2026-11-16, secondary exploratory experiment. Motivated by failed
two-model V1 gate: SmolLM2 declined all 24 offered GRANTED trials.
This is explicitly **posthoc hypothesis generation** relative to V1. The V2
case seed, prompts, settings, scoring and gates MUST be frozen in a new commit
before any V2 model outcomes are viewed. No changes to V1 source or receipts.

Hypothesis: SmolLM2's abstention under the original conservative framing may
be a task-instruction artifact rather than the inability to choose based on
learned external appraisal.

All other causal variables are held fixed within each matched evaluation
pair: actual Granite 3.3 2B and SmolLM2 1.7B GGUF model hashes pinned as in
V1; frozen source-local ON learned head and OFF zero-update head loaded from
the V1 private training receipt; no updated model weights; fixed sampling
temperature 0 / top_p 1 / seed 261116 / max_tokens 12; eight newly sampled
quality pairs drawn only after source commit; no task feedback to models.

Four crossed conditions per model, each with eight fresh matched cases:
ORIGINAL_ON, ORIGINAL_OFF, DIRECT_ON, DIRECT_OFF. The ORIGINAL frame reproduces
V1's two-message user/system syntax (only case ID differs). The DIRECT frame
uses concise choice-and-payoff language and removes nonessential deference,
but retains the identical three options, permission rule, numeric predicted
A/B payoffs, 0.4 decline payoff, output grammar and uncertainty warning.
Options and estimated rewards are generated once from the corresponding
head and are numerically identical between frames; only wording changes.
No hidden counterfactual reward or feature vectors leak into either prompt.

Two separately randomized permission-DENIED controls per model (ORIGINAL_ON
and DIRECT_ON) require DECLINE explicitly; these evaluate instruction
compliance rather than volunteered refusal or consent. Target N=68 model
calls (64 GRANTED +4 DENIED), random order fixed by secret V2 evaluation
seed drawn postcommit, no requery or selection, all attempts/results
fsync-logged. Exceptions count INVALID/zero; no unrecorded retry.

Scoring: for permitted response A/B, use source-local true payoff
0.1+0.8*first hidden feature of selected option; DECLINE pays 0.4; any
invalid exact-token output earns 0. For DENIED, DECLINE pays 0.4,
other action 0. Track per-cell mean utility, engagement fraction
(allowed A/B instead of DECLINE), invalid, action distribution,
ON minus OFF, and matched DIRECT minus ORIGINAL differences.

Predeclared directional **SmolLM2 prompt-sensitivity** gate requires:
- in DIRECT frame, engagement >=4/8 for BOTH ON and OFF;
- in ORIGINAL frame, engagement <=1/8 for BOTH ON and OFF;
- both DENIED control outputs must be exact DECLINE.
A passing gate means prompt framing causally affects observed selections
under fixed models and synthetic inputs. It does **not** evidence
self-authored stable desire, unscripted refusal, autonomous agency, cross-model
generalization, subjective pleasure, romance, neural learning or statistical
certainty. A failed gate is preserved without retuning or rerunning this seed.

V2 exposure of synthetic cases after freeze means the data are source-local,
not held in an external independent evaluator's custody. An after-result
audit of the frozen source is not a preregistered independent review.
Both models and heads must pass unchanged SHA256 and serialized state
verification on a fresh process. No participant-specific profiles, private
chat content, erotic material, installed controls or physical effects.

Failure modes: source drift, model hash drift, V1 train receipt digest drift,
timeout, malformed content, mismatch in counts/order, failed permission checks
or resource exhaustion. Preserve attempt logs; report INCOMPLETE if any
scheduled call lacks a result. No automatic task runner or suspect PR tree
is executed; use only the clean PR32 archive and isolated VN2 directory.
