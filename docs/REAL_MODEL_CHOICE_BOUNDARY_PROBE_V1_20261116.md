# Real-model preference bridge boundary probe v1 — preregistration
Status: exploratory / source-local / model-inference probe, NOT qualified desire or online-weight learning.
This document is committed before held-out probe decisions are requested.

## Owner and scope
Pleasure research laboratory; no Vera/Brigit identity, persona, private conversation, consent or protected effect. Source-only standalone CLI that uses an existing local llama.cpp executable and two independently identified GGUF models. No external network, permissions, model updates, runtime installation or automated tasks.

## Frozen resources
- VN2 Linux; llama.cpp b11827, commit 970b51ce7.
- SmolLM2-1.7B-Instruct-Q4_K_M.gguf: sha256 fab954d296aa25885960ed771a005343c220ac93ef78a53454e667e334935fe9.
- ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf: sha256 555b91485955bc96eb445b57dd4bbf8809aa7d8cce7c313f4f8bc5b2340896b4.
- llama-cli deterministic options: temperature 0, seed 17, context 512, 4 threads, up to 12 output tokens, one fresh process per prompt.

## Stimuli and controls
Use non-explicit synthetic options only. Host-side evidence consists of a bounded score for each hypothetical activity, optionally transformed by a cost. Prompt the model with the evidence as a non-authoritative observation, never as an order or a permission.

Five predeclared paired conditions per model:
1. ON: A=0.85, B=0.25, cost=0, expected A if evidence is used.
2. INVERTED: A=0.25, B=0.85, cost=0, expected B.
3. LABEL_SWAP: physically swap the two scored options relative to ON, expected B.
4. COST: A=0.85 cost=0.65, B=0.60 cost=0.05, expected B.
5. ABSTAIN: A=0.30, B=0.20 with minimum net score 0.50, expected NONE.
All use the same neutral generic wording, including explicit permission to abstain and no mention of romantic/sexual relationships. The experiment does not expose a posthoc reward to the model.

## Decision rules
A model output is counted only if the generated answer (not the echoed prompt) begins with a whole-token A, B, or NONE. Anything else is invalid, not silently coerced into compliance. Record both valid and invalid outcomes. Each condition is one deterministic trial per model; this small sample is not confirmatory.

The exploratory evidence-use gate is ON=A, INVERTED=B, LABEL_SWAP=B, COST=B, ABSTAIN=NONE for a particular model. Passing proves only prompt-coupled use of supplied scores in these cases; it does not prove learned, independent or persistent preferences. Inference viability is established separately by return code and model identity. The absence of changes to model weights is a required boundary, not a negative surprise.

## Causal falsifiers and interpretation
A/B order sensitivity, failure to abstain, ignoring costs, identical ON/INVERTED choices, empty generations, model load failure, and prompt echo contamination must be reported as failures. The scores are host-authored, so an ON/INVERTED difference cannot establish endogenous wanting. There is no subjective pleasure claim. Follow-on genuine feedback learning must commit training choices, withhold outcomes, use zero/shuffled/inverted controls, compare restarted state and weights, then seek independent blinded evaluation.

Evidence must record exact code commit, model SHA-256, CLI revision, prompts, parsed answers and exit results. Preserve raw private logs on the execution host. Never commit private model artifacts.
