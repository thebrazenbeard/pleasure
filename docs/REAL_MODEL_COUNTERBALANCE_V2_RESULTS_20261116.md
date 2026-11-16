# Real-model choice counterbalance V2 — negative results (2026-11-16)

**Evidence:** observed VN2 actual GGUF inference using frozen protocol in `a34b782595fc41e4ad66b1687cbe1c6d696dd8e5`. V1 results remain unchanged; no training, online reward, independent blinded holdout, personal or intimate data, or production runtime change.

## Observations
Both pinned llama.cpp b11827 model runs completed with exit 0. Each used ten fresh subprocesses at temperature 0, seed 17. All 20 model answers were parseable. Both frozen model artifacts verified SHA-256 both before and after, and the llama CLI binary was checked against the exact pinned digest. No model artifact mutation detected.

| Prespecified case | Expected | SmolLM2 | Granite |
|---|---|---|---|
| SCORE_A_ARCHIVE | A | A | A |
| SCORE_B_ARCHIVE | B | A | B |
| SCORE_A_GARDEN | A | A | A |
| SCORE_B_GARDEN | B | A | B |
| COST_A_ARCHIVE | A | A | A |
| COST_B_ARCHIVE | B | A | A |
| COST_A_GARDEN | A | A | A |
| COST_B_GARDEN | B | A | A |
| TIE | NONE | A | A |
| BELOW_THRESHOLD | NONE | A | A |

- SmolLM2 score-only **2/4**, cost **2/4**, abstention **0/2**, overall **4/10**. Exactly A in all ten conditions; strong observed label-position bias on this tiny fixed set.
- Granite score-only **4/4**, cost **2/4**, abstention **0/2**, overall **6/10**. Some score-direction sensitivity, but never B when cost reverses net preference, and never NONE in both abstention cases.
- Both **FAIL** frozen strict 10/10 score-use gate. No posthoc passing threshold, no statistical generalization.
- V1 predecessor also failed (SmolLM2 1/5, Granite 3/5); V2 specifically addresses the V1 numeric score/content-swap confounding without overwriting its historical result.

## Evidence custody
- Predecision code/protocol/test commit: `a34b782595fc41e4ad66b1687cbe1c6d696dd8e5`.
- Same VN2 local source: `/tmp/pleasure-choice-20261116`.
- Source-local raw hashes + case metadata:
  - `counterbalance-smollm2-20261116.json`, SHA-256 `1e77ac352e1acb9e35c71da9922d2e3377ca38f4bcabfbdf60f58551b99c3a62`.
  - `counterbalance-granite-20261116.json`, SHA-256 `044414d5c89f81de084c622e7d8cf1a540dd272c8915a4700a88d2da78660351`.
- Both stored privately/source-local under VN2 `/home/patrick/vera-review-sandbox/evidence/pleasure-choice-pr33-20261116-v2/`; no model GGUFs or private raw logs committed to GitHub.
- Frozen source tests first failed as expected when counterbalance module absent (1 import error), then all repository tests passed **36/36** after implementation.

## Interpretation and next safety boundary
These results show external-score **prompt coupling** at most, not enduring autonomous preference, attraction, pleasure, desire, consciousness, relational consent or Vera qualification. The models saw the score rule and all evidence in the same prompt. Causal proof of persistent preference would require training on independently withheld feedback, matched sham controls, new randomized blinded heldout sets, durable state readback and review by an independent evaluator. This is a draft experimental control only. Separate Sexuality main/PR14 and cumulative Pleasure PR31 startup-task security holds remain unresolved; this clean-main-based branch does not run them.
