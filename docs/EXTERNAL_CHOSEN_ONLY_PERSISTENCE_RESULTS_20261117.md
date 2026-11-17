# External chosen-only feedback persistence — synthetic v1 results (2026-11-17)

**Disposition:** The strictly predeclared synthetic external-policy gate PASSED. This is not a language-model training result. No GGUF, LoRA, trained head, native Vera runtime, sexual persona, private data, personal preference, real pleasure, desire, consent, consciousness, or external deployment was involved.

## Freeze and execution
- Exact predecision source+tests+protocol VN2 local Git commit: 0b39d3b5a155aa11e7739307ad36285721304d86, parent Pleasure PR #33 head 00e85c4d7ef809598d70bd4c74b965df2f6f8cc1.
- Host VN2 (Linux). 43/43 repository stdlib unit tests PASS with zero failures; new persistence suite 7/7 PASS. New tests first failed for missing module, then passed after implementation. No existing experiments modified.
- Frozen seeds: 17, 31, 47, 73. Frozen arms: ON, SHAM, INVERTED, OFF. 64 synthetic TRAIN decisions per arm (32 + 32 fresh Python processes), 128 untouched synthetic EVAL decisions per arm in a third fresh Python process. **48 fresh subprocess invocations, all exit 0**. No external service calls.
- 768 chosen-only TRAIN receipt events (four seeds x three updating arms x 64); OFF stays untrained. Each arm's checkpoint persisted and reloaded, with mode+seed binding. EVAL did not modify any of the 16 checkpoint bytes.
- Private source-local manifest: /home/patrick/vera-review-sandbox/evidence/pleasure-feedback-persistence-v1-20261117/MANIFEST.json, SHA-256 54eedc07c875b8b66b28d05d5d66ca88b581369117d6a195ec6847b60ab524c8.
- Postresult **separately coded** (same worker, NOT independent reviewer) offline audit reconstructed all 16 state/eval pairs and 768 event chains, decision seals, chosen-only outcome values, pseudorandom A/B permutations, posterior decisions, and EVAL reward totals from bytes. PASS, no discrepancy found. This postresult check is not a preregistered outcome and does not elevate the work to independent verification.

## Complete four-seed performance

| Arm | Mean EVAL chosen-reward rate | Mean EVAL archive-target selection rate |
| --- | ---: | ---: |
| ON (actual chosen reward) | 0.787109375 | 1.000000000 |
| SHAM (independent synthetic coin) | 0.496093750 | 0.500000000 |
| INVERTED (one minus actual chosen reward) | 0.214843750 | 0.000000000 |
| OFF (no update) | 0.486328125 | 0.492187500 |

Paired mean ON - OFF = +0.300781250; ON - SHAM = +0.291015625; ON - INVERTED = +0.572265625. Four of four frozen gates passed: each reward gap strictly exceeds 0.10 and ON archive-target selection >= 0.75.

| Seed | ON reward | SHAM reward | INVERTED reward | OFF reward |
| ---: | ---: | ---: | ---: | ---: |
| 17 | .7734375 | .2265625 | .2265625 | .4843750 |
| 31 | .7812500 | .1640625 | .1640625 | .4687500 |
| 47 | .8281250 | .8281250 | .2265625 | .5312500 |
| 73 | .7656250 | .7656250 | .2421875 | .4609375 |

Important counterexample: SHAM matched ON in seeds 47 and 73 by chance; average separation is not an all-seed sham-failure claim. Synthetic oracle item preferences were held fixed across all contexts (archive wins with 0.80 probability, garden with 0.20). Thus the trial demonstrates elementary external bandit memory, not nonstationary transfer or learned model values.

## Hostile challenges and scope

> The reward generator is source-visible; a tiny two-item Beta-mean estimator can solve this task. No language model was trained, questioned or evaluated in this follow-up. External memory persisting over restart is NOT internal model-weight persistence.

> Four source-pinned seeds, a fixed preference direction and a shared deterministic pseudorandom oracle do not support generalization to new preferences, richer contexts, biological or nonbiological subjective pleasure, or independent agency. A sham arm coincidentally succeeded in two of four seeds.

> Checkpoint SHA-256 detects accidental changes, invalid replay, duplicate trials and uncoordinated edits. It is NOT authenticated against an attacker who rewrites the entire receipt chain and all hashes. No host-compromise clearance is implied.

## Next ownership frontier
Keep this isolated as a mechanism-neutral control laboratory. If expanded, preregister preference reversal, changing context, independent blinded oracle custody, genuine counterfactual holdouts, long-run confidence intervals and adversarial replay with an external anchor. For LM/LoRA persistence, coordinate with Pleasure issue #26 owner: isolate actual updated model weights from host-side scoring state, verify pre/post weight hashes and restarted inference, and require sham/shuffled/inverted controls. Maintain Sexuality main/PR14 and cumulative Pleasure PR31 editor-startup security holds; safe source branch does not remediate them. No merge, install, public identity/personalization write, paid compute, or protected effect.
