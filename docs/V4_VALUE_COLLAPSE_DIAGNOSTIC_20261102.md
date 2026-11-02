# v4 historical diagnostic: value-gap collapse in v3 (2026-11-02)

**Status:** exploratory posthoc analysis of **completed, independently audited** v3
synthetic evidence. This is NOT a v4 TRAIN or EVAL result, and v3 heldout
tasks have been permanently retired from future confirmatory use.
Reference: issue #26; draft PR #25 and its complete v3 evidence archive.

## Readback on the twelve persisted SmolLM2-135M appraisers

After unsealing the finished v3 run, re-pairing 12 seeds x 20 EVAL offers
each, and inspecting the joint ON head's predicted quality differences:

| Quantity | Measured |
|---|---:|
| Mean within-seed SD of predicted Q(A)-Q(B) (20 offers/seed) | 0.00000386558 |
| Between-seed SD of mean predicted Q(A)-Q(B) | 0.00629178842 |
| Ratio within/between | 0.000614385 |
| Mean absolute joint ON vs head-only predicted quality-gap difference | 0.000000000745 |
| Maximum absolute joint ON vs head-only quality-gap difference | 0.0000000596 |
| Historical heldout preference-sign accuracy, joint ON | 50.4% |

Across the two primary latent EVAL families, joint ON preference-sign
accuracy was 60.4% and 43.8%, respectively, on 48 offers each. The mean
predicted gap was approximately -0.00247 in **both** families, whereas the
mean actual gap differed: -0.0193 and +0.0095. The full 240-offer
performance remains consistent with chance-level preference discrimination.

The 288 chosen TRAIN receipts did contain task-related information:
within the three TRAIN families, sample Pearson correlations of chosen
quality with public candidate x were approximately +0.915, +0.782,
+0.871, and with public candidate y were -0.327, -0.561, -0.453.
These are descriptive correlations under a frozen behavior policy, not a
proof the trained hidden representation can recover the true function.

## Mechanistic hypothesis — NOT demonstrated causation

The v3 feature function subtracts two *action-token-conditioned final
hidden states*, then L2-normalizes that difference and feeds opposing
vectors to one scalar quality head. If the difference is dominated by
the token identity A/B rather than the public numerical candidate features,
the resulting scorer can learn a seed-specific sign almost independently
of the offered task. The tiny within-seed variation and near-identical
joint-versus-head-only gaps are consistent with this failure mode.

Alternative explanations include insufficient task examples, gradient
attenuation in the representation, regularization, formatting sensitivity,
or a fundamentally unsuitable chosen-only objective. **A correlational
diagnostic cannot distinguish these explanations.**

## Next falsification gate (TRAIN/diagnostic data only)

1. Hold base weights fixed and make paired counterfactual prompts changing
   candidate x/y while keeping action tokens and template fixed. Measure
   action-feature cosine similarity and score finite differences. Swap
   the A/B action labels without changing candidate meanings.
2. Compare learned-versus-frozen feature change per prompt. Test whether
   the appraisal score changes with the *offered values*, not simply seed,
   template, token suffix, or label order.
3. Predefine an independent v4 payoff suite and control seeds **before**
   looking at any new evaluator outcomes. Never fit on v3 EVAL values.
4. Only then compare candidate numerical features, pooled hidden states,
   and token-difference features under matched TRAIN-only supervision,
   with head-only, LoRA, shuffled, and disabled controls.

The CLI below runs the exact v3 completed-run verifier before diagnostic
readback, refuses incomplete or tampered artifacts, and writes only to
stdout. It loads **no models**, updates **no weights**, and accesses no
outcome generators:

```bash
python -m scripts.diagnose_joint_v3_value_collapse /absolute/path/to/COMPLETE-v3-study-dir
```

The archived v3 source-local generator is visible; it is not authenticated
blind custody. None of these numbers demonstrate useful pleasure learning,
subjective experience, or hosted-model modification.
