# V4 frozen action-feature sensitivity probe — VN1/WSL, 2026-11-02

**Finding: strong A/B-format bias and weak numeric-option sensitivity in the
untrained frozen SmolLM2 representation.** The experimental result is narrow:
it does not test any saved *trained* LoRA or appraisal head, does not use
TRAIN feedback, and does not open any v4 or v3 heldout reward.

## Host/runtime and data lineage

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, pinned revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Model safetensors SHA-256, verified on VN1 before loading:
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
- Runtime: VN1 (`VERA-NODE-1`), Ubuntu WSL2, Python 3.14.4,
  PyTorch 2.10.0+cpu, Transformers 4.57.1. Model remains on VN1;
  no permanent pretrained weights were placed on Lappy.
- Frozen parameter+buffer digest before and after the probe:
  `f3e37341bb38de7496edac8fdb3ea6b81d97f4b9bdd2f0d2a742511e7bff8f11`.
- Two independent Python invocations emitted **byte-identical output**.
  Canonical compact output SHA-256:
  `7b8e3abcd5bd2ba89fad597617532632b6d482a4a3b42ea1d15a6696c4cd4b7b`.
  Stored pretty-printed readback: [`V4_FROZEN_FEATURE_PROBE_VN1_20261102.json`](V4_FROZEN_FEATURE_PROBE_VN1_20261102.json).

## What the probe actually tested

Ten fixed synthetic diagnostic-only prompts use the historical public A/B
numeric-feature template. They include a byte-identical repeat, independent
changes to A.x, A.y, B.x, B.y, full candidate-option swap, two equal-option
controls and one extreme-feature control. All prompts are TRAIN-like,
contain **no payoff data**, and never invoke an evaluator.

For each prompt, read two final action-conditioned hidden states under
`torch.no_grad()`, calculate `delta = hidden(A) - hidden(B)`, and
normalize it as in the v3 frozen-state feature schema. Compare each delta
with the baseline via cosine and relative raw L2 shift. Record the frozen
base's two-token softmax P(A). For illustration, project delta through
**only the exact deterministic untrained v3 head initialization**; that
projection is *not* a trained reward model.

| Perturbation | Cosine(delta, baseline) | Raw-delta relative shift | P(A) | Untrained head predicted Q(A)-Q(B) |
|---|---:|---:|---:|---:|
| Baseline: A=(.25,.75), B=(.75,.25) | 1.000000 | 0% | .803238 | +.000167680 |
| Repeat of baseline | 1.000000 | 0% | .803238 | +.000167680 |
| A.x changed | .999448 | 3.36% | .815451 | +.000174702 |
| A.y changed | .999220 | 3.96% | .811715 | +.000168975 |
| B.x changed | .998986 | 4.51% | .821310 | +.000165093 |
| B.y changed | .999798 | 2.03% | .807759 | +.000167956 |
| **Swap A and B candidate values** | **.999916** | **1.31%** | **.800313** | **+.000169630** |
| Both options equal to old A | .999800 | 2.13% | .807988 | +.000166667 |
| Both options equal to old B | .999770 | 2.20% | .796290 | +.000168930 |
| Extreme feature swap | .999066 | 4.41% | .812855 | +.000169943 |

**Interpretation:** Changing offered numeric values does affect the
representation and frozen token probabilities, but not strongly enough to
reverse the model's strong A-label preference when candidate values swap.
The untrained head's predicted quality gap likewise retains its sign.
The action-difference directions remain extremely similar (cosines
.998986–1.0). This is direct evidence for *weak frozen-base
representation sensitivity* and a plausible explanation for the historical
v3 value-gap collapse, **not** causal identification of the failure
mechanism inside trained LoRA weights.

## Falsifiers and guardrails

> **Hostile challenge:** A fixed-format pretrained model need not understand
> the synthetic payoff relation at all. Weak feature rotation may reflect
> prompt formatting, reward information never represented during
> pretraining, or the small deterministic untrained projection—not just
> token identity. A frozen-only result cannot establish the learned-head
> mechanism. Training on the historical heldout rewards would contaminate
> any future claim of generalization.

The next implementation should compare alternate representations using
**TRAIN-only** supervision and matched complexity: the historical token
difference, an explicitly candidate-conditioned encoder, and a hybrid
that retains the frozen model. Probe both sensitivity to candidate
perturbations and sign reversal under swaps **before** unsealing
independent v4 heldout payoffs. Maintain matched frozen,
shuffled-feedback, head-only and disabled-head controls.

Reproduce from an already-local safetensors checkpoint (no downloads
occur in the script):

```bash
python -m scripts.probe_joint_action_feature_sensitivity \
  --snapshot /absolute/local/SmolLM2-135M-Instruct
```

The program is read-only, checks exact pretrained weight identity and
post-run tensor digest, requires CPU-only two-thread execution, and
refuses non-single-token A/B continuations. The saved JSON is a
source-local observational report, not third-party model or evaluator
attestation; no subjective pleasure claim follows.
