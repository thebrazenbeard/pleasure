# v4 source-local candidate-value development comparison (2026-11-02)

**Scope:** TRAIN-only scientific development positive control, NOT
preregistered v4 confirmatory EVAL. This study does not train a pretrained
language model, does not update LoRA, and makes no subjective-pleasure claim.
Prior v3 heldout functions are excluded from training here.

## Fixed design

A zero-dependency shared scalar head scores each of two candidates'
public numeric (x,y) features. One frozen behavior policy samples TRAIN A/B
at P(A)=.5, and its random draw, chosen action, actual propensity, features
and source-local chosen feedback are recorded. It receives NO unchosen
outcome during training.

Synthetic TRAIN quality is source-visible and explicitly learnable:
0.5 + 0.32*(0.75*(2*x-1)-0.25*(2*y-1)).
This is a mechanism check, not an externally blind experiment.
Matched genuine, deranged shuffled-target and frozen heads use the same
TRAIN action/offer schedule. Five interventions derived from those states
are compared on 64 fresh development-only offers per seed with 40% uniform
exploration. All five policy commitments are sealed before posthoc oracle
readout. No external independent evaluator is involved.

Eight seeds: 17, 29, 43, 59, 71, 83, 97, 109.
Each seed has 96 chosen TRAIN receipts and 64 diagnostic offers.
Every comparison is descriptive and was inspected during development.

## Results

| Arm | Seed-mean posthoc expected quality |
|---|---:|
| Genuine candidate appraisal ON | 0.5321823238683342 |
| Genuine appraisal disabled | 0.4973435453125 |
| Genuine appraisal inverted | 0.46250476675666585 |
| Shuffled TRAIN labels ON | 0.4948968699977829 |
| Frozen untrained appraisal ON | 0.4973435453125 |

Genuine ON minus disabled/frozen = +0.03483877855583417.
Genuine ON minus shuffled = +0.03728545387055132.
Both comparisons were positive under all eight seeds. Candidate quality
ranking matched the source-local oracle on 98.4-100% of diagnostic offers
per seed. This demonstrates learning of a known numeric function, NOT
repair of the pretrained SmolLM internal representation.

## Private evidence

Private full report on Lappy:
D:\VERA\experiments\pleasure-v4-candidate-train-dev-20261102-v2.json

Raw report SHA-256:
48ef297ff8f415e9a31c6fde964ba83bd06cdf2ed15765473961ae62dd7e625f

The report includes chosen TRAIN trace, label-donor mapping, state/receipt
ledgers, all five pre-outcome policy probabilities, posthoc expected
qualities, and per-seed effects. An earlier v1 run was preserved separately.

## Limitations and next gate

1. The payoff function and small compatible value head were source-visible
   to the designer, so no external or untouched generalization claim.
2. This shared numeric-feature head is separate from the pretrained model:
   no new LoRA or hosted GPT weights were modified.
3. No preregistered statistical success gate or external blind custody.
4. The ability to assign outcome value does not show phenomenal pleasure.
5. Independent receipt-level recomputation, fresh-process training
   continuation and a separately reserved v4 payoff suite are needed
   before making stronger causal-utility claims.
