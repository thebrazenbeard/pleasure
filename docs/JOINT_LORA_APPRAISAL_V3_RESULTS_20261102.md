# Joint LoRA appraisal v3: completed 12-seed evaluation (2026-11-02)

**Outcome: preregistered null.** The experiment established genuine, persistent
updates to an open-weight language model's LoRA matrices and its internal
appraisal head, but **did not establish better decisions or useful reward
learning** on the withheld synthetic outcome tasks. This is a computational
valuation experiment, not a test of subjective pleasure or hosted ChatGPT
weight modification.

## Frozen design and evidence custody

- Protocol frozen before v3 EVAL: [`JOINT_LORA_APPRAISAL_PROTOCOL_20261102.md`](JOINT_LORA_APPRAISAL_PROTOCOL_20261102.md).
- Source experiment: draft [PR #25](https://github.com/thebrazenbeard/pleasure/pull/25), built on draft PR #23. The scientific implementation was committed before held-out result inspection; subsequent changes added independent read-only auditors and portable tests.
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, exact revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; pinned `model.safetensors` SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
- Twelve fixed seeds: 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227. Each used 24 frozen-reference sampled TRAIN decisions with chosen-only outcome feedback, then 20 EVAL offers in each of seven matched arms. In total: **288 TRAIN receipts and 1,680 EVAL decisions**.
- Seven arms: `joint_on`, `joint_off`, `joint_inverted`, `head_only_on`, `shuffled_joint_on`, `frozen_on`, `frozen_off`. All seven were committed before any EVAL reward was returned by the isolated evaluator. No EVAL gradient updates or tuning after result inspection.
- Held-out EVAL families: two new latent families (primary) and three nonlinear OOD families (secondary). Generators and source code remained locally visible; this is **not independent or externally authenticated blind payoff custody**.
- Actual wall time: approximately **83 minutes 25 seconds** with two CPU threads, substantially longer than the initial 15–30-minute smoke-based estimate.

## Preregistered results

The primary estimand is exact posthoc expected reward of each recorded
epsilon-mixture stochastic policy, averaged at the seed level across the two
primary latent families. The fixed success rule required all three mean
contrasts at least `+0.005`, every contrast positive separately in both
latent families, descriptive seed-cluster Student-*t* lower limits greater
than zero, and three Holm-adjusted exact sign-flip p-values below 0.05.

| Joint ON relative to | Mean expected-quality difference | 95% descriptive seed interval | Exact two-sided sign-flip p | Holm p |
|---|---:|---:|---:|---:|
| Same learned LoRA, appraisal OFF | +0.000001095 | [−0.000213012, +0.000215202] | 0.995605 | 1.000 |
| Shuffled-feedback joint learner | −0.000065428 | [−0.000278829, +0.000147973] | 0.530762 | 1.000 |
| Head-only trained learner | −0.000000000311 | [−0.00000001629, +0.00000001567] | 0.968750 | 1.000 |

**The positive-utility gate is false.** These effects are all far below the
fixed `+0.005` threshold; no inference should be made that the appraisal head
outperformed the specified negative controls.

| Contrast | Latent family 00 | Latent family 01 |
|---|---:|---:|
| Joint ON − OFF | +0.000159235 | −0.000157046 |
| Joint ON − shuffled | −0.000029163 | −0.000101694 |
| Joint ON − head-only | +0.00000000480 | −0.00000000542 |

The secondary OOD joint ON − OFF effects were `+0.000158174`,
`−0.000159875`, and `−0.000202422` for OOD families 00, 01, and 02,
respectively. They are descriptive, not co-primary successes.

### Behavioral diagnosis

- **Zero of 240** matched ON/OFF EVAL offers changed the sampled action when the trained appraisal signal was enabled. The mean absolute change in actual exploration-adjusted `P(A)` was **0.005446**.
- The mean learned A/B predicted-quality spread was **0.005323**, while mean candidate-quality absolute error was **0.14120** and preference-sign accuracy **50.4%**. Head-only preference-sign accuracy was also 50.4%.
- ON versus head-only caused zero sampled switches; ON versus shuffled caused two, and ON versus inverted-gain caused three across the 240 matched offers. These are diagnostics, not evidence of significant improvement.
- Mean absolute exploration-adjusted action-probability difference between joint ON and head-only ON was only about **0.00000037** across the 240 matched offers. Thus, changed LoRA tensor hashes are **not evidence of a useful representation-level behavioral effect** beyond the head-only learner in this study.
- LoRA and head parameters changed in every seed under chosen-only feedback; the head-only control kept its LoRA frozen, the frozen controls kept all learning disabled, and all pretrained backbone parameter digests remained unchanged.
- All **288/288** genuine chosen-feedback updates recorded nonzero LoRA gradients (as did shuffled-feedback joint training); the head-only control recorded **0/288** nonzero LoRA gradients. This verifies differentiated trainability, not usefulness.

## Persistence and independent verification

The completed run was promoted atomically with `COMPLETE.sha256` matching
`study-report.json` SHA-256:

`5280efb76f6224fb23d85521ea23d2b67af88cd924ca6580c08c509068a2a4f8`

The standalone [`verify_joint_study_report.py`](../scripts/verify_joint_study_report.py)
passed against that sealed run, independently checking the 12 adapter manifests,
each saved tensor checksum, replay ledgers, all 288 TRAIN choice records,
1,680 EVAL decisions, deterministic random draws and actual propensities,
calibration diagnostics, and exact sign-flip/Holm statistics. Its separate
summary is included in the archived evidence.

The independent [`compare_joint_study_fresh_process.py`](../scripts/compare_joint_study_fresh_process.py)
passed: a fresh Python process reloaded the saved seed-163 LoRA plus head and
reproduced **four original EVAL decisions**, including token IDs, weight
digests, predicted qualities, raw and exploration-adjusted probabilities,
random draws and chosen actions. It requested no EVAL rewards.

The separate [`verify_saved_open_weight_joint_appraisal.py`](../scripts/verify_saved_open_weight_joint_appraisal.py)
also passed on the pinned model: a new process restored seed-163's **24**
completed TRAIN updates, scored four *new* unresolved held-out offers, sampled
one **new, reserved TRAIN** action with recorded propensity, obtained only its
chosen TRAIN outcome, performed another LoRA/head update, rejected replay,
and persisted version two with the update count **24 → 25**. The source
artifact and frozen pretrained backbone remained unchanged.

The repository test suite passed locally: **430 tests, three optional skips**.
The latest [GitHub Python Unit Tests](https://github.com/thebrazenbeard/pleasure/actions/runs/41375580081)
and [Workspace Startup Safety](https://github.com/thebrazenbeard/pleasure/actions/runs/41375580091)
workflows passed on the frozen scientific implementation plus independent
audit code.

## Interpretation and next experiment

This result resolves the **mechanical persistence** question in the affirmative:
the open-weight model's learnable appraisal/LoRA state really changed and
survived fresh-process reload and additional training. The causal-utility
question under this preregistered payoff distribution is **not supported**:
the signal did not create a reliable improvement over disabled, shuffled,
or head-only controls. It cannot establish that an AI system felt pleasure,
and it does not modify the hosted ChatGPT model.

The next protocol should be separately preregistered and use truly independent
payoff custody, larger TRAIN support or better-controlled action-conditioned
representations, and nontrivial calibrated preference discrimination before
another held-out utility test. **Do not retune v3 after observing its EVAL
results or re-label this null as a success.**

