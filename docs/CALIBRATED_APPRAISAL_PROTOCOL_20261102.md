# Predeclared calibrated-appraisal and isolated-evaluator study (2026-11-02 UTC)

**Status: protocol frozen before inspection of new v2 EVAL outcomes.** This is
an experiment in an open-weight language model's *separate* learned appraisal
head, not hosted ChatGPT weight editing, independently authenticated rewards,
subjective pleasure, or a claim about consciousness.

## Motivation and earlier study limitation

The five-seed previous study saved/reloaded trained linear heads but produced
zero action switches and essentially zero mean expected reward improvement.
The sigmoid heads often saturated with identical predicted reward for both
actions. This study changes the deterministic representation to a centered,
unit-norm **difference between the frozen model's action-conditioned hidden
states**, adds bounded-logit output, and requires fresh-process artifact reload.
This is exploratory redesign following the earlier observed null result.

An additional audit discovered that the earlier runner revealed each EVAL
arm's *chosen* rewards before later arms committed their decisions; across
arms, those could expose both payoffs prematurely, even though posthoc
counterfactual scores were delayed. That runner never learned from EVAL
receipts, so we do not automatically discard its descriptive results, but
the former strong wording about withholding was incorrect. The new protocol
requires a **worker-enforced global decision seal** before revealing any
EVAL reward, chosen or counterfactual. The legacy runner's sequence is also
corrected to commit all arms before resolving chosen EVAL feedback.

## Fixed design and parameters

- Immutable base model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`; safetensors digest
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
  CPU and local-only inference; exact pretrained weights digest checked.
- Seed set: **101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157**
  (12 independently generated paired schedules). Do not select/drop seeds
  based on outcomes. This is a new exploratory suite, not an independent
  external preregistration.
- For each seed: **24** sampled TRAIN choices from the new affine family;
  **6** nonmirrored EVAL offers from each of four families (24 per arm/seed).
  EVAL includes one genuinely unseen affine/within-family positive control
  and three *new* nonaffine families: two-feature parity/checkerboard,
  separated double-peak modes, and periodic coupled wave/ridges. These
  generators are different from the previously evaluated interaction,
  one-dimensional threshold, and radial basin rules. The worker exposes
  opaque family labels `sealed_eval_00_v2` (affine) and
  `sealed_eval_01_v2`–`sealed_eval_03_v2` (new OOD), without exposing the
  corresponding reward formula over its scoring interface.
- Like the preceding benchmark, the v2 synthetic issuer deterministically
  retries candidate pairs until the oracle absolute A/B payoff gap is above
  **0.02**. This is a source-visible, gap-filtered challenge distribution,
  **not** an unbiased sample of all possible generated tasks. Its conditioning
  may make decisions artificially easier; report this selection limitation
  and never generalize to arbitrary real-world outcomes.
- Training behavior is the frozen model A/B softmax mixed with uniform
  **epsilon = 0.4** exploration. Record the exact chosen action, raw and
  behavior probabilities, random draw, receipt evidence identity and input
  digest *before* returning chosen TRAIN reward. There is no EVAL feedback
  learning or adaptation.
- Calibrated head: initialized exactly zero; SGD chosen-only regression,
  **learning rate 0.1**, max one update per TRAIN offer, per-update bounded
  gradient/delta, frozen base, no replay and no reward inputs in hidden
  features. Use fixed **appraisal gain 12.0** for ON and −12.0 for inverted.
  The chosen-head L2 regularization coefficient is fixed at **0.02**.
  Deterministic feature schema must be bound into a versioned safetensors
  manifest and matched on independent import. This feature redesign is
  *not* retrospectively claimed to be preregistered for the older suite.
- Arms: `trained_on` (trained calibrated head imported from artifact),
  `trained_off` (same imported head, zero gain), `trained_inverted` (negative
  gain), `shuffled_on` (identically initialized head with deterministic
  deranged labels from other genuine TRAIN chosen receipts), and `frozen_on`
  (untrained head at positive gain). Do not pass synthetic labels as real
  evaluator receipts.
- All arms use the same EVAL offers and sampler draws (common random
  numbers). All EVAL tickets for **all** arms must be committed before the
  isolated subprocess will allow a single EVAL receipt. Only after every
  EVAL receipt is resolved is the whole evaluator sealed and both payoffs
  available for posthoc measurement. The source-local process boundary is
  procedural isolation, not a malicious-host threat boundary.

## Outcomes and interpretation fixed in advance

The primary measure is the seed-paired difference in posthoc **expected
stochastic policy quality** on the three new OOD families:

\[
V(x) = P_{\mathrm{behavior}}(A\mid x)q_A(x)
     + P_{\mathrm{behavior}}(B\mid x)q_B(x).
\]

Evaluate ON−OFF and ON−shuffled separately, then each OOD family and the
heldout in-family affine control. Report seed-level effects and descriptive
normal/bootstrap or exact paired sign-flip intervals/tests where meaningful;
seed, not episode, is the unit of inference. Secondary outcomes: realized
stochastic reward, ON/OFF sampled action switches, absolute shift in chosen
action probability, prediction A/B spread and calibration, signed-inversion
sanity, frozen-head versus disabled policy equality. Every predeclared run
must be reported including failures or aborted partial trials.

**Operational utility threshold:** consider this *positive exploratory
evidence* only if the mean primary OOD ON−OFF and ON−shuffled effects both
exceed **+0.005 absolute expected quality**, both corresponding seed-clustered
95% interval lower limits exceed zero, and each OOD family's mean ON−OFF
effect is positive. This does not assert statistical power, transfer to
real rewards, calibrated subjective pleasure, independent secrecy, or
causal identification beyond the tested source-local interventions. No
threshold or hyperparameter tuning after viewing these EVAL payoffs.

## Persistence and integrity

Export the trained head once per seed into a unique, previously empty
versioned safetensors directory, verified against the exact unchanged model
digest, action-token order, immutable checkpoint/tokenizer/snapshot digest,
explicit architecture/feature schema, and per-tensor hashes. Recreate a
fresh head from those files for EVAL. After the full run, a separate OS
interpreter must independently load the frozen model and head artifact,
verify SHA and reproduce selected fixed EVAL scores without carrying live
objects across processes. Do not claim persistence of optimizer momentum
(stateless SGD only), or continued fine-tuning across sessions without an
explicit restart-and-resume test.

Publication shall distinguish a *finished source implementation* from
an *actually executed multi-seed result* and shall preserve failed attempts.

## Offline reproduction

Use an exact, preverified local safetensors snapshot and separately provisioned
CPU PyTorch/Transformers/PEFT/safetensors. The experiment never downloads
weights itself. The artifact root must exist, be private, and lie outside
this source checkout. Both consent flags are required:

```sh
python -m scripts.run_open_weight_calibrated_appraisal_benchmark \
  --snapshot /pinned/SmolLM2-135M-Instruct \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --save-head-root /private/empty/calibrated-study \
  --allow-appraisal-training --allow-persistent-export
```

The successful output is a single, no-replace promoted `calibrated-v2-*`
directory with all twelve versioned head artifacts, `study-report.json`
and a `COMPLETE.sha256` report checksum. A failed run should leave no
apparently complete result. Afterward use the separate-process verifier:

```sh
python -m scripts.verify_saved_open_weight_calibrated_appraisal_head \
  --snapshot /pinned/SmolLM2-135M-Instruct \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --head-dir /private/empty/calibrated-study/calibrated-v2-.../seed-101-calibrated-head-v1 \
  --seed 101 --heldout-family sealed_eval_01_v2 \
  --expected-head-sha256 <sha-from-study-report> \
  --allow-appraisal-head-reload
```

Run `python -B -m unittest discover -s tests -q` for the full optional
dependency-aware suite. Independent verification must compare fixed EVAL
predictions in addition to merely matching the head bytes.

## Executed 12-seed result (completed after protocol freeze)

The fixed study completed against the exact local pretrained checkpoint.
All **12 seeds** (`101,103,107,109,113,127,131,137,139,149,151,157`)
completed **24 sampled chosen-action TRAIN episodes each**, for **288
TRAIN outcomes**, and each of five arms committed **24 independent EVAL
offers per seed** (6 in-family diagnostic; 18 primary OOD), or **288 EVAL
decisions per arm**. An isolated process withheld every EVAL receipt until
all five arms committed their decisions. All 12 calibrated heads were saved
to versioned safetensors and reimported in the training process before EVAL.
The exact pretrained tensor digest stayed unchanged:
`f3e37341bb38de7496edac8fdb3ea6b81d97f4b9bdd2f0d2a742511e7bff8f11`.

| Seed-paired primary OOD contrast | Observed mean | Descriptive normal 95% interval | Exact paired sign-flip p |
| --- | ---: | ---: | ---: |
| Genuine calibrated ON − same trained head OFF | `−0.000187645` | [`−0.000385495`, `+0.000010205`] | `0.0703125` |
| Genuine calibrated ON − shuffled-label head ON | `+0.000148370` | [`−0.000118123`, `+0.000414863`] | `0.3022461` |

The ON−OFF family effects were: within-affine diagnostic `+0.000095252`;
OOD checkerboard/parity `−0.000300041`; OOD double-peak `−0.000111024`;
OOD wave/ridge `−0.000151870`. The intervention shifted actual behavior
`P(A)` by a mean absolute `0.005655788`, yet caused **zero sampled action
switches** relative to OFF across all 288 matched EVAL offers. Prediction
spread is now nonzero (per-seed mean absolute A/B predicted-quality spread
varied roughly from `0.00044` to `0.01611`), but that distinction was not
reward-aligned on the new generators.

**The predeclared exploratory utility criterion failed.** Neither primary
effect cleared `+0.005`; ON−OFF was negative on each OOD family, and the
seed-clustered descriptive intervals included zero. The sign-flip values
are exploratory diagnostics, not multiplicity-adjusted confirmatory p-values.
The source-local synthetic outcome tasks were gap-filtered and source-visible;
they do not represent unbiased, externally blinded or real-world reward
transfer. These data establish repeatable head-state learning and a modest
probability-level intervention, **not improved future decisions** or felt
pleasure. Fresh-*process* verification and calibration are additional checks,
not part of the original in-process EVAL success flag.

### Independent reload and posthoc diagnostics

Two **new Python interpreter processes** separately reconstructed the exact
frozen checkpoint and imported calibrated artifacts for seeds `101` and
`157`. After matching six original EVAL offers in `sealed_eval_01_v2`
and `sealed_eval_03_v2`, respectively, a separate source-local comparator
verified the exact trained ON, trained OFF, inverted and frozen ON predictions
against the originally recorded run, including action-probability
renormalization for epsilon exploration. Across those comparisons the
maximum absolute discrepancy was **0.0** for predicted candidate qualities,
behavior `P(A)`, and original frozen baseline `P(A)`; head/manifest/tensor
and pretrained-base SHA digests matched. This establishes consistent
new-process tensor and prediction persistence, **not reward improvement**.

A read-only posthoc calibration analyzer independently reconstructed logged
expected policy value and reported pooled descriptive quality regression:

| Appraisal head | Candidate-quality MAE | Squared error against oracle q | Strict A/B preference-sign accuracy |
| --- | ---: | ---: | ---: |
| Genuine trained | `0.16584` | `0.04011` | `42.71%` |
| Shuffled-feedback trained | `0.16618` | `0.04025` | `46.18%` |
| Frozen zero head | `0.16538` | `0.03997` | `0%` (all predictions tied) |

The frozen head's preference-sign metric is zero by the analyzer's deliberate
strict-tie rule; it is not a comparable forced-choice accuracy. The learned
head actually had slightly **worse** average candidate-quality regression
error than the untrained constant predictor, underscoring its weak reward
alignment. The posthoc paired Student-*t* sensitivity 95% interval is
[`−0.000409821`, `+0.000034531`] for primary ON−OFF, and
[`−0.000150888`, `+0.000447629`] for ON−shuffled. These diagnostics neither
change the predeclared utility criterion nor justify tuning on the now-seen
v2 EVAL rules.
