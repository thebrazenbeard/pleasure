# Read-only custody verification of v4 candidate development evidence

**Verified subject:** TRAIN-only eight-seed v4 candidate-value development
report stored privately on Lappy. NOT an externally independent or blind
payoff service. No hosted or pretrained model weights were changed.

- Report: D:\VERA\experiments\pleasure-v4-candidate-train-dev-20261102-v2.json
- Exact original file SHA-256:
  48ef297ff8f415e9a31c6fde964ba83bd06cdf2ed15765473961ae62dd7e625f
- Benchmark source parent branch: research/pleasure-v4-wave1-controls-20261102
  (initial head f1d9eb576430faf6e78d51008dd70be1ecf32696)
- Evidence class: SOURCE_LOCAL_V4_CANDIDATE_TRAIN_DEVELOPMENT_NOT_CONFIRMATORY

## Independently recomputed

The standalone stdlib verifier does not import the development runner,
candidate head or its synthetic payoff function. It recomputes all seeded
candidate-feature draws, logged chosen actions, exact TRAIN outcome scores,
SHA-256 IDs, pre-feedback commitments, per-step **chosen-only** SGD
weights, shuffled derangements and donor evidence mappings, consumed
receipt ledgers, bounded exported state data and digests.

On the development diagnostics it regenerates candidate offers, raw
pre-exploration probabilities, fixed epsilon adjustment, posthoc expected
rewards, the all-arm decision SHA seal, mean values and contrasts.

It passed on all **8 seeds, 768 TRAIN chosen receipts, 512 paired
development diagnostic offers across five controls**. Recomputed effects:

- Genuine appraisal ON minus disabled/frozen: +0.03483877855583417
- Genuine appraisal ON minus shuffled: +0.03728545387055132
- Genuine ON mean expected reward: 0.5321823238683342
- Inverted appraisal mean expected reward: 0.46250476675666585

Six focused tests verify that the separate verifier accepts matching
JSON evidence and rejects tampered chosen reward, shuffled donor, saved
head weights, policy probabilities, aggregate summary, incomplete
diagnostic rows and altered global decision seal.

## Important limits

This is *independent verification logic* authored by the same assistant,
executed on the same authorized host, not an independent third-party
auditor. It demonstrates mathematical consistency with independently
reconstructed numeric traces, not authenticity of the publisher,
host trust, a third-party blinded payoff, or independent statistical
confirmation of generalization.

The v4 development suite and report were inspected during design.
Its source-visible affine reward is intentionally learnable by the
explicit candidate-value scorer. These development offers must never be
relabelled as untouched v4 EVAL. Another study needs a genuinely new,
preregistered, separately safeguarded payoff and prospective controls.

### Reproduce (no optional model packages)

    python -m scripts.verify_candidate_development_report \
      /absolute/path/to/pleasure-v4-candidate-train-dev-20261102-v2.json \
      --expected-sha256 \
      48ef297ff8f415e9a31c6fde964ba83bd06cdf2ed15765473961ae62dd7e625f

No source or reward artifact is modified by this command.
