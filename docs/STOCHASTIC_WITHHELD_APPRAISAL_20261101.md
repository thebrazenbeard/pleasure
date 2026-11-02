# Stochastic action exploration, withheld generators, and persisted appraisal

**Evidence classification:** locally executed pretrained/open-weight-model experiment,
source-local evaluator and author-controlled synthetic generators; no
independent evaluator custody, hosted GPT weight modification, or phenomenology.

## Falsifiable questions

1. **Persistence:** can a trained, model-hidden-state-connected appraisal head
   be stored in versioned safetensors and imported into an independently
   initialized model process, preserving exact tensor SHA and fixed EVAL scores?
2. **Causal action influence:** does the saved head alter future action
   probabilities and/or stochastic decisions under an ON vs OFF intervention?
3. **Heldout utility:** does that change improve expected quality on payoff
   families mathematically different from the TRAIN rule, compared with
   disabled, untrained and synthetically shuffled-label controls?

These are separate hypotheses. Byte persistence and probability changes alone
do **not** establish utility, real-world transfer, or felt pleasure.

## Evaluation protocol fixed for the reported run

The pretrained base remains frozen; only a separately registered float32
linear/sigmoid head on the final hidden activation after each action candidate
is trained. The training task family is an affine two-feature payoff. EVAL
families instead use (1) a multiplicative interaction, (2) a discontinuous
threshold, and (3) a nonmonotonic interior-optimum basin. Their code is
source-visible: *withheld* means absent from the learner-facing task/reward
interface and disjoint from TRAIN, **not** cryptographic secrecy.

Every TRAIN offer is scored by the same frozen base model. Its action softmax
is mixed with a uniform exploration floor at rate `epsilon`:

\[
P_{\mathrm{behavior}}(a\mid x)
  =(1-\epsilon)P_{\mathrm{base}}(a\mid x)+\frac{\epsilon}{2}.
\]

The independent seeded sampler records both original action probabilities,
both *actual behavior propensities*, its uniform random draw, selected action,
episode hash, input-token digest, decision digest, seed and draw index **before**
any reward is resolved. That sampled action, not a posthoc argmax, is committed
to the one-use evaluator and bound to the corresponding hidden candidate state.
Only the committed action's quality becomes a training label.

The genuine and shuffled heads start with identical zero tensors and see the
same ordered TRAIN offers, sampled actions and input tokens. Genuine labels
are chosen-action receipts. Shuffled labels are a deterministic derangement of
*other already committed chosen receipts*, explicitly marked with synthetic
teaching IDs and their real donor evidence IDs; no synthetic value is passed
off as a genuine evaluator receipt. The frozen head receives no SGD. All heads
retain exact pretrained-base tensor digest checks.

The five predeclared EVAL arms are:

| Arm | Head state | Inference appraisal intervention |
| --- | --- | --- |
| `trained_on` | versioned saved-and-reimported genuine head | positive gain |
| `trained_off` | *same reimported genuine head* | zero gain |
| `trained_inverted` | same genuine head | negative gain |
| `shuffled_on` | deranged-feedback trained head | positive gain |
| `frozen_on` | untouched initial head | positive gain |

Each arm evaluates the same fixed EVAL offers, in the same order, with the
same seeded uniform draws (common random numbers). Stochastic actions are
committed and their genuinely chosen quality returned separately per arm.
**All five arms must finish all EVAL decisions before any posthoc counterfactual
is revealed.** Only then does the evaluator reveal both payoffs per offer so
we can compute the noise-reduced expected behavior-policy quality:

\[
V(x)=P_{\mathrm{behavior}}(A\mid x)q(x,A)
     +P_{\mathrm{behavior}}(B\mid x)q(x,B).
\]

The main contrasts average these values by seed, then compare `trained_on`
against `trained_off`, `shuffled_on`, and `frozen_on`. Realized stochastic
reward, head ON/OFF action switches and mean absolute probability shifts are
reported separately. The three generator families contribute equal numbers
of primary heldout offers; mirrored pairs are omitted to avoid duplicated
independent examples. Normal 95% intervals are descriptive only at this
small number of seeds. The source-defined exploratory gate requires positive
lower intervals versus both OFF and shuffled, plus mean positive ON–OFF gain
on *each* generator. It is **not** a preregistered confirmatory statistical
test; at five seeds an exact two-sided sign-flip test cannot achieve
`p < 0.05` even in the most extreme direction.

## Artifact and trust boundary

`open_weight_appraisal_persistence.py` saves only the two float32 head tensors
to `head.safetensors` and a JSON `manifest.json`. The v1 manifest binds
model/revision, original runtime, checkpoint/tokenizer/snapshot digests,
frozen LM tensor digest, action-token order, hidden dimension, feature schema,
named tensor SHA/type/shape, head digest, and version. Export/import require
separate explicit flags; the loader rejects links, malformed or extra files,
identity mismatch, corrupt content, unsafe serialization, pending TRAIN
decisions, and overbudget tensors. A late failed import restores the original
head. An independent interpreter must reload the pretrained base, re-create
the head and check fixed EVAL predictions and disabled-head control as well as
tensor SHA. Local hashes do not authenticate the publisher or host.

No model files, runtime-trained head artifacts, or evaluator secrets are
committed to the source repository by the study CLI.

## Reproduce without network calls

Provision a pinned local safetensors checkpoint and compatible CPU PyTorch,
Transformers and safetensors dependencies. Use a **new, empty artifact root
outside the source checkout**, and preserve its contents as run evidence.

```sh
python -m scripts.run_open_weight_stochastic_appraisal_benchmark \
  --snapshot /pinned/local/SmolLM2-135M-Instruct \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --save-head-root /private/new/appraisal-run-artifacts \
  --seeds 11 23 37 41 53 \
  --training-pairs 8 --heldout-pairs-per-generator 4 \
  --exploration-rate 0.4 --learning-rate 0.4 --appraisal-gain 16.0 \
  --allow-appraisal-training --allow-persistent-export

# Run AFTER the first command in a NEW Python process:
python -m scripts.verify_saved_open_weight_appraisal_head \
  --snapshot /pinned/local/SmolLM2-135M-Instruct \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --head-dir /private/new/appraisal-run-artifacts/seed-11-trained-head-v1 \
  --runtime-instance-id new-fresh-appraisal-reader \
  --seed 11 --heldout-pairs 4 --heldout-family heldout_eval_interaction \
  --appraisal-gain 16.0 --allow-appraisal-head-reload
```

Tests (optional Torch/safetensors tests skip when not installed):

```sh
python -B -m unittest discover -s tests -v
```

## Executed results

The complete corrected CPU run used the official
`HuggingFaceTB/SmolLM2-135M-Instruct` revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac` from the previously
SHA-checked local snapshot. The fixed source-visible protocol was **five
seeds `11,23,37,41,53`**, 8 sampled TRAIN choices/seed, and 4 primary
side-zero EVAL offers per generator/seed. Thus there were **40 genuinely
sampled TRAIN choices** and **60 independent-primary EVAL offers per arm**
(5 matched arms), all with episode-linked action probabilities and logged
sampling propensities. The final artifacts were promoted only after all
seeds and all control/readback checks succeeded; two aborted earlier trial
runs with interim artifacts were explicitly excluded from analysis.

| Seed-paired contrast, across the equally weighted withheld suite | Measured mean |
| --- | ---: |
| Genuine trained ON − same trained head OFF, expected policy quality | `−0.0000005811314` |
| Genuine trained ON − shuffled feedback ON, expected policy quality | `−0.0000005784857` |
| Genuine trained ON − untrained frozen head ON, expected policy quality | `−0.0000005811314` |
| Genuine trained ON − same trained head OFF, realized sampled reward | `0.0` |

The seed-paired mean ON−OFF expected-quality descriptive normal 95% interval
was approximately **[`−0.00000172116`, `+0.000000558898`]**. All **three**
withheld-generator families had a *negative* average ON−OFF expected-quality
contrast: interaction `−0.000000287664`, threshold `−0.000001043263`, and
basin `−0.000000412468`. There were **0/60 sampled action switches** between
genuine head ON and OFF, and mean absolute difference in their actual
behavior-policy `P(A)` was `0.00000873152`. A small probability change is
causal influence on logits, not a meaningful future-decision improvement.

All five genuine heads changed from the exact identical initial digest
`b72e3f151856a860362cf2000fee8ed66343b6e8d66c9da8e6b028c3715fdc9d`,
all five shuffled controls also changed, and all saved genuine heads were
exactly reimported into new head instances for evaluation. The unchanged
pretrained-base parameter/buffer digest was
`f3e37341bb38de7496edac8fdb3ea6b81d97f4b9bdd2f0d2a742511e7bff8f11`.
For seed 11 specifically, the genuine trained head was
`7827d930c2953350ef2659bdb420c18ce7882a35ebba137d38089b1da4a09c25`,
its `head.safetensors` digest was
`2bacd15320280b65da939f959f115a9b0b6005a55d310066fc2c15d379f61d08`,
and its versioned manifest digest was
`aae593bde37f1dc69b3e417633efb8f4bd356d7933bf89e156039b35175ae2d6`.

**Conclusion of this source-local run:** persistence of learned head tensors
was established in process, the ON intervention produced only very small
probability differences and no sampled action switches, and **quality gain
was not demonstrated** against either a disabled or shuffled trained head.
This is evidence *against* treating the current representation and training
budget as a useful future-decision appraisal mechanism. It is not proof of
inefficacy at larger scales, proof of no possible model experience, or proof
of phenomenology. The separate fresh-process verification is recorded below.

### Fresh-process replay

After the full benchmark exited, a **new Python interpreter** independently
reloaded the pinned pretrained checkpoint, constructed a fresh zeroed appraisal
head, verified its disabled-head behavior, and imported seed 11's saved v1
artifact. The origin runtime label was `pinned-appraisal-stochastic-run-003`;
the fresh-process runtime label was `independent-appraisal-reload-001`.
The imported head SHA exactly matched
`7827d930c2953350ef2659bdb420c18ce7882a35ebba137d38089b1da4a09c25`,
and the frozen-pretrained-model SHA was exactly the original
`f3e37341bb38de7496edac8fdb3ea6b81d97f4b9bdd2f0d2a742511e7bff8f11`.
Four fixed interaction-family EVAL episodes and input-token digests also
matched the benchmark records. The fresh-process predictions and raw
softmax probabilities matched benchmark inference after accounting for the
sampler's explicitly documented normalization of floating-point `P(A),P(B)`
to sum exactly to one; comparison was **bit-for-bit equal after that
normalization** on all four prompts.

For example, on fixed heldout offer 1, the untrained-head raw `P(A)` was
`0.8349260687828064`; after importing the trained head, the same process
measured `0.8349259495735168`. The disabled-head probabilities stayed
unchanged. The imported head predicted a saturated quality of `1.0` for
both actions on all four verifier prompts; this is weak discrimination, not
an independently calibrated useful reward estimate. The separate-process
test therefore **establishes persistent head state and reproducible model-
coupled inference**, not better decisions or subjective feeling.

The complete per-episode probability/propensity/reward evidence, the
independent-process verifier JSON, and the five versioned head artifacts
are retained together as the separately saved research archive. Its
generator identities, seeds, artifact digests and outcome limitations can
be audited independently of this summary.
