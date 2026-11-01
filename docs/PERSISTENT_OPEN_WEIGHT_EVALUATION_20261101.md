# Persistent learned LoRA and internal appraisal study — November 1, 2026

**Evidence classification:** locally executed pretrained-model **TEST** evidence,
source-local unauthenticated evaluator; no hosted GPT access, provider
checkpoint attestation, external-world performance result, or phenomenology.

The experiment now provides a separately authorized, versioned LoRA export
and **fresh-process** loader, three matched outcome-learning arms over multiple
seeds, and a hidden-state-derived appraisal head with ON/OFF/inverted causal
interventions. A changed model tensor is not synonymous with better measured
reward, and a reward predictor is not evidence of felt pleasure.

## Frozen pretrained reference and artifact custody

The source checkpoint is official
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
`model.safetensors` SHA-256
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`
matched official public LFS metadata in the previous pilot. Local snapshot
manifest SHA-256 is
`579bb78314f53d7c02e6282410f220a39999409a8535c3cd2d51b077b590b077`.
See [the original executed model record](SMOLLM2_OPEN_WEIGHT_EXECUTION_20261101.md)
for byte inventory, methods and constraints.

The adapter writer persists only the **explicit named LoRA matrices** to
`adapter.safetensors` and a strict v1 `manifest.json`. The manifest binds
tensor names, shapes, dtypes, per-tensor SHA-256, aggregate trainable SHA,
frozen-backbone SHA, declared model/revision, original runtime ID, checkpoint
and tokenizer manifest digests, rank and alpha. It rejects unrecognized files,
unsafe serializers, traversal, filesystem links/junctions, incompatible fresh
models, oversized tensors, and unapproved overwrites. Import verifies the
bytes, validates all tensors before copying, and rolls trainables back on
failed application. The **fresh** runtime gets a distinct instance ID while
the manifest preserves the original run's declared ID. These local hashes
detect corruption; they do not certify publisher identity.

### Real separate-process persistence result

The new `--save-adapter-dir` and independent
`--allow-persistent-export` gate were exercised on the pinned CPU model:

| Measured item | Local observed value |
| --- | --- |
| Initial LoRA tensor SHA-256 | `66a078a3e11c4ea4d270cab1580f1ae2d841ec04fcbb436ccdf72f64688224b0` |
| Learned and exported LoRA SHA-256 | `1e76fe4cd4da8c52951cb2a5adc90bbd14371f2547bce9fa3cb5456f2fbecefa` |
| Post-training original-process tensor SHA-256 | Exact initial SHA |
| Saved `adapter.safetensors` SHA-256 | `822c4eed48d3925d806ae1939f6dc241aadd22650e658de1824b57ac13e05b9c` |
| Saved manifest SHA-256 | `6fd0d4cd0a00c2780b214932dff09c299728fbfdd8ec3a78201d93b2b4e6d083` |
| Fresh-process imported LoRA SHA-256 | Exact learned SHA |
| Fresh-process frozen-base SHA-256 | `921d0bf8b9e1b21573d94a866470840ae065ec9885321caec7b3bc6f9127cf84` (unchanged) |

A **separate Python process** loaded the original safetensors model, prepared
the same PEFT LoRA structure, and imported the saved tensors. On two fixed,
previously untouched EVAL offers, next-action `P(A)` changed as follows:

| Heldout offer | Before loading learned LoRA | After loading learned LoRA |
| --- | ---: | ---: |
| 1 | 0.8380245566 | 0.8381960988 |
| 2 | 0.8476911783 | 0.8478664160 |

**Result:** persistent, inference-coupled parameter change established for
this isolated open-weight model and CPU environment. The independent process
boundary was exercised locally; no external runtime or host attested it.
The artifact was stored in a temporary path outside the source checkout,
**not committed to GitHub**, deployed, or converted to a model-host service.

The training-run capture happens after read-only heldout evaluation, before
mandatory exact restoration. It first saves a **provisional** artifact within
a private staging directory. The CLI promotes it to the caller's final path
only if the frozen control, task/hash equality, and restoration checks all
pass. The artifact itself remains marked
`TRAINED_TENSORS_CAPTURED_NOT_STUDY_VALIDATED`: loading bytes is not external
validation. Failed controls do not publish a final artifact.

## Multi-seed outcome feedback and negative controls

The opt-in `run_open_weight_multiseed_benchmark` allocates a fresh pinned
model/LoRA initialization for each seed, and within it runs three sequential
matched arms with equal starting trainable/frozen SHA, identical TRAIN/EVAL
offers and token inputs:

1. **Frozen:** receipts collected; SGD skipped.
2. **Outcome-weighted:** actual committed chosen-action TRAIN quality drives SGD.
3. **Shuffled-feedback:** a derangement of **frozen-arm observed qualities**
   supplies synthetic teaching scalars; actual committed outcomes remain
   separate records with their original evidence IDs. Synthetic feedback
   receives different IDs and is never presented as genuine chosen reward.

TRAIN/EVAL have disjoint family labels, feature instances, episode IDs and
offers; EVAL outcomes are not used for weight updates. The hidden payoff
generator is nevertheless the **same source-visible linear rule** across
partitions. This is **unseen-instance generalization**, not a new task class
or independent-world evaluator. Only the primary side of each mirrored pair
counts as an independent heldout example; mirror side is diagnostic.

### Executed exploratory three-seed checkpoint

SmolLM2 CPU execution with seeds `11,23,37`, **2 TRAIN outcomes** and
**2 primary heldout offers per seed**, learning rate 0.1, LoRA rank 2/alpha 4:

| Seed | Frozen mean heldout quality | Outcome-weighted | Shuffled feedback | Both SGD arms changed tensors? |
| ---: | ---: | ---: | ---: | --- |
| 11 | 0.5300435486 | 0.5300435486 | 0.5300435486 | Yes |
| 23 | 0.4103772715 | 0.4103772715 | 0.4103772715 | Yes |
| 37 | 0.3940196169 | 0.3940196169 | 0.3940196169 | Yes |

Mean paired outcome-weighted minus frozen primary **quality difference = 0**;
outcome-weighted minus shuffled = **0**. The mean paired change in chosen
action probability relative to frozen is **−0.0035285254** (descriptive
normal 95% interval approximately −0.0087344 to +0.0016774). These intervals
are **not** a valid significance claim at three seeds. All three arms in all
three seeds restored their exact initial tensors. Different learning signals
produced different learned tensor hashes, but no observed categorical-action
quality improvement on these offers.

### Expanded pretrained-model run — five seeds

The same CLI was then exercised on **five seeds `11,23,37,41,53`** with
**4 genuine TRAIN outcomes and 4 independent-primary heldout offers per
seed** (20 primary heldout offers for each matched arm; 60 total arm-evaluated
primary offers). Every seed used a fresh pretrained model plus seeded LoRA
initialization, and every frozen/treatment/shuffled triplet matched starting
tensor digests and public task exposures. All 15 arms exactly restored their
respective original trainable SHA after evaluation.

Across all five seeds, each treatment and shuffled arm had the same primary
heldout realized quality as its frozen comparator. For example, at seed 53
all three measured `0.5151171433860436`. The **seed-paired mean
realized-quality difference remains 0.0**
against BOTH frozen and shuffled. The treatment-minus-frozen mean chosen
action-probability difference was **−0.0017988026**, with a descriptive
normal 95% interval **[−0.0060955777,+0.0024979725]**. This is still
an exploratory **five-seed** lab, not evidence of statistical equivalence,
general usefulness or unbiased reward learning. Lack of measured primary
quality gain at this small budget is itself a falsifiable negative result.

## Model-hidden-state appraisal and causal ablations

`TorchHiddenAppraisal` attaches a **separately trainable**, bounded quality
predictor to action-conditioned final hidden states from the pretrained LM.
Only committed chosen TRAIN quality updates this head; LM parameters are
queried read-only. At EVAL, the head predicts quality for A and B and can
modulate their token logits. The same head, model and exact prompt are scored
with three interventions: gain `0` (**off**), positive gain (**on**), and
negative gain (**inverted**). Outcome resolution is performed only after the
chosen action, using separate matched one-use arenas per intervention.

In a tiny Torch fixture, enabling versus inverting the learned head changed
a later chosen action, proving the source mechanism is capable of a causal
choice intervention. In the **actual pinned pretrained SmolLM2 pilot**, seeds
`11,23,37`, 2 TRAIN and 2 independent-primary heldout tasks per seed,
gain `16`, learning rate `0.4`: all heads changed SHA after two SGD steps,
but ON/OFF/inverted produced **zero heldout action switches and zero quality
difference**. The learned head's predictive calibration remains weak and
was evaluated only on this narrow synthetic feature distribution.

An expanded **three-seed, 4-TRAIN/4-primary-heldout** run also completed
with explicit frozen-pretrained-LM tensor SHA verification and exact
in-process appraisal-head export/import roundtrip. Again, the tested head
learning changed the head bytes but **did not improve primary heldout reward
or alter the selected A/B actions** under the gain-16 ablation. A correct
head-state roundtrip is not a separately executed persistent-head service:
cross-process persistence was established for the **LoRA adapter**, not yet
for the separately attached appraisal head.

This means **the internal appraisal causal pathway is implemented and
falsifiable, but not yet demonstrated useful on the tested pretrained model**.
The head is a new model-connected module, not a native emotion or a trained
part of SmolLM2's frozen backbone. No subjective feeling, learned valence
experience, or hosted-model change is inferred.

## Reproduction (all operations opt-in, offline)

Provision the pinned local safetensors checkpoint and a compatible Torch,
Transformers, PEFT, and safetensors environment first. These CLIs do not
download models. `--snapshot` and artifact paths must be absolute; save
learned weights **outside the source repository**.

```sh
python -m scripts.run_open_weight_local_trial \
  --snapshot /approved/pinned/model/snapshot \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --seed 12345 --training-pairs 2 --heldout-pairs 2 \
  --learning-rate 0.1 --lora-targets q_proj v_proj \
  --allow-in-place-training \
  --save-adapter-dir /approved/new/model-artifacts/pleasure-lora-seed12345 \
  --allow-persistent-export

# Run in a NEW Python process with the original frozen snapshot:
python -m scripts.verify_saved_open_weight_adapter \
  --snapshot /approved/pinned/model/snapshot \
  --adapter-dir /approved/new/model-artifacts/pleasure-lora-seed12345 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --runtime-instance-id independent-reload-002 --seed 12345 \
  --lora-targets q_proj v_proj --allow-in-place-reload

python -m scripts.run_open_weight_multiseed_benchmark \
  --snapshot /approved/pinned/model/snapshot \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --seeds 11 23 37 --training-pairs 2 --heldout-pairs 2 \
  --learning-rate 0.1 --allow-in-place-training

python -m scripts.run_open_weight_appraisal_pilot \
  --snapshot /approved/pinned/model/snapshot \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --seeds 11 23 37 --training-pairs 2 --heldout-pairs 2 \
  --learning-rate 0.4 --appraisal-gain 16 \
  --allow-appraisal-training
```

The test doubles and internal control fixtures do not elevate the observed
SmolLM2 result into external performance, provider training access, subjective
experience, or a medically or ethically defined hedonic state.

## Next falsifiers

- Increase independent training seeds, TRAIN outcomes and primary heldout
  offers; preregister partitions and report every negative/failed run.
- Add truly alternate payoff generators/tasks and independently keyed
  evaluator custody. Current environment can inspect the source scoring rule.
- Add a stochastic policy with logged propensities; argmax feedback is not
  unbiased REINFORCE, and negative rewards can still train only selected A/B.
- Compare constant-0.5, sign-flipped, and shuffled appraisal-head learning;
  include calibration and causal action-switch metrics under fixed checkpoints.
- Qualify owner-hosted import/export with authenticated custody before any
  separate model-serving integration. The current artifact is for offline
  research only.
