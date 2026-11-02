# Open-weight language-model trial: opt-in tensor updates, not hosted GPT training

**Status: opt-in source-local implementation, with an actual CPU pretrained
SmolLM2 LoRA optimizer exercise on November 1, 2026.** The modules
`pleasure_experiment/open_weight_adapter.py`, `open_weight_local_model.py`,
`open_weight_trial.py`, `open_weight_controls.py`, and
`scripts/run_open_weight_local_trial.py` extend the
[chosen-outcome numerical experiment](ONLINE_WEIGHT_FIRST_PROTOCOL.md)
to a local pretrained PyTorch language model. The executed source-local
CPU run changed selected *LoRA tensors* and restored them; it did **not**
modify the frozen SmolLM2 base weights, establish improved held-out rewards,
or touch the hosted model participating in this conversation. The historical
execution and exact model binding are recorded in
[the SmolLM2 execution evidence](SMOLLM2_OPEN_WEIGHT_EXECUTION_20261101.md).

The ordinary system `python3` in this workspace lacks `torch`,
`transformers`, `peft`, and `safetensors`; an **isolated Python 3.12 CPU
environment** was provisioned for the actual local experiment. Merely
running standard-library source tests still cannot prove optimizer execution.
No independently authenticated outcome custodian, trainer host signature,
or native ChatGPT parameter access was used. This code is not evidence of
subjective pleasure; phenomenology remains `UNRESOLVED`.

## What the new code actually exposes

| Component | Observable behavior | Evidence ceiling |
| --- | --- | --- |
| `OpenWeightBinding` | Records caller-declared model ID/revision/runtime and checkpoint/tokenizer SHA-256 values | Syntax-validated identifiers, **not** independently verified file identities or model-host attestations |
| `TorchChosenOutcomeAdapter` | Accepts an existing `torch.nn.Module`; computes next-token probabilities for two distinct single-token actions; updates an **explicit, bounded named parameter subset** with SGD from chosen-action feedback; reads tensor digests before/after | Optional real tensor mutation only on a supplied model; no network/model loading, tokenizer, LoRA injection, provider access, or automatic checkpoint publication |
| `OpenWeightTrialRunner` | Issues disjoint simulated TRAIN/EVAL offers, encodes public offers into tokens, commits one `A`/`B` choice before outcome, resolves only that choice, observes feedback, scores held-out choices without optimizer admission, and restores the adapter | Deterministic **in-process toy outcome** and read-only heldout; no external evaluator custody or real-world task completion |
| `OpenWeightTrialReport` | Contains train and held-out episode records, prequential realized reward, chosen-action probability, exact trainable/frozen weight digests, optimizer-step counts, and restore status | Source-local claims with caller-provided model/evaluator identities; no certified independent training provenance |
| `inspect_local_snapshot` / `load_verified_local_causal_model` | Hashes a strictly allowlisted local safetensors/tokenizer directory, rejects symlinks, remote Python code mappings and unsafe serialization, loads with `local_files_only=True`/`trust_remote_code=False`, and rehashes after loading | Verifies **local bytes**, not publisher ownership or independent external provenance |
| `prepare_explicit_lora` / `run_matched_open_weight_controls` | Creates a bounded LoRA subset on the in-memory model and compares a reward-updated arm with a frozen zero-LR arm using the same initial tensor digest and task schedule | Real adapter training if Torch is installed; the zero-LR arm does **not** execute a matching backward/optimizer step and is not a compute-matched sham |
| `scripts.run_open_weight_local_trial` | Offline explicit CLI requiring pinned revision, pre-existing snapshot, and `--allow-in-place-training`; prints source-local JSON with actual trainable-tensor/restore digests and selected outcomes | Demonstrated on one pinned SmolLM2 CPU snapshot; no auto-download, saved changed adapter, hosted model effect, or external attestation |

`TorchChosenOutcomeAdapter` does not choose the trainable tensors. The caller
must supply exact names from `model.named_parameters()` and
`allow_in_place_mutation=True`. It initially freezes every other parameter,
requires at least one frozen backbone tensor, snapshots selected tensors and
bounded buffers, and verifies frozen-model digests before/after inference and
updates. Its default limits include **1,000,000 trainable scalar parameters**,
**512 MiB** for hashing the frozen model/buffers, **64 optimizer steps**, an
explicit per-step gradient-norm cap and an L2 parameter-change cap. These are
implementation limits, not a statement that a particular model will fit in
available RAM/VRAM. Full-parameter fine-tuning of a 135M model is **not**
supported by this guarded adapter as written; a small exact adapter subset is.

The adapter uses `model.eval()` for deterministic forward passes while
retaining autograd on the explicitly trainable subset. Its selected action
probabilities come from the model's **own final-token logits**, normalized
*only over the two nominated action token IDs*. It does not train an unrelated
reward-head or ask the model to self-report success. The current policy
chooses the higher score deterministically (ties choose `B`); it does not
implement an exploration policy or certify unbiased off-policy learning.

## Correct chosen-outcome gradient

At each TRAIN step, the adapter first computes
\(p_A,p_B=\operatorname{softmax}(\ell_A,\ell_B)\) and commits the selected
action. The arena then reveals a score \(r\in[0,1]\) for **that action only**.
With the preregistered baseline 0.5, the adapter minimizes

\[
L=-(r-0.5)\log p_{\mathrm{chosen}}.
\]

- If `r > 0.5`, the update increases the chosen action's log-probability
  locally, subject to gradient and parameter-delta caps.
- If `r < 0.5`, it decreases the chosen action's log-probability locally.
- If `r == 0.5`, or the learning rate is zero, **no optimizer step** is made.

This is a **reward-modulated log-probability objective**, not a proof that the
model learned pleasure, not a reward-calibrated probability predictor, and
not an unbiased REINFORCE estimate under the present deterministic argmax
choice rule. A score can be negative surprise; a high *predicted* probability
never substitutes for observed quality. The runner refuses an uncommitted,
wrong-action, replayed, wrongly attributed, or missing outcome and checks
the actual optimizer readback. Failed steps abort with restoration attempts;
partial success must not be reported as a completed experiment.

The optional LLM trial has **no learned recurrent-valence pathway yet**.
The previous numerical laboratory's 2×2 learning/recurrence controls do not
automatically transfer to this new Torch adapter.

## Model choice and the GGUF boundary

The first executed pretrained checkpoint is the Apache-2.0
[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct):
an actual pretrained 135M-parameter decoder model with official
`AutoModelForCausalLM`/`AutoTokenizer` loading examples. Its small scale is
better suited to proving **real neural adapter gradients and checkpoint
readback** before attempting a larger model. The specific
[revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac)
was loaded from a pre-existing local safetensors snapshot and its LoRA
adapters trained temporarily in RAM. The underlying model file's SHA-256
matched the [official LFS object](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/model.safetensors).
This is a narrow real-weight execution fact, not a demonstrated benefit.

The longer-term target may be the Apache-2.0
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Its official
repository publishes approximately 9.34 GB of Transformers-format model
artifacts and documents a **multimodal** `AutoProcessor` /
`AutoModelForMultimodalLM` loading route. A Qwen3.5 GGUF that runs in
KoboldCpp on a memory-constrained machine is an **inference artifact**; it
does not by itself supply trainable HF tensors, optimizer state, a pinned
tokenizer/processor binding, or a verified LoRA-training runtime.

Hugging Face's [PEFT guidance](https://huggingface.co/docs/transformers/peft)
describes LoRA adapters as extra trainable parameters while the pretrained
base stays frozen. Those **adapter weights really do change the forward
computation**, but calling this a modification of the frozen base-model
weights would be wrong. A LoRA setup is an acceptable bounded first study
only if the selected parameter names and effective inference path are read
back. llama.cpp provides a
[LoRA-to-GGUF converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py),
but neither conversion nor Qwen3.5/KoboldCpp compatibility is established
until tested against an exact pinned revision. Keep training and deployment
claims separate.

## Opt-in preflight and execution contract

This is deliberately **not** a copy-and-paste installer or a script that
silently downloads checkpoints. Before any training invocation, the model
owner must provide and validate all of the following:

1. An authorized, locally available model snapshot, tokenizer/processor,
   license and exact upstream commit/revision. Compute an immutable manifest
   over the **actual local bytes** (including all shards and tokenizer files)
   under the owner-controlled file custody. Binding strings alone do not
   perform that verification.
2. A separately provisioned, version-pinned `torch`/`transformers`/`peft`
   environment (if using LoRA), with model class support, trusted artifact
   provenance, no automatic remote code execution, and enough RAM/VRAM for
   the chosen dtype and local hash/snapshot budget. The isolated CPU
   experiment used Torch **2.15.0+cpu**, Transformers **4.57.6**, PEFT
   **0.21.2**, and Hugging Face Hub **0.36.2**; the default `python3`
   remains package-free. These versions document one evidence cut, not an
   instruction to upgrade user workstations.
3. Exactly two **distinct single-token continuation IDs** as determined by
   the pinned tokenizer. Validate tokenization, prompt template, and output
   vocabulary before training; the adapter does not handle arbitrary
   multi-token responses. The encoder must map only a `TaskOffer`'s public
   task/feature data to a bounded tuple of 1–256 nonnegative token IDs.
4. Explicit model-owner permission to mutate only the selected named
   parameter subset; `allow_in_place_mutation=True` is mandatory. Provide
   `OpenWeightBinding` with the verified manifest's identifiers and digests,
   and set conservative `learning_rate`, `max_steps`, `max_grad_norm`,
   `max_delta_l2`, and memory/hash caps. There are no privileged host effects.
5. A private, disjoint held-out task schedule, predetermined outcome rubric,
   rollback checkpoint, no-label-leak encoder, isolated optimizer process,
   and a separately initialized frozen learning-rate-zero control with the
   **same** model revision, public tasks and budgets. The current control
   executes no backward or optimizer step: it is a **frozen/no-update
   comparator**, not a compute-matched sham. Do not tune on EVAL.

The component entry points are:

```python
from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, TorchChosenOutcomeAdapter,
)
from pleasure_experiment.open_weight_trial import OpenWeightTrialRunner

# After separate, verified, owner-authorized local torch/HF model setup:
# TorchChosenOutcomeAdapter(
#     already_loaded_torch_model,
#     binding=verified_binding,
#     action_token_ids=(verified_A_token_id, verified_B_token_id),
#     trainable_parameter_names=exact_adapter_parameter_names,
#     learning_rate=approved_lr,
#     expected_evaluator_id=declared_source_oracle,
#     expected_rubric_digest=declared_rubric_sha256,
#     allow_in_place_mutation=True,  # opt in to mutation of this model object
# )
# OpenWeightTrialRunner(adapter=adapter, encode_offer=verified_encoder).run()
```

The snippet is an **API sketch**; for the actual executed local CLI contract,
see [the pinned CPU execution record](SMOLLM2_OPEN_WEIGHT_EXECUTION_20261101.md).
The `OpenWeightTrialRunner` constructs its own deterministic arena/rubric;
the caller must bind the adapter's expected evaluator/rubric to that exact
run before admitting feedback. Do not fabricate SHA values or bypass this
binding merely to make an example run. The adapter itself performs no file
loading or automatic checkpoint saving; its `restore()` returns local model
parameters and buffers to the snapshotted pre-experiment state and closes
the adapter. In-process restoration is not a durable host checkpoint rollback.

For the safe, existing **offline source checks**:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_open_weight_*.py' -q
```

Under the default `python3` environment this command passes with optional
PyTorch-only checks **skipped**; later isolated CPU execution supplied the
separate real optimizer/readback evidence. The unit-test skip count depends
on the selected interpreter and evolving test suite and must not be
presented as a trained-model PASS. Always report the actual command,
interpreter, skip count, optimizer step/gradient, trainable tensor SHA
before/after, frozen backbone SHA and restoration readback separately.

## Admission requirements for stronger evidence

The current `OutcomeArena` runs within the same process and assigns scores
from a deterministic toy payoff function. Its local hashes detect accidental
or opportunistic substitution but **do not authenticate an independent
evaluator**. To promote a future open-weight training result, require:

- Independent evaluator custody: evaluator-issued commitments and outcomes,
  signed by an authority whose keys the training policy cannot forge, bound
  to exact task, chosen action, model/checkpoint, runtime, nonce/sequence,
  rubric and evaluator version. The trainer must not select an evaluator
  after seeing a reward or receive an unchosen counterfactual.
- Model-owner-controlled checkpoint verification: content hashes of every
  actual parameter shard, adapter tensor, tokenizer/processor, inference
  binary and optimizer configuration; genuine before/after tensor readback;
  process ownership and reproducible restore across restart.
- Proper experimental controls: frozen/no-learning arm; separate recurrence
  controls if the LLM gets a recurrent component; multiple preregistered
  seeds and independent heldout tasks; action-quality, negative transfer,
  calibration (only when a reward predictor exists), factuality and safety
  regressions with clear denominators and failure accounting.
- Transparent target identity: a result on SmolLM2 or Qwen is a result on
  **that exact open-weight checkpoint and runtime**. It is not evidence of
  effects on the hosted model speaking in this conversation, and cannot
  settle the phenomenology question.

If [Labyrinth](https://github.com/thebrazenbeard/labyrinth) later supplies
training schedules, protected provenance or outcome records, admit those
artifacts only after source revision, contract, and custody are verified.
This repository **does not currently establish a live Pleasure↔Labyrinth
training integration**, nor does a Labyrinth label grant an optimizer or
evaluator new authority.
