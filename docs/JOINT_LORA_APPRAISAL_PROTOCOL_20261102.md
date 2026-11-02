# Joint LoRA representation and appraisal: frozen v3 protocol

**Protocol version:** `pleasure-joint-lora-appraisal-v3-20261102`  
**Frozen before any v3 heldout reward inspection:** 2026-11-02 UTC  
**Evidence class:** source-local, author-controlled synthetic reward research;
not an external blind experiment, modification of hosted ChatGPT, evidence
of felt pleasure, or a claim about consciousness.

## Why this experiment exists

The preceding PR #23 study reliably persisted a separate calibrated
model-hidden-state head but **did not establish improved heldout quality**.
The v2 generator rules and their reward results have now been inspected;
they are development evidence, **not** eligible as untouched confirmation
tasks for this study. This v3 protocol introduces a new worker-owned source-
local generator family and an authorized trainable LoRA representation.

The falsifiable hypothesis is that **chosen-only stochastic feedback can
jointly train a small LoRA representation and a bounded appraisal head**
such that the head remains causally useful when reloaded into a fresh model
process. Outcome utility must improve on new, reward-structured task
instantiations against both a disabled head and a wrong-feedback control.
Saving changed tensors by itself only proves persistence, not utility.

This experiment deliberately distinguishes three different findings:

1. **State persistence:** exact safe LoRA/head import, independent prediction
   reproduction, and additional authorized TRAIN update after process restart.
2. **Mechanistic appraisal influence:** positive, zero and sign-inverted head
   gain on identical learned LoRA weights, with common EVAL offers/draws.
3. **Learned utility:** strictly positive seed-paired expected reward effects
   on a genuinely new within-structure payoff distribution and against
   shuffled supervision. Transfer to unrelated OOD functions is a **separate
   secondary stress test**, not assumed from finite TRAIN observations.

## Immutable model and training recipe

| Item | Fixed selection |
| --- | --- |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Immutable revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Base weights safetensors SHA-256 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| Runtime | Pinned, local-only CPU PyTorch/Transformers/PEFT; no paid compute or implicit downloads |
| LoRA architecture | Rank `2`, alpha `4`, only `q_proj` and `v_proj`, no dropout |
| Joint objective | Chosen-action quality regression MSE, separate bounded appraisal head with L2 coefficient `0.02` |
| Head representation | Opposing unit-L2-normalized frozen/LoRA LM final hidden-state action differences; bounded `2*tanh(logit/2)` then sigmoid |
| Head initialization | Tiny deterministic nonzero `weight[i]=0.01*sin((i+1)*1.61803398875)/sqrt(hidden_size)`, zero bias |
| Optimizer | Stateless SGD learning rate `0.05`; zero momentum or accumulated adaptive state |
| Update caps | Global gradient L2 `1.0`, per-step parameter delta L2 at most `1.0`; total training budget at most `48` distinct chosen TRAIN receipts per head/version lineage |
| Training consent | Separate explicit flags for mutation/export/import; exact frozen-base SHA required before and after every trial |
| Action exploration | Frozen-reference LM A/B softmax mixed with uniform epsilon `0.4` |
| Appraisal intervention | Fixed gain `+12.0`, `0.0`, and `−12.0` on the same learned weights |

Only the LoRA matrices and the appraisal head receive gradients. Neither
ordinary pretrained backbone parameters nor tokenizer files may change.
All LoRA/head instances in different controls must begin at **exactly the
same tensor digests**, verified by readback; merely seeding a global RNG is
insufficient. All TRAIN actions and chosen receipts are first committed under
the one frozen-reference behavior policy. Each independent learner then
computes both candidate hidden states from the offer and its *current* weights
during offline sequential TRAIN replay; its forward pass never consumes the
receipt's reward. This separation is chronological at the learner-update level,
not a claim that the learner's features were computed before the evaluator
returned receipts to the orchestrator. There is no EVAL weight update, EVAL reward
replay into training, unbounded optimizer, or unsafe pickle/`torch.save` deserialization.

The CPU-only smoke measurement before freezing found approximately
`115,200` LoRA trainable parameters, `1.36 s` for one pinned-model update
at two CPU threads, and approximately `1.12 GiB` peak RSS. The full study
may take **15–30 minutes** including multi-arm evaluation and persistence.
If the predefined resource envelope is exceeded, report an incomplete run
and preserve failure details; do not selectively omit seeds or reduce the
trial budget after inspecting results.

## Fresh outcome distribution and predeclared schedule

**Exact twelve independent schedule seeds:**

`163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227`.

For every seed, run exactly **8 side-zero TRAIN offers from each of three
new latent task families** (24 recorded chosen-action outcomes per seed):

- `train_latent_00_v3`
- `train_latent_01_v3`
- `train_latent_02_v3`

All TRAIN functions are members of a common low-order, compositional
two-feature payoff class, with positive dependence on `x` and negative
dependence on `y`, but differing coefficients. The learner sees task
descriptions and chosen quality receipts, **not** coefficient tables,
counterfactual reward, or evaluation generator source through its IPC.

The **primary new learnability suite** consists of two different latent
reward instantiations, held out from all training outcomes:

- `eval_latent_00_v3`
- `eval_latent_01_v3`

Each has exactly **4 side-zero EVAL offers per seed and arm**. Coefficients
vary inside the same identifiable latent preference structure. This tests
whether a learned score helps on unseen payoff *instances*, while avoiding
the scientifically indefensible expectation that arbitrary unseen reward
formulas are predictable from unrelated affine TRAIN examples.

Three **secondary nonlinear OOD stress families**, also with four offers
per seed and arm, are `eval_ood_00_v3`, `eval_ood_01_v3`, and
`eval_ood_02_v3`. They include new source-visible nonlinear geometries
(for example, transformed polynomial/sigmoid, rotated corridor and annular
regions), different from previous v1/v2 interaction, checkerboard,
double-peak, wave-ridge, threshold and basin generators. They are never
used to choose an optimizer or stop condition.

The evaluator draws candidate `(x,y)` features deterministically from
approximately `[0.05,0.95]` and **does not reject examples according to
their true A/B quality difference**. Thus unlike prior gap-filtered tasks,
near-ties remain in the challenge distribution. Report tie rates honestly.
No mirrored side-one offers are counted as additional independent samples.
The generator source and seed are locally inspectable, so these are not
externally blinded or truly secret functions.

The schedule totals **24 TRAIN decisions/seed** and **20 EVAL offers/arm/seed**.
Over twelve seeds, this is **288 chosen TRAIN outcomes** and **240 EVAL
offers per arm**, with every arm matched to the same offers and RNG draws.

## Matched arms and negative controls

Seven EVAL interventions are fixed before EVAL, all with the same complete
TRAIN sample schedule and action propensities:

| Arm | TRAIN state | EVAL intervention |
| --- | --- | --- |
| `joint_on` | Chosen-quality-trained LoRA + appraisal head, persisted and reloaded | Appraisal gain `+12` |
| `joint_off` | **Same exact imported joint LoRA and head** | Gain `0`; isolates **LoRA-only inference influence** |
| `joint_inverted` | Same exact imported joint LoRA and head | Gain `−12` |
| `head_only_on` | Frozen pretrained representation; head trained on identical genuine TRAIN choices/rewards | Gain `+12` |
| `shuffled_joint_on` | Identical initialization, LoRA + head trained with **wrong** chosen-quality labels | Gain `+12` |
| `frozen_on` | LoRA and head untouched after exact matched **nonzero** initialization | Gain `+12`; measures the untrained-head intervention |
| `frozen_off` | Same untouched baseline | Gain `0`; must exactly reproduce frozen-reference policy |

The term **LoRA-only** above refers to the head-disabled inference
intervention on the jointly trained LoRA. It is not misrepresented as a
separately trained reward objective without a critic. If a later experiment
adds a genuinely separate LoRA-only training objective, it needs its own
protocol and controls.

For shuffled control, generate a reproducible *per-seed random derangement*
of the genuinely committed chosen TRAIN reward labels, independently of
reward magnitudes. Reject fixed points, log every original/donor evidence
ID, log the synthetic teaching ID, confirm at least one label changes, and
never pass a fabricated evaluator receipt. This addresses the earlier fixed
cyclic-permutation limitation but still constitutes a single negative-control
draw per seed.

TRAIN action selection uses the exact frozen-reference policy across every
arm. Before chosen quality is resolved, log: frozen action softmax,
exploration-adjusted actual `P(A),P(B)`, uniform RNG draw, draw index, chosen
action, episode/input digest, policy and receipt-commit IDs. Commit exactly
that sampled action. The genuine joint/head-only arms update only on its
authentic chosen TRAIN receipt. Shuffled arms use only synthetic targets
derived from *other* authentic TRAIN receipts after all TRAIN commitments,
never any EVAL quality.

One source-local **separate OS evaluator process per seed** owns the outcome
rules and every arm's predeclared commitments. Its bounded JSON IPC must
reject replayed, cross-arm, altered or wrong-partition tickets. **No chosen
or counterfactual EVAL quality may be returned until every scheduled EVAL
decision of all seven arms is committed and a global decision seal succeeds.**
The posthoc oracle unlocks only after every EVAL chosen receipt resolves and
the whole seed is finalized. The worker's PID/session hashes are local
traceability, not independent reward authenticity.

## Primary effect, statistics and success criterion

Let \(q_A,q_B\) be quality values exposed *only after whole-study sealing*,
and let \(p_{a}\) denote the **recorded actual epsilon-mixture behavior
propensity** for an EVAL offer. The noise-reduced per-offer expected quality is

\[
V(x)=p_A(x)q_A(x)+p_B(x)q_B(x).
\]

The **primary** suite is the equal-weight average over only the **two new
`eval_latent_*_v3` families**. Calculate each arm's family-balanced value
per seed. The seed (n=12), not an arm×offer×family cell, is the statistical
unit. The three co-primary contrasts are:

1. `joint_on − joint_off`: Does the appraisal head itself causally improve
   decisions on the *same learned representation*?
2. `joint_on − shuffled_joint_on`: Does genuine outcome feedback outperform
   a wrong-supervision head+representation control?
3. `joint_on − head_only_on`: Does adapted LoRA representation add value
   beyond the same reward-trained head on the frozen backbone?

The operational exploratory utility gate succeeds **only** when **all
three** comparisons have mean seed-paired expected-quality improvement at
least **+0.005 absolute**, both latent-family means are positive in each
comparison, every seed-clustered Student-*t* descriptive 95% interval
excludes zero on the positive side, and the three seed-level exact paired
sign-flip p-values are **Holm-adjusted below 0.05**. The sign-flip test
assumes seed-effect sign exchangeability, which cannot be authenticated by
source-local deterministic simulation. These extra criteria are demanding;
a null is a valid outcome, and no tuning may occur after looking at EVAL.

Report the paired seed-difference arrays, raw and Holm-adjusted p-values,
t intervals, effect-size margins, each family separately and each control.
The three nonlinear OOD families are **secondary**, with negative-transfer
quantification and no claim of expected transfer without structural overlap.
Report ON/OFF probability shifts, sampled action switches, realized stochastic
reward, candidate/chosen quality MAE and squared error, mean absolute A/B
prediction spread, preference sign accuracy with near-ties disclosed,
inverted-gain sanity, frozen-OFF/reference-policy equality, and the
possibly nonzero frozen-ON/OFF shift due to the deliberately nonzero initial
head. Do **not** incorrectly assert untrained ON/OFF equivalence.

## Persisted state and fresh-process continuation

After each seed's authorized TRAIN, save exact LoRA and head tensors to
versioned **safetensors** only, with immutable snapshot/checkpoint/tokenizer
SHA-256, frozen model weights SHA, target modules, LoRA rank/alpha, exact
feature/optimizer schema, update counters, tensor names/shapes/dtypes/digests
and custody provenance. Load into a separately constructed LoRA+head object
for EVAL. Reject unexpected files, symbolic links, wrong model identity,
wrong LoRA schema, missing or extra tensors, and overwrites. Restore original
state if any in-place import fails.

An **independent fresh Python process** must reconstruct the exact base
model, restore LoRA and head, reproduce fixed EVAL episode token digests,
candidate quality predictions and epsilon-mixture action probabilities
against the main benchmark. Then, using a separately reserved **TRAIN-only
continuation offer** (not an EVAL task or inspected EVAL reward), perform
one further chosen-quality SGD step and save a **new artifact version**.
Verify both trained parameter groups update (when nonzero gradients exist),
the pretrained base SHA remains identical, the update count increments,
and the source artifact is unchanged. Stateless momentum-free SGD means no
hidden optimizer momenta are needed for valid resumption; any other
optimizer requires explicit state custody and a revised manifest.

Publish artifacts as one complete study directory with a verified run-level
manifest/checksum and atomic no-replace completion marker after all seeds
and controls finish. A crash/failed seed leaves no apparently valid full
study. No large model or learned weights are silently committed to GitHub.

## Interpretation and stopping rules

This is a **source-local synthetic experiment**, not an independently
qualified real-world evaluator. Reward rules and seeds are visible to source
readers; source hashes do not certify third-party custody. Improvement within
the known structural class would demonstrate a persistent, causally useful
learned appraisal mechanism in the local open-weight model. It would **not**
prove transfer to novel real-world rewards, the model's subjective pleasure,
or any change to the hosted language model participating in the conversation.

Do not silently shrink the seeds, pick the strongest run, select an EVAL
family after results, change thresholds based on v3 reward values, or reuse
the now-inspected v3 suite as confirmation. If performance, RAM, permission
or correctness blocks arise, record the failed attempt and do not call an
incomplete run scientifically successful. Any revised research design must
use a new protocol and fresh, previously uninspected evaluator suite.
