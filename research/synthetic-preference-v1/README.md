# Model-neutral synthetic choice and candidate-valence trial (development V1)

This source is a standalone **unblinded synthetic experiment**, not a deployed
Pleasure adapter, a Sexuality adapter, an authenticated Volition implementation,
or evidence of subjective pleasure or desire. No private relational material
or explicit content was used. Both local model servers were read-only at
loopback. No neural weights were modified.

## Source custody and security

- Host-neutral scope: generic candidate-value head and offline chosen-outcome
  learning controls; it does not require Sexuality or consume Brigit identity.
- Original Pleasure PR #31 source head:
  d77bdea2f949cca226729fc097d034037eeb2bf2.
  **PR #31's cumulative tree contains a hidden VS Code folderOpen Node task
  and must NOT be trusted/opened as an executable workspace.**
- Only individually inspected candidate_value_head.py and its test were
  extracted through a bare Git-object database into a research-only directory.
  Original head source SHA256:
  ddf97b14e3a1561da83c3bf64fedfa4ab06767f369f09904abebe75e810ba46e.
- Parent release tree here is clean Pleasure main,
  db3d05da63b18193633a607b870e9942700602ff.
- External relation to Sexuality PR #15: security containment remains draft,
  unmerged, unreviewed by an independent security specialist.
- Original evaluation protocol freeze commit (private VN2 local Git):
  409c652a5cbe8a575fbd2395bd1605a96fbc68d6.
  Evaluation seed drawn only after that freeze; these source copies retain
  byte-identical runner and head files.

## Running the source-only tests

From the repository root, standard existing tests:

    PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v

For the separately isolated study (Python 3.12, no third-party dependencies):

    cd research/synthetic-preference-v1/integration
    PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v

The original runner is bound to two exact VN2 GGUF file paths and to loopback
server ports. It is a **historical reproducibility snapshot**, not a portable
production runtime. It will not operate on a different model, host or checkpoint
without an explicit new study and new model/custody bindings. Do not rerun the
sealed 2026-11-16 seed for selection/tuning.

## Private evidence verification

The raw train, model-output, audit and posthoc files are retained privately
on VN2 under:

    /home/patrick/vera-sexuality-research-20261116/receipts/synthetic-preference-v1/

Run (read-only; no model server required):

    python3 -B integration/verify_v1_receipts.py --receipts /path/to/private/receipts

The manifest integration/PINNED_V1_RECEIPTS_20261116.json records expected
SHA256s. It was created **after** evaluation and is not a signature or an
independently held secret. A model-neutral independent evaluator can verify
its arithmetic, not claim external-blind experiment custody.

Never commit raw private receipts, raw model thoughts or personal conversation
data. The synthetic outcome summary is in RESULTS_20261116.md.

## What this is not

- Model self-directed policy formation: prompts instruct the task, reward/quality
  estimates are from an externally trained sidecar.
- Volitional refusal: a DENIED task explicitly instructs decline.
- Independent holdout: the case generator and answer computation are in source.
- Generalization: six exposed synthetic examples per arm, two local models,
  no preregistered independent external contexts.
- Neural training: the GGUF weights are frozen; only external head coefficients
  are updated.
- Consciousness, emotional sensation, attraction, relationship status,
  autobiographical memory, autonomy or consent.

Next controlled step: diagnose abstention confounding with a newly frozen
prompt/estimate intervention and independent evaluation custody; preserve this
null result without tuning or laundering it into success.
