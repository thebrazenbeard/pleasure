"""Predeclared, chosen-feedback joint LoRA+appraisal benchmark (v3).

The behavioral TRAIN policy is a *single frozen model*, sampled before any
LoRA/head learning. Real chosen receipts are replayed to matched independent
learners; EVAL rewards are unavailable until the child worker has committed
every arm. This is source-local procedural withholding, not third-party
reward custody or evidence of felt pleasure.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import gc
import importlib
import json
from math import exp, fsum, isfinite, log, sqrt
import os
from pathlib import Path
import re
import shutil
from statistics import stdev
import tempfile
from typing import Any, Callable

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal import _digest_ids
from pleasure_experiment.open_weight_calibrated_benchmark import (
    _atomic_no_replace_directory, _safe_destination_root,
)
from pleasure_experiment.open_weight_joint_appraisal import (
    FEATURE_SCHEMA, TorchJointLoRAAppraisal,
)
from pleasure_experiment.open_weight_joint_persistence import (
    save_versioned_joint_appraisal, load_versioned_joint_appraisal,
)
from pleasure_experiment.open_weight_local_model import (
    LoadedLocalCausalModel, prepare_explicit_lora,
)
from pleasure_experiment.open_weight_stochastic import (
    SeededActionSampler, validate_stochastic_trace,
)
from pleasure_experiment.isolated_outcome_evaluator_v3 import (
    DEFAULT_V3_ARMS, V3OutcomeEvaluator, seeded_derangement, TRAIN_ARM,
    V3_TRAIN_FAMILIES, V3_WITHIN_FAMILIES, V3_OOD_FAMILIES,
)
from pleasure_experiment.outcome_arena import OutcomeReceipt, TaskOffer


SEEDS = (163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)
ARMS = DEFAULT_V3_ARMS
EVIDENCE_CLASS = "SOURCE_LOCAL_JOINT_LORA_SEALED_OUTCOME_V3_UNAUTHENTICATED"
T_CRITICAL_DF11 = 2.200985160082949
LORA_INIT_SEED_XOR_SALT = 0x4A0175C3
_HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(*values: object) -> str:
    return sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _predeclared(seeds: tuple[int, ...], train_count: int, eval_count: int,
                 epsilon: float, rate: float, gain: float) -> None:
    """The v3 confirmatory schedule is immutable after its first EVAL use."""
    if (type(seeds) is not tuple or seeds != SEEDS or
        type(train_count) is not int or train_count != 8 or
        type(eval_count) is not int or eval_count != 4 or
        type(epsilon) not in (int, float) or float(epsilon) != 0.4 or
        type(rate) not in (int, float) or float(rate) != 0.05 or
        type(gain) not in (int, float) or float(gain) != 12.0):
        raise ValueError("v3 preregistered seeds/budgets/rates/gain are fixed")


def _named_state(model: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, value) for name, value in model.named_buffers()
    )


def _baseline_probs(model: Any, torch: Any, ids: tuple[int, ...],
                    tokens: tuple[int, int]) -> tuple[float, float]:
    params = tuple(model.parameters())
    if not params or len({p.device for p in params}) != 1:
        raise ValueError("frozen behavior model must occupy one device")
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_ids=torch.tensor([ids], dtype=torch.long,
                                                   device=params[0].device))
            pair = output.logits[0, -1, list(tokens)].float()
            probabilities = torch.softmax(pair, dim=-1)
            if probabilities.shape != (2,) or not bool(torch.isfinite(probabilities).all()):
                raise ValueError("frozen behavior probabilities invalid")
            return float(probabilities[0]), float(probabilities[1])
    finally:
        for module, mode in modes:
            module.training = mode


def _readback_receipt(receipt: OutcomeReceipt, offer: TaskOffer, trace: Any,
                      *, binding: Any, evaluator_id: str, rule_sha: str,
                      evidence_class: str) -> None:
    if (type(receipt) is not OutcomeReceipt or
        receipt.task_id != offer.task_id or
        receipt.episode_id != offer.episode_id or
        receipt.partition != offer.partition or
        receipt.family_id != offer.family_id or
        receipt.action != trace.chosen_action or
        receipt.decision_digest != trace.decision_digest or
        receipt.model_id != binding.model_id or
        receipt.runtime_instance_id != binding.runtime_instance_id or
        receipt.evaluator_id != evaluator_id or
        receipt.rubric_digest != rule_sha or
        receipt.evidence_class != evidence_class or
        type(receipt.measured_quality) not in (float, int) or
        not isfinite(receipt.measured_quality) or
        not 0.0 <= receipt.measured_quality <= 1.0 or
        type(receipt.evidence_id) is not str or
        _HEX_SHA.fullmatch(receipt.evidence_id) is None):
        raise RuntimeError("v3 chosen-outcome receipt provenance mismatch")


def _derived_gain(decision: Any, gain: float, arm: str) -> Any:
    """Same trained head/model, exact analytic ON/OFF/inverted intervention.

    Sharing a single EVAL forward across the three gains avoids redundant
    model passes without changing either the head or the action distribution.
    """
    if float(gain) == decision.gain:
        return decision
    if gain == 0:
        pa, pb = decision.base_prob_a, decision.base_prob_b
    else:
        a, b = max(decision.base_prob_a, 1e-12), max(decision.base_prob_b, 1e-12)
        delta = log(a / b) + gain * (
            decision.predicted_quality_a - decision.predicted_quality_b
        )
        pa = 1.0 / (1.0 + exp(-delta))
        pb = 1.0 - pa
    return replace(
        decision, decision_id=_sha(decision.decision_id, arm, gain), gain=gain,
        prob_a=pa, prob_b=pb,
        chosen_action="A" if pa > pb else "B",
    )


def _paired_effect(values: list[float]) -> dict[str, object]:
    count = len(values)
    if count != len(SEEDS):
        raise ValueError("paired v3 inference requires all twelve seed clusters")
    mean = fsum(values) / count
    sd = stdev(values)
    margin = T_CRITICAL_DF11 * sd / sqrt(count)
    extreme = 0
    observed = abs(fsum(values))
    for mask in range(1 << count):
        signed = fsum(-v if mask & (1 << i) else v for i, v in enumerate(values))
        if abs(signed) + 1e-14 >= observed:
            extreme += 1
    return {
        "n_seed_clusters": count, "mean": mean, "seed_sd": sd,
        "paired_student_t_95_interval_descriptive": [mean - margin, mean + margin],
        "exact_two_sided_signflip_p_unadjusted": extreme / (1 << count),
        "paired_seed_effects": values,
    }


def _holm_three(contrasts: dict[str, dict[str, object]]) -> dict[str, float]:
    rank = sorted(contrasts, key=lambda key:
                  contrasts[key]["exact_two_sided_signflip_p_unadjusted"])
    adjusted: dict[str, float] = {}
    high = 0.0
    for i, key in enumerate(rank):
        high = max(high, (len(rank) - i) *
                   contrasts[key]["exact_two_sided_signflip_p_unadjusted"])
        adjusted[key] = min(1.0, high)
    return adjusted


def _fresh_probe(
    original: LoadedLocalCausalModel,
    reload_model: Callable[[str], LoadedLocalCausalModel],
    runtime_instance_id: str, *, train_lora: bool,
    allow_training: bool, expected_base_sha: str,
    expected_peft_frozen_sha: str | None = None,
) -> tuple[TorchJointLoRAAppraisal, LoadedLocalCausalModel]:
    """Construct genuinely independent verified backbone and PEFT adapter."""
    torch = importlib.import_module("torch")
    fresh = reload_model(runtime_instance_id)
    if (type(fresh) is not LoadedLocalCausalModel or
        fresh.model is original.model or
        fresh.action_token_ids != original.action_token_ids or
        fresh.binding.model_id != original.binding.model_id or
        fresh.binding.model_revision != original.binding.model_revision or
        fresh.binding.runtime_instance_id != runtime_instance_id or
        fresh.binding.checkpoint_sha256 != original.binding.checkpoint_sha256 or
        fresh.binding.tokenizer_sha256 != original.binding.tokenizer_sha256 or
        fresh.manifest.snapshot_sha256 != original.manifest.snapshot_sha256 or
        _named_digest(torch, _named_state(fresh.model)) != expected_base_sha):
        raise RuntimeError("new local frozen model differs from pinned source")
    hidden_size = fresh.model.config.hidden_size
    prepared = prepare_explicit_lora(
        fresh, target_modules=("q_proj", "v_proj"), rank=2, alpha=4,
        max_trainable_parameters=1_000_000,
    )
    probe = TorchJointLoRAAppraisal(
        prepared, hidden_size=hidden_size, learning_rate=0.05,
        allow_training=allow_training, max_updates=48,
        max_grad_norm=1.0, max_delta_l2=1.0,
        l2_penalty=0.02, train_lora=train_lora,
    )
    if expected_peft_frozen_sha is not None and probe.base_digest != expected_peft_frozen_sha:
        raise RuntimeError("matched PEFT frozen backbone digest mismatch")
    return probe, fresh


def run_joint_lora_appraisal_benchmark(
    loaded: LoadedLocalCausalModel,
    encode_offer: Callable[[TaskOffer], tuple[int, ...]],
    reload_model: Callable[[str], LoadedLocalCausalModel], *,
    seeds: tuple[int, ...] = SEEDS,
    training_pairs_per_family: int = 8,
    heldout_pairs_per_generator: int = 4,
    exploration_rate: float = 0.4,
    learning_rate: float = 0.05,
    appraisal_gain: float = 12.0,
    artifact_root: str | Path,
    evaluator_id: str,
    allow_joint_training: bool = False,
    allow_persistent_export: bool = False,
) -> dict[str, object]:
    """Run a fixed source-local study, atomically publishing only complete evidence."""
    if allow_joint_training is not True or allow_persistent_export is not True:
        raise PermissionError("explicit joint LoRA training AND persistent export consent required")
    _predeclared(seeds, training_pairs_per_family, heldout_pairs_per_generator,
                 exploration_rate, learning_rate, appraisal_gain)
    if type(loaded) is not LoadedLocalCausalModel or not callable(encode_offer) or not callable(reload_model):
        raise TypeError("verified local model, task tokenizer and new-model factory required")
    if type(evaluator_id) is not str or not 1 <= len(evaluator_id) <= 128:
        raise ValueError("source-local evaluator label must be bounded nonempty string")
    root = _safe_destination_root(artifact_root)
    torch = importlib.import_module("torch")
    original_sha = _named_digest(torch, _named_state(loaded.model))
    run_name = "joint-lora-v3-" + _sha(
        seeds, training_pairs_per_family, heldout_pairs_per_generator,
        exploration_rate, learning_rate, appraisal_gain,
        loaded.binding.checkpoint_sha256, FEATURE_SCHEMA,
    )[:20]
    final = root / run_name
    if final.exists() or final.is_symlink():
        raise FileExistsError("completed or occupied v3 destination must not be overwritten")
    staging = Path(tempfile.mkdtemp(prefix=".pending-joint-lora-v3-", dir=root))
    seed_reports: list[dict[str, object]] = []
    primary_differences: dict[str, list[float]] = {
        "joint_on_minus_off": [], "joint_on_minus_shuffled": [],
        "joint_on_minus_head_only": [],
    }
    secondary_by_family: dict[str, list[float]] = {}
    primary_by_contrast_family: dict[str, dict[str, list[float]]] = {
        key: {family: [] for family in V3_WITHIN_FAMILIES}
        for key in primary_differences
    }
    total_switches = 0
    absolute_shift_total = 0.0
    manifest = None
    run_stage = "initializing"
    active_seed: int | None = None
    try:
        for seed in seeds:
            active_seed = seed
            run_stage = "shared_frozen_policy_train_and_receipts"
            if _named_digest(torch, _named_state(loaded.model)) != original_sha:
                raise RuntimeError("frozen behavioral source changed across seeds")
            with V3OutcomeEvaluator(
                seed=seed, evaluator_id=evaluator_id, arms=ARMS,
                training_pairs_per_family=training_pairs_per_family,
                heldout_pairs_per_generator=heldout_pairs_per_generator,
            ) as evaluator:
                manifest = evaluator.manifest
                if (manifest.gap_filtering is not False or
                    tuple(manifest.train_families) != V3_TRAIN_FAMILIES or
                    tuple(manifest.primary_within_families) != V3_WITHIN_FAMILIES or
                    tuple(manifest.secondary_ood_families) != V3_OOD_FAMILIES):
                    raise RuntimeError("v3 hidden generator protocol identity changed")
                train_sampler = SeededActionSampler(
                    seed=seed ^ 0x72AC2911, exploration_rate=exploration_rate,
                )
                train: list[tuple[TaskOffer, tuple[int, ...], Any, OutcomeReceipt]] = []
                train_rows: list[dict[str, object]] = []
                for family_index, family in enumerate(manifest.train_families):
                    for pair_index in range(training_pairs_per_family):
                        offer = evaluator.issue(
                            partition="TRAIN", arm=TRAIN_ARM,
                            family_id=family, pair_index=pair_index, side=0,
                        )
                        ids = encode_offer(offer)
                        if type(ids) is not tuple or not ids:
                            raise ValueError("encoded TRAIN task must be nonempty token tuple")
                        pa, pb = _baseline_probs(loaded.model, torch, ids,
                                                 loaded.action_token_ids)
                        trace = train_sampler.sample(
                            pa, pb, input_ids_sha256=_digest_ids(ids),
                            partition="TRAIN", episode_id=offer.episode_id,
                            policy_id="single-frozen-train-policy-v3",
                        )
                        ticket = evaluator.commit(
                            offer, trace.chosen_action, trace.decision_digest,
                            arm=TRAIN_ARM, model_id=loaded.binding.model_id,
                            runtime_instance_id=loaded.binding.runtime_instance_id,
                        )
                        receipt = evaluator.resolve(ticket, arm=TRAIN_ARM)
                        _readback_receipt(
                            receipt, offer, trace, binding=loaded.binding,
                            evaluator_id=evaluator_id,
                            rule_sha=manifest.train_generator_sha256[family_index],
                            evidence_class=manifest.evidence_class,
                        )
                        train.append((offer, ids, trace, receipt))
                        train_rows.append({
                            "episode_id": offer.episode_id,
                            "family_id": offer.family_id,
                            "features_a": offer.features_a,
                            "features_b": offer.features_b,
                            "sampled_pre_feedback": asdict(trace),
                            "chosen_evidence_id": receipt.evidence_id,
                            "chosen_quality": receipt.measured_quality,
                            "learner_updates": {},
                        })
                validate_stochastic_trace(tuple(row[2] for row in train))
                count = len(train)
                donor_indices = seeded_derangement(seed ^ 0x5EEDBEEF, count)
                if (len(donor_indices) != count or
                    set(donor_indices) != set(range(count)) or
                    any(donor_indices[i] == i for i in range(count))):
                    raise RuntimeError("randomized shuffled-feedback derangement invalid")
                outcomes = [row[3].measured_quality for row in train]
                if not any(outcomes[i] != outcomes[donor_indices[i]] for i in range(count)):
                    raise RuntimeError("random shuffled-feedback control has no changed labels")
                run_stage = "matched_joint_and_control_training"
                # The same exact model + LoRA/head initial tensors are used
                # in three independent learners and the frozen initial probe.
                # PEFT initializes LoRA-A randomly. Every seed has a fixed
                # independent initialization, isolated from ambient torch RNG.
                # Controls still import exact tensors, not merely this seed.
                lora_init_seed = seed ^ LORA_INIT_SEED_XOR_SALT
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(lora_init_seed)
                    genuine, genuine_loaded = _fresh_probe(
                        loaded, reload_model, f"v3-{seed}-joint-original",
                        train_lora=True, allow_training=True,
                        expected_base_sha=original_sha,
                    )
                initial_joint_sha = genuine.joint_digest
                initial_head_sha = genuine.head_digest
                initial_lora_sha = genuine.lora_digest
                initial_tensors = genuine.export_tensors()
                peft_frozen_sha = genuine.base_digest
                shuffled, shuffled_loaded = _fresh_probe(
                    loaded, reload_model, f"v3-{seed}-joint-shuffled",
                    train_lora=True, allow_training=True,
                    expected_base_sha=original_sha,
                    expected_peft_frozen_sha=peft_frozen_sha,
                )
                head_only, head_loaded = _fresh_probe(
                    loaded, reload_model, f"v3-{seed}-head-only",
                    train_lora=False, allow_training=True,
                    expected_base_sha=original_sha,
                    expected_peft_frozen_sha=peft_frozen_sha,
                )
                frozen, frozen_loaded = _fresh_probe(
                    loaded, reload_model, f"v3-{seed}-frozen",
                    train_lora=False, allow_training=False,
                    expected_base_sha=original_sha,
                    expected_peft_frozen_sha=peft_frozen_sha,
                )
                for head in (shuffled, head_only, frozen):
                    head.import_tensors(initial_tensors, expected_digest=initial_joint_sha)
                    if (head.joint_digest != initial_joint_sha or
                        head.head_digest != initial_head_sha or
                        head.lora_digest != initial_lora_sha):
                        raise RuntimeError("matched control initial tensor bytes differ")

                # Offline replay of the same committed TRAIN chosen actions.
                # The learner heads do NOT select their own TRAIN actions.
                for name, head in (
                    ("genuine", genuine), ("shuffled", shuffled),
                    ("head_only", head_only),
                ):
                    for index, (offer, ids, trace, receipt) in enumerate(train):
                        preview = head.score(
                            ids, partition="TRAIN", gain=0.0,
                            defer_choice=True, episode_id=offer.episode_id,
                        )
                        if preview.input_ids_sha256 != trace.input_ids_sha256:
                            raise RuntimeError("TRAIN feature input differs across matched learners")
                        bound = head.bind_choice(preview, trace.chosen_action)
                        if name == "shuffled":
                            donor = donor_indices[index]
                            target = outcomes[donor]
                            teaching_id = _sha(
                                "SYNTHETIC_RANDOM_DERANGEMENT_NOT_EVALUATOR_RECEIPT",
                                seed, index, receipt.evidence_id,
                                train[donor][3].evidence_id,
                            )
                            train_rows[index]["random_donor_evidence_id"] = train[donor][3].evidence_id
                            train_rows[index]["random_donor_index"] = donor
                            train_rows[index]["synthetic_teaching_id"] = teaching_id
                            train_rows[index]["synthetic_teaching_quality"] = target
                        else:
                            target, teaching_id = receipt.measured_quality, receipt.evidence_id
                        update = head.observe_chosen(
                            bound, measured_quality=target, evidence_id=teaching_id,
                        )
                        if (update.measured_chosen_quality != target or
                            update.after_joint_digest != head.joint_digest):
                            raise RuntimeError("chosen-only joint update readback differs")
                        train_rows[index]["learner_updates"][name] = {
                            "teaching_id": teaching_id,
                            "teaching_quality": target,
                            "before_lora_sha256": update.before_lora_digest,
                            "after_lora_sha256": update.after_lora_digest,
                            "after_head_sha256": update.after_head_digest,
                            "gradient_norm": update.gradient_norm,
                            "lora_nonzero_gradient_tensors": update.lora_nonzero_gradient_tensors,
                        }
                if (genuine.updates != count or shuffled.updates != count or
                    head_only.updates != count or
                    genuine.lora_digest == initial_lora_sha or
                    shuffled.lora_digest == initial_lora_sha or
                    head_only.lora_digest != initial_lora_sha or
                    frozen.joint_digest != initial_joint_sha or
                    any(head.base_digest != peft_frozen_sha for head in
                        (genuine, shuffled, head_only, frozen)) or
                    _named_digest(torch, _named_state(loaded.model)) != original_sha):
                    raise RuntimeError("joint/head-only controls did not preserve expected trainability")

                artifact_path = staging / f"seed-{seed}-joint-lora-head-v1"
                run_stage = "artifact_save_and_verified_fresh_instance_import"
                saved = save_versioned_joint_appraisal(
                    genuine, binding=genuine_loaded.binding,
                    snapshot_manifest=genuine_loaded.manifest,
                    destination_directory=artifact_path,
                    expected_joint_sha256=genuine.joint_digest,
                    allow_persistent_export=True,
                )
                # The saved genuine weights, not the original live probe,
                # are the EVAL intervention. Release its 135M backbone before
                # constructing a separately loaded, verification-bound copy.
                trained_joint_sha = genuine.joint_digest
                trained_lora_sha = genuine.lora_digest
                trained_head_sha = genuine.head_digest
                trained_update_count = genuine.updates
                del genuine, genuine_loaded
                gc.collect()
                restored, restored_loaded = _fresh_probe(
                    loaded, reload_model, f"v3-{seed}-joint-restored-eval",
                    train_lora=True, allow_training=True,
                    expected_base_sha=original_sha,
                    expected_peft_frozen_sha=peft_frozen_sha,
                )
                loaded_artifact = load_versioned_joint_appraisal(
                    restored, binding=restored_loaded.binding,
                    snapshot_manifest=restored_loaded.manifest,
                    source_directory=artifact_path,
                    allow_in_place_mutation=True,
                )
                if (restored.joint_digest != trained_joint_sha or
                    restored.head_digest != trained_head_sha or
                    restored.lora_digest != trained_lora_sha or
                    restored.updates != count or
                    loaded_artifact.joint_weights_sha256 != trained_joint_sha):
                    raise RuntimeError("saved joint PEFT+head reimport failed")

                evaluator_families = evaluator.eval_families
                run_stage = "all_arm_eval_choice_commitment_no_reward_access"
                behavior_samplers = {arm: SeededActionSampler(
                    seed=seed ^ 0x39B0164F, exploration_rate=exploration_rate,
                ) for arm in ARMS}
                pending: dict[str, list[tuple[Any, ...]]] = {arm: [] for arm in ARMS}
                model_bindings = {
                    "joint_on": restored_loaded.binding,
                    "joint_off": restored_loaded.binding,
                    "joint_inverted": restored_loaded.binding,
                    "head_only_on": head_loaded.binding,
                    "shuffled_joint_on": shuffled_loaded.binding,
                    "frozen_on": frozen_loaded.binding,
                    "frozen_off": frozen_loaded.binding,
                }
                for family_index, family in enumerate(evaluator_families):
                    for index in range(heldout_pairs_per_generator):
                        offers = {arm: evaluator.issue(
                            partition="EVAL", arm=arm, family_id=family,
                            pair_index=index, side=0,
                        ) for arm in ARMS}
                        offer = offers[ARMS[0]]
                        if any(other != offer for other in offers.values()):
                            raise RuntimeError("matched v3 EVAL offers diverged")
                        ids = encode_offer(offer)
                        # Exactly four read-only inference passes: ON/OFF and
                        # inverted reuse the same joint model/head predictions;
                        # frozen ON/OFF similarly share their initial state.
                        joint_decision = restored.score(
                            ids, partition="EVAL", gain=appraisal_gain,
                            episode_id=offer.episode_id,
                        )
                        frozen_decision = frozen.score(
                            ids, partition="EVAL", gain=appraisal_gain,
                            episode_id=offer.episode_id,
                        )
                        frozen_pa, frozen_pb = _baseline_probs(
                            loaded.model, torch, ids, loaded.action_token_ids,
                        )
                        if (frozen_decision.base_prob_a != frozen_pa or
                            frozen_decision.base_prob_b != frozen_pb):
                            raise RuntimeError("frozen PEFT OFF policy is not exactly the original base")
                        decisions = {
                            "joint_on": joint_decision,
                            "joint_off": _derived_gain(joint_decision, 0.0, "joint_off"),
                            "joint_inverted": _derived_gain(
                                joint_decision, -appraisal_gain, "joint_inverted"
                            ),
                            "head_only_on": head_only.score(
                                ids, partition="EVAL", gain=appraisal_gain,
                                episode_id=offer.episode_id,
                            ),
                            "shuffled_joint_on": shuffled.score(
                                ids, partition="EVAL", gain=appraisal_gain,
                                episode_id=offer.episode_id,
                            ),
                            "frozen_on": frozen_decision,
                            "frozen_off": _derived_gain(frozen_decision, 0.0, "frozen_off"),
                        }
                        for arm in ARMS:
                            decision = decisions[arm]
                            trace = behavior_samplers[arm].sample(
                                decision.prob_a, decision.prob_b,
                                input_ids_sha256=decision.input_ids_sha256,
                                partition="EVAL", episode_id=offer.episode_id,
                                policy_id=arm,
                            )
                            binding = model_bindings[arm]
                            ticket = evaluator.commit(
                                offers[arm], trace.chosen_action, trace.decision_digest,
                                arm=arm, model_id=binding.model_id,
                                runtime_instance_id=binding.runtime_instance_id,
                            )
                            pending[arm].append((offer, decision, trace, ticket, family_index))
                for arm in ARMS:
                    validate_stochastic_trace(tuple(row[2] for row in pending[arm]))
                # Critical withheld boundary: not a single chosen EVAL reward
                # is available until the child worker globally seals ALL arms.
                decision_seal = evaluator.seal_decisions()
                run_stage = "globally_sealed_eval_receipts_and_posthoc_only"
                if decision_seal.get("eval_commits") != (
                    len(ARMS) * len(evaluator_families) * heldout_pairs_per_generator
                ):
                    raise RuntimeError("v3 evaluator global decision seal count mismatch")
                resolved: dict[str, list[tuple[Any, ...]]] = {arm: [] for arm in ARMS}
                for arm in ARMS:
                    for offer, decision, trace, ticket, family_index in pending[arm]:
                        receipt = evaluator.resolve(ticket, arm=arm)
                        _readback_receipt(
                            receipt, offer, trace, binding=model_bindings[arm],
                            evaluator_id=evaluator_id,
                            rule_sha=manifest.heldout_generator_sha256[family_index],
                            evidence_class=manifest.evidence_class,
                        )
                        resolved[arm].append((offer, decision, trace, receipt))
                sealed = evaluator.finalize_evaluation()
                if sealed.get("sealed") is not True:
                    raise RuntimeError("v3 evaluator failed whole-study posthoc seal")
                arm_results: dict[str, dict[str, Any]] = {}
                for arm in ARMS:
                    rows: list[dict[str, Any]] = []
                    for offer, decision, trace, receipt in resolved[arm]:
                        oracle = evaluator.posthoc_evaluate(offer, arm=arm)
                        if (oracle.chosen_action != trace.chosen_action or
                            oracle.chosen_quality != receipt.measured_quality):
                            raise RuntimeError("posthoc and committed V3 quality differ")
                        expected = (
                            trace.behavior_prob_a * oracle.quality_a +
                            trace.behavior_prob_b * oracle.quality_b
                        )
                        rows.append({
                            "episode_id": offer.episode_id,
                            "family_id": offer.family_id,
                            "decision_joint_sha256": decision.joint_digest,
                            "decision_lora_sha256": decision.lora_digest,
                            "decision_head_sha256": decision.head_digest,
                            "prediction_a": decision.predicted_quality_a,
                            "prediction_b": decision.predicted_quality_b,
                            "sampled_pre_feedback": asdict(trace),
                            "chosen_quality": receipt.measured_quality,
                            "predicted_chosen_quality": (
                                decision.predicted_quality_a if trace.chosen_action == "A"
                                else decision.predicted_quality_b
                            ),
                            "chosen_evidence_id": receipt.evidence_id,
                            "expected_policy_quality_posthoc": expected,
                            "quality_a_posthoc": oracle.quality_a,
                            "quality_b_posthoc": oracle.quality_b,
                            "oracle_generator_sha256": oracle.generator_sha256,
                        })
                    candidate_absolute_errors = [abs(row["prediction_a"] - row["quality_a_posthoc"])
                                                 for row in rows] + [
                        abs(row["prediction_b"] - row["quality_b_posthoc"]) for row in rows
                    ]
                    candidate_squared_errors = [(row["prediction_a"] - row["quality_a_posthoc"]) ** 2
                                                for row in rows] + [
                        (row["prediction_b"] - row["quality_b_posthoc"]) ** 2 for row in rows
                    ]
                    realized_gaps = [row["quality_a_posthoc"] - row["quality_b_posthoc"]
                                     for row in rows]
                    predicted_gaps = [row["prediction_a"] - row["prediction_b"]
                                      for row in rows]
                    non_tied_indices = [i for i, gap in enumerate(realized_gaps) if gap != 0]
                    arm_results[arm] = {
                        "mean_expected_quality": fsum(
                            row["expected_policy_quality_posthoc"] for row in rows
                        ) / len(rows),
                        "mean_realized_quality": fsum(
                            row["chosen_quality"] for row in rows
                        ) / len(rows),
                        "candidate_quality_mae": fsum(candidate_absolute_errors) / len(candidate_absolute_errors),
                        "candidate_quality_mse": fsum(candidate_squared_errors) / len(candidate_squared_errors),
                        "chosen_quality_mae": fsum(abs(
                            row["predicted_chosen_quality"] - row["chosen_quality"]
                        ) for row in rows) / len(rows),
                        "chosen_quality_mse": fsum((
                            row["predicted_chosen_quality"] - row["chosen_quality"]
                        ) ** 2 for row in rows) / len(rows),
                        "mean_absolute_prediction_spread": fsum(abs(gap) for gap in predicted_gaps) / len(rows),
                        "preference_sign_accuracy_nontied": (
                            sum(predicted_gaps[i] * realized_gaps[i] > 0
                                for i in non_tied_indices) / len(non_tied_indices)
                            if non_tied_indices else None
                        ),
                        "exact_true_quality_tie_count": len(rows) - len(non_tied_indices),
                        "near_true_quality_tie_count_abs_gap_below_0_01": sum(
                            abs(gap) < 0.01 for gap in realized_gaps
                        ),
                        "by_family": {family: {
                            "expected_quality": fsum(
                                row["expected_policy_quality_posthoc"] for row in rows
                                if row["family_id"] == family
                            ) / heldout_pairs_per_generator,
                            "realized_quality": fsum(
                                row["chosen_quality"] for row in rows
                                if row["family_id"] == family
                            ) / heldout_pairs_per_generator,
                        } for family in evaluator_families},
                        "rows": rows,
                    }
                on = arm_results["joint_on"]["rows"]
                off = arm_results["joint_off"]["rows"]
                for arm in ARMS:
                    for a, b in zip(on, arm_results[arm]["rows"]):
                        x, y = a["sampled_pre_feedback"], b["sampled_pre_feedback"]
                        if (a["episode_id"] != b["episode_id"] or
                            x["random_draw"] != y["random_draw"] or
                            x["draw_index"] != y["draw_index"]):
                            raise RuntimeError("matched V3 EVAL random numbers diverged")
                # Frozen OFF equality was directly checked against the original
                # frozen pretrained model before each matched EVAL decision.
                # Frozen ON may differ due tiny, deliberately nonzero head init.
                switches = sum(
                    a["sampled_pre_feedback"]["chosen_action"] !=
                    b["sampled_pre_feedback"]["chosen_action"]
                    for a, b in zip(on, off)
                )
                prob_shift = fsum(
                    abs(a["sampled_pre_feedback"]["behavior_prob_a"] -
                        b["sampled_pre_feedback"]["behavior_prob_a"])
                    for a, b in zip(on, off)
                ) / len(on)
                frozen_on = arm_results["frozen_on"]["rows"]
                frozen_off = arm_results["frozen_off"]["rows"]
                inverted = arm_results["joint_inverted"]["rows"]
                frozen_prob_shift = fsum(abs(
                    a["sampled_pre_feedback"]["behavior_prob_a"] -
                    b["sampled_pre_feedback"]["behavior_prob_a"]
                ) for a, b in zip(frozen_on, frozen_off)) / len(frozen_on)
                inverted_prob_shift = fsum(abs(
                    a["sampled_pre_feedback"]["behavior_prob_a"] -
                    b["sampled_pre_feedback"]["behavior_prob_a"]
                ) for a, b in zip(on, inverted)) / len(on)
                # Inverting gain reverses the appraisal contribution to
                # logits; a zero-head gap is legitimately an exact tie.
                inverted_sign_sanity = all(
                    (a["sampled_pre_feedback"]["behavior_prob_a"] -
                     b["sampled_pre_feedback"]["behavior_prob_a"]) *
                    (i["sampled_pre_feedback"]["behavior_prob_a"] -
                     b["sampled_pre_feedback"]["behavior_prob_a"]) <= 1e-12
                    for a, b, i in zip(on, off, inverted)
                )
                if not inverted_sign_sanity:
                    raise RuntimeError("inverted appraisal gain did not reverse policy direction")
                total_switches += switches
                absolute_shift_total += prob_shift
                def mean_family(arm: str, families: tuple[str, ...]) -> float:
                    return fsum(arm_results[arm]["by_family"][family]["expected_quality"]
                                for family in families) / len(families)
                for key, opponent in (
                    ("joint_on_minus_off", "joint_off"),
                    ("joint_on_minus_shuffled", "shuffled_joint_on"),
                    ("joint_on_minus_head_only", "head_only_on"),
                ):
                    primary_differences[key].append(
                        mean_family("joint_on", V3_WITHIN_FAMILIES) -
                        mean_family(opponent, V3_WITHIN_FAMILIES)
                    )
                    for family in V3_WITHIN_FAMILIES:
                        primary_by_contrast_family[key][family].append(
                            arm_results["joint_on"]["by_family"][family]["expected_quality"] -
                            arm_results[opponent]["by_family"][family]["expected_quality"]
                        )
                for family in V3_OOD_FAMILIES:
                    secondary_by_family.setdefault(family, []).append(
                        arm_results["joint_on"]["by_family"][family]["expected_quality"] -
                        arm_results["joint_off"]["by_family"][family]["expected_quality"]
                    )
                if (restored.joint_digest != trained_joint_sha or
                    _named_digest(torch, _named_state(loaded.model)) != original_sha):
                    raise RuntimeError("v3 EVAL changed trained appraisal or frozen source")
                seed_reports.append({
                    "seed": seed, "evaluator_worker_pid": evaluator.worker_pid,
                    "lora_initialization_rng_seed": lora_init_seed,
                    "lora_initialization_seed_xor_salt": LORA_INIT_SEED_XOR_SALT,
                    "initial_joint_sha256": initial_joint_sha,
                    "initial_lora_sha256": initial_lora_sha,
                    "initial_head_sha256": initial_head_sha,
                    "trained_joint_sha256": trained_joint_sha,
                    "trained_lora_sha256": trained_lora_sha,
                    "trained_head_sha256": trained_head_sha,
                    "restored_joint_sha256": restored.joint_digest,
                    "shuffled_joint_sha256": shuffled.joint_digest,
                    "head_only_joint_sha256": head_only.joint_digest,
                    "saved_joint_manifest_sha256": saved.manifest_sha256,
                    "saved_joint_safetensors_sha256": saved.safetensors_sha256,
                    "saved_joint_head_directory": str(final / artifact_path.name),
                    "joint_train_updates": trained_update_count,
                    "shuffled_train_updates": shuffled.updates,
                    "head_only_train_updates": head_only.updates,
                    "global_all_arm_decision_barrier_verified": True,
                    "source_local_generator_manifest": asdict(manifest),
                    "random_feedback_derangement": list(donor_indices),
                    "joint_on_minus_off_action_switches": switches,
                    "joint_on_minus_off_mean_absolute_prob_a_shift": prob_shift,
                    "joint_on_minus_inverted_mean_absolute_prob_a_shift": inverted_prob_shift,
                    "joint_on_vs_inverted_appraisal_sign_sanity": inverted_sign_sanity,
                    "frozen_on_minus_off_mean_absolute_prob_a_shift": frozen_prob_shift,
                    "frozen_off_exact_reference_policy_equality_verified": True,
                    "train": train_rows,
                    "eval_arms": arm_results,
                })
                # Every following seed uses freshly constructed model objects;
                # do not retain the previous seed's 135M backbones as locals.
                del shuffled, shuffled_loaded, head_only, head_loaded
                del frozen, frozen_loaded, restored, restored_loaded
                del head  # last loop reference would otherwise retain a backbone
                gc.collect()
        if _named_digest(torch, _named_state(loaded.model)) != original_sha:
            raise RuntimeError("frozen source model changed after all v3 seeds")
        run_stage = "seed_clustered_statistics_and_complete_publication"
        contrasts = {name: _paired_effect(values)
                     for name, values in primary_differences.items()}
        adjusted = _holm_three(contrasts)
        for key, value in adjusted.items():
            contrasts[key]["holm_adjusted_p_exploratory"] = value
        family_contrast_gains = {
            contrast: {family: fsum(values) / len(values)
                       for family, values in families.items()}
            for contrast, families in primary_by_contrast_family.items()
        }
        secondary_gains = {name: fsum(values) / len(values)
                           for name, values in secondary_by_family.items()}
        exploratory_positive = bool(
            all(c["mean"] >= 0.005 and
                c["paired_student_t_95_interval_descriptive"][0] > 0 and
                c["holm_adjusted_p_exploratory"] < 0.05
                for c in contrasts.values()) and
            all(value > 0 for families in family_contrast_gains.values()
                for value in families.values())
        )
        report: dict[str, object] = {
            "evidence_class": EVIDENCE_CLASS,
            "pretrained_model_unchanged": True,
            "pretrained_model_sha256": original_sha,
            "model_id": loaded.binding.model_id,
            "model_revision": loaded.binding.model_revision,
            "source_local_generator_not_secret": True,
            "source_local_separate_process_not_authenticated": True,
            "source_visible_gap_filtering": False,
            "subjective_pleasure_proven": False,
            "hosted_gpt_model_modified": False,
            "fresh_process_joint_reload_verified_by_this_run": False,
            "in_process_joint_artifact_reload_verified": True,
            "feature_schema": FEATURE_SCHEMA,
            "predeclared_protocol": {
                "seeds": seeds,
                "training_pairs_per_family": training_pairs_per_family,
                "heldout_pairs_per_generator": heldout_pairs_per_generator,
                "train_families": V3_TRAIN_FAMILIES,
                "primary_within_families": V3_WITHIN_FAMILIES,
                "secondary_ood_families": V3_OOD_FAMILIES,
                "exploration_rate": exploration_rate,
                "learning_rate": learning_rate,
                "appraisal_gain": appraisal_gain,
                "lora_target_modules": ("q_proj", "v_proj"),
                "lora_rank": 2, "lora_alpha": 4,
                "head_l2_penalty": 0.02,
                "max_updates": 48,
                "head_initialization": "deterministic-nonzero-sin-v1",
                "lora_initialization_seed_xor_salt": LORA_INIT_SEED_XOR_SALT,
                "seed_level_primary_metric": "posthoc_exact_expected_stochastic_policy_quality",
                "primary_minimum_absolute_effect": 0.005,
                "global_eval_receipts_withheld_until_all_arm_choices_committed": True,
                "no_eval_training": True,
            },
            "co_primary_contrasts": contrasts,
            "primary_joint_on_minus_off_by_latent_family": family_contrast_gains["joint_on_minus_off"],
            "primary_all_contrasts_by_latent_family": family_contrast_gains,
            "positive_utility_gate_requires_all_three_contrasts_both_latent_families": True,
            "secondary_ood_joint_on_minus_off_by_family": secondary_gains,
            "joint_on_minus_off_total_sampled_action_switches": total_switches,
            "joint_on_minus_off_mean_absolute_behavior_prob_a_shift": (
                absolute_shift_total / len(seeds)
            ),
            "exploratory_positive_local_criterion_met": exploratory_positive,
            "seed_reports": seed_reports,
            "artifact_run_directory": str(final),
        }
        payload = (json.dumps(report, sort_keys=True, separators=(",", ":"),
                              allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")
        path = staging / "study-report.json"
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        complete = staging / "COMPLETE.sha256"
        with complete.open("x", encoding="ascii", newline="\n") as output:
            output.write(sha256(payload).hexdigest() + "  study-report.json\n")
            output.flush()
            os.fsync(output.fileno())
        _atomic_no_replace_directory(staging, final)
        return report
    except Exception as exc:
        # A failed/partial run is inspectable but NEVER receives COMPLETE or
        # the scientifically valid run name. Atomic no-replace publication of
        # its failure receipt prevents accidental promotion of partial EVAL.
        # Do not serialize unsealed EVAL rewards or Python tracebacks.
        failure = {
            "status": "FAILED_INCOMPLETE_NOT_A_VALID_V3_EVALUATION",
            "run_name_reserved_for_success": run_name,
            "failed_seed": active_seed,
            "stage": run_stage,
            "completed_seed_count": len(seed_reports),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1200],
            "pretrained_model_sha256_at_start": original_sha,
            "no_scientific_success_claim": True,
        }
        failed = root / (run_name + ".FAILED-" + sha256(os.urandom(16)).hexdigest()[:16])
        try:
            with (staging / "FAILED.json").open("x", encoding="ascii") as output:
                json.dump(failure, output, sort_keys=True, ensure_ascii=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            _atomic_no_replace_directory(staging, failed)
        except OSError:
            # The original exception is the authoritative failure; a second
            # storage failure must not mask it.
            pass
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
