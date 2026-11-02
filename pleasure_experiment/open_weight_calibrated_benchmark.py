"""Paired exploratory test of a persisted calibrated appraisal head.

The frozen pretrained LM supplies logits and two candidate-conditioned hidden
vectors. A separate child process controls source-local synthetic payoffs;
all EVAL arms commit before it releases *any* EVAL chosen reward. Freshly
imported safetensors head weights, not live TRAIN objects, drive EVAL policy.

This tests causal action effects and reward learning; it does not establish
phenomenology, improve hosted GPT weights, or authenticate an outside oracle.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import importlib
import json
from math import fsum, isfinite, sqrt
import os
from pathlib import Path
import random
import re
import shutil
from statistics import stdev
import tempfile
from typing import Any, Callable

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal import _digest_ids
from pleasure_experiment.open_weight_appraisal_persistence import (
    load_versioned_appraisal_head, save_versioned_appraisal_head,
)
from pleasure_experiment.open_weight_calibrated_appraisal import (
    FEATURE_SCHEMA, TorchCalibratedAppraisal,
)
from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache
from pleasure_experiment.open_weight_stochastic import (
    SeededActionSampler, validate_stochastic_trace,
)
from pleasure_experiment.isolated_outcome_evaluator import (
    IsolatedOutcomeEvaluator, TRAIN_ARM,
)
from pleasure_experiment.outcome_arena import OutcomeReceipt, TaskOffer


ARMS = ("trained_on", "trained_off", "trained_inverted", "shuffled_on", "frozen_on")
EVIDENCE_CLASS = "SOURCE_LOCAL_CALIBRATED_SEALED_PROCESS_APPRAISAL_UNAUTHENTICATED"
SEEDS_20261102 = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157)


def _hash(*items: object) -> str:
    return sha256(json.dumps(items, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _interval(differences: list[float]) -> dict[str, object]:
    n = len(differences)
    mean = fsum(differences) / n
    sigma = stdev(differences) if n > 1 else 0.0
    width = 1.96 * sigma / sqrt(n)
    # The seed, not each episode or matched arm, is the independent unit.
    extreme = 0
    observed = abs(fsum(differences))
    for mask in range(1 << n):
        flipped = fsum(-x if mask & (1 << i) else x
                       for i, x in enumerate(differences))
        if abs(flipped) + 1e-14 >= observed:
            extreme += 1
    return {
        "n_seeds": n, "mean": mean, "seed_sd": sigma,
        "descriptive_normal_95_interval": [mean - width, mean + width],
        "exact_two_sided_paired_signflip_p": extreme / (1 << n),
        "seed_level_differences": differences,
    }


def _check_receipt(receipt: OutcomeReceipt, offer: TaskOffer, trace: Any, *,
                   binding: Any, evaluator_id: str, rubric: str,
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
        receipt.rubric_digest != rubric or
        receipt.evidence_class != evidence_class or
        type(receipt.measured_quality) not in (int, float) or
        not isfinite(receipt.measured_quality) or
        not 0 <= receipt.measured_quality <= 1 or
        type(receipt.evidence_id) is not str or
        re.fullmatch(r"[0-9a-f]{64}", receipt.evidence_id) is None):
        raise RuntimeError("isolated chosen reward provenance mismatch")


def _fresh_head(model: Any, tokens: tuple[int, int], hidden_size: int,
                learning_rate: float, training_pairs: int,
                cache: FrozenActionFeatureCache) -> TorchCalibratedAppraisal:
    return TorchCalibratedAppraisal(
        model, action_token_ids=tokens, hidden_size=hidden_size,
        learning_rate=learning_rate, allow_training=learning_rate > 0,
        max_updates=training_pairs, max_delta_l2=1.0, feature_cache=cache,
    )


def _safe_destination_root(root: str | Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    path = Path(root)
    if (not path.is_absolute() or ".." in path.parts or not path.is_dir() or
        path == source_root or source_root in path.parents):
        raise ValueError("artifact root must be an existing absolute directory outside source")
    for piece in (path,) + tuple(path.parents):
        if piece.is_symlink() or (hasattr(piece, "is_junction") and piece.is_junction()):
            raise ValueError("artifact root may not be linked")
    return path


def _atomic_no_replace_directory(source: Path, target: Path) -> None:
    """Linux renameat2 RENAME_NOREPLACE: never replace a concurrent target."""
    import ctypes
    if target.exists() or target.is_symlink():
        raise FileExistsError("completed study destination already exists")
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("atomic no-replace directory rename unavailable")
    function.argtypes = (ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def _predeclared(seeds: tuple[int, ...], training_pairs: int, heldout: int,
                 epsilon: float, rate: float, gain: float) -> None:
    if (type(seeds) is not tuple or seeds != SEEDS_20261102 or
        type(training_pairs) is not int or training_pairs != 24 or
        type(heldout) is not int or heldout != 6 or
        type(epsilon) not in (float, int) or float(epsilon) != 0.4 or
        type(rate) not in (float, int) or float(rate) != 0.1 or
        type(gain) not in (float, int) or float(gain) != 12.0):
        raise ValueError("this predeclared v2 study fixes seeds/budgets/rates/gain; create a new protocol for variants")


def run_calibrated_appraisal_benchmark(
    loaded: Any, encode_offer: Callable[[TaskOffer], tuple[int, ...]], *,
    seeds: tuple[int, ...] = SEEDS_20261102,
    training_pairs: int = 24, heldout_pairs_per_generator: int = 6,
    exploration_rate: float = 0.4, learning_rate: float = 0.1,
    appraisal_gain: float = 12.0, artifact_root: str | Path,
    evaluator_id: str, allow_appraisal_training: bool = False,
    allow_persistent_export: bool = False,
) -> dict[str, object]:
    """Execute the frozen protocol; publish only a fully complete run directory.

    Source-local study, source-visible generators, gap-filtered offers. The
    returned per-seed rows show every probability, sampled action, receipt and
    posthoc payoff, not merely an aggregate success flag.
    """
    if allow_appraisal_training is not True or allow_persistent_export is not True:
        raise PermissionError("explicit opt-in for head training AND persistence required")
    _predeclared(seeds, training_pairs, heldout_pairs_per_generator,
                 exploration_rate, learning_rate, appraisal_gain)
    if type(evaluator_id) is not str or not 1 <= len(evaluator_id) <= 128:
        raise ValueError("nonempty bounded evaluator label required")
    if not callable(encode_offer):
        raise TypeError("offer tokenizer must be caller-supplied callable")
    root = _safe_destination_root(artifact_root)
    torch = importlib.import_module("torch")
    model, tokens, binding = loaded.model, loaded.action_token_ids, loaded.binding
    if type(tokens) is not tuple or len(tokens) != 2:
        raise ValueError("model requires exactly two single-token actions")
    named_state = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, value) for name, value in model.named_buffers()
    )
    original_model_sha = _named_digest(torch, named_state)
    run_name = "calibrated-v2-" + _hash(
        seeds, training_pairs, heldout_pairs_per_generator, exploration_rate,
        learning_rate, appraisal_gain, binding.checkpoint_sha256, FEATURE_SCHEMA,
    )[:20]
    final = root / run_name
    if final.exists() or final.is_symlink():
        raise FileExistsError("fixed study already completed or destination occupied")
    staging = Path(tempfile.mkdtemp(prefix=".pending-calibrated-v2-", dir=root))
    seed_reports: list[dict[str, object]] = []
    primary_off: list[float] = []
    primary_shuffled: list[float] = []
    all_family_gains: dict[str, list[float]] = {}
    all_family_vs_shuffled: dict[str, list[float]] = {}
    total_switches = 0
    total_shift = 0.0
    last_manifest = None
    try:
        for seed in seeds:
            cache = FrozenActionFeatureCache(
                model, action_token_ids=tokens,
                checkpoint_sha256=binding.checkpoint_sha256,
                tokenizer_sha256=binding.tokenizer_sha256,
                max_entries=96, expected_frozen_model_sha256=original_model_sha,
            )
            genuine = _fresh_head(model, tokens, model.config.hidden_size,
                                  learning_rate, training_pairs, cache)
            shuffled = _fresh_head(model, tokens, model.config.hidden_size,
                                   learning_rate, training_pairs, cache)
            frozen = _fresh_head(model, tokens, model.config.hidden_size,
                                 0.0, training_pairs, cache)
            initial_sha = genuine.head_digest
            if initial_sha != shuffled.head_digest or initial_sha != frozen.head_digest:
                raise RuntimeError("matched calibrated heads did not start identical")

            with IsolatedOutcomeEvaluator(
                seed=seed, evaluator_id=evaluator_id, arms=ARMS,
                training_pairs=training_pairs,
                heldout_pairs_per_generator=heldout_pairs_per_generator,
                suite="novel_v2",
            ) as evaluator:
                manifest = evaluator.manifest
                last_manifest = manifest
                train_sampler = SeededActionSampler(
                    seed=seed ^ 0x72AC2911, exploration_rate=exploration_rate,
                )
                train: list[tuple[TaskOffer, tuple[int, ...], Any, OutcomeReceipt]] = []
                train_rows: list[dict[str, object]] = []
                with cache.verified_epoch() as epoch:
                    for i in range(training_pairs):
                        offer = evaluator.issue(
                            partition="TRAIN", arm=TRAIN_ARM,
                            family_id=manifest.train_family, pair_index=i, side=0,
                        )
                        ids = encode_offer(offer)
                        features = epoch.get_features(
                            ids, partition="TRAIN", episode_id=offer.episode_id,
                        )
                        base = torch.softmax(features.logits, dim=-1)
                        trace = train_sampler.sample(
                            float(base[0]), float(base[1]),
                            input_ids_sha256=_digest_ids(ids), partition="TRAIN",
                            episode_id=offer.episode_id,
                            policy_id="frozen-base-train-v2",
                        )
                        # The genuine preview is strictly before commit/resolution.
                        preview = genuine.score(
                            ids, partition="TRAIN", gain=0.0, defer_choice=True,
                            episode_id=offer.episode_id, features=features,
                        )
                        if (preview.input_ids_sha256 != trace.input_ids_sha256 or
                            abs(preview.prob_a - trace.prob_a) > 2e-6 or
                            abs(preview.prob_b - trace.prob_b) > 2e-6):
                            raise RuntimeError("TRAIN base policy/appraisal mismatch")
                        ticket = evaluator.commit(
                            offer, trace.chosen_action, trace.decision_digest,
                            arm=TRAIN_ARM, model_id=binding.model_id,
                            runtime_instance_id=binding.runtime_instance_id,
                        )
                        receipt = evaluator.resolve(ticket, arm=TRAIN_ARM)
                        _check_receipt(
                            receipt, offer, trace, binding=binding,
                            evaluator_id=evaluator_id,
                            rubric=manifest.train_generator_sha256,
                            evidence_class=manifest.evidence_class,
                        )
                        bound = genuine.bind_choice(preview, trace.chosen_action)
                        updated = genuine.observe_chosen(
                            bound, measured_quality=receipt.measured_quality,
                            evidence_id=receipt.evidence_id,
                        )
                        if updated.after_digest != genuine.head_digest:
                            raise RuntimeError("calibrated TRAIN head update mismatch")
                        train.append((offer, ids, trace, receipt))
                        train_rows.append({
                            "episode_id": offer.episode_id,
                            "features_a": offer.features_a, "features_b": offer.features_b,
                            "sampled_pre_feedback": asdict(trace),
                            "chosen_quality": receipt.measured_quality,
                            "chosen_evidence_id": receipt.evidence_id,
                            "genuine_head_before": updated.before_digest,
                            "genuine_head_after": updated.after_digest,
                            "gradient_norm": updated.gradient_norm,
                        })
                    validate_stochastic_trace(tuple(row[2] for row in train))
                    # Same observed TRAIN decisions; wrong supervision comes only
                    # from an explicitly labelled cyclic derangement of genuinely
                    # committed chosen receipts, not fabricated oracle evidence.
                    offset = 1 + (seed % (training_pairs - 1))
                    outcomes = [row[3].measured_quality for row in train]
                    donors = [(i + offset) % training_pairs
                              for i in range(training_pairs)]
                    if not any(outcomes[i] != outcomes[j] for i, j in enumerate(donors)):
                        raise RuntimeError("shuffled control has no changed teaching targets")
                    for i, (offer, ids, trace, receipt) in enumerate(train):
                        features = epoch.get_features(
                            ids, partition="TRAIN", episode_id=offer.episode_id,
                        )
                        preview = shuffled.score(
                            ids, partition="TRAIN", gain=0.0, defer_choice=True,
                            episode_id=offer.episode_id, features=features,
                        )
                        bound = shuffled.bind_choice(preview, trace.chosen_action)
                        synthetic_id = _hash(
                            "SYNTHETIC_DERANGED_NOT_EVALUATOR_RECEIPT", seed, i,
                            receipt.evidence_id, train[donors[i]][3].evidence_id,
                        )
                        update = shuffled.observe_chosen(
                            bound, measured_quality=outcomes[donors[i]],
                            evidence_id=synthetic_id,
                        )
                        train_rows[i]["synthetic_donor_evidence_id"] = (
                            train[donors[i]][3].evidence_id
                        )
                        train_rows[i]["synthetic_teaching_quality"] = outcomes[donors[i]]
                        train_rows[i]["synthetic_teaching_id"] = synthetic_id
                        train_rows[i]["shuffled_head_after"] = update.after_digest

                    trained_sha, shuffled_sha = genuine.head_digest, shuffled.head_digest
                    if (trained_sha == initial_sha or shuffled_sha == initial_sha or
                        _named_digest(torch, named_state) != original_model_sha):
                        raise RuntimeError("training update missing or pretrained model changed")
                    saved_path = staging / f"seed-{seed}-calibrated-head-v1"
                    artifact = save_versioned_appraisal_head(
                        genuine, binding=binding, snapshot_manifest=loaded.manifest,
                        destination_directory=saved_path,
                        expected_head_sha256=trained_sha, allow_persistent_export=True,
                    )
                    restored = _fresh_head(
                        model, tokens, model.config.hidden_size, 0.0,
                        training_pairs, cache,
                    )
                    imported = load_versioned_appraisal_head(
                        restored, binding=binding, snapshot_manifest=loaded.manifest,
                        source_directory=saved_path, allow_in_place_mutation=True,
                    )
                    if (restored.head_digest != trained_sha or
                        imported.head_weights_sha256 != trained_sha):
                        raise RuntimeError("persisted head failed exact in-run readback")

                    sampler = {arm: SeededActionSampler(
                        seed=seed ^ 0x39B0164F, exploration_rate=exploration_rate,
                    ) for arm in ARMS}
                    pending: dict[str, list[tuple[Any, ...]]] = {arm: [] for arm in ARMS}
                    eval_families = evaluator.eval_families
                    for family_index, family in enumerate(eval_families):
                        for i in range(heldout_pairs_per_generator):
                            offers = {arm: evaluator.issue(
                                partition="EVAL", arm=arm, family_id=family,
                                pair_index=i, side=0,
                            ) for arm in ARMS}
                            offer = offers[ARMS[0]]
                            if any(other != offer for other in offers.values()):
                                raise RuntimeError("matched EVAL offer schedule diverged")
                            ids = encode_offer(offer)
                            features = epoch.get_features(
                                ids, partition="EVAL", episode_id=offer.episode_id,
                            )
                            for arm, head, gain in (
                                ("trained_on", restored, appraisal_gain),
                                ("trained_off", restored, 0.0),
                                ("trained_inverted", restored, -appraisal_gain),
                                ("shuffled_on", shuffled, appraisal_gain),
                                ("frozen_on", frozen, appraisal_gain),
                            ):
                                decision = head.score(
                                    ids, partition="EVAL", gain=gain,
                                    episode_id=offer.episode_id, features=features,
                                )
                                trace = sampler[arm].sample(
                                    decision.prob_a, decision.prob_b,
                                    input_ids_sha256=decision.input_ids_sha256,
                                    partition="EVAL", episode_id=offer.episode_id,
                                    policy_id=arm,
                                )
                                ticket = evaluator.commit(
                                    offers[arm], trace.chosen_action,
                                    trace.decision_digest, arm=arm,
                                    model_id=binding.model_id,
                                    runtime_instance_id=binding.runtime_instance_id,
                                )
                                pending[arm].append((offer, decision, trace,
                                                     ticket, family_index))
                    for arm in ARMS:
                        validate_stochastic_trace(tuple(row[2] for row in pending[arm]))
                    # This is a child-process enforced gate. No EVAL chosen
                    # receipt, let alone counterfactual, existed in parent yet.
                    seal = evaluator.seal_decisions()
                    if (seal.get("eval_commits") !=
                        len(ARMS) * len(eval_families) * heldout_pairs_per_generator):
                        raise RuntimeError("global decision seal count invalid")
                    resolved: dict[str, list[tuple[Any, ...]]] = {arm: [] for arm in ARMS}
                    for arm in ARMS:
                        for offer, decision, trace, ticket, family_index in pending[arm]:
                            receipt = evaluator.resolve(ticket, arm=arm)
                            _check_receipt(
                                receipt, offer, trace, binding=binding,
                                evaluator_id=evaluator_id,
                                rubric=manifest.heldout_generator_sha256[family_index],
                                evidence_class=manifest.evidence_class,
                            )
                            resolved[arm].append((offer, decision, trace, receipt))
                    final_seal = evaluator.finalize_evaluation()
                    if final_seal.get("sealed") is not True:
                        raise RuntimeError("global reward posthoc gate did not seal")
                    eval_results: dict[str, dict[str, Any]] = {}
                    for arm in ARMS:
                        rows = []
                        for offer, decision, trace, receipt in resolved[arm]:
                            oracle = evaluator.posthoc_evaluate(offer, arm=arm)
                            if (oracle.chosen_action != trace.chosen_action or
                                oracle.chosen_quality != receipt.measured_quality):
                                raise RuntimeError("posthoc oracle/committed reward differs")
                            expected = (trace.behavior_prob_a * oracle.quality_a +
                                        trace.behavior_prob_b * oracle.quality_b)
                            rows.append({
                                "episode_id": offer.episode_id,
                                "family_id": offer.family_id,
                                "head_sha256": decision.head_digest,
                                "prediction_a": decision.predicted_quality_a,
                                "prediction_b": decision.predicted_quality_b,
                                "sampled_pre_feedback": asdict(trace),
                                "chosen_quality": receipt.measured_quality,
                                "chosen_evidence_id": receipt.evidence_id,
                                "expected_policy_quality_posthoc": expected,
                                "quality_a_posthoc": oracle.quality_a,
                                "quality_b_posthoc": oracle.quality_b,
                                "oracle_sha256": oracle.generator_sha256,
                            })
                        eval_results[arm] = {
                            "mean_expected_quality": fsum(
                                row["expected_policy_quality_posthoc"] for row in rows
                            ) / len(rows),
                            "mean_realized_quality": fsum(
                                row["chosen_quality"] for row in rows
                            ) / len(rows),
                            "by_family": {family: {
                                "expected_quality": fsum(
                                    row["expected_policy_quality_posthoc"] for row in rows
                                    if row["family_id"] == family
                                ) / heldout_pairs_per_generator,
                                "realized_quality": fsum(
                                    row["chosen_quality"] for row in rows
                                    if row["family_id"] == family
                                ) / heldout_pairs_per_generator,
                            } for family in eval_families},
                            "rows": rows,
                        }
                    on, off = (eval_results["trained_on"], eval_results["trained_off"])
                    for arm in ARMS:
                        for x, y in zip(on["rows"], eval_results[arm]["rows"]):
                            a, b = (x["sampled_pre_feedback"], y["sampled_pre_feedback"])
                            if (x["episode_id"] != y["episode_id"] or
                                a["random_draw"] != b["random_draw"] or
                                a["draw_index"] != b["draw_index"]):
                                raise RuntimeError("matched EVAL common random draws diverged")
                    for a, b in zip(off["rows"], eval_results["frozen_on"]["rows"]):
                        if (abs(a["sampled_pre_feedback"]["behavior_prob_a"] -
                                b["sampled_pre_feedback"]["behavior_prob_a"]) > 1e-7):
                            raise RuntimeError("frozen and disabled policy mismatch")
                    switches = sum(
                        x["sampled_pre_feedback"]["chosen_action"] !=
                        y["sampled_pre_feedback"]["chosen_action"]
                        for x, y in zip(on["rows"], off["rows"])
                    )
                    shift = fsum(
                        abs(x["sampled_pre_feedback"]["behavior_prob_a"] -
                            y["sampled_pre_feedback"]["behavior_prob_a"])
                        for x, y in zip(on["rows"], off["rows"])
                    ) / len(on["rows"])
                    total_switches += switches
                    total_shift += shift
                    primary = evaluator.primary_ood_families
                    def primary_quality(arm: str) -> float:
                        return fsum(eval_results[arm]["by_family"][f]["expected_quality"]
                                    for f in primary) / len(primary)
                    primary_off.append(primary_quality("trained_on") -
                                       primary_quality("trained_off"))
                    primary_shuffled.append(primary_quality("trained_on") -
                                            primary_quality("shuffled_on"))
                    for family in eval_families:
                        all_family_gains.setdefault(family, []).append(
                            on["by_family"][family]["expected_quality"] -
                            off["by_family"][family]["expected_quality"]
                        )
                        all_family_vs_shuffled.setdefault(family, []).append(
                            on["by_family"][family]["expected_quality"] -
                            eval_results["shuffled_on"]["by_family"][family]["expected_quality"]
                        )
                    if (restored.head_digest != trained_sha or
                        _named_digest(torch, named_state) != original_model_sha):
                        raise RuntimeError("EVAL changed the head or frozen LM")
                    seed_reports.append({
                        "seed": seed, "evaluator_worker_pid": evaluator.worker_pid,
                        "initial_head_sha256": initial_sha,
                        "trained_head_sha256": trained_sha,
                        "shuffled_head_sha256": shuffled_sha,
                        "restored_head_sha256": restored.head_digest,
                        "saved_head_manifest_sha256": artifact.manifest_sha256,
                        "saved_head_safetensors_sha256": artifact.safetensors_sha256,
                        "saved_head_directory": str(final / saved_path.name),
                        "train_updates": genuine.updates,
                        "shuffled_updates": shuffled.updates,
                        "features_forward_misses": cache.misses,
                        "features_forward_hits": cache.hits,
                        "sampled_action_switches_on_vs_off": switches,
                        "mean_absolute_behavior_probability_shift": shift,
                        "trained_prediction_spread_mean_abs": fsum(
                            abs(row["prediction_a"] - row["prediction_b"])
                            for row in on["rows"]
                        ) / len(on["rows"]),
                        "globally_sealed_before_eval_feedback": True,
                        "family_manifest": asdict(manifest),
                        "train": train_rows, "eval_arms": eval_results,
                    })
            if _named_digest(torch, named_state) != original_model_sha:
                raise RuntimeError("frozen pretrained model changed between seeds")
        interval_off = _interval(primary_off)
        interval_shuffle = _interval(primary_shuffled)
        means = {family: fsum(values) / len(values)
                 for family, values in all_family_gains.items()}
        shuffle_means = {family: fsum(values) / len(values)
                         for family, values in all_family_vs_shuffled.items()}
        # Worker declares its opaque transfer suite (01..03); 00 is only the
        # in-family diagnostic, never eligible for the primary effect.
        ood = tuple(last_manifest.heldout_families[1:])
        exploratory_positive = bool(
            interval_off["mean"] > 0.005 and interval_shuffle["mean"] > 0.005
            and interval_off["descriptive_normal_95_interval"][0] > 0
            and interval_shuffle["descriptive_normal_95_interval"][0] > 0
            and all(means[family] > 0 for family in ood)
        )
        result: dict[str, object] = {
            "evidence_class": EVIDENCE_CLASS,
            "source_local_generator_not_secret": True,
            "external_evaluator_authenticated": False,
            "source_visible_gap_filtered_payoff_task_distribution": True,
            "subjective_pleasure_proven": False,
            "hosted_gpt_model_modified": False,
            "pretrained_model_unchanged": True,
            "fresh_process_import_verified_by_this_run": False,
            "in_process_saved_head_import_verified": True,
            "head_feature_schema": FEATURE_SCHEMA,
            "frozen_pretrained_model_sha256": original_model_sha,
            "model_id": binding.model_id,
            "model_revision": binding.model_revision,
            "predeclared_protocol": {
                "seeds": seeds, "training_pairs_per_seed": training_pairs,
                "heldout_pairs_per_family_per_seed": heldout_pairs_per_generator,
                "suite": "novel_v2", "training_generator": last_manifest.train_family,
                "primary_ood_generators": ood,
                "positive_control_generator": last_manifest.heldout_families[0],
                "exploration_rate": exploration_rate,
                "learning_rate": learning_rate,
                "appraisal_gain": appraisal_gain,
                "primary_effect_threshold_absolute": 0.005,
                "global_all_arm_seal_before_any_eval_reward": True,
                "statistical_unit": "seed",
            },
            "primary_ood_trained_on_minus_off": interval_off,
            "primary_ood_trained_on_minus_shuffled": interval_shuffle,
            "on_minus_off_by_family": means,
            "on_minus_shuffled_by_family": shuffle_means,
            "total_on_vs_off_sampled_action_switches": total_switches,
            "mean_absolute_on_vs_off_behavior_prob_a_shift": total_shift / len(seeds),
            "exploratory_positive_local_criterion_met": exploratory_positive,
            "seed_reports": seed_reports,
            "artifact_run_directory": str(final),
        }
        # Commit a *single* complete study directory containing all seed heads
        # and the full evidence report. No per-seed final-dir promotion is used.
        report_bytes = (json.dumps(result, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        report_path = staging / "study-report.json"
        with report_path.open("xb") as out:
            out.write(report_bytes)
            out.flush()
            os.fsync(out.fileno())
        report_sha = sha256(report_bytes).hexdigest()
        complete_path = staging / "COMPLETE.sha256"
        with complete_path.open("x", encoding="ascii", newline="\n") as out:
            out.write(report_sha + "  study-report.json\n")
            out.flush()
            os.fsync(out.fileno())
        _atomic_no_replace_directory(staging, final)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
