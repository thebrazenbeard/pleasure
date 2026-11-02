"""Matched stochastic contextual-bandit appraisal study on withheld payoff rules.

The pretrained language model is read-only. TRAIN records sampled A/B actions
with their *actual exploration propensities* before chosen feedback. Genuine
and deranged-label heads see identical training actions and inputs. EVAL
includes structurally different payoff generators; their counterfactuals
become available only after the arms' evaluation decisions have been sealed.

This is a source-local unauthenticated simulator, not an independent blind
evaluation or evidence of subjective experience. The primary outcome is
paired expected policy value on withheld generators, not a trained-tensor SHA.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import importlib
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
import re
import shutil
import tempfile
import weakref
from statistics import stdev
from typing import Any, Callable

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal, _digest_ids
from pleasure_experiment.open_weight_appraisal_persistence import (
    load_versioned_appraisal_head, save_versioned_appraisal_head,
)
from pleasure_experiment.open_weight_stochastic import (
    SeededActionSampler, validate_stochastic_trace,
)
from pleasure_experiment.outcome_arena import OutcomeReceipt, TaskOffer
from pleasure_experiment.withheld_payoffs import (
    EVAL_FAMILIES, TRAIN_FAMILY, WithheldPayoffArena,
    EVIDENCE_CLASS as WITHHELD_EVIDENCE_CLASS,
)


ARMS = ("trained_on", "trained_off", "trained_inverted", "shuffled_on", "frozen_on")
EVIDENCE_CLASS = "SOURCE_LOCAL_STOCHASTIC_WITHHELD_APPRAISAL_UNAUTHENTICATED"


class _PrivateStudyStaging:
    """Remove unvalidated staging on normal completion or failed execution."""

    def __init__(self, root: Path):
        self.name = tempfile.mkdtemp(prefix=".pending-stochastic-appraisal-", dir=root)
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.name,
                                           ignore_errors=True)

    def cleanup(self) -> None:
        self._finalizer()


def _sha(*items: object) -> str:
    return sha256(json.dumps(items, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("ascii")).hexdigest()


def _paired_interval(differences: list[float]) -> dict[str, object]:
    n = len(differences)
    mean = fsum(differences) / n
    if n < 2:
        return {"n_seeds": n, "mean": mean, "sd": None,
                "descriptive_normal_95_interval": None}
    sigma = stdev(differences)
    width = 1.96 * sigma / sqrt(n)
    return {"n_seeds": n, "mean": mean, "sd": sigma,
            "descriptive_normal_95_interval": [mean - width, mean + width]}


def _receipt_check(
    receipt: OutcomeReceipt, offer: TaskOffer, trace: Any,
    *, model_id: str, runtime_instance_id: str,
    expected_evaluator_id: str, expected_rule_digest: str,
) -> None:
    if (
        type(receipt) is not OutcomeReceipt or
        receipt.task_id != offer.task_id or
        receipt.episode_id != offer.episode_id or
        receipt.partition != offer.partition or
        receipt.family_id != offer.family_id or
        receipt.action != trace.chosen_action or
        receipt.decision_digest != trace.decision_digest or
        receipt.model_id != model_id or
        receipt.runtime_instance_id != runtime_instance_id or
        receipt.evaluator_id != expected_evaluator_id or
        receipt.rubric_digest != expected_rule_digest or
        receipt.evidence_class != WITHHELD_EVIDENCE_CLASS or
        type(receipt.measured_quality) not in (float, int) or
        not isfinite(receipt.measured_quality) or
        not 0 <= receipt.measured_quality <= 1 or
        type(receipt.evidence_id) is not str or
        re.fullmatch(r"[0-9a-f]{64}", receipt.evidence_id) is None
    ):
        raise RuntimeError("committed chosen outcome failed full provenance checks")


def _base_probabilities(model: Any, torch: Any, ids: tuple[int, ...],
                        tokens: tuple[int, int]) -> tuple[float, float]:
    parameters = tuple(model.parameters())
    if not parameters or len({p.device for p in parameters}) != 1:
        raise ValueError("model must be local and on one device")
    original_modes = tuple((part, part.training) for part in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], dtype=torch.long,
                                                   device=parameters[0].device)).logits
            relevant = logits[0, -1, list(tokens)]
            probabilities = torch.softmax(relevant.float(), dim=-1)
            return (float(probabilities[0]), float(probabilities[1]))
    finally:
        for part, original in original_modes:
            part.training = original


def _probe(model: Any, tokens: tuple[int, int], hidden_size: int,
           rate: float, train_pairs: int) -> TorchHiddenAppraisal:
    return TorchHiddenAppraisal(
        model, action_token_ids=tokens, hidden_size=hidden_size,
        learning_rate=rate, allow_training=rate > 0,
        max_updates=train_pairs, max_delta_l2=1.0,
    )


def run_stochastic_appraisal_benchmark(
    loaded: Any, encode_offer: Callable[[TaskOffer], tuple[int, ...]], *,
    seeds: tuple[int, ...], training_pairs: int, heldout_pairs_per_generator: int,
    exploration_rate: float, learning_rate: float, appraisal_gain: float,
    artifact_root: str | Path, evaluator_id: str,
    allow_appraisal_training: bool = False,
    allow_persistent_export: bool = False,
) -> dict[str, object]:
    """Execute a predeclared, matched heldout study and save each learned head.

    The returned record is deliberately verbose: every TRAIN probability,
    sampled action, receipt identity, and genuine/synthetic teaching target is
    independently inspectable. The predeclared parameters are never optimized
    against EVAL payoffs. EVAL gets posthoc counterfactuals only after sealing.
    """
    if allow_appraisal_training is not True or allow_persistent_export is not True:
        raise PermissionError("explicit training AND persistent export gates required")
    if (type(seeds) is not tuple or not 2 <= len(seeds) <= 32 or
        len(set(seeds)) != len(seeds) or
        any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("seeds must be 2..32 distinct uint32 values")
    if type(training_pairs) is not int or not 4 <= training_pairs <= 64:
        raise ValueError("training pairs must be between 4 and 64")
    if (type(heldout_pairs_per_generator) is not int or
        not 2 <= heldout_pairs_per_generator <= 64):
        raise ValueError("heldout pairs per generator must be between 2 and 64")
    for name, value, low, high in (
        ("exploration_rate", exploration_rate, 0.01, 1.0),
        ("learning_rate", learning_rate, 1e-8, 1.0),
        ("appraisal_gain", appraisal_gain, 1e-8, 16.0),
    ):
        if type(value) not in (int, float) or not isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite in [{low},{high}]")
    root = Path(artifact_root)
    source_root = Path(__file__).resolve().parents[1]
    if (not root.is_absolute() or ".." in root.parts or not root.is_dir() or
        root == source_root or source_root in root.parents):
        raise ValueError("artifact root must be an existing absolute directory outside source")
    for piece in (root,) + tuple(root.parents):
        if piece.is_symlink() or (hasattr(piece, "is_junction") and piece.is_junction()):
            raise ValueError("artifact root may not be linked")
    if any((root / f"seed-{seed}-trained-head-v1").exists() for seed in seeds):
        raise FileExistsError("a final head artifact already exists; no overwrite allowed")
    if type(evaluator_id) is not str or not evaluator_id:
        raise ValueError("nonempty source-local evaluator label required")
    torch = importlib.import_module("torch")
    model = loaded.model
    binding = loaded.binding
    tokens = loaded.action_token_ids
    if type(tokens) is not tuple or len(tokens) != 2:
        raise ValueError("two single-token actions required")
    hidden_size = model.config.hidden_size
    named_state = tuple(model.named_parameters()) + tuple(
        ("buffer:" + name, buf) for name, buf in model.named_buffers()
    )
    original_model_digest = _named_digest(torch, named_state)
    report_seeds: list[dict[str, object]] = []
    per_seed_contrasts = {name: [] for name in (
        "on_minus_off_expected", "on_minus_shuffled_expected",
        "on_minus_frozen_expected", "on_minus_off_realized",
    )}
    # Never publish a final learned-head directory before all five EVAL arms
    # and all seeds have completed the matched-control checks. A private
    # GC/exception unwinding removes unvalidated artifacts on failed runs.
    provisional = _PrivateStudyStaging(root)
    provisional_root = Path(provisional.name)
    for seed in seeds:
        torch.manual_seed(seed)
        genuine = _probe(model, tokens, hidden_size, learning_rate, training_pairs)
        shuffled = _probe(model, tokens, hidden_size, learning_rate, training_pairs)
        frozen = _probe(model, tokens, hidden_size, 0.0, training_pairs)
        initial_head = genuine.head_digest
        if not initial_head == shuffled.head_digest == frozen.head_digest:
            raise RuntimeError("matched appraisal arms did not start from equal tensors")
        manifest_arena = WithheldPayoffArena(seed=seed, evaluator_id=evaluator_id)
        generator_manifest = manifest_arena.manifest
        train_sampler = SeededActionSampler(
            seed=seed ^ 0x72AC2911, exploration_rate=exploration_rate,
        )
        training: list[tuple[TaskOffer, tuple[int, ...], Any, OutcomeReceipt]] = []
        for i in range(training_pairs):
            offer = manifest_arena.issue(
                partition="TRAIN", family_id=TRAIN_FAMILY, pair_index=i, side=0,
            )
            ids = encode_offer(offer)
            if type(ids) is not tuple or not ids:
                raise ValueError("encoded task must be nonempty tuple of tokens")
            pa, pb = _base_probabilities(model, torch, ids, tokens)
            trace = train_sampler.sample(
                pa, pb, input_ids_sha256=_digest_ids(ids), partition="TRAIN",
                episode_id=offer.episode_id, policy_id="frozen-base-train",
            )
            ticket = manifest_arena.commit(
                offer, trace.chosen_action, trace.decision_digest,
                model_id=binding.model_id,
                runtime_instance_id=binding.runtime_instance_id,
            )
            receipt = manifest_arena.resolve(ticket)
            _receipt_check(
                receipt, offer, trace, model_id=binding.model_id,
                runtime_instance_id=binding.runtime_instance_id,
                expected_evaluator_id=evaluator_id,
                expected_rule_digest=generator_manifest.train_generator_sha256,
            )
            training.append((offer, ids, trace, receipt))
        validate_stochastic_trace(tuple(row[2] for row in training))
        # Derangement of already committed chosen outcomes. There is NEVER a
        # fabricated complete OutcomeReceipt with a different teaching label.
        offset = 1 + (seed % (training_pairs - 1))
        targets = [row[3].measured_quality for row in training]
        donor_indices = [(i + offset) % training_pairs for i in range(training_pairs)]
        if not any(targets[i] != targets[j] for i, j in enumerate(donor_indices)):
            raise RuntimeError("deranged-feedback control has no differing labels")
        train_rows: list[dict[str, object]] = []
        for i, (offer, ids, trace, receipt) in enumerate(training):
            updates: dict[str, object] = {}
            for arm, head, value, evidence_id in (
                ("genuine", genuine, receipt.measured_quality, receipt.evidence_id),
                ("shuffled", shuffled, targets[donor_indices[i]],
                 _sha("SYNTHETIC_DERANGED_FEEDBACK_NOT_OUTCOME", seed, i,
                      receipt.evidence_id, training[donor_indices[i]][3].evidence_id)),
            ):
                preview = head.score(ids, partition="TRAIN", gain=0.0,
                                     defer_choice=True)
                if (preview.input_ids_sha256 != trace.input_ids_sha256 or
                    abs(preview.prob_a - trace.prob_a) > 2e-6 or
                    abs(preview.prob_b - trace.prob_b) > 2e-6):
                    raise RuntimeError("training appraisal/base policy inputs diverged")
                bound = head.bind_choice(preview, trace.chosen_action)
                if not bound.choice_bound or bound.chosen_action != receipt.action:
                    raise RuntimeError("sampled choice was not bound to chosen features")
                update = head.observe_chosen(
                    bound, measured_quality=value, evidence_id=evidence_id,
                )
                if (update.evidence_id != evidence_id or
                    update.measured_chosen_quality != value or
                    update.after_digest != head.head_digest):
                    raise RuntimeError("chosen-only head update readback mismatch")
                updates[arm] = {"teaching_quality": value,
                                "teaching_evidence_id": evidence_id,
                                "head_before": update.before_digest,
                                "head_after": update.after_digest,
                                "optimizer_step_executed": update.optimizer_step_executed}
            train_rows.append({
                "episode_id": offer.episode_id, "generator_id": offer.family_id,
                "features_a": offer.features_a, "features_b": offer.features_b,
                "stochastic_pre_feedback": asdict(trace),
                "committed_chosen_evidence_id": receipt.evidence_id,
                "actual_chosen_quality": receipt.measured_quality,
                "synthetic_donor_evidence_id": training[donor_indices[i]][3].evidence_id,
                "updates": updates,
            })
        trained_sha = genuine.head_digest
        shuffled_sha = shuffled.head_digest
        if trained_sha == initial_head or shuffled_sha == initial_head:
            raise RuntimeError("head did not change despite permitted SGD updates")
        if _named_digest(torch, named_state) != original_model_digest:
            raise RuntimeError("appraisal training changed the frozen language model")

        destination = provisional_root / f"seed-{seed}-trained-head-v1"
        artifact = save_versioned_appraisal_head(
            genuine, binding=binding, snapshot_manifest=loaded.manifest,
            destination_directory=destination, expected_head_sha256=trained_sha,
            allow_persistent_export=True,
        )
        # EVAL uses a newly created head populated from the SAVED ARTIFACT,
        # not merely the still-live training head.
        restored = _probe(model, tokens, hidden_size, 0.0, training_pairs)
        imported = load_versioned_appraisal_head(
            restored, binding=binding, snapshot_manifest=loaded.manifest,
            source_directory=destination, allow_in_place_mutation=True,
        )
        if restored.head_digest != trained_sha or imported.head_weights_sha256 != trained_sha:
            raise RuntimeError("persisted appraisal head failed exact tensor reimport")
        results: dict[str, dict[str, object]] = {}
        all_offers: dict[str, tuple[TaskOffer, ...]] = {}
        sealed_evaluators: dict[str, WithheldPayoffArena] = {}
        arm_pending: dict[str, list[tuple[Any, ...]]] = {}
        for arm, head, gain in (
            ("trained_on", restored, appraisal_gain),
            ("trained_off", restored, 0.0),
            ("trained_inverted", restored, -appraisal_gain),
            ("shuffled_on", shuffled, appraisal_gain),
            ("frozen_on", frozen, appraisal_gain),
        ):
            evaluator = WithheldPayoffArena(seed=seed, evaluator_id=evaluator_id)
            sampler = SeededActionSampler(
                seed=seed ^ 0x39B0164F, exploration_rate=exploration_rate,
            )
            pending = []
            for family_index, family in enumerate(EVAL_FAMILIES):
                for i in range(heldout_pairs_per_generator):
                    offer = evaluator.issue(
                        partition="EVAL", family_id=family, pair_index=i, side=0,
                    )
                    ids = encode_offer(offer)
                    decision = head.score(ids, partition="EVAL", gain=gain)
                    trace = sampler.sample(
                        decision.prob_a, decision.prob_b,
                        input_ids_sha256=decision.input_ids_sha256,
                        partition="EVAL", episode_id=offer.episode_id,
                        policy_id=arm,
                    )
                    ticket = evaluator.commit(
                        offer, trace.chosen_action, trace.decision_digest,
                        model_id=binding.model_id,
                        runtime_instance_id=binding.runtime_instance_id,
                    )
                    # EVAL reward stays inside the oracle until every arm
                    # has committed. Otherwise different arms might expose
                    # both q(A) and q(B) before later policy choices.
                    pending.append((offer, decision, trace, ticket, family_index))
            validate_stochastic_trace(tuple(item[2] for item in pending))
            offers = tuple(item[0] for item in pending)
            if arm != ARMS[0] and offers != all_offers[ARMS[0]]:
                raise RuntimeError("evaluation offers differed between arms")
            all_offers[arm] = offers
            sealed_evaluators[arm] = evaluator
            arm_pending[arm] = pending

        # Neither *chosen* EVAL rewards nor counterfactuals may be revealed
        # until every matched arm has committed every EVAL offer. In the
        # previous runner chosen EVAL feedback was resolved per arm, which
        # could reconstruct both q(A) and q(B) across matched arms early.
        if set(arm_pending) != set(ARMS):
            raise RuntimeError("not all predeclared evaluation arms committed")
        for arm in ARMS:
            evaluator = sealed_evaluators[arm]
            resolved_rows = []
            for offer, decision, trace, ticket, family_index in arm_pending[arm]:
                receipt = evaluator.resolve(ticket)
                _receipt_check(
                    receipt, offer, trace, model_id=binding.model_id,
                    runtime_instance_id=binding.runtime_instance_id,
                    expected_evaluator_id=evaluator_id,
                    expected_rule_digest=(
                        generator_manifest.heldout_generator_sha256[family_index]
                    ),
                )
                resolved_rows.append((offer, decision, trace, receipt))
            arm_pending[arm] = resolved_rows
        for evaluator in sealed_evaluators.values():
            evaluator.finalize_evaluation()
        for arm in ARMS:
            evaluator = sealed_evaluators[arm]
            pending = arm_pending[arm]
            rows = []
            for offer, decision, trace, receipt in pending:
                oracle = evaluator.posthoc_evaluate(offer)
                if (oracle.chosen_action != trace.chosen_action or
                    oracle.chosen_quality != receipt.measured_quality):
                    raise RuntimeError("sealed posthoc score differs from committed reward")
                expected = (trace.behavior_prob_a * oracle.quality_a +
                            trace.behavior_prob_b * oracle.quality_b)
                rows.append({
                    "episode_id": offer.episode_id, "generator_id": offer.family_id,
                    "decision_head_sha256": decision.head_digest,
                    "predicted_quality_a": decision.predicted_quality_a,
                    "predicted_quality_b": decision.predicted_quality_b,
                    "stochastic_pre_feedback": asdict(trace),
                    "committed_evidence_id": receipt.evidence_id,
                    "realized_quality": receipt.measured_quality,
                    "expected_policy_quality_posthoc": expected,
                    "quality_a_posthoc": oracle.quality_a,
                    "quality_b_posthoc": oracle.quality_b,
                    "oracle_regret_posthoc": oracle.oracle_quality - expected,
                    "oracle_generator_sha256": oracle.generator_sha256,
                })
            results[arm] = {
                "mean_expected_quality": fsum(
                    row["expected_policy_quality_posthoc"] for row in rows
                ) / len(rows),
                "mean_realized_quality": fsum(
                    row["realized_quality"] for row in rows
                ) / len(rows),
                "by_family": {family: {
                    "expected_quality": fsum(
                        row["expected_policy_quality_posthoc"] for row in rows
                        if row["generator_id"] == family
                    ) / heldout_pairs_per_generator,
                    "realized_quality": fsum(
                        row["realized_quality"] for row in rows
                        if row["generator_id"] == family
                    ) / heldout_pairs_per_generator,
                } for family in EVAL_FAMILIES},
                "rows": rows,
            }
        on_rows = results["trained_on"]["rows"]
        off_rows = results["trained_off"]["rows"]
        frozen_rows = results["frozen_on"]["rows"]
        for arm in ARMS:
            for on_row, other_row in zip(on_rows, results[arm]["rows"]):
                left = on_row["stochastic_pre_feedback"]
                right = other_row["stochastic_pre_feedback"]
                if (on_row["episode_id"] != other_row["episode_id"] or
                    left["random_draw"] != right["random_draw"] or
                    left["draw_index"] != right["draw_index"]):
                    raise RuntimeError("matched EVAL arms lost common random numbers")
        for off_row, frozen_row in zip(off_rows, frozen_rows):
            a, b = (off_row["stochastic_pre_feedback"],
                    frozen_row["stochastic_pre_feedback"])
            if (abs(a["behavior_prob_a"] - b["behavior_prob_a"]) > 1e-7 or
                abs(a["behavior_prob_b"] - b["behavior_prob_b"]) > 1e-7):
                raise RuntimeError("frozen-head and disabled-head policies diverged")
        action_switches = sum(
            row["stochastic_pre_feedback"]["chosen_action"] !=
            off_row["stochastic_pre_feedback"]["chosen_action"]
            for row, off_row in zip(on_rows, off_rows)
        )
        absolute_action_probability_shift = fsum(
            abs(row["stochastic_pre_feedback"]["behavior_prob_a"] -
                off_row["stochastic_pre_feedback"]["behavior_prob_a"])
            for row, off_row in zip(on_rows, off_rows)
        ) / len(on_rows)
        expected_on = results["trained_on"]["mean_expected_quality"]
        for key, comparator, metric in (
            ("on_minus_off_expected", "trained_off", "mean_expected_quality"),
            ("on_minus_shuffled_expected", "shuffled_on", "mean_expected_quality"),
            ("on_minus_frozen_expected", "frozen_on", "mean_expected_quality"),
            ("on_minus_off_realized", "trained_off", "mean_realized_quality"),
        ):
            main = expected_on if metric == "mean_expected_quality" else results["trained_on"][metric]
            per_seed_contrasts[key].append(main - results[comparator][metric])
        if restored.head_digest != trained_sha or _named_digest(torch, named_state) != original_model_digest:
            raise RuntimeError("EVAL changed learned head or frozen pretrained language model")
        report_seeds.append({
            "seed": seed, "initial_head_sha256": initial_head,
            "learned_head_sha256": trained_sha,
            "shuffled_head_sha256": shuffled_sha,
            "persisted_head_manifest_sha256": artifact.manifest_sha256,
            "persisted_head_safetensors_sha256": artifact.safetensors_sha256,
            "saved_head_directory": artifact.artifact_directory,
            "restored_head_sha256": restored.head_digest,
            "persistence_in_process_verified": True,
            "trained_on_vs_off_sampled_action_switches": action_switches,
            "trained_on_vs_off_mean_absolute_behavior_prob_a_shift": (
                absolute_action_probability_shift
            ),
            "common_eval_random_numbers_verified": True,
            "frozen_head_equals_disabled_behavior_verified": True,
            "generator_manifest": asdict(generator_manifest),
            "train": train_rows, "eval_arms": results,
        })
    if _named_digest(torch, named_state) != original_model_digest:
        raise RuntimeError("frozen model changed across seeds")
    contrasts = {key: _paired_interval(values) for key, values in per_seed_contrasts.items()}
    all_family_gains = {
        family: fsum(
            row["eval_arms"]["trained_on"]["by_family"][family]["expected_quality"] -
            row["eval_arms"]["trained_off"]["by_family"][family]["expected_quality"]
            for row in report_seeds
        ) / len(report_seeds) for family in EVAL_FAMILIES
    }
    family_vs_shuffled = {
        family: fsum(
            row["eval_arms"]["trained_on"]["by_family"][family]["expected_quality"] -
            row["eval_arms"]["shuffled_on"]["by_family"][family]["expected_quality"]
            for row in report_seeds
        ) / len(report_seeds) for family in EVAL_FAMILIES
    }
    # Exploratory improvement gate fixed in source for this run: seed-paired
    # normal interval > 0 against BOTH OFF and shuffled, with positive mean
    # on every withheld generator. Descriptive CI is not a p-value or proof.
    lower_off = contrasts["on_minus_off_expected"]["descriptive_normal_95_interval"]
    lower_shuffle = contrasts["on_minus_shuffled_expected"]["descriptive_normal_95_interval"]
    exploratory_positive = bool(
        lower_off is not None and lower_shuffle is not None and
        lower_off[0] > 0 and lower_shuffle[0] > 0 and
        all(gain > 0 for gain in all_family_gains.values())
    )
    # Promote final paths only after every scientific control and exact
    # tensor readback. The rename is same-filesystem and rejects overwrites;
    # recover already-promoted directories if a later rename unexpectedly
    # fails. Every manifest remains labeled not-study-validated on its own.
    promotions: list[tuple[Path, Path]] = []
    try:
        for row in report_seeds:
            source = Path(row["saved_head_directory"])
            target = root / source.name
            if target.exists() or target.is_symlink():
                raise FileExistsError("final appraisal destination appeared during run")
            source.rename(target)
            promotions.append((source, target))
        for row, (_, target) in zip(report_seeds, promotions):
            row["saved_head_directory"] = str(target)
    except Exception:
        for source, target in reversed(promotions):
            target.rename(source)
        raise
    finally:
        provisional.cleanup()
    return {
        "evidence_class": EVIDENCE_CLASS,
        "model_id_declared": binding.model_id,
        "model_revision_declared": binding.model_revision,
        "frozen_pretrained_model_sha256": original_model_digest,
        "predeclared_protocol": {
            "seeds": seeds, "training_pairs_per_seed": training_pairs,
            "heldout_primary_pairs_per_generator_per_seed": heldout_pairs_per_generator,
            "train_generator": TRAIN_FAMILY,
            "withheld_generator_families": EVAL_FAMILIES,
            "exploration_rate": exploration_rate,
            "learning_rate": learning_rate, "appraisal_gain": appraisal_gain,
            "feedback_design": "shared sampled chosen-only TRAIN actions; deranged synthetic targets",
            "primary_metric": "seed-paired_posthoc_exact_expected_behavior_policy_quality",
            "posthoc_counterfactual_only_after_eval_seal": True,
        },
        "contrasts": contrasts,
        "trained_on_minus_off_by_withheld_generator": all_family_gains,
        "trained_on_minus_shuffled_by_withheld_generator": family_vs_shuffled,
        "total_trained_on_vs_off_sampled_action_switches": sum(
            row["trained_on_vs_off_sampled_action_switches"] for row in report_seeds
        ),
        "mean_absolute_trained_on_vs_off_behavior_prob_a_shift": fsum(
            row["trained_on_vs_off_mean_absolute_behavior_prob_a_shift"]
            for row in report_seeds
        ) / len(report_seeds),
        "exploratory_positive_local_criterion_met": exploratory_positive,
        "cross_process_import_verified_by_this_run": False,
        "offline_pretrained_model_unchanged": True,
        "source_local_generator_not_secret": True,
        "external_evaluator_authenticated": False,
        "subjective_pleasure_proven": False,
        "seed_reports": report_seeds,
    }
