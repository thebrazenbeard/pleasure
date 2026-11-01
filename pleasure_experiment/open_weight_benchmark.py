"""Multi-seed, matched open-weight outcome-learning versus adverse controls.

All outcomes are source-local simulated measurements. TRAIN and EVAL have
different family IDs and episode IDs; the EVAL family is never used for SGD.
The shuffled arm is an explicit intervention: it receives synthetic teaching
values permuted from *frozen baseline* chosen outcomes, while the actual
committed chosen outcome stays intact in the separate observation record.

This module does not load a model, verify its publisher, or save any weights.
An authorized caller supplies exclusive fresh adapters with exact baselines
and can capture the learned adapter before this module restores each arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite, sqrt
from statistics import stdev
from typing import Callable
import json

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, OpenWeightOutcome, OpenWeightRestore, OpenWeightUpdate,
    TorchChosenOutcomeAdapter,
    EVIDENCE_CLASS as ADAPTER_EVIDENCE_CLASS,
    PHENOMENOLOGY as ADAPTER_PHENOMENOLOGY,
)
from pleasure_experiment.open_weight_controls import _readback
from pleasure_experiment.open_weight_trial import (
    ActionModelAdapter, MAX_HELDOUT_PAIRS, MAX_TRAINING_PAIRS,
    _check_choice, _check_outcome, _chosen_probability, _input_digest,
    _schedule_digest, _token_ids,
)
from pleasure_experiment.outcome_arena import OutcomeArena, OutcomeReceipt, TaskOffer


ARMS = ("frozen", "outcome_weighted", "shuffled_feedback")
EVIDENCE_CLASS = "SOURCE_LOCAL_MULTI_SEED_OPEN_WEIGHT_BENCHMARK_UNAUTHENTICATED"
TRAIN_FAMILY = "open_weight_benchmark_train"
HELDOUT_FAMILY = "open_weight_benchmark_holdout"


def _synthetic_id(*values: object) -> str:
    return sha256(json.dumps(values, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":")).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class BenchmarkTrainingRecord:
    episode_id: str
    input_ids_sha256: str
    chosen_action: str
    chosen_probability: float
    actual_measured_quality: float
    teaching_quality: float
    committed_evidence_id: str
    teaching_evidence_id: str
    donor_evidence_id: str | None
    weights_before_sha256: str
    weights_after_sha256: str
    optimizer_step_executed: bool


@dataclass(frozen=True)
class BenchmarkHeldoutRecord:
    episode_id: str
    input_ids_sha256: str
    chosen_action: str
    chosen_probability: float
    measured_quality: float


@dataclass(frozen=True)
class BenchmarkArmReport:
    arm: str
    seed: int
    binding: OpenWeightBinding
    initial_weights_sha256: str
    learned_weights_sha256: str
    restored_weights_sha256: str
    frozen_backbone_sha256: str
    train_schedule_sha256: str
    heldout_schedule_sha256: str
    training: tuple[BenchmarkTrainingRecord, ...]
    heldout_primary: tuple[BenchmarkHeldoutRecord, ...]
    heldout_mirrored_diagnostic: tuple[BenchmarkHeldoutRecord, ...]
    mean_primary_quality: float
    mean_primary_chosen_probability: float
    optimizer_steps: int
    restored: bool
    model_backend: str
    learned_callback_invoked: bool
    evidence_class: str = EVIDENCE_CLASS


@dataclass(frozen=True)
class BenchmarkSeedReport:
    seed: int
    frozen: BenchmarkArmReport
    outcome_weighted: BenchmarkArmReport
    shuffled_feedback: BenchmarkArmReport
    treatment_minus_frozen_quality: float
    treatment_minus_shuffled_quality: float
    treatment_minus_frozen_chosen_probability: float
    control_baselines_matched: bool
    train_and_heldout_schedules_matched: bool
    input_exposure_matched: bool
    heldout_family_disjoint: bool


@dataclass(frozen=True)
class DescriptivePairedContrast:
    number_of_seeds: int
    mean_difference: float
    sample_standard_deviation: float | None
    approximate_normal_95_interval: tuple[float, float] | None
    statistically_significant_claimed: bool = False


@dataclass(frozen=True)
class MultiSeedOpenWeightBenchmark:
    seed_reports: tuple[BenchmarkSeedReport, ...]
    treatment_vs_frozen_quality: DescriptivePairedContrast
    treatment_vs_shuffled_quality: DescriptivePairedContrast
    treatment_vs_frozen_chosen_probability: DescriptivePairedContrast
    training_pairs: int
    heldout_pairs: int
    evaluator_id_declared: str
    rubric_digest_declared: str
    all_arms_restored: bool
    all_baselines_and_schedules_matched: bool
    evidence_class: str = EVIDENCE_CLASS
    externally_authenticated: bool = False
    hosted_model_modified: bool = False
    phenomenology: str = "UNRESOLVED"


def _contrast(values: tuple[float, ...]) -> DescriptivePairedContrast:
    avg = fsum(values) / len(values)
    if len(values) < 2:
        return DescriptivePairedContrast(len(values), avg, None, None)
    sd = stdev(values)
    half_width = 1.96 * sd / sqrt(len(values))
    return DescriptivePairedContrast(
        len(values), avg, sd, (avg - half_width, avg + half_width),
    )


def _offers(arena: OutcomeArena, training_pairs: int, heldout_pairs: int):
    training = tuple(arena.issue(
        partition="TRAIN", family_id=TRAIN_FAMILY,
        pair_index=i, side=i % 2,
    ) for i in range(training_pairs))
    heldout = tuple(arena.issue(
        partition="EVAL", family_id=HELDOUT_FAMILY,
        pair_index=i, side=side,
    ) for i in range(heldout_pairs) for side in (0, 1))
    if {o.episode_id for o in training} & {o.episode_id for o in heldout}:
        raise RuntimeError("TRAIN and heldout episode IDs overlap")
    if {(o.features_a, o.features_b) for o in training} & {
        (o.features_a, o.features_b) for o in heldout
    }:
        raise RuntimeError("TRAIN and heldout offers contain the same task features")
    return training, heldout


def _validate_update(
    update: OpenWeightUpdate, outcome: OpenWeightOutcome, choice: object,
    adapter: ActionModelAdapter, frozen_digest: str, prior_step: int,
) -> None:
    changed = update.before_digest != update.after_digest
    if (
        type(update) is not OpenWeightUpdate
        or update.evidence_class != ADAPTER_EVIDENCE_CLASS
        or update.phenomenology != ADAPTER_PHENOMENOLOGY
        or update.evidence_id != outcome.evidence_id
        or update.decision_digest != choice.decision_digest
        or update.chosen_action != choice.chosen_action
        or update.measured_quality != outcome.measured_quality
        or update.before_digest != choice.weights_digest
        or update.after_digest != adapter.weights_digest
        or update.step_index != prior_step + 1
        or update.frozen_backbone_digest != frozen_digest
        or adapter.frozen_backbone_digest != frozen_digest
        or update.update_detected is not changed
        or type(update.optimizer_step_executed) is not bool
        or type(update.optimizer_applied) is not bool
        or update.optimizer_applied is not changed
        or (changed and not update.optimizer_step_executed)
        or adapter.pending
    ):
        raise RuntimeError("benchmark optimizer outcome/readback mismatch")


def _run_arm(
    *, arm: str, seed: int, adapter: ActionModelAdapter,
    encode_offer: Callable[[TaskOffer], tuple[int, ...]],
    training_pairs: int, heldout_pairs: int, evaluator_id: str,
    baseline: tuple[OpenWeightBinding, str, str] | None,
    frozen_donors: tuple[BenchmarkTrainingRecord, ...] | None,
    on_learned_weights: Callable[[int, str, str], None] | None,
) -> BenchmarkArmReport:
    phase = "VALIDATE"
    first_readback = None
    train_records: list[BenchmarkTrainingRecord] = []
    heldout_records: list[tuple[TaskOffer, BenchmarkHeldoutRecord]] = []
    captured = False
    error: BaseException | None = None
    try:
        first_readback = _readback(adapter)
        binding, original_weights, original_frozen = first_readback
        if baseline is not None and first_readback != baseline:
            raise ValueError("benchmark arms must start with exact matched binding/tensors")
        arena = OutcomeArena(seed=seed, evaluator_id=evaluator_id)
        train_offers, eval_offers = _offers(arena, training_pairs, heldout_pairs)
        shift = 1 + ((seed ^ 0x9E3779B9) % (training_pairs - 1))
        if arm == "shuffled_feedback":
            if frozen_donors is None or len(frozen_donors) != training_pairs:
                raise ValueError("shuffled-feedback arm requires frozen observed donor outcomes")
        for index, offer in enumerate(train_offers):
            phase = "TRAIN_INFER"
            ids = _token_ids(encode_offer(offer))
            before, step = adapter.weights_digest, adapter.step_index
            choice = adapter.infer(
                input_ids=ids, offer_nonce=offer.nonce,
                episode_id=offer.episode_id, source_partition="TRAIN",
            )
            _check_choice(choice, binding=binding, offer=offer,
                          ids=ids, weights_digest=before, step=step)
            if (adapter.weights_digest != before or not adapter.pending
                or adapter.frozen_backbone_digest != original_frozen):
                raise RuntimeError("TRAIN inference mutated model state")
            phase = "COMMIT_RESOLVE"
            ticket = arena.commit(
                offer, choice.chosen_action, choice.decision_digest,
                model_id=binding.model_id,
                runtime_instance_id=binding.runtime_instance_id,
            )
            actual = arena.resolve(ticket)
            _check_outcome(offer, choice, ticket, actual,
                           evaluator_id, arena.rubric_digest)
            teaching, teaching_id, donor_id = (
                actual.measured_quality, actual.evidence_id, None
            )
            if arm == "shuffled_feedback":
                donor = frozen_donors[(index + shift) % training_pairs]
                teaching, donor_id = donor.actual_measured_quality, donor.committed_evidence_id
                teaching_id = _synthetic_id(
                    "EXPLICIT_SYNTHETIC_PERMUTED_FROZEN_BASELINE_FEEDBACK_V1",
                    actual.evidence_id, donor_id, choice.decision_digest, index,
                )
                if donor.episode_id == offer.episode_id:
                    raise RuntimeError("shuffled feedback donor was not deranged")
            feedback = OpenWeightOutcome(
                evidence_id=teaching_id, decision_digest=actual.decision_digest,
                episode_id=actual.episode_id, chosen_action=actual.action,
                evaluator_id=actual.evaluator_id,
                rubric_digest=actual.rubric_digest, measured_quality=teaching,
            )
            phase = "TRAIN_UPDATE"
            update = adapter.observe(feedback)
            _validate_update(update, feedback, choice, adapter, original_frozen, step)
            if arm == "frozen" and (
                update.optimizer_step_executed or update.update_detected
            ):
                raise RuntimeError("frozen arm unexpectedly optimized model tensors")
            train_records.append(BenchmarkTrainingRecord(
                offer.episode_id, _input_digest(ids), choice.chosen_action,
                _chosen_probability(choice), actual.measured_quality,
                teaching, actual.evidence_id, teaching_id, donor_id,
                update.before_digest, update.after_digest,
                update.optimizer_step_executed,
            ))
        for offer in eval_offers:
            phase = "HELDOUT"
            ids = _token_ids(encode_offer(offer))
            before, step = adapter.weights_digest, adapter.step_index
            choice = adapter.infer(
                input_ids=ids, offer_nonce=offer.nonce,
                episode_id=offer.episode_id, source_partition="EVAL",
            )
            _check_choice(choice, binding=binding, offer=offer,
                          ids=ids, weights_digest=before, step=step)
            if (adapter.weights_digest != before or adapter.pending
                or adapter.frozen_backbone_digest != original_frozen):
                raise RuntimeError("EVAL inference mutated model state")
            ticket = arena.commit(
                offer, choice.chosen_action, choice.decision_digest,
                model_id=binding.model_id, runtime_instance_id=binding.runtime_instance_id,
            )
            outcome = arena.resolve(ticket)
            _check_outcome(offer, choice, ticket, outcome,
                           evaluator_id, arena.rubric_digest)
            if (adapter.weights_digest != before or adapter.step_index != step
                or adapter.pending or adapter.frozen_backbone_digest != original_frozen):
                raise RuntimeError("heldout EVAL changed model state")
            heldout_records.append((offer, BenchmarkHeldoutRecord(
                offer.episode_id, _input_digest(ids), choice.chosen_action,
                _chosen_probability(choice), outcome.measured_quality,
            )))
        phase = "LEARNED_CAPTURE"
        learned = adapter.weights_digest
        learned_step = adapter.step_index
        if adapter.pending or adapter.frozen_backbone_digest != original_frozen:
            raise RuntimeError("learned model has pending choice or changed frozen backbone")
        if on_learned_weights is not None and arm == "outcome_weighted":
            on_learned_weights(seed, arm, learned)
            if (adapter.weights_digest != learned or adapter.step_index != learned_step
                or adapter.pending or adapter.frozen_backbone_digest != original_frozen):
                raise RuntimeError("learned adapter capture mutated live model state")
            captured = True
        train_digest = _schedule_digest(train_offers)
        heldout_digest = _schedule_digest(eval_offers)
        selected = tuple(rec for offer, rec in heldout_records
                         if offer.side == offer.pair_index % 2)
        if len(selected) != heldout_pairs:
            raise RuntimeError("heldout primary schedule count diverged")
        all_heldout = tuple(rec for _, rec in heldout_records)
    except Exception as exc:
        error = exc
    finally:
        try:
            receipt = adapter.restore()
            if first_readback is not None:
                _, initial, frozen = first_readback
                restored = (
                    type(receipt) is OpenWeightRestore and receipt.closed
                    and receipt.optimizer_state_cleared
                    and receipt.restored_digest == initial
                    and receipt.frozen_backbone_digest == frozen
                    and adapter.weights_digest == initial
                    and adapter.frozen_backbone_digest == frozen
                    and not adapter.pending
                )
            else:
                restored = type(receipt) is OpenWeightRestore and receipt.closed
        except Exception as restore_exc:
            if error is None:
                error = RuntimeError(f"benchmark restoration failed: {restore_exc}")
            else:
                error = RuntimeError(
                    f"benchmark {phase} failed ({error}) and restoration failed ({restore_exc})"
                )
            restored = False
    if error is not None:
        raise RuntimeError(f"benchmark arm {arm} aborted at {phase}: {error}") from error
    if not restored:
        raise RuntimeError(f"benchmark arm {arm} exact restoration mismatch")
    assert first_readback is not None
    binding, initial, frozen = first_readback
    return BenchmarkArmReport(
        arm=arm, seed=seed, binding=binding,
        initial_weights_sha256=initial,
        learned_weights_sha256=learned,
        restored_weights_sha256=adapter.weights_digest,
        frozen_backbone_sha256=frozen,
        train_schedule_sha256=train_digest,
        heldout_schedule_sha256=heldout_digest,
        training=tuple(train_records), heldout_primary=selected,
        heldout_mirrored_diagnostic=all_heldout,
        mean_primary_quality=fsum(x.measured_quality for x in selected) / len(selected),
        mean_primary_chosen_probability=fsum(x.chosen_probability for x in selected) / len(selected),
        optimizer_steps=sum(x.optimizer_step_executed for x in train_records),
        restored=True,
        model_backend=(
            "SOURCE_LOCAL_TORCH_ADAPTER_UNAUTHENTICATED"
            if type(adapter) is TorchChosenOutcomeAdapter else "SOURCE_ONLY_TEST_DOUBLE_OR_UNKNOWN"
        ),
        learned_callback_invoked=captured,
    )


def run_multiseed_open_weight_benchmark(
    *, adapter_factory: Callable[[float, int, str], ActionModelAdapter],
    encode_offer: Callable[[TaskOffer], tuple[int, ...]],
    seeds: tuple[int, ...], training_pairs: int, heldout_pairs: int,
    learning_rate: float,
    evaluator_id: str = "source-local-open-weight-oracle",
    on_learned_weights: Callable[[int, str, str], None] | None = None,
) -> MultiSeedOpenWeightBenchmark:
    """Sequential fresh adapters for three matched arms per independent seed.

    Frozen is run first to collect genuine chosen-outcome donor receipts. The
    shuffled arm uses a derangement of those scalar values with distinct
    *synthetic* feedback evidence IDs, preserving each actual outcome receipt.
    Descriptive paired normal intervals are NOT a hypothesis test or proof
    of generalization, and the source-local evaluator is untrusted.
    """
    if not callable(adapter_factory) or not callable(encode_offer):
        raise TypeError("adapter_factory and encode_offer must be callable")
    if type(seeds) is not tuple or not 1 <= len(seeds) <= 32:
        raise ValueError("seeds must be 1..32 distinct uint32 integers")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be distinct uint32 integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct uint32 integers")
    if type(training_pairs) is not int or not 2 <= training_pairs <= MAX_TRAINING_PAIRS:
        raise ValueError("training_pairs must be 2..64 for a deranged feedback control")
    if type(heldout_pairs) is not int or not 1 <= heldout_pairs <= MAX_HELDOUT_PAIRS:
        raise ValueError("heldout_pairs must be 1..64")
    try:
        valid_rate = (type(learning_rate) in (float, int)
                      and isfinite(float(learning_rate))
                      and 0 < float(learning_rate) <= 1)
    except (OverflowError, ValueError):
        valid_rate = False
    if not valid_rate:
        raise ValueError("learning_rate must be finite in (0,1]")
    if on_learned_weights is not None and not callable(on_learned_weights):
        raise TypeError("on_learned_weights must be callable")
    # Validate identities/seeds before creating the first opt-in mutating adapter.
    prototype = OutcomeArena(seed=seeds[0], evaluator_id=evaluator_id)
    for seed in seeds[1:]:
        OutcomeArena(seed=seed, evaluator_id=evaluator_id)
    evidence = []
    for seed in seeds:
        args = dict(
            seed=seed, encode_offer=encode_offer, training_pairs=training_pairs,
            heldout_pairs=heldout_pairs, evaluator_id=evaluator_id,
            on_learned_weights=on_learned_weights,
        )
        frozen = _run_arm(
            arm="frozen", adapter=adapter_factory(0.0, seed, "frozen"),
            baseline=None, frozen_donors=None, **args,
        )
        baseline = (frozen.binding, frozen.initial_weights_sha256,
                    frozen.frozen_backbone_sha256)
        treatment = _run_arm(
            arm="outcome_weighted",
            adapter=adapter_factory(float(learning_rate), seed, "outcome_weighted"),
            baseline=baseline, frozen_donors=None, **args,
        )
        shuffled = _run_arm(
            arm="shuffled_feedback",
            adapter=adapter_factory(float(learning_rate), seed, "shuffled_feedback"),
            baseline=baseline, frozen_donors=frozen.training, **args,
        )
        arms = (frozen, treatment, shuffled)
        match_schedule = (
            len({arm.train_schedule_sha256 for arm in arms}) == 1
            and len({arm.heldout_schedule_sha256 for arm in arms}) == 1
        )
        exposure = (
            len({tuple(x.input_ids_sha256 for x in arm.training) for arm in arms}) == 1
            and len({tuple(x.input_ids_sha256 for x in arm.heldout_mirrored_diagnostic)
                 for arm in arms}) == 1
        )
        if not match_schedule or not exposure:
            raise RuntimeError("benchmark matched task/input exposure diverged")
        if len({arm.model_backend for arm in arms}) != 1:
            raise RuntimeError("benchmark arms used nonmatching model backend classes")
        if frozen.optimizer_steps or frozen.learned_weights_sha256 != frozen.initial_weights_sha256:
            raise RuntimeError("frozen baseline performed training")
        evidence.append(BenchmarkSeedReport(
            seed=seed, frozen=frozen, outcome_weighted=treatment,
            shuffled_feedback=shuffled,
            treatment_minus_frozen_quality=(
                treatment.mean_primary_quality - frozen.mean_primary_quality
            ),
            treatment_minus_shuffled_quality=(
                treatment.mean_primary_quality - shuffled.mean_primary_quality
            ),
            treatment_minus_frozen_chosen_probability=(
                treatment.mean_primary_chosen_probability - frozen.mean_primary_chosen_probability
            ),
            control_baselines_matched=True,
            train_and_heldout_schedules_matched=True,
            input_exposure_matched=True,
            heldout_family_disjoint=True,
        ))
    rows = tuple(evidence)
    return MultiSeedOpenWeightBenchmark(
        seed_reports=rows,
        treatment_vs_frozen_quality=_contrast(tuple(x.treatment_minus_frozen_quality for x in rows)),
        treatment_vs_shuffled_quality=_contrast(tuple(x.treatment_minus_shuffled_quality for x in rows)),
        treatment_vs_frozen_chosen_probability=_contrast(tuple(
            x.treatment_minus_frozen_chosen_probability for x in rows
        )),
        training_pairs=training_pairs, heldout_pairs=heldout_pairs,
        evaluator_id_declared=evaluator_id,
        rubric_digest_declared=prototype.rubric_digest,
        all_arms_restored=True,
        all_baselines_and_schedules_matched=True,
    )
