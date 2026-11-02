"""Matched source-local two-arm chosen-outcome open-weight study.

Runs one opted-in adapter with outcome-weighted SGD and a fresh zero-learning
rate control on matched TRAIN and heldout episodes. The caller owns the
adapter factory, model checkpoints and tokenizer. No model is downloaded,
authenticated, installed or loaded by this module.

Controls identify local policy differences, not an unbiased causal reward
estimate, authenticated host state, hosted GPT training or felt pleasure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

from pleasure_experiment.open_weight_trial import (
    ActionModelAdapter,
    MAX_HELDOUT_PAIRS,
    MAX_TRAINING_PAIRS,
    OpenWeightTrialReport,
    OpenWeightTrialRunner,
)
from pleasure_experiment.open_weight_adapter import OpenWeightBinding
from pleasure_experiment.outcome_arena import OutcomeArena, TaskOffer


EVIDENCE_CLASS = "SOURCE_LOCAL_MATCHED_OPEN_WEIGHT_CONTROLS_UNAUTHENTICATED"


@dataclass(frozen=True)
class MatchedOpenWeightControlsReport:
    """Two actual runner reports; tensor provenance is adapter-dependent."""

    outcome_weighted: OpenWeightTrialReport
    zero_learning_rate: OpenWeightTrialReport
    primary_realized_quality_difference: float
    primary_chosen_probability_difference: float
    train_schedule_matched: bool
    heldout_schedule_matched: bool
    input_exposure_matched: bool
    initial_tensor_digest_matched: bool
    frozen_backbone_digest_matched: bool
    backend_kind: str
    observed_trainable_tensor_change: bool
    confirmed_pretrained_llm_training: bool = False
    hosted_model_modified: bool = False
    externally_authenticated: bool = False
    phenomenology: str = "UNRESOLVED"
    evidence_class: str = EVIDENCE_CLASS


def _readback(adapter: ActionModelAdapter) -> tuple[OpenWeightBinding, str, str]:
    binding = adapter.binding
    if type(binding) is not OpenWeightBinding:
        raise TypeError("adapter factory must return an exact OpenWeightBinding")
    weights = adapter.weights_digest
    frozen = adapter.frozen_backbone_digest
    if (
        type(weights) is not str or len(weights) != 64
        or any(ch not in "0123456789abcdef" for ch in weights)
        or type(frozen) is not str or len(frozen) != 64
        or any(ch not in "0123456789abcdef" for ch in frozen)
    ):
        raise ValueError("initial trainable/frozen tensor digest missing or invalid")
    if type(adapter.step_index) is not int or adapter.step_index != 0 or adapter.pending:
        raise ValueError("fresh baseline adapter required")
    return binding, weights, frozen


def _exposure(report: OpenWeightTrialReport, *, train: bool) -> tuple[str, ...]:
    episodes = report.training if train else report.heldout
    return tuple(ep.input_ids_sha256 for ep in episodes)


def _restore_failed_initialization(adapter: ActionModelAdapter) -> None:
    """A factory already opted into mutations; never leak a half-open model."""
    try:
        receipt = adapter.restore()
        if not receipt.closed:
            raise RuntimeError("adapter did not close after initialization failure")
    except Exception as exc:
        raise RuntimeError(
            "failed adapter initialization also failed to restore model state"
        ) from exc


def _run_constructed_arm(
    adapter: ActionModelAdapter,
    *,
    encode_offer: Callable[[TaskOffer], tuple[int, ...]],
    seed: int,
    training_pairs: int,
    heldout_pairs: int,
    evaluator_id: str,
    on_learned_weights: Callable[[str], None] | None = None,
) -> OpenWeightTrialReport:
    try:
        runner = OpenWeightTrialRunner(
            adapter=adapter, encode_offer=encode_offer,
            seed=seed, training_pairs=training_pairs,
            heldout_pairs=heldout_pairs, evaluator_id=evaluator_id,
            on_learned_weights=on_learned_weights,
        )
    except Exception:
        _restore_failed_initialization(adapter)
        raise
    # Once the run begins, OpenWeightTrialRunner owns finally-restoration
    # and exposes an explicit unsuccessful-restoration abort if necessary.
    return runner.run()


def run_matched_open_weight_controls(
    *,
    adapter_factory: Callable[[float], ActionModelAdapter],
    encode_offer: Callable[[TaskOffer], tuple[int, ...]],
    learning_rate: float,
    seed: int = 12345,
    training_pairs: int = 8,
    heldout_pairs: int = 8,
    evaluator_id: str = "source-local-open-weight-oracle",
    on_learned_weights: Callable[[str], None] | None = None,
) -> MatchedOpenWeightControlsReport:
    """Compare a bounded SGD arm to a *separate* zero-LR arm, sequentially.

    The factory must return a fresh adapter and restored baseline model at
    each call. If the second adapter is not matched, its unused state is
    restored and the study is rejected. An outcome-dependent improvement is
    never assumed; quality differences may be negative or zero. Unknown/test
    double backends stay explicitly source-only.
    """
    if not callable(adapter_factory) or not callable(encode_offer):
        raise TypeError("adapter_factory and encode_offer must be callables")
    if on_learned_weights is not None and not callable(on_learned_weights):
        raise TypeError("on_learned_weights must be callable")
    if type(learning_rate) not in (int, float):
        raise ValueError("learning_rate must be finite and positive")
    try:
        gain = float(learning_rate)
    except (ValueError, OverflowError) as exc:
        raise ValueError("learning_rate must be finite and positive") from exc
    if not isfinite(gain) or not 0.0 < gain <= 1.0:
        raise ValueError("learning_rate must be finite in (0, 1]")
    # Validate all public scalars before invoking the mutating factory.
    # Bad counts/seed must not leave an unused model with altered flags/cache.
    if type(training_pairs) is not int or not 1 <= training_pairs <= MAX_TRAINING_PAIRS:
        raise ValueError("training_pairs out of bounded range")
    if type(heldout_pairs) is not int or not 1 <= heldout_pairs <= MAX_HELDOUT_PAIRS:
        raise ValueError("heldout_pairs out of bounded range")
    OutcomeArena(seed=seed, evaluator_id=evaluator_id)

    treatment_adapter = adapter_factory(gain)
    try:
        treatment_binding, treatment_weights, treatment_backbone = _readback(
            treatment_adapter
        )
    except Exception:
        _restore_failed_initialization(treatment_adapter)
        raise
    treatment = _run_constructed_arm(
        treatment_adapter, encode_offer=encode_offer,
        seed=seed, training_pairs=training_pairs, heldout_pairs=heldout_pairs,
        evaluator_id=evaluator_id,
        on_learned_weights=on_learned_weights,
    )
    if (
        treatment.binding != treatment_binding
        or treatment.initial_weights_digest != treatment_weights
        or treatment.frozen_backbone_digest != treatment_backbone
        or not treatment.restored
    ):
        raise RuntimeError("treatment baseline or local restoration readback mismatch")

    zero_adapter = adapter_factory(0.0)
    try:
        zero_binding, zero_weights, zero_backbone = _readback(zero_adapter)
        if (
            zero_binding != treatment_binding
            or zero_weights != treatment_weights
            or zero_backbone != treatment_backbone
        ):
            raise ValueError("zero-LR control must start at exact matched model baseline")
    except Exception:
        # An adapter factory already opted in to mutation. Even a rejected
        # second arm must not leave a model with altered training flags.
        try:
            zero_adapter.restore()
        except Exception as exc:
            raise RuntimeError("unmatched control baseline and restoration failed") from exc
        raise
    zero = _run_constructed_arm(
        zero_adapter, encode_offer=encode_offer,
        seed=seed, training_pairs=training_pairs, heldout_pairs=heldout_pairs,
        evaluator_id=evaluator_id,
    )
    if (
        zero.learned_weights_digest != zero.initial_weights_digest
        or zero.observed_weight_changes != 0
        or zero.optimizer_steps_reported != 0
        or any(
            ep.update.update_detected
            or ep.update.optimizer_applied
            or ep.update.optimizer_step_executed
            for ep in zero.training
        )
    ):
        raise ValueError("zero-learning-rate control improperly changed trainable model weights")
    if treatment.backend_kind != zero.backend_kind:
        raise ValueError("treatment and control use different adapter backend classes")
    if (
        treatment.train_schedule_sha256 != zero.train_schedule_sha256
        or treatment.heldout_schedule_sha256 != zero.heldout_schedule_sha256
        or treatment.rubric_digest_declared != zero.rubric_digest_declared
        or treatment.evaluator_id_declared != zero.evaluator_id_declared
    ):
        raise ValueError("treatment and control task schedule/rubric diverged")
    if (
        _exposure(treatment, train=True) != _exposure(zero, train=True)
        or _exposure(treatment, train=False) != _exposure(zero, train=False)
    ):
        raise ValueError("treatment and control received non-matching pre-outcome prompts")
    if not zero.restored:
        raise RuntimeError("zero-LR control failed local restoration")

    return MatchedOpenWeightControlsReport(
        outcome_weighted=treatment,
        zero_learning_rate=zero,
        primary_realized_quality_difference=(
            treatment.heldout_primary.mean_realized_quality
            - zero.heldout_primary.mean_realized_quality
        ),
        primary_chosen_probability_difference=(
            treatment.heldout_primary.mean_chosen_action_probability
            - zero.heldout_primary.mean_chosen_action_probability
        ),
        train_schedule_matched=True,
        heldout_schedule_matched=True,
        input_exposure_matched=True,
        initial_tensor_digest_matched=True,
        frozen_backbone_digest_matched=True,
        backend_kind=treatment.backend_kind,
        observed_trainable_tensor_change=(
            treatment.backend_kind == "SOURCE_LOCAL_TORCH_ADAPTER_UNAUTHENTICATED"
            and treatment.observed_weight_changes > 0
        ),
    )
