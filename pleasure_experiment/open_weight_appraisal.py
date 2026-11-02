"""Source-local model-hidden-state appraisal head with explicit interventions.

This is a separately trainable *probe* on an LLM's own hidden activations,
not a native feeling, not an LLM backbone/LoRA update, and not a hosted-GPT
intervention. A candidate's hidden vector is computed from the prefix plus
that candidate token, before the external outcome is revealed. Only a
committed TRAIN choice's observed quality may update the head. Its predicted
qualities can be used as a causal intervention on a later A/B decision;
zero-gain, positive-gain, and sign-inverted evaluations ablate that influence.

The caller must exclusively own the model, supply an authenticated outcome
contract if required, and keep training and heldout episodes separate. This
class only validates local decision/feedback linkage, not external outcomes.
Optional torch is imported only when a probe is constructed. Exported state
is a mapping of plain CPU float32 tensors suitable for *safe* safetensors
serialization by a separate artifact writer (never use pickle/torch.save).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import importlib
import json
from math import isfinite
import re
from secrets import token_hex
from threading import Lock
from typing import Any

from pleasure_experiment.open_weight_adapter import _named_digest


EVIDENCE_CLASS = "SOURCE_LOCAL_HIDDEN_STATE_APPRAISAL_UNAUTHENTICATED"
PHENOMENOLOGY = "UNRESOLVED"
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_MAX_INPUT_TOKENS = 256


def _bounded_float(value: object, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return result


def _ids(ids: tuple[int, ...]) -> tuple[int, ...]:
    if (
        type(ids) is not tuple or not 1 <= len(ids) <= _MAX_INPUT_TOKENS
        or any(type(x) is not int or not 0 <= x <= 2**31 - 1 for x in ids)
    ):
        raise ValueError("input_ids must be 1..256 bounded token IDs")
    return ids


def _digest_ids(ids: tuple[int, ...]) -> str:
    return sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class AppraisalDecision:
    decision_id: str
    input_ids_sha256: str
    partition: str
    chosen_action: str
    gain: float
    base_prob_a: float
    base_prob_b: float
    prob_a: float
    prob_b: float
    predicted_quality_a: float
    predicted_quality_b: float
    head_digest: str
    choice_bound: bool = True
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY


@dataclass(frozen=True)
class AppraisalUpdate:
    decision_id: str
    evidence_id: str
    measured_chosen_quality: float
    before_digest: str
    after_digest: str
    gradient_norm: float
    delta_l2: float
    optimizer_step_executed: bool
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY

    @property
    def changed(self) -> bool:
        return self.before_digest != self.after_digest


class TorchHiddenAppraisal:
    """Bounded scalar quality-prediction head on action-conditioned LM states.

    The LM itself remains read-only. Positive and negative `gain` values alter
    action selection from the same frozen LM logits and same trained head.
    This supplies a direct *mechanism* intervention, not evidence of quality
    improvement; that requires matched independent heldout outcome evaluation.
    """

    def __init__(
        self,
        model: Any,
        *,
        action_token_ids: tuple[int, int],
        hidden_size: int,
        learning_rate: float,
        allow_training: bool = False,
        max_updates: int = 64,
        max_grad_norm: float = 1.0,
        max_delta_l2: float = 1.0,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("optional PyTorch is required for hidden appraisal") from exc
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a caller-supplied torch.nn.Module")
        if (
            type(action_token_ids) is not tuple or len(action_token_ids) != 2
            or any(type(n) is not int or n < 0 for n in action_token_ids)
            or action_token_ids[0] == action_token_ids[1]
        ):
            raise ValueError("two distinct action token IDs required")
        if type(hidden_size) is not int or not 1 <= hidden_size <= 8192:
            raise ValueError("hidden_size must be in [1, 8192]")
        rate = _bounded_float(learning_rate, "learning_rate", 0.0, 1.0)
        if type(allow_training) is not bool or type(max_updates) is not int or not 1 <= max_updates <= 4096:
            raise ValueError("invalid explicit training gate or update count")
        grad = _bounded_float(max_grad_norm, "max_grad_norm", 1e-8, 10.0)
        delta = _bounded_float(max_delta_l2, "max_delta_l2", 1e-8, 10.0)
        self._lock = Lock()
        self._torch, self._model, self._tokens = torch, model, action_token_ids
        self._head = torch.nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)
        with torch.no_grad():
            self._head.weight.zero_()
            self._head.bias.zero_()
        self._rate, self._consent, self._max_updates = rate, allow_training, max_updates
        self._grad_cap, self._delta_cap = grad, delta
        self._pending: tuple[AppraisalDecision, Any] | None = None
        self._unbound: tuple[AppraisalDecision, Any, Any] | None = None
        self._seen_evidence: set[str] = set()
        self._updates = 0

    @property
    def head_digest(self) -> str:
        with self._lock:
            return self._digest()

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    def _digest(self) -> str:
        return _named_digest(self._torch, tuple(self._head.named_parameters()))

    def _model_forward(self, ids: tuple[int, ...]) -> Any:
        torch = self._torch
        parameters = tuple(self._model.parameters())
        if not parameters:
            raise ValueError("model must expose a parameter device")
        devices = {p.device for p in parameters}
        if len(devices) != 1:
            raise ValueError("model must reside on exactly one device")
        token_batch = torch.tensor([ids], dtype=torch.long, device=parameters[0].device)
        # Force deterministic inference without permanently changing any
        # individual submodule's possibly mixed training mode.
        old_modes = tuple((module, module.training) for module in self._model.modules())
        try:
            self._model.eval()
            with torch.no_grad():
                return self._model(
                    input_ids=token_batch, output_hidden_states=True, use_cache=False,
                )
        finally:
            for module, original in old_modes:
                module.training = original

    def _features(self, ids: tuple[int, ...]) -> tuple[Any, Any, Any]:
        torch = self._torch
        prefix = self._model_forward(ids)
        try:
            logits = prefix.logits[0, -1, list(self._tokens)]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("model did not expose two action logits") from exc
        if logits.numel() != 2 or not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite or missing model action logits")
        choices = []
        for token in self._tokens:
            output = self._model_forward(ids + (token,))
            hidden = getattr(output, "hidden_states", None)
            if not hidden:
                raise ValueError("model must expose hidden_states for appraisal")
            value = hidden[-1][0, -1, :].detach().to(device="cpu", dtype=torch.float32)
            if (
                value.ndim != 1 or value.numel() != self._head.in_features
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("hidden feature size or finiteness mismatch")
            choices.append(value.contiguous())
        return logits.detach().to(device="cpu", dtype=torch.float32), choices[0], choices[1]

    def score(
        self,
        input_ids: tuple[int, ...],
        *,
        gain: float = 0.0,
        partition: str = "EVAL",
        defer_choice: bool = False,
    ) -> AppraisalDecision:
        ids = _ids(input_ids)
        strength = _bounded_float(gain, "gain", -16.0, 16.0)
        if partition not in ("TRAIN", "EVAL") or type(partition) is not str:
            raise ValueError("partition must be TRAIN or EVAL")
        if type(defer_choice) is not bool or (defer_choice and partition != "TRAIN"):
            raise ValueError("defer_choice requires a TRAIN decision and explicit boolean")
        with self._lock:
            if self._pending is not None or self._unbound is not None:
                raise ValueError("another TRAIN decision remains pending")
            if partition == "TRAIN":
                if not self._consent:
                    raise PermissionError("explicit appraisal training consent required")
                if self._updates >= self._max_updates:
                    raise ValueError("appraisal update budget exhausted")
            logits, feat_a, feat_b = self._features(ids)
            with self._torch.no_grad():
                pred_a = self._head(feat_a).sigmoid().reshape(())
                pred_b = self._head(feat_b).sigmoid().reshape(())
                adjusted = logits + strength * self._torch.stack((pred_a, pred_b)).sub(0.5)
                base_prob = self._torch.softmax(logits, dim=-1)
                probs = self._torch.softmax(adjusted, dim=-1)
            if not bool(self._torch.isfinite(probs).all()):
                raise ValueError("nonfinite appraisal intervention")
            decision = AppraisalDecision(
                decision_id=sha256((token_hex(16) + _digest_ids(ids)).encode("ascii")).hexdigest(),
                input_ids_sha256=_digest_ids(ids),
                partition=partition,
                chosen_action=(
                    "UNBOUND" if defer_choice
                    else "A" if float(probs[0]) > float(probs[1]) else "B"
                ),
                gain=strength,
                base_prob_a=float(base_prob[0]), base_prob_b=float(base_prob[1]),
                prob_a=float(probs[0]), prob_b=float(probs[1]),
                predicted_quality_a=float(pred_a), predicted_quality_b=float(pred_b),
                head_digest=self._digest(),
                choice_bound=not defer_choice,
            )
            if partition == "TRAIN":
                if defer_choice:
                    self._unbound = (decision, feat_a, feat_b)
                else:
                    selected = feat_a if decision.chosen_action == "A" else feat_b
                    self._pending = (decision, selected)
            return decision

    def bind_choice(
        self, decision: AppraisalDecision, chosen_action: str,
    ) -> AppraisalDecision:
        """Bind an externally sampled TRAIN action to its pre-outcome feature.

        `score(..., defer_choice=True)` exposes action probabilities without
        selecting an action. A caller can use a seeded stochastic policy to
        draw from those probabilities, record the draw and propensity, then
        bind exactly one of A/B here. No feedback or state import is permitted
        while the decision is unbound; only the returned bound decision can
        receive chosen-action feedback. Probability provenance and stochastic
        draw correctness must be checked by the experiment orchestrator.
        """
        if type(chosen_action) is not str or chosen_action not in ("A", "B"):
            raise ValueError("chosen_action must be exactly A or B")
        with self._lock:
            if (
                type(decision) is not AppraisalDecision
                or decision.partition != "TRAIN" or decision.choice_bound
                or self._pending is not None or self._unbound is None
                or self._unbound[0] is not decision
            ):
                raise ValueError("no matching unbound TRAIN decision")
            if self._digest() != decision.head_digest:
                raise ValueError("appraisal head changed before choice binding")
            bound = replace(decision, chosen_action=chosen_action, choice_bound=True)
            feature = self._unbound[1] if chosen_action == "A" else self._unbound[2]
            self._pending = (bound, feature)
            self._unbound = None
            return bound

    def abandon_unbound(self, decision: AppraisalDecision) -> None:
        """Release an unbound TRAIN proposal without recording an outcome.

        A benchmark must treat abandoned proposals as a failed arm and should
        never quietly drop them from its denominators.
        """
        with self._lock:
            if self._unbound is None or self._unbound[0] is not decision:
                raise ValueError("no matching unbound TRAIN decision")
            self._unbound = None

    def observe_chosen(
        self,
        decision: AppraisalDecision,
        *,
        measured_quality: float,
        evidence_id: str,
    ) -> AppraisalUpdate:
        value = _bounded_float(measured_quality, "measured_quality", 0.0, 1.0)
        if type(evidence_id) is not str or _SHA.fullmatch(evidence_id) is None:
            raise ValueError("evidence_id must be a SHA-256 digest")
        with self._lock:
            if (
                type(decision) is not AppraisalDecision
                or decision.partition != "TRAIN" or not decision.choice_bound
            ):
                raise ValueError("only a committed TRAIN decision can receive feedback")
            if self._pending is None or decision is not self._pending[0]:
                raise ValueError("outcome does not match one pending TRAIN decision")
            if evidence_id in self._seen_evidence:
                raise ValueError("replayed appraisal outcome evidence")
            before = self._digest()
            if before != decision.head_digest:
                raise ValueError("appraisal weights changed before feedback")
            feature = self._pending[1]
            self._pending = None
            old = tuple(p.detach().clone() for p in self._head.parameters())
            gradient_norm = delta_l2 = 0.0
            executed = False
            try:
                if self._rate:
                    optimizer = self._torch.optim.SGD(self._head.parameters(), lr=self._rate)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self._head(feature).sigmoid().reshape(())
                    loss = (prediction - value).square()
                    loss.backward()
                    gradient_norm = float(self._torch.nn.utils.clip_grad_norm_(
                        self._head.parameters(), self._grad_cap,
                    ))
                    if not isfinite(gradient_norm):
                        raise ValueError("nonfinite appraisal gradient")
                    optimizer.step()
                    delta_l2 = float(self._torch.sqrt(sum(
                        (p.detach() - baseline).square().sum()
                        for p, baseline in zip(self._head.parameters(), old)
                    )))
                    if not isfinite(delta_l2) or delta_l2 > self._delta_cap:
                        raise ValueError("appraisal update exceeded bounded delta")
                    if any(not bool(self._torch.isfinite(p).all()) for p in self._head.parameters()):
                        raise ValueError("nonfinite appraisal head state")
                    executed = True
            except Exception:
                with self._torch.no_grad():
                    for p, baseline in zip(self._head.parameters(), old):
                        p.copy_(baseline)
                raise
            finally:
                for p in self._head.parameters():
                    p.grad = None
            self._seen_evidence.add(evidence_id)
            self._updates += 1
            return AppraisalUpdate(
                decision_id=decision.decision_id, evidence_id=evidence_id,
                measured_chosen_quality=value, before_digest=before,
                after_digest=self._digest(), gradient_norm=gradient_norm,
                delta_l2=delta_l2, optimizer_step_executed=executed,
            )

    def export_tensors(self) -> dict[str, Any]:
        """Detached CPU clones for an external safetensors writer, never pickle."""
        with self._lock:
            return {
                name: p.detach().to(device="cpu", dtype=self._torch.float32).clone()
                for name, p in self._head.named_parameters()
            }

    def import_tensors(self, tensors: dict[str, Any], *, expected_digest: str) -> None:
        """Replace head tensors only after complete shape/dtype/hash validation."""
        if type(expected_digest) is not str or _SHA.fullmatch(expected_digest) is None:
            raise ValueError("expected_digest must be SHA-256")
        with self._lock:
            if self._pending is not None or self._unbound is not None:
                raise ValueError("cannot import appraisal tensors while TRAIN pending")
            current = dict(self._head.named_parameters())
            if type(tensors) is not dict or set(tensors) != set(current):
                raise ValueError("appraisal tensor names mismatch")
            checked = []
            for name, existing in current.items():
                candidate = tensors[name]
                if (
                    type(candidate) is not self._torch.Tensor
                    or candidate.device.type != "cpu" or candidate.dtype != self._torch.float32
                    or candidate.shape != existing.shape or not candidate.is_contiguous()
                    or not bool(self._torch.isfinite(candidate).all())
                ):
                    raise ValueError("appraisal tensor type, shape, or finiteness mismatch")
                checked.append((name, candidate.clone()))
            if _named_digest(self._torch, tuple(checked)) != expected_digest:
                raise ValueError("appraisal imported tensor hash mismatch")
            original = {name: p.detach().clone() for name, p in current.items()}
            try:
                with self._torch.no_grad():
                    for name, tensor in checked:
                        current[name].copy_(tensor)
                if self._digest() != expected_digest:
                    raise RuntimeError("appraisal tensor readback mismatch")
            except Exception:
                with self._torch.no_grad():
                    for name, tensor in original.items():
                        current[name].copy_(tensor)
                raise


def compare_appraisal_ablations(
    probe: TorchHiddenAppraisal, input_ids: tuple[int, ...], *, gain: float,
) -> dict[str, AppraisalDecision]:
    """Same model, same trained head: zero, positive and sign-inverted gain.

    This demonstrates whether the *mechanism* can causally change inference.
    It does not establish improved realized quality on unseen outcomes.
    """
    value = _bounded_float(gain, "gain", 1e-8, 16.0)
    return {
        "off": probe.score(input_ids, partition="EVAL", gain=0.0),
        "on": probe.score(input_ids, partition="EVAL", gain=value),
        "inverted": probe.score(input_ids, partition="EVAL", gain=-value),
    }
