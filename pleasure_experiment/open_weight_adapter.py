"""Optional PyTorch causal-LM action adapter with genuine bounded SGD.

This module does NOT download, instantiate, deserialize, or grant access to a
model. An authorized caller must supply a local torch.nn.Module and explicitly
approve in-place training of an exact named adapter parameter subset. All
identities, checkpoint/tokenizer hashes, and measured outcomes are
caller-declared: local content hashes are not host attestations.
The caller must own an EXCLUSIVE, offline model/trainer instance throughout
the experiment; an in-process lock here cannot prevent other threads or hosts
from touching a separately exposed model reference. Never train a concurrently
served model with this adapter.

The constrained action space is TWO distinct single-token continuations with
deterministic argmax selection. The reward-weighted chosen-action loglikelihood
objective is -(observed_quality - 0.5) * log P(chosen | A,B); this is NOT an
unbiased on-policy REINFORCE estimator. It changes trainable model policy
tensors, not a separate self-reported reward head. It is not
an authenticated external experiment, a hosted GPT weight update, or evidence
of subjective pleasure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from secrets import token_hex
from threading import Lock
import importlib
import json
import re
from typing import Any


EVIDENCE_CLASS = "SOURCE_ONLY_OPEN_WEIGHT_NUMERIC_TENSOR_READBACK_UNAUTHENTICATED"
PHENOMENOLOGY = "UNRESOLVED"
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9._:/-]{1,160}\Z")


def _id(value: str, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a nonempty bounded identifier")
    return value


def _digest_id(value: str, field: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _real(value: object, field: str, low: float, high: float) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number in [{low}, {high}]")
    try:
        numeric = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not isfinite(numeric) or not low <= numeric <= high:
        raise ValueError(f"{field} must be a finite number in [{low}, {high}]")
    return 0.0 if numeric == 0.0 else numeric


def _hash(fields: object) -> str:
    return sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class OpenWeightBinding:
    """Caller-declared provenance, NOT a verified model/checkpoint identity."""

    model_id: str
    model_revision: str
    runtime_instance_id: str
    checkpoint_sha256: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "runtime_instance_id"):
            _id(getattr(self, name), name)
        for name in ("checkpoint_sha256", "tokenizer_sha256"):
            _digest_id(getattr(self, name), name)


@dataclass(frozen=True)
class OpenWeightChoice:
    binding: OpenWeightBinding
    episode_id: str
    source_partition: str
    chosen_action: str
    decision_digest: str
    weights_digest: str
    frozen_backbone_digest: str
    input_ids_digest: str
    offer_nonce_digest: str
    session_nonce_digest: str
    prob_a: float
    prob_b: float
    score_a: float
    score_b: float
    step_index: int
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY


@dataclass(frozen=True)
class OpenWeightOutcome:
    evidence_id: str
    decision_digest: str
    episode_id: str
    chosen_action: str
    evaluator_id: str
    rubric_digest: str
    measured_quality: float

    def __post_init__(self) -> None:
        _digest_id(self.evidence_id, "evidence_id")
        _digest_id(self.decision_digest, "decision_digest")
        _id(self.episode_id, "episode_id")
        if self.chosen_action not in ("A", "B") or type(self.chosen_action) is not str:
            raise ValueError("chosen_action must be A or B")
        _id(self.evaluator_id, "evaluator_id")
        _digest_id(self.rubric_digest, "rubric_digest")
        object.__setattr__(self, "measured_quality", _real(
            self.measured_quality, "measured_quality", 0.0, 1.0
        ))


@dataclass(frozen=True)
class OpenWeightUpdate:
    decision_digest: str
    evidence_id: str
    chosen_action: str
    measured_quality: float
    prediction_before: float
    before_digest: str
    after_digest: str
    frozen_backbone_digest: str
    step_index: int
    gradient_norm: float
    delta_l2: float
    optimizer_applied: bool
    optimizer_step_executed: bool
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY

    @property
    def update_detected(self) -> bool:
        return self.after_digest != self.before_digest


@dataclass(frozen=True)
class OpenWeightRestore:
    before_digest: str
    restored_digest: str
    frozen_backbone_digest: str
    optimizer_state_cleared: bool
    cancelled_pending: bool
    closed: bool
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY


def _tensor_byte_chunks(torch: Any, param: Any):
    """Stream large backbones with bounded CPU chunks, including bf16 bytes."""
    tensor = param.detach()
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError("only contiguous dense model tensor storage is supported")
    flat = tensor.reshape(-1)
    for offset in range(0, flat.numel(), 262144):
        segment = flat[offset:offset + 262144].cpu().contiguous()
        try:
            byte_tensor = segment.view(torch.uint8).reshape(-1)
            try:
                yield byte_tensor.numpy().tobytes()
            except (RuntimeError, TypeError):
                # NumPy is not a required direct dependency; this fallback is
                # bounded by the segment size, never the complete backbone.
                yield bytes(byte_tensor.tolist())
        except (RuntimeError, TypeError) as exc:
            raise ValueError("unsupported model parameter storage") from exc


def _named_digest(torch: Any, named: tuple[tuple[str, Any], ...]) -> str:
    digest = sha256()
    for name, param in named:
        byte_length = param.numel() * param.element_size()
        descriptor = json.dumps(
            (name, str(param.dtype), tuple(param.shape), byte_length),
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(4, "big"))
        digest.update(descriptor)
        digest.update(byte_length.to_bytes(8, "big"))
        for chunk in _tensor_byte_chunks(torch, param):
            digest.update(chunk)
    return digest.hexdigest()


class TorchChosenOutcomeAdapter:
    """Opt-in in-place update of an exact caller-selected policy tensor subset.

    Adapter names are exact torch named_parameter names (not a proof that
    those tensors are LoRA). The base parameters and their requires_grad flags
    are restored; the caller must discard any preexisting KV caches after an
    update. This object cannot certify an externally loaded checkpoint.
    """

    __slots__ = (
        "_lock", "_torch", "_model", "_binding", "_action_tokens",
        "_trainable", "_frozen", "_frozen_state", "_original_flags", "_saved_trainable",
        "_saved_buffers", "_saved_gradients",
        "_base_trainable_digest", "_frozen_digest", "_learning_rate",
        "_expected_weights_digest",
        "_max_grad_norm", "_max_delta_l2", "_max_steps", "_optimizer",
        "_session_nonce", "_expected_evaluator", "_expected_rubric",
        "_step_index", "_seen_episodes", "_seen_evidence", "_pending",
        "_closed", "_original_training", "_original_module_training",
        "_original_use_cache",
    )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("open-weight adapter state is guarded")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("open-weight adapter state is guarded")

    def __init__(
        self,
        model: Any,
        *,
        binding: OpenWeightBinding,
        action_token_ids: tuple[int, int],
        trainable_parameter_names: tuple[str, ...],
        learning_rate: float,
        expected_evaluator_id: str,
        expected_rubric_digest: str,
        allow_in_place_mutation: bool = False,
        max_steps: int = 64,
        max_grad_norm: float = 1.0,
        max_delta_l2: float = 1.0,
        max_trainable_parameters: int = 1_000_000,
        max_frozen_parameter_bytes: int = 512 * 1024 * 1024,
        max_buffer_snapshot_bytes: int = 16 * 1024 * 1024,
        trial_session_nonce: str | None = None,
    ) -> None:
        if allow_in_place_mutation is not True:
            raise PermissionError("explicit in-place model parameter training consent required")
        if type(binding) is not OpenWeightBinding:
            raise TypeError("binding must be OpenWeightBinding")
        binding = OpenWeightBinding(**binding.__dict__)
        if (type(action_token_ids) is not tuple or len(action_token_ids) != 2 or
            any(type(x) is not int or x < 0 for x in action_token_ids) or
            action_token_ids[0] == action_token_ids[1]):
            raise ValueError("A and B must be distinct nonnegative single-token IDs")
        if (type(trainable_parameter_names) is not tuple or
            not trainable_parameter_names or
            len(set(trainable_parameter_names)) != len(trainable_parameter_names)):
            raise ValueError("explicit nonempty unique trainable names are required")
        for name in trainable_parameter_names:
            _id(name, "trainable_parameter_name")
        lr = _real(learning_rate, "learning_rate", 0.0, 1.0)
        gradient_cap = _real(max_grad_norm, "max_grad_norm", 0.000001, 100.0)
        delta_cap = _real(max_delta_l2, "max_delta_l2", 0.000001, 100.0)
        if type(max_steps) is not int or not 1 <= max_steps <= 10_000:
            raise ValueError("max_steps must be 1..10000")
        if (type(max_trainable_parameters) is not int or
            not 1 <= max_trainable_parameters <= 10_000_000):
            raise ValueError("max_trainable_parameters must be 1..10000000")
        if (type(max_frozen_parameter_bytes) is not int or
            not 1 <= max_frozen_parameter_bytes <= 16 * 1024 ** 3):
            raise ValueError("max_frozen_parameter_bytes must be 1..16 GiB")
        if (type(max_buffer_snapshot_bytes) is not int or
            not 0 <= max_buffer_snapshot_bytes <= 256 * 1024 ** 2):
            raise ValueError("max_buffer_snapshot_bytes must be 0..256 MiB")
        _id(expected_evaluator_id, "expected_evaluator_id")
        _digest_id(expected_rubric_digest, "expected_rubric_digest")
        session_nonce = token_hex(16) if trial_session_nonce is None else _id(
            trial_session_nonce, "trial_session_nonce"
        )
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch unavailable: no open-weight optimizer can run in this environment"
            ) from exc
        if not isinstance(model, torch.nn.Module):
            raise TypeError("caller must supply a torch.nn.Module")
        params = tuple(model.named_parameters())
        by_name = dict(params)
        requested = frozenset(trainable_parameter_names)
        if len(by_name) != len(params) or not requested.issubset(by_name):
            raise ValueError("trainable tensor names must exactly resolve on the supplied model")
        selected = tuple((name, param) for name, param in params if name in requested)
        frozen = tuple((name, param) for name, param in params if name not in requested)
        buffers = tuple(("buffer:" + name, buffer)
                        for name, buffer in model.named_buffers())
        frozen_state = frozen + buffers
        if not frozen:
            raise ValueError("the model backbone must contain frozen non-adapter parameters")
        if sum(p.numel() * p.element_size() for _, p in frozen_state) > max_frozen_parameter_bytes:
            raise ValueError("frozen backbone exceeds explicit local hash budget")
        if sum(buf.numel() * buf.element_size() for _, buf in buffers) > max_buffer_snapshot_bytes:
            raise ValueError("model buffers exceed exact local rollback snapshot budget")
        if any(param.device.type == "meta" for _, param in params) or any(
            buffer.device.type == "meta" for _, buffer in model.named_buffers()
        ):
            raise ValueError("meta/unmaterialized tensors cannot be trained")
        if any(not param.is_floating_point() for _, param in selected):
            raise ValueError("selected adapter tensors must be floating point")
        count = sum(param.numel() for _, param in selected)
        if not 1 <= count <= max_trainable_parameters:
            raise ValueError("trainable parameter count exceeds bounded adapter budget")
        devices = {str(param.device) for _, param in params}
        if len(devices) != 1:
            raise ValueError("all model parameters must share one device in this pilot")
        orig_flags = tuple((name, param.requires_grad) for name, param in params)
        saved = tuple((name, param.detach().clone()) for name, param in selected)
        saved_gradients = tuple(
            (name, param.grad.detach().clone() if param.grad is not None else None)
            for name, param in selected
        )
        saved_buffers = tuple((name, buffer.detach().clone()) for name, buffer in buffers)
        frozen_digest = _named_digest(torch, frozen_state)
        initial_digest = _named_digest(torch, selected)
        original_training = bool(model.training)
        original_module_training = tuple((module, bool(module.training))
                                         for module in model.modules())
        config = getattr(model, "config", None)
        original_cache = getattr(config, "use_cache", None) if config is not None else None
        # Mutate only after all input/tensor checks have passed.
        try:
            for name, param in params:
                param.requires_grad_(name in requested)
            model.eval()
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False
            optimizer = torch.optim.SGD(tuple(param for _, param in selected), lr=lr)
        except Exception:
            for name, flag in orig_flags:
                by_name[name].requires_grad_(flag)
            for module, flag in original_module_training:
                module.training = flag
            if config is not None and original_cache is not None:
                config.use_cache = original_cache
            raise
        initial = dict(
            _lock=Lock(), _torch=torch, _model=model, _binding=binding,
            _action_tokens=action_token_ids, _trainable=selected,
            _frozen=frozen, _frozen_state=frozen_state,
            _original_flags=orig_flags, _saved_trainable=saved,
            _saved_buffers=saved_buffers, _saved_gradients=saved_gradients,
            _base_trainable_digest=initial_digest, _frozen_digest=frozen_digest,
            _expected_weights_digest=initial_digest,
            _learning_rate=lr, _max_grad_norm=gradient_cap,
            _max_delta_l2=delta_cap, _max_steps=max_steps, _optimizer=optimizer,
            _session_nonce=session_nonce, _expected_evaluator=expected_evaluator_id,
            _expected_rubric=expected_rubric_digest, _step_index=0,
            _seen_episodes=set(), _seen_evidence=set(), _pending=None,
            _closed=False, _original_training=original_training,
            _original_module_training=original_module_training,
            _original_use_cache=original_cache,
        )
        for name, value in initial.items():
            object.__setattr__(self, name, value)

    @property
    def binding(self) -> OpenWeightBinding:
        return OpenWeightBinding(**self._binding.__dict__)

    @property
    def weights_digest(self) -> str:
        with self._lock:
            return _named_digest(self._torch, self._trainable)

    @property
    def step_index(self) -> int:
        with self._lock:
            return self._step_index

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def frozen_backbone_digest(self) -> str:
        return self._frozen_digest

    def _forward_pair(self, input_ids: tuple[int, ...], *, with_grad: bool) -> Any:
        torch = self._torch
        # Do not use cached past_key_values or model.generate; forward on the
        # complete supplied context after every update.
        device = self._trainable[0][1].device
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        if with_grad:
            output = self._model(input_ids=tokens)
        else:
            with torch.no_grad():
                output = self._model(input_ids=tokens)
        logits = getattr(output, "logits", output)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or (
            logits.shape[0] != 1 or logits.shape[1] != len(input_ids)
        ):
            raise ValueError("model must return next-token logits [1, sequence, vocab]")
        if max(self._action_tokens) >= logits.shape[-1]:
            raise ValueError("action tokens outside model vocabulary")
        pair = torch.stack((logits[0, -1, self._action_tokens[0]],
                            logits[0, -1, self._action_tokens[1]]))
        if not bool(torch.isfinite(pair).all().item()):
            raise ValueError("model produced nonfinite action logits")
        return pair

    def _check_optimizer_membership(self) -> None:
        expected = tuple(id(param) for _, param in self._trainable)
        actual = tuple(
            id(param) for group in self._optimizer.param_groups
            for param in group["params"]
        )
        if actual != expected or any(
            group["lr"] != self._learning_rate for group in self._optimizer.param_groups
        ):
            raise RuntimeError("optimizer parameter membership or learning rate changed")
        if any(not param.requires_grad for _, param in self._trainable) or any(
            param.requires_grad for _, param in self._frozen
        ):
            raise RuntimeError("model trainable/frozen parameter flags changed")

    def infer(
        self, *,
        input_ids: tuple[int, ...],
        offer_nonce: str,
        episode_id: str,
        source_partition: str,
        score_bias: tuple[float, float] = (0.0, 0.0),
    ) -> OpenWeightChoice:
        if (type(input_ids) is not tuple or not 1 <= len(input_ids) <= 2048 or
            any(type(token) is not int or token < 0 for token in input_ids)):
            raise ValueError("input_ids must be a bounded nonempty tuple of nonnegative IDs")
        if type(score_bias) is not tuple or len(score_bias) != 2:
            raise ValueError("score_bias must be a pair")
        bias_a, bias_b = (_real(v, "score_bias", -1.0, 1.0) for v in score_bias)
        _id(offer_nonce, "offer_nonce")
        _id(episode_id, "episode_id")
        if type(source_partition) is not str or source_partition not in ("TRAIN", "EVAL"):
            raise ValueError("source_partition must be TRAIN or EVAL")
        with self._lock:
            if self._closed:
                raise RuntimeError("adapter already restored and closed")
            if self._pending is not None:
                raise RuntimeError("a TRAIN outcome must resolve before another inference")
            self._check_optimizer_membership()
            if episode_id in self._seen_episodes:
                raise ValueError("episode already inferred in this adapter")
            if source_partition == "TRAIN" and self._step_index >= self._max_steps:
                raise RuntimeError("online update budget exhausted")
            if _named_digest(self._torch, self._frozen_state) != self._frozen_digest:
                raise RuntimeError("frozen base model parameter readback mismatch")
            before = _named_digest(self._torch, self._trainable)
            if before != self._expected_weights_digest:
                raise RuntimeError("adapter weights changed outside observed optimizer steps")
            pair = self._forward_pair(input_ids, with_grad=False)
            if (_named_digest(self._torch, self._trainable) != before or
                _named_digest(self._torch, self._frozen_state) != self._frozen_digest):
                raise RuntimeError("inference mutated trainable or frozen model state")
            probabilities = self._torch.softmax(pair, dim=0)
            a = float(probabilities[0].item())
            b = float(probabilities[1].item())
            score_a, score_b = a + bias_a, b + bias_b
            action = "A" if score_a > score_b else "B"
            fields = [
                "OPEN_WEIGHT_CHOSEN_ACTION_V1", self._binding.__dict__,
                self._action_tokens, input_ids, episode_id, offer_nonce,
                self._session_nonce, source_partition, self._step_index,
                before, self._frozen_digest, a.hex(), b.hex(),
                score_a.hex(), score_b.hex(), action,
            ]
            choice = OpenWeightChoice(
                binding=OpenWeightBinding(**self._binding.__dict__), episode_id=episode_id,
                source_partition=source_partition, chosen_action=action,
                decision_digest=_hash(fields), weights_digest=before,
                frozen_backbone_digest=self._frozen_digest,
                input_ids_digest=_hash(input_ids),
                offer_nonce_digest=sha256(offer_nonce.encode("utf-8")).hexdigest(),
                session_nonce_digest=sha256(self._session_nonce.encode("utf-8")).hexdigest(),
                prob_a=a, prob_b=b, score_a=score_a, score_b=score_b,
                step_index=self._step_index,
            )
            self._seen_episodes.add(episode_id)
            if source_partition == "TRAIN":
                object.__setattr__(self, "_pending", (choice, input_ids))
            return replace(choice, binding=OpenWeightBinding(**choice.binding.__dict__))

    def observe(self, outcome: OpenWeightOutcome) -> OpenWeightUpdate:
        if type(outcome) is not OpenWeightOutcome:
            raise TypeError("outcome must be OpenWeightOutcome")
        fixed = replace(outcome)
        with self._lock:
            if self._closed:
                raise RuntimeError("adapter already restored and closed")
            if self._pending is None:
                raise RuntimeError("there is no pending TRAIN choice")
            choice, input_ids = self._pending
            self._check_optimizer_membership()
            if (fixed.decision_digest != choice.decision_digest
                or fixed.episode_id != choice.episode_id
                or fixed.chosen_action != choice.chosen_action):
                raise ValueError("outcome does not match pending chosen action")
            if (fixed.evaluator_id != self._expected_evaluator
                or fixed.rubric_digest != self._expected_rubric):
                raise ValueError("outcome evaluator/rubric declaration mismatch")
            if fixed.evidence_id in self._seen_evidence:
                raise ValueError("replayed outcome evidence identifier")
            if _named_digest(self._torch, self._frozen_state) != self._frozen_digest:
                raise RuntimeError("backbone changed before update")
            before = _named_digest(self._torch, self._trainable)
            if before != choice.weights_digest:
                raise RuntimeError("trainable weights changed after action commitment")
            chosen_index = 0 if choice.chosen_action == "A" else 1
            pre_prediction = choice.prob_a if chosen_index == 0 else choice.prob_b
            grad_norm, delta = 0.0, 0.0
            performed = False
            snapshots = tuple(param.detach().clone() for _, param in self._trainable)
            try:
                if self._learning_rate > 0 and fixed.measured_quality != 0.5:
                    pair = self._forward_pair(input_ids, with_grad=True)
                    logits_prob = self._torch.softmax(pair.detach(), dim=0)
                    current_prob = float(logits_prob[chosen_index].item())
                    if abs(current_prob - pre_prediction) > 1e-6:
                        raise RuntimeError("policy changed between commitment and update")
                    loss = -(fixed.measured_quality - 0.5) * self._torch.log_softmax(
                        pair, dim=0
                    )[chosen_index]
                    if not bool(self._torch.isfinite(loss).item()):
                        raise ValueError("nonfinite reward-weighted action loss")
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradients = [param.grad for _, param in self._trainable
                                 if param.grad is not None]
                    if not gradients:
                        raise RuntimeError("chosen-action loss does not reach trainable policy")
                    grad_norm = float(self._torch.nn.utils.clip_grad_norm_(
                        [param for _, param in self._trainable], self._max_grad_norm
                    ).item())
                    if not isfinite(grad_norm):
                        raise ValueError("nonfinite chosen-action gradient")
                    self._optimizer.step()
                    performed = True
                    squares = sum(float(self._torch.sum((param.detach() - saved) ** 2).item())
                                  for saved, (_, param) in zip(snapshots, self._trainable))
                    delta = squares ** 0.5
                    if not isfinite(delta) or delta > self._max_delta_l2:
                        raise RuntimeError("online SGD exceeded parameter-change budget")
                after = _named_digest(self._torch, self._trainable)
                if _named_digest(self._torch, self._frozen_state) != self._frozen_digest:
                    raise RuntimeError("frozen backbone mutated during optimizer step")
            except Exception:
                # Never leave a failed forward/optimizer in an apparently
                # usable model state. Reset every state surface we own and
                # verify the frozen backbone; no success receipt is minted.
                try:
                    self._rollback_locked()
                except Exception as rollback_exc:
                    raise RuntimeError(
                        "optimizer failure with unverified model rollback"
                    ) from rollback_exc
                raise
            self._optimizer.zero_grad(set_to_none=True)
            object.__setattr__(self, "_pending", None)
            object.__setattr__(self, "_step_index", self._step_index + 1)
            object.__setattr__(self, "_expected_weights_digest", after)
            self._seen_evidence.add(fixed.evidence_id)
            return OpenWeightUpdate(
                decision_digest=choice.decision_digest, evidence_id=fixed.evidence_id,
                chosen_action=choice.chosen_action, measured_quality=fixed.measured_quality,
                prediction_before=pre_prediction, before_digest=before,
                after_digest=after, frozen_backbone_digest=self._frozen_digest,
                step_index=self._step_index, gradient_norm=grad_norm,
                delta_l2=delta, optimizer_applied=after != before,
                optimizer_step_executed=performed,
            )

    def _rollback_locked(self) -> tuple[str, str]:
        """Close and restore owned state; reject unverifiable frozen parameter drift.

        The full frozen backbone is hashed but not copied because a real
        language model may contain gigabytes of immutable parameters. A
        modified frozen PARAMETER cannot be safely repaired by this adapter;
        its digest mismatch must stop all further use and prevent PASS.
        """
        object.__setattr__(self, "_closed", True)
        object.__setattr__(self, "_pending", None)
        try:
            with self._torch.no_grad():
                for name, saved in self._saved_trainable:
                    dict(self._trainable)[name].copy_(saved)
                for name, saved in self._saved_buffers:
                    dict(self._frozen_state)[name].copy_(saved)
            for name, flag in self._original_flags:
                dict(self._trainable + self._frozen)[name].requires_grad_(flag)
            self._optimizer.state.clear()
            self._optimizer.zero_grad(set_to_none=True)
            for name, saved_gradient in self._saved_gradients:
                param = dict(self._trainable)[name]
                param.grad = (None if saved_gradient is None else saved_gradient.clone())
            for module, flag in self._original_module_training:
                module.training = flag
            config = getattr(self._model, "config", None)
            if config is not None and self._original_use_cache is not None:
                config.use_cache = self._original_use_cache
            object.__setattr__(self, "_expected_weights_digest", self._base_trainable_digest)
            restored = _named_digest(self._torch, self._trainable)
            frozen = _named_digest(self._torch, self._frozen_state)
        except Exception as exc:
            raise RuntimeError("exact model rollback operation failed") from exc
        if restored != self._base_trainable_digest or frozen != self._frozen_digest:
            raise RuntimeError("exact frozen backbone or adapter baseline rollback unverified")
        all_params = dict(self._trainable + self._frozen)
        if (any(all_params[name].requires_grad != flag
                for name, flag in self._original_flags)
                or any(module.training != flag
                       for module, flag in self._original_module_training)):
            raise RuntimeError("model trainability or execution mode rollback unverified")
        if (config is not None and self._original_use_cache is not None
                and config.use_cache != self._original_use_cache):
            raise RuntimeError("model inference cache setting rollback unverified")
        return restored, frozen

    def restore(self) -> OpenWeightRestore:
        with self._lock:
            before = _named_digest(self._torch, self._trainable)
            cancelled = self._pending is not None
            restored, frozen = self._rollback_locked()
            return OpenWeightRestore(
                before_digest=before, restored_digest=restored,
                frozen_backbone_digest=frozen,
                optimizer_state_cleared=not self._optimizer.state,
                cancelled_pending=cancelled, closed=True,
            )

    def __enter__(self) -> TorchChosenOutcomeAdapter:
        if self._closed:
            raise RuntimeError("adapter already restored and closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.restore()
        return False
