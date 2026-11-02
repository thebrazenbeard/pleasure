"""Opt-in jointly trainable PEFT LoRA representation and appraisal head.

Chosen-action TRAIN feedback backpropagates through *fresh, differentiable*
action-conditioned model hidden vectors into an explicit PEFT LoRA allowlist
and an independent bounded quality head. Neither candidate vectors nor LM
gradients are borrowed from the frozen/no-grad feature cache. Pretrained
parameters and buffers are exact-hashed as immutable before and after every
training operation. There is no evaluator trust proof here: the orchestrator
must authenticate/associate the chosen receipt with the sampled decision.

This is an open-weight local training mechanism, not hosted GPT modification,
proof of reward improvement, a feeling, or a phenomenology assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import importlib
import json
from math import isfinite, sin, sqrt
import re
from secrets import token_hex
from threading import Lock
from typing import Any

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal import _bounded_float, _digest_ids, _ids
from pleasure_experiment.open_weight_local_model import PreparedLocalLoRA


FEATURE_SCHEMA = "joint-relative-action-unit-hidden-tanh2-v1"
EVIDENCE_CLASS = "SOURCE_LOCAL_JOINT_LORA_APPRAISAL_UNAUTHENTICATED"
PHENOMENOLOGY = "UNRESOLVED"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EPISODE = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")


@dataclass(frozen=True)
class JointAppraisalDecision:
    decision_id: str
    episode_id: str
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
    lora_digest: str
    base_digest: str
    joint_digest: str
    update_index: int
    choice_bound: bool = True
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY


@dataclass(frozen=True)
class JointAppraisalUpdate:
    decision_id: str
    evidence_id: str
    episode_id: str
    measured_chosen_quality: float
    before_head_digest: str
    after_head_digest: str
    before_lora_digest: str
    after_lora_digest: str
    before_joint_digest: str
    after_joint_digest: str
    frozen_base_digest: str
    gradient_norm: float
    delta_l2: float
    lora_gradient_tensors: int
    lora_nonzero_gradient_tensors: int
    optimizer_step_executed: bool
    update_index: int
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY

    @property
    def head_changed(self) -> bool:
        return self.before_head_digest != self.after_head_digest

    @property
    def lora_changed(self) -> bool:
        return self.before_lora_digest != self.after_lora_digest


class TorchJointLoRAAppraisal:
    """One-owner stateless-SGD LoRA+head update with replayable custody state.

    A source-local caller must own the prepared PEFT model exclusively. No
    optimizer momentum is carried across processes, but update counts and all
    accepted TRAIN episode/evidence IDs **must** be restored with tensors.
    """

    feature_schema = FEATURE_SCHEMA

    def __init__(
        self, prepared: PreparedLocalLoRA, *, hidden_size: int,
        learning_rate: float = 0.05, allow_training: bool = False,
        max_updates: int = 48, max_grad_norm: float = 1.0,
        max_delta_l2: float = 1.0, l2_penalty: float = 0.02,
        train_lora: bool = True,
    ) -> None:
        if type(prepared) is not PreparedLocalLoRA:
            raise TypeError("exact prepared local LoRA record required")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("optional PyTorch required for joint LoRA") from exc
        if not isinstance(prepared.model, torch.nn.Module):
            raise TypeError("prepared LoRA model must be torch module")
        if type(hidden_size) is not int or not 1 <= hidden_size <= 8192:
            raise ValueError("hidden_size out of bounded range")
        if type(allow_training) is not bool or type(train_lora) is not bool:
            raise ValueError("explicit allow_training/train_lora booleans required")
        if type(max_updates) is not int or not 1 <= max_updates <= 4096:
            raise ValueError("max_updates must be 1..4096")
        rate = _bounded_float(learning_rate, "learning_rate", 0.0, 1.0)
        grad_cap = _bounded_float(max_grad_norm, "max_grad_norm", 1e-8, 10.0)
        delta_cap = _bounded_float(max_delta_l2, "max_delta_l2", 1e-8, 10.0)
        regularizer = _bounded_float(l2_penalty, "l2_penalty", 0.0, 1.0)
        if (type(prepared.action_token_ids) is not tuple or
            len(prepared.action_token_ids) != 2 or
            any(type(token) is not int or token < 0
                for token in prepared.action_token_ids) or
            prepared.action_token_ids[0] == prepared.action_token_ids[1]):
            raise ValueError("two distinct action tokens required")
        names = tuple(prepared.trainable_parameter_names)
        if type(names) is not tuple or not names or len(set(names)) != len(names):
            raise ValueError("exact unique LoRA trainable names required")
        parameter_map = dict(prepared.model.named_parameters())
        if not set(names).issubset(parameter_map):
            raise ValueError("LoRA trainable name missing from model")
        if any(".lora_A." not in name and ".lora_B." not in name for name in names):
            raise ValueError("only explicit LoRA A/B names may train")
        selected = tuple((name, parameter_map[name]) for name in names)
        if (sum(value.numel() for _, value in selected) > 1_000_000 or
            any(value.dtype != torch.float32 or value.device.type != "cpu"
                or not value.requires_grad for _, value in selected)):
            raise ValueError("LoRA must be bounded, trainable CPU float32")
        if any(name not in names and param.requires_grad
               for name, param in parameter_map.items()):
            raise ValueError("unexpected non-LoRA trainable pretrained parameters")
        frozen = tuple((name, param) for name, param in
                       prepared.model.named_parameters() if name not in names)
        buffers = tuple(("buffer:" + name, buffer) for name, buffer in
                        prepared.model.named_buffers())
        if not frozen or any(param.device.type != "cpu" for _, param in frozen + buffers):
            raise ValueError("nonempty CPU frozen model required")
        if not all(torch.isfinite(param).all().item() for _, param in selected):
            raise ValueError("nonfinite LoRA adapter parameters")
        self._torch = torch
        self._prepared = prepared
        self._model = prepared.model
        self._tokens = prepared.action_token_ids
        self._lora = selected
        self._frozen = frozen + buffers
        self._base_sha = _named_digest(torch, self._frozen)
        self._lock = Lock()
        self._train_lora = train_lora
        if not train_lora:
            for _, param in self._lora:
                param.requires_grad_(False)
        self._head = torch.nn.Linear(hidden_size, 1, dtype=torch.float32)
        # Nonzero deterministic head initialization makes the very first
        # chosen-quality gradient reach PEFT LoRA B (which starts at zero).
        with torch.no_grad():
            for i in range(hidden_size):
                self._head.weight[0, i] = (
                    0.01 * sin((i + 1) * 1.61803398875) / sqrt(hidden_size)
                )
            self._head.bias.zero_()
        self._rate = rate
        self._allow_training = allow_training
        self._max_updates = max_updates
        self._grad_cap = grad_cap
        self._delta_cap = delta_cap
        self._l2_penalty = regularizer
        self._updates = 0
        self._seen_evidence: set[str] = set()
        self._seen_episodes: set[str] = set()
        self._unbound: tuple[JointAppraisalDecision, tuple[int, ...]] | None = None
        self._pending: tuple[JointAppraisalDecision, tuple[int, ...]] | None = None
        self._closed = False
        self._optimizer = torch.optim.SGD(
            (tuple(param for _, param in self._lora) if train_lora else ()) +
            tuple(self._head.parameters()), lr=rate,
        )
        config = getattr(self._model, "peft_config", None)
        if not isinstance(config, dict) or len(config) != 1:
            raise ValueError("exactly one explicit PEFT adapter configuration required")
        value = next(iter(config.values()))
        targets = getattr(value, "target_modules", None)
        if (type(targets) not in (set, list, tuple) or not targets or
            not all(type(item) is str for item in targets)):
            raise ValueError("PEFT LoRA target modules must be explicit names")
        self._lora_config = {
            "rank": prepared.rank, "alpha": prepared.alpha,
            "target_modules": tuple(sorted(targets)),
        }

    @property
    def prepared(self) -> PreparedLocalLoRA:
        return self._prepared

    @property
    def action_token_ids(self) -> tuple[int, int]:
        return self._tokens

    @property
    def lora_config(self) -> dict[str, object]:
        return dict(self._lora_config)

    @property
    def optimizer_spec(self) -> dict[str, object]:
        return {
            "algorithm": "stateless-SGD", "learning_rate": self._rate,
            "max_grad_norm": self._grad_cap,
            "max_delta_l2": self._delta_cap,
            "l2_penalty": self._l2_penalty, "max_updates": self._max_updates,
            "train_lora": self._train_lora,
        }

    @property
    def base_digest(self) -> str:
        return self._base_sha

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None or self._unbound is not None

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    def _head_named(self) -> tuple[tuple[str, Any], ...]:
        return tuple(("head." + name, tensor)
                     for name, tensor in self._head.named_parameters())

    def _lora_named(self) -> tuple[tuple[str, Any], ...]:
        return tuple(("lora." + name, tensor) for name, tensor in self._lora)

    def _joint_named(self) -> tuple[tuple[str, Any], ...]:
        return self._head_named() + self._lora_named()

    @property
    def head_digest(self) -> str:
        with self._lock:
            return _named_digest(self._torch, self._head_named())

    @property
    def lora_digest(self) -> str:
        with self._lock:
            return _named_digest(self._torch, self._lora_named())

    @property
    def joint_digest(self) -> str:
        with self._lock:
            return _named_digest(self._torch, self._joint_named())

    def _guard(self) -> None:
        if self._closed:
            raise RuntimeError("joint LoRA session closed after failure")
        if _named_digest(self._torch, self._frozen) != self._base_sha:
            self._closed = True
            raise RuntimeError("frozen pretrained model/buffers changed")
        optimizer_members = tuple(id(param) for group in self._optimizer.param_groups
                                  for param in group["params"])
        expected = (tuple(id(param) for _, param in self._lora)
                    if self._train_lora else ()) + tuple(
            id(param) for param in self._head.parameters()
        )
        if (optimizer_members != expected or any(
            group["lr"] != self._rate for group in self._optimizer.param_groups
        )):
            self._closed = True
            raise RuntimeError("joint optimizer parameter membership or rate changed")
        if (any(param.requires_grad != self._train_lora
                for _, param in self._lora) or
            any(param.requires_grad for _, param in self._frozen) or
            any(not param.requires_grad for param in self._head.parameters())):
            self._closed = True
            raise RuntimeError("unexpected frozen or trainable parameter flags")

    def _forward(self, ids: tuple[int, ...], *, grad: bool) -> Any:
        torch = self._torch
        batch = torch.tensor([ids], dtype=torch.long, device="cpu")
        modes = tuple((part, part.training) for part in self._model.modules())
        cache_config = getattr(self._model, "config", None)
        original_cache = getattr(cache_config, "use_cache", None)
        try:
            self._model.eval()
            if cache_config is not None and hasattr(cache_config, "use_cache"):
                cache_config.use_cache = False
            if grad:
                return self._model(input_ids=batch, output_hidden_states=True,
                                   use_cache=False)
            with torch.no_grad():
                return self._model(input_ids=batch, output_hidden_states=True,
                                   use_cache=False)
        finally:
            for module, old in modes:
                module.training = old
            if cache_config is not None and original_cache is not None:
                cache_config.use_cache = original_cache

    def _features(self, ids: tuple[int, ...], *, grad: bool) -> tuple[Any, Any, Any]:
        torch = self._torch
        if grad:
            logits = None
        else:
            prefix = self._forward(ids, grad=False)
            logits = prefix.logits[0, -1, list(self._tokens)].float()
            if logits.shape != (2,) or not bool(torch.isfinite(logits).all()):
                raise ValueError("missing or nonfinite base action logits")
        states = []
        for token in self._tokens:
            output = self._forward(ids + (token,), grad=grad)
            h = getattr(output, "hidden_states", None)
            if not h:
                raise ValueError("action-conditioned hidden states missing")
            state = h[-1][0, -1, :].float()
            if (state.ndim != 1 or state.numel() != self._head.in_features or
                not bool(torch.isfinite(state).all())):
                raise ValueError("nonfinite action hidden features or width mismatch")
            states.append(state)
        delta = states[0] - states[1]
        # Unlike the v2 detached cache, this exact transformation is in the
        # autograd graph when grad=True: LoRA receives chosen-loss gradients.
        normalized = delta / torch.linalg.vector_norm(delta).clamp_min(1e-5)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("invalid normalized differentiable hidden difference")
        return logits, normalized, -normalized

    def _quality(self, vector: Any) -> Any:
        torch = self._torch
        raw = self._head(vector).reshape(())
        return (2.0 * torch.tanh(raw / 2.0)).sigmoid()

    def score(self, input_ids: tuple[int, ...], *, gain: float = 0.0,
              partition: str = "EVAL", defer_choice: bool = False,
              episode_id: str) -> JointAppraisalDecision:
        ids = _ids(input_ids)
        strength = _bounded_float(gain, "gain", -16.0, 16.0)
        if type(partition) is not str or partition not in ("TRAIN", "EVAL"):
            raise ValueError("partition must be TRAIN or EVAL")
        if type(episode_id) is not str or _EPISODE.fullmatch(episode_id) is None:
            raise ValueError("bounded episode_id required")
        if type(defer_choice) is not bool or (defer_choice and partition != "TRAIN"):
            raise ValueError("deferred choice requires TRAIN")
        with self._lock:
            self._guard()
            if self._pending is not None or self._unbound is not None:
                raise ValueError("another TRAIN choice remains pending")
            if partition == "TRAIN":
                if not self._allow_training or self._rate == 0:
                    raise PermissionError("explicit nonzero-rate joint training required")
                if self._updates >= self._max_updates:
                    raise ValueError("joint optimizer update budget exhausted")
                if episode_id in self._seen_episodes:
                    raise ValueError("TRAIN episode replay detected")
            baseline = _named_digest(self._torch, self._joint_named())
            logits, feature_a, feature_b = self._features(ids, grad=False)
            with self._torch.no_grad():
                qa, qb = self._quality(feature_a), self._quality(feature_b)
                probabilities = self._torch.softmax(
                    logits + strength * self._torch.stack((qa, qb)).sub(.5), dim=-1,
                )
                base = self._torch.softmax(logits, dim=-1)
            if (not bool(self._torch.isfinite(probabilities).all()) or
                _named_digest(self._torch, self._joint_named()) != baseline):
                raise RuntimeError("joint inference changed trainable tensors")
            self._guard()
            result = JointAppraisalDecision(
                decision_id=sha256((token_hex(16) + _digest_ids(ids) + episode_id)
                                   .encode("ascii")).hexdigest(),
                episode_id=episode_id, input_ids_sha256=_digest_ids(ids),
                partition=partition, chosen_action=(
                    "UNBOUND" if defer_choice else
                    "A" if float(probabilities[0]) > float(probabilities[1]) else "B"
                ), gain=strength,
                base_prob_a=float(base[0]), base_prob_b=float(base[1]),
                prob_a=float(probabilities[0]), prob_b=float(probabilities[1]),
                predicted_quality_a=float(qa), predicted_quality_b=float(qb),
                head_digest=_named_digest(self._torch, self._head_named()),
                lora_digest=_named_digest(self._torch, self._lora_named()),
                base_digest=self._base_sha, joint_digest=baseline,
                update_index=self._updates, choice_bound=not defer_choice,
            )
            if partition == "TRAIN":
                if defer_choice:
                    self._unbound = (result, ids)
                else:
                    self._pending = (result, ids)
            return result

    def bind_choice(self, decision: JointAppraisalDecision,
                    chosen_action: str) -> JointAppraisalDecision:
        if type(chosen_action) is not str or chosen_action not in ("A", "B"):
            raise ValueError("chosen action must be A/B")
        with self._lock:
            self._guard()
            if (type(decision) is not JointAppraisalDecision or
                decision.partition != "TRAIN" or decision.choice_bound or
                self._unbound is None or self._unbound[0] is not decision or
                self._pending is not None or
                _named_digest(self._torch, self._joint_named()) != decision.joint_digest):
                raise ValueError("no identical pre-outcome unbound TRAIN decision")
            bound = replace(decision, choice_bound=True, chosen_action=chosen_action)
            self._pending = (bound, self._unbound[1])
            self._unbound = None
            return bound

    def abandon_unbound(self, decision: JointAppraisalDecision) -> None:
        with self._lock:
            if self._unbound is None or self._unbound[0] is not decision:
                raise ValueError("no matching unbound TRAIN decision")
            self._unbound = None

    def observe_chosen(
        self, decision: JointAppraisalDecision, *,
        measured_quality: float, evidence_id: str,
    ) -> JointAppraisalUpdate:
        value = _bounded_float(measured_quality, "measured_quality", 0.0, 1.0)
        if type(evidence_id) is not str or _SHA.fullmatch(evidence_id) is None:
            raise ValueError("chosen receipt evidence_id must be lowercase SHA256")
        with self._lock:
            self._guard()
            if (type(decision) is not JointAppraisalDecision or
                decision.partition != "TRAIN" or not decision.choice_bound or
                self._pending is None or self._pending[0] is not decision or
                decision.chosen_action not in ("A", "B") or
                decision.update_index != self._updates or
                evidence_id in self._seen_evidence or
                decision.episode_id in self._seen_episodes):
                raise ValueError("only matching unconsumed chosen TRAIN outcome can update")
            before_h = _named_digest(self._torch, self._head_named())
            before_l = _named_digest(self._torch, self._lora_named())
            before = _named_digest(self._torch, self._joint_named())
            if (before_h != decision.head_digest or
                before_l != decision.lora_digest or
                before != decision.joint_digest):
                raise RuntimeError("joint tensors changed between decision and receipt")
            selected = (tuple(param for _, param in self._lora)
                        if self._train_lora else ()) + tuple(
                self._head.parameters()
            )
            snapshots = tuple(param.detach().clone() for param in selected)
            grad_norm = delta = 0.0
            lora_nonzero = lora_gradients = 0
            try:
                self._optimizer.zero_grad(set_to_none=True)
                # Brand-new autograd forwards! Cached/detached feature use here
                # would silently remove the entire LoRA learning mechanism.
                _, feature_a, feature_b = self._features(self._pending[1], grad=True)
                chosen = feature_a if decision.chosen_action == "A" else feature_b
                prediction = self._quality(chosen)
                # The entire current LoRA/head state was SHA checked before
                # this fresh differentiable forward. Recomputed pre-feedback
                # quality must still agree with the original decision, so a
                # mode/dropout/cache discrepancy cannot silently teach from
                # a different representation than the committed policy used.
                expected_prediction = (
                    decision.predicted_quality_a if decision.chosen_action == "A"
                    else decision.predicted_quality_b
                )
                if abs(float(prediction.detach()) - expected_prediction) > 2e-5:
                    raise RuntimeError("joint differentiable TRAIN replay differs from pre-feedback quality")
                loss = (prediction - value).square() + (
                    self._l2_penalty * self._head.weight.square().sum()
                )
                if not bool(self._torch.isfinite(loss)):
                    raise ValueError("nonfinite chosen-only joint objective")
                loss.backward()
                for _, param in self._lora:
                    if param.grad is not None:
                        lora_gradients += 1
                        if (bool(self._torch.isfinite(param.grad).all()) and
                            bool((param.grad != 0).any())):
                            lora_nonzero += 1
                if self._train_lora and (lora_gradients == 0 or lora_nonzero == 0):
                    raise RuntimeError("chosen quality failed to reach LoRA autograd graph")
                grad_norm = float(self._torch.nn.utils.clip_grad_norm_(
                    selected, self._grad_cap,
                ))
                if not isfinite(grad_norm):
                    raise ValueError("nonfinite joint gradient")
                self._optimizer.step()
                delta = float(self._torch.sqrt(sum(
                    (param.detach() - old).square().sum()
                    for param, old in zip(selected, snapshots)
                )))
                if (not isfinite(delta) or delta > self._delta_cap or
                    any(not bool(self._torch.isfinite(param).all())
                        for param in selected)):
                    raise RuntimeError("joint optimizer exceeded finite bounded delta")
                self._guard()
                after_h = _named_digest(self._torch, self._head_named())
                after_l = _named_digest(self._torch, self._lora_named())
                after = _named_digest(self._torch, self._joint_named())
                if after == before:
                    raise RuntimeError("joint SGD produced no trainable tensor update")
            except Exception:
                with self._torch.no_grad():
                    for param, old in zip(selected, snapshots):
                        param.copy_(old)
                self._optimizer.zero_grad(set_to_none=True)
                self._pending = None
                self._closed = True
                raise
            self._optimizer.zero_grad(set_to_none=True)
            self._pending = None
            self._seen_evidence.add(evidence_id)
            self._seen_episodes.add(decision.episode_id)
            self._updates += 1
            return JointAppraisalUpdate(
                decision_id=decision.decision_id,
                evidence_id=evidence_id, episode_id=decision.episode_id,
                measured_chosen_quality=value,
                before_head_digest=before_h, after_head_digest=after_h,
                before_lora_digest=before_l, after_lora_digest=after_l,
                before_joint_digest=before, after_joint_digest=after,
                frozen_base_digest=self._base_sha,
                gradient_norm=grad_norm, delta_l2=delta,
                lora_gradient_tensors=lora_gradients,
                lora_nonzero_gradient_tensors=lora_nonzero,
                optimizer_step_executed=True, update_index=self._updates,
            )

    def export_tensors(self) -> dict[str, Any]:
        with self._lock:
            self._guard()
            if self._pending is not None or self._unbound is not None:
                raise ValueError("cannot export with pending TRAIN choice")
            return {name: param.detach().to("cpu", dtype=self._torch.float32).clone()
                    .contiguous() for name, param in self._joint_named()}

    def export_training_state(self) -> dict[str, object]:
        with self._lock:
            self._guard()
            if self._pending is not None or self._unbound is not None:
                raise ValueError("cannot export TRAIN replay ledger while pending")
            return {
                "updates": self._updates,
                "seen_evidence_ids": tuple(sorted(self._seen_evidence)),
                "seen_episode_ids": tuple(sorted(self._seen_episodes)),
            }

    def import_tensors(
        self, tensors: dict[str, Any], *, expected_digest: str,
        updates_at_export: int = 0,
        seen_evidence_ids: tuple[str, ...] = (),
        seen_episode_ids: tuple[str, ...] = (),
    ) -> None:
        if type(expected_digest) is not str or _SHA.fullmatch(expected_digest) is None:
            raise ValueError("expected joint tensor digest must be SHA256")
        if (type(updates_at_export) is not int or
            not 0 <= updates_at_export <= self._max_updates or
            type(seen_evidence_ids) is not tuple or
            type(seen_episode_ids) is not tuple or
            len(seen_evidence_ids) != updates_at_export or
            len(seen_episode_ids) != updates_at_export or
            len(set(seen_evidence_ids)) != len(seen_evidence_ids) or
            len(set(seen_episode_ids)) != len(seen_episode_ids) or
            any(type(item) is not str or _SHA.fullmatch(item) is None
                for item in seen_evidence_ids) or
            any(type(item) is not str or _EPISODE.fullmatch(item) is None
                for item in seen_episode_ids)):
            raise ValueError("replayed TRAIN ledger or update count malformed")
        with self._lock:
            self._guard()
            if self._pending is not None or self._unbound is not None:
                raise ValueError("cannot import tensors during pending TRAIN")
            named = dict(self._joint_named())
            if type(tensors) is not dict or set(tensors) != set(named):
                raise ValueError("joint head/LoRA tensor names mismatch")
            checked = []
            for name, current in named.items():
                value = tensors[name]
                if (type(value) is not self._torch.Tensor or
                    value.device.type != "cpu" or
                    value.dtype != self._torch.float32 or
                    value.shape != current.shape or not value.is_contiguous() or
                    not bool(self._torch.isfinite(value).all())):
                    raise ValueError("joint imported tensor shape/dtype/finiteness invalid")
                checked.append((name, value.clone()))
            if _named_digest(self._torch, tuple(checked)) != expected_digest:
                raise ValueError("joint imported tensor digest mismatch")
            originals = {name: param.detach().clone() for name, param in named.items()}
            old_updates, old_evidence, old_episodes = (
                self._updates, self._seen_evidence.copy(), self._seen_episodes.copy(),
            )
            try:
                with self._torch.no_grad():
                    for name, value in checked:
                        named[name].copy_(value)
                self._updates = updates_at_export
                self._seen_evidence = set(seen_evidence_ids)
                self._seen_episodes = set(seen_episode_ids)
                self._optimizer.state.clear()  # explicitly stateless SGD
                self._optimizer.zero_grad(set_to_none=True)
                if (_named_digest(self._torch, self._joint_named()) != expected_digest or
                    _named_digest(self._torch, self._frozen) != self._base_sha):
                    raise RuntimeError("joint import readback or pretrained base mismatch")
            except Exception:
                with self._torch.no_grad():
                    for name, old in originals.items():
                        named[name].copy_(old)
                self._updates = old_updates
                self._seen_evidence = old_evidence
                self._seen_episodes = old_episodes
                self._optimizer.state.clear()
                self._optimizer.zero_grad(set_to_none=True)
                raise
