"""Bounded action-relative reward appraisal on frozen language-model states.

The earlier head regressed unnormalized final hidden vectors with sigmoid-MSE;
large magnitudes and high rates drove some outputs to exactly 1.0 for BOTH
actions. This opt-in alternative computes opposing unit-L2 A/B differences
before feedback. A smooth tanh caps the quality logit to [-2, 2], so outputs
cannot numerically saturate to 0 or 1 while preserving useful gradients.

Only committed chosen TRAIN outcomes may update this separate head. The
underlying pretrained LM is inference-only; neither mechanism nor changed
weights is evidence of subjective experience or generalized reward utility.
Optional verified cached features avoid repeated model forwards per arm.
Its feature schema differs from the legacy head and MUST be manifest-bound.
"""

from __future__ import annotations

from contextvars import ContextVar
from math import isfinite
from typing import Any

from pleasure_experiment.open_weight_appraisal import (
    TorchHiddenAppraisal, _bounded_float, _digest_ids, _ids,
)


FEATURE_SCHEMA = "action-centered-unit-difference-bounded-logit-2-v1"
EVIDENCE_CLASS = "SOURCE_LOCAL_CALIBRATED_RELATIVE_APPRAISAL_UNAUTHENTICATED"


def _new_bounded_linear(torch: Any, width: int, l2_penalty: float) -> Any:
    class _BoundedLinear(torch.nn.Linear):
        def forward(self, features: Any) -> Any:
            linear = torch.nn.functional.linear(features, self.weight, self.bias)
            return 2.0 * torch.tanh(linear / 2.0)

    result = _BoundedLinear(width, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        result.weight.zero_()
        result.bias.zero_()
    if l2_penalty:
        # The inherited chosen-only optimizer computes MSE; these per-tensor
        # gradient hooks add d/dw [lambda/2 * ||w||^2] BEFORE its gradient
        # clipping, yielding actual L2-penalized reward regression. No LM
        # parameters are in the optimizer or attached to these hooks.
        for parameter in result.parameters():
            parameter.register_hook(
                lambda grad, p=parameter: grad + l2_penalty * p.detach()
            )
    return result


class TorchCalibratedAppraisal(TorchHiddenAppraisal):
    """Same score/bind/observe/export/import API as TorchHiddenAppraisal.

    `score(..., episode_id=..., features=...)` can consume a trusted
    FrozenActionFeatures bundle from a shared frozen-model feature cache.
    Its decisions and gradients remain chosen-only and replay protected by
    the inherited implementation. Reward authenticity is the caller's job.
    """

    feature_schema = FEATURE_SCHEMA

    def __init__(
        self, model: Any, *, action_token_ids: tuple[int, int],
        hidden_size: int, learning_rate: float,
        allow_training: bool = False, max_updates: int = 64,
        max_grad_norm: float = 1.0, max_delta_l2: float = 1.0,
        feature_cache: Any | None = None, l2_penalty: float = 0.02,
    ) -> None:
        regularization = _bounded_float(l2_penalty, "l2_penalty", 0.0, 1.0)
        if feature_cache is not None:
            from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache
            if (type(feature_cache) is not FrozenActionFeatureCache or
                feature_cache._model is not model or
                feature_cache._tokens != action_token_ids):
                raise ValueError("feature cache must bind this same frozen LM and action IDs")
        super().__init__(
            model, action_token_ids=action_token_ids, hidden_size=hidden_size,
            learning_rate=learning_rate, allow_training=allow_training,
            max_updates=max_updates, max_grad_norm=max_grad_norm,
            max_delta_l2=max_delta_l2,
        )
        self._head = _new_bounded_linear(self._torch, hidden_size, regularization)
        self.l2_penalty = regularization
        self._feature_cache = feature_cache
        self._score_override: ContextVar[Any | None] = ContextVar(
            f"calibrated-appraisal-override-{id(self)}", default=None,
        )

    def _features(self, ids: tuple[int, ...]) -> tuple[Any, Any, Any]:
        raw = self._score_override.get()
        logits, feature_a, feature_b = (
            super()._features(ids) if raw is None else raw
        )
        torch = self._torch
        delta = feature_a.float() - feature_b.float()
        if (delta.ndim != 1 or delta.numel() != self._head.in_features or
            not bool(torch.isfinite(delta).all())):
            raise ValueError("calibrated action-relative feature is invalid")
        norm = float(torch.linalg.vector_norm(delta))
        if not isfinite(norm):
            raise ValueError("calibrated relative feature norm is nonfinite")
        # Avoid amplifying cancellation noise when actions are indistinguishable.
        relative = torch.zeros_like(delta) if norm < 1e-5 else delta / norm
        if not bool(torch.isfinite(relative).all()):
            raise ValueError("relative appraisal feature normalization failed")
        return logits, relative.contiguous(), (-relative).contiguous()

    def score(
        self, input_ids: tuple[int, ...], *, gain: float = 0.0,
        partition: str = "EVAL", defer_choice: bool = False,
        episode_id: str | None = None, features: Any | None = None,
    ) -> Any:
        """Score a fresh LM prefix or one exact verified cache bundle."""
        ids = _ids(input_ids)
        _bounded_float(gain, "gain", -16.0, 16.0)
        if episode_id is not None and (
            type(episode_id) is not str or not 1 <= len(episode_id) <= 256 or
            episode_id != episode_id.strip()
        ):
            raise ValueError("episode_id must be a nonempty bounded string")
        if features is None and self._feature_cache is not None and episode_id is not None:
            features = self._feature_cache.get_features(
                ids, partition=partition, episode_id=episode_id,
            )
        if features is None:
            return super().score(
                ids, gain=gain, partition=partition, defer_choice=defer_choice,
            )
        if self._feature_cache is None or episode_id is None:
            raise ValueError("explicit cache and episode identity required")
        from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatures

        torch = self._torch
        if (
            type(features) is not FrozenActionFeatures or
            features.partition != partition or
            features.episode_id != episode_id or
            features.input_ids_sha256 != _digest_ids(ids) or
            features.action_token_ids != self._tokens
        ):
            raise ValueError("cached action features do not match this decision")
        # Caller-supplied dataclasses are never provenance. Check original
        # issuance and tensor-by-tensor equality against the cache's private
        # canonical storage; verify LM SHA outside a read-only cache epoch.
        trusted = self._feature_cache.verify_issued(
            features, input_ids=ids, partition=partition, episode_id=episode_id,
        )
        logits, a, b = trusted.logits, trusted.feature_a, trusted.feature_b
        if any(
            type(tensor) is not torch.Tensor or tensor.device.type != "cpu" or
            tensor.dtype != torch.float32 or not tensor.is_contiguous() or
            not bool(torch.isfinite(tensor).all())
            for tensor in (logits, a, b)
        ) or tuple(logits.shape) != (2,) or any(
            tuple(tensor.shape) != (self._head.in_features,) for tensor in (a, b)
        ):
            raise ValueError("cached appraisal tensors have invalid shape or type")
        token = self._score_override.set((logits.clone(), a.clone(), b.clone()))
        try:
            return super().score(
                ids, gain=gain, partition=partition, defer_choice=defer_choice,
            )
        finally:
            self._score_override.reset(token)

