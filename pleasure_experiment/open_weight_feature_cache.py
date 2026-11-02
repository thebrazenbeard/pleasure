"""Bounded, model-digest-bound cache of frozen language-model A/B features.

The cache stores only inference-derived logits and action-conditioned hidden
vectors. It never stores rewards, oracle coefficients, optimizer tensors,
chosen feedback, or payoff labels. It does tag each cache entry with an
isolated TRAIN/EVAL partition. A copied feature bundle can be used
by an independently trained appraisal head for fast matched-arm scoring.

Callers must exclusively own the frozen model. Ordinary ``get_features``
verifies its complete named parameter/buffer SHA before every cache access.
For a larger read-only scoring batch, ``verified_epoch`` checks the SHA at
entry *and exit*, avoiding redundant 257-MB model hashing across five arms.
This epoch boundary assumes exclusive ownership and no model mutation while
open; a transient mutation restored within that interval is not detectable.
Neither source-local hashes nor thread locks authenticate an external host.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import re
import secrets
from threading import RLock, get_ident
from typing import Any, Iterator

from pleasure_experiment.open_weight_adapter import _named_digest
from pleasure_experiment.open_weight_appraisal import _digest_ids, _ids


EVIDENCE_CLASS = "SOURCE_LOCAL_FROZEN_LM_ACTION_FEATURE_CACHE_UNAUTHENTICATED"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: str, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _episode(value: str) -> str:
    if type(value) is not str or not 1 <= len(value) <= 256 or value != value.strip():
        raise ValueError("episode_id must be a nonempty bounded identifier")
    return value


@dataclass(frozen=True)
class FrozenActionFeatures:
    """Detached CPU clones. The tensor values can be copied, not trusted as proof."""

    logits: Any
    feature_a: Any
    feature_b: Any
    input_ids_sha256: str
    partition: str
    episode_id: str
    frozen_model_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    action_token_ids: tuple[int, int]
    issuance_id: str
    evidence_class: str = EVIDENCE_CLASS


class _FeatureEpoch:
    def __init__(self, cache: "FrozenActionFeatureCache", owner_thread: int) -> None:
        self._cache = cache
        self._thread = owner_thread
        self._closed = False

    def get_features(
        self, input_ids: tuple[int, ...], *, partition: str, episode_id: str,
    ) -> FrozenActionFeatures:
        if self._closed or get_ident() != self._thread:
            raise RuntimeError("frozen model feature epoch is closed or owned by another thread")
        return self._cache._get_verified(input_ids, partition=partition,
                                         episode_id=episode_id, epoch=self)


class FrozenActionFeatureCache:
    """Cache cross-arm feature inputs, not weights, rewards or outcomes.

    A model-digest mismatch permanently invalidates this cache instance. The
    caller must reconstruct a new cache against a fresh frozen model, rather
    than silently relabeling stale features as a new checkpoint.
    """

    def __init__(
        self, model: Any, *, action_token_ids: tuple[int, int],
        checkpoint_sha256: str, tokenizer_sha256: str,
        max_entries: int = 256,
        expected_frozen_model_sha256: str | None = None,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("optional PyTorch required for action features") from exc
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a caller-owned torch.nn.Module")
        if (type(action_token_ids) is not tuple or len(action_token_ids) != 2 or
            any(type(token) is not int or not 0 <= token < 2**31
                for token in action_token_ids) or
            action_token_ids[0] == action_token_ids[1]):
            raise ValueError("two distinct nonnegative action token IDs required")
        if type(max_entries) is not int or not 1 <= max_entries <= 4096:
            raise ValueError("max_entries must be between 1 and 4096")
        self._torch, self._model, self._tokens = torch, model, action_token_ids
        self._checkpoint = _digest(checkpoint_sha256, "checkpoint")
        self._tokenizer = _digest(tokenizer_sha256, "tokenizer")
        self._limit = max_entries
        self._lock = RLock()
        self._items: OrderedDict[tuple[str, str, str], FrozenActionFeatures] = OrderedDict()
        self._epoch: _FeatureEpoch | None = None
        self._tainted = False
        self._hits = self._misses = 0
        self._model_sha = self._current_model_sha()
        if expected_frozen_model_sha256 is not None and (
            self._model_sha != _digest(expected_frozen_model_sha256, "expected frozen model")
        ):
            raise ValueError("frozen model does not match expected digest")

    @property
    def frozen_model_sha256(self) -> str:
        return self._model_sha

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    def _current_model_sha(self) -> str:
        state = tuple(self._model.named_parameters()) + tuple(
            ("buffer:" + name, buf) for name, buf in self._model.named_buffers()
        )
        if not state:
            raise ValueError("model must have named parameters or buffers")
        return _named_digest(self._torch, state)

    def _verify_model(self) -> None:
        if self._tainted:
            raise RuntimeError("frozen feature cache invalidated by model mutation")
        if self._current_model_sha() != self._model_sha:
            self._items.clear()
            self._tainted = True
            raise RuntimeError("frozen model tensors changed; cached features rejected")

    @staticmethod
    def _query(input_ids: tuple[int, ...], partition: str, episode_id: str):
        ids = _ids(input_ids)
        if type(partition) is not str or partition not in ("TRAIN", "EVAL"):
            raise ValueError("partition must be TRAIN or EVAL")
        event = _episode(episode_id)
        return ids, (partition, event, _digest_ids(ids))

    def _forward(self, ids: tuple[int, ...]) -> Any:
        torch = self._torch
        params = tuple(self._model.parameters())
        if not params or len({p.device for p in params}) != 1:
            raise ValueError("frozen model must reside on exactly one device")
        batch = torch.tensor([ids], dtype=torch.long, device=params[0].device)
        modes = tuple((module, module.training) for module in self._model.modules())
        try:
            self._model.eval()
            with torch.no_grad():
                return self._model(input_ids=batch, output_hidden_states=True,
                                   use_cache=False)
        finally:
            for module, mode in modes:
                module.training = mode

    def _compute(self, ids: tuple[int, ...], partition: str,
                 episode_id: str) -> FrozenActionFeatures:
        torch = self._torch
        prefix = self._forward(ids)
        try:
            logits = prefix.logits[0, -1, list(self._tokens)].detach().to(
                device="cpu", dtype=torch.float32,
            ).contiguous()
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("model did not expose A/B next-token logits") from exc
        if logits.ndim != 1 or logits.numel() != 2 or not bool(torch.isfinite(logits).all()):
            raise ValueError("model action logits missing or nonfinite")
        vectors = []
        for action in self._tokens:
            output = self._forward(ids + (action,))
            if not getattr(output, "hidden_states", None):
                raise ValueError("model does not expose action-conditioned hidden states")
            hidden = output.hidden_states[-1][0, -1, :].detach().to(
                device="cpu", dtype=torch.float32,
            ).contiguous()
            if hidden.ndim != 1 or hidden.numel() < 1 or hidden.numel() > 8192 or (
                not bool(torch.isfinite(hidden).all())
            ):
                raise ValueError("action hidden vector missing or nonfinite")
            vectors.append(hidden)
        if vectors[0].shape != vectors[1].shape:
            raise ValueError("action feature dimensions differ")
        return FrozenActionFeatures(
            logits=logits.clone(), feature_a=vectors[0].clone(),
            feature_b=vectors[1].clone(), input_ids_sha256=_digest_ids(ids),
            partition=partition, episode_id=episode_id,
            frozen_model_sha256=self._model_sha,
            checkpoint_sha256=self._checkpoint,
            tokenizer_sha256=self._tokenizer,
            action_token_ids=self._tokens,
            issuance_id=secrets.token_hex(24),
        )

    @staticmethod
    def _copies(record: FrozenActionFeatures) -> FrozenActionFeatures:
        return FrozenActionFeatures(
            logits=record.logits.detach().clone(),
            feature_a=record.feature_a.detach().clone(),
            feature_b=record.feature_b.detach().clone(),
            input_ids_sha256=record.input_ids_sha256,
            partition=record.partition, episode_id=record.episode_id,
            frozen_model_sha256=record.frozen_model_sha256,
            checkpoint_sha256=record.checkpoint_sha256,
            tokenizer_sha256=record.tokenizer_sha256,
            action_token_ids=record.action_token_ids,
            issuance_id=record.issuance_id,
        )

    def verify_issued(
        self, features: FrozenActionFeatures, *, input_ids: tuple[int, ...],
        partition: str, episode_id: str,
    ) -> FrozenActionFeatures:
        """Verify a caller-supplied bundle against an authoritative cache entry.

        Verify complete identity, the per-entry unpredictable issuance ID and
        the *exact bytes* of all three detached CPU feature tensors. A forged,
        modified or relabeled bundle is rejected, including relabeling EVAL as
        TRAIN when a separate genuine TRAIN entry exists. Only cloned trusted
        cache values are returned; no caller-provided tensors feed a head.

        An owned ``verified_epoch`` amortizes full frozen-model hashing;
        otherwise this method hashes the model on every invocation.
        """
        with self._lock:
            ids, key = self._query(input_ids, partition, episode_id)
            if self._epoch is None:
                self._verify_model()
            elif self._epoch._thread != get_ident() or self._epoch._closed:
                raise RuntimeError("feature epoch owned by a different thread")
            elif self._tainted:
                raise RuntimeError("frozen model feature cache is invalidated")
            if type(features) is not FrozenActionFeatures:
                raise TypeError("an issued frozen action feature bundle is required")
            trusted = self._items.get(key)
            if trusted is None:
                raise ValueError("feature bundle was not issued for this input and partition")
            metadata = (
                "input_ids_sha256", "partition", "episode_id",
                "frozen_model_sha256", "checkpoint_sha256", "tokenizer_sha256",
                "action_token_ids", "issuance_id", "evidence_class",
            )
            if any(getattr(features, field) != getattr(trusted, field)
                   for field in metadata):
                raise ValueError("feature bundle issuance or metadata mismatch")
            if features.input_ids_sha256 != _digest_ids(ids):
                raise ValueError("feature bundle input digest mismatch")
            for field in ("logits", "feature_a", "feature_b"):
                candidate, source = getattr(features, field), getattr(trusted, field)
                if (type(candidate) is not self._torch.Tensor or
                    candidate.device != source.device or
                    candidate.dtype != source.dtype or
                    candidate.shape != source.shape or
                    candidate.requires_grad or
                    not bool(self._torch.equal(candidate, source))):
                    raise ValueError(f"feature bundle {field} was modified")
            return self._copies(trusted)

    def _get_verified(self, ids: tuple[int, ...], *, partition: str,
                      episode_id: str, epoch: _FeatureEpoch | None) -> FrozenActionFeatures:
        with self._lock:
            if epoch is not self._epoch:
                raise RuntimeError("feature cache epoch ownership mismatch")
            ids, key = self._query(ids, partition, episode_id)
            if epoch is None:
                self._verify_model()
            elif self._tainted:
                raise RuntimeError("frozen model feature cache is invalidated")
            if key in self._items:
                self._hits += 1
                self._items.move_to_end(key)
                return self._copies(self._items[key])
            entry = self._compute(ids, partition, episode_id)
            if epoch is None:
                self._verify_model()
            self._items[key] = entry
            self._misses += 1
            if len(self._items) > self._limit:
                self._items.popitem(last=False)
            return self._copies(entry)

    def get_features(
        self, input_ids: tuple[int, ...], *, partition: str,
        episode_id: str,
    ) -> FrozenActionFeatures:
        """Verify frozen SHA on lookup; safe for one-off uncached scoring."""
        return self._get_verified(input_ids, partition=partition,
                                  episode_id=episode_id, epoch=None)

    @contextmanager
    def verified_epoch(self) -> Iterator[_FeatureEpoch]:
        """Read-only multi-arm batch with exact frozen SHA before and after.

        Caller must not mutate the LM or serve this epoch to another thread.
        Cached features may be reused by trained/shuffled/frozen heads, but
        actual outcomes and the model's training state are not cached.
        """
        with self._lock:
            if self._epoch is not None:
                raise RuntimeError("nested or concurrent feature epochs forbidden")
            self._verify_model()
            scope = _FeatureEpoch(self, get_ident())
            self._epoch = scope
        try:
            yield scope
        finally:
            with self._lock:
                scope._closed = True
                self._epoch = None
                self._verify_model()

    def chosen_training_feature(
        self, features: FrozenActionFeatures, *, chosen_action: str,
        expected_episode_id: str,
    ) -> Any:
        """Return only a chosen TRAIN hidden vector; never an EVAL feature.

        This is *not* a reward receipt validator. The training runner still
        must bind the sampled action to its committed outcome evidence.
        """
        if type(features) is not FrozenActionFeatures or features.partition != "TRAIN":
            raise PermissionError("only TRAIN features may enter a training update")
        if type(chosen_action) is not str or chosen_action not in ("A", "B"):
            raise ValueError("chosen_action must be A or B")
        if (features.episode_id != _episode(expected_episode_id) or
            features.frozen_model_sha256 != self._model_sha or
            features.checkpoint_sha256 != self._checkpoint or
            features.tokenizer_sha256 != self._tokenizer or
            features.action_token_ids != self._tokens or
            features.evidence_class != EVIDENCE_CLASS):
            raise ValueError("TRAIN feature identity does not match this model and episode")
        with self._lock:
            if self._epoch is None:
                self._verify_model()
            elif self._epoch._thread != get_ident() or self._tainted:
                raise RuntimeError("stale or cross-thread frozen feature epoch")
            # Never use a caller-mutated feature tensor as a gradient input.
            # A forged EVAL->TRAIN relabel cannot authorize unseen vectors:
            # only an original TRAIN cache entry supplies the chosen values.
            key = ("TRAIN", features.episode_id, features.input_ids_sha256)
            stored = self._items.get(key)
            if (stored is None or stored.partition != "TRAIN" or
                stored.frozen_model_sha256 != self._model_sha or
                stored.issuance_id != features.issuance_id):
                raise ValueError("TRAIN feature was not issued by this cache")
            return (stored.feature_a if chosen_action == "A" else
                    stored.feature_b).detach().clone()

    def invalidate(self) -> None:
        """Drop cached features; does not reauthorize a changed frozen model."""
        with self._lock:
            if self._epoch is not None:
                raise RuntimeError("cannot invalidate during a read-only epoch")
            self._items.clear()
