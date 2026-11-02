"""Fresh source-local v3 synthetic reward custody in a separate process.

Compared with v1/v2, v3 deliberately samples unfiltered random task pairs:
the oracle is never consulted during offer generation. Three TRAIN payoff
families share an explicitly learnable low-order coordinate structure, two
heldout families change its coefficients, and three unseen nonlinear payoff
rules are secondary transfer stress tests. All formulas are source-visible,
not independently secret or authenticated, and no experiential claim follows.

This module reuses the audited, bounded JSON IPC *transport* and one-use
commitment checks from the v2 worker without altering or executing v2
generator rules. It starts its own child module and maintains its own suite,
schedule and source-integrity manifest. Never move evaluators across hosts
or trust same-user source inspection as a malicious-host security boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
import inspect
import json
from math import tanh
import os
from pathlib import Path
from queue import Queue
import re
from secrets import token_hex
import subprocess
import sys
from threading import Lock, Thread
from typing import Any

from pleasure_experiment.outcome_arena import (
    CommitTicket, OutcomeReceipt, TaskOffer, _hex, _unit,
)
from pleasure_experiment.withheld_payoffs import PosthocOutcome
from pleasure_experiment.isolated_outcome_evaluator import (
    IsolatedOutcomeEvaluator, _NovelPayoffArena, _WorkerStudy,
    _checked_json, _model_record, _worker_call,
    PROTOCOL, EVIDENCE_CLASS, TRAIN_ARM, MAX_MESSAGE_BYTES, MAX_ARMS,
)


V3_VERSION = "PLEASURE_JOINT_REPRESENTATION_GENERATORS_V3"
V3_EVIDENCE_CLASS = "SOURCE_LOCAL_ISOLATED_OUTCOME_V3_UNAUTHENTICATED"
V3_TRAIN_FAMILIES = (
    "train_latent_00_v3", "train_latent_01_v3", "train_latent_02_v3",
)
V3_WITHIN_FAMILIES = ("eval_latent_00_v3", "eval_latent_01_v3")
V3_OOD_FAMILIES = ("eval_ood_00_v3", "eval_ood_01_v3", "eval_ood_02_v3")
V3_EVAL_FAMILIES = V3_WITHIN_FAMILIES + V3_OOD_FAMILIES
V3_ALL_FAMILIES = V3_TRAIN_FAMILIES + V3_EVAL_FAMILIES
DEFAULT_V3_ARMS = (
    "joint_on", "joint_off", "joint_inverted", "head_only_on",
    "shuffled_joint_on", "frozen_on", "frozen_off",
)
_LABEL = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_TRAIN_PAIRS_PER_FAMILY = 64
MAX_HELDOUT_PAIRS_PER_FAMILY = 64


def _reject_duplicate_fields(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ValueError("v3 IPC duplicate JSON object key")
        result[key] = value
    return result


def _strict_json(line: str) -> dict[str, Any]:
    if not line or len(line.encode("utf-8")) > MAX_MESSAGE_BYTES or not line.endswith("\n"):
        raise ValueError("v3 IPC line malformed or exceeds size bound")
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"v3 IPC nonfinite JSON constant: {value}")
    value = json.loads(line, object_pairs_hook=_reject_duplicate_fields,
                       parse_constant=reject_nonfinite)
    if type(value) is not dict:
        raise ValueError("v3 IPC top-level object required")
    return value


@dataclass(frozen=True)
class V3GeneratorManifest:
    train_families: tuple[str, ...]
    heldout_families: tuple[str, ...]
    primary_within_families: tuple[str, ...]
    secondary_ood_families: tuple[str, ...]
    train_generator_sha256: tuple[str, ...]
    heldout_generator_sha256: tuple[str, ...]
    suite_sha256: str
    evidence_class: str = V3_EVIDENCE_CLASS
    gap_filtering: bool = False
    external_attestation: bool = False


def _validated_v3(seed: object, evaluator_id: object, arms: object,
                  training_pairs_per_family: object,
                  heldout_pairs_per_generator: object) -> tuple[str, ...]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("evaluator seed must be a uint32")
    if type(evaluator_id) is not str or not 1 <= len(evaluator_id) <= 128:
        raise ValueError("nonempty bounded evaluator label required")
    if (type(arms) not in (tuple, list) or not 1 <= len(arms) <= MAX_ARMS or
        any(type(arm) is not str or _LABEL.fullmatch(arm) is None or
            arm == TRAIN_ARM for arm in arms) or len(set(arms)) != len(arms)):
        raise ValueError("1..8 distinct bounded non-TRAIN arms required")
    if (type(training_pairs_per_family) is not int or
        not 1 <= training_pairs_per_family <= MAX_TRAIN_PAIRS_PER_FAMILY):
        raise ValueError("training_pairs_per_family outside bounded schedule")
    if (type(heldout_pairs_per_generator) is not int or
        not 1 <= heldout_pairs_per_generator <= MAX_HELDOUT_PAIRS_PER_FAMILY):
        raise ValueError("heldout_pairs_per_generator outside bounded schedule")
    return tuple(arms)


def _v3_quality(family: str, features: tuple[float, float]) -> float:
    """Entirely new source-visible rules; the learner IPC never exposes q.

    Shared TRAIN and within-family EVAL tasks have positive x and negative y
    influence with mild interaction. Secondary OOD tasks require genuinely
    different nonlinear preference shapes, without any TRAIN label leakage.
    """
    x, y = features
    u, v = 2.0 * x - 1.0, 2.0 * y - 1.0
    if family in V3_TRAIN_FAMILIES + V3_WITHIN_FAMILIES:
        params = {
            V3_TRAIN_FAMILIES[0]: (0.50, 0.29, 0.75, 0.30, 0.04),
            V3_TRAIN_FAMILIES[1]: (0.49, 0.24, 0.62, 0.42, 0.03),
            V3_TRAIN_FAMILIES[2]: (0.51, 0.27, 0.67, 0.37, -0.03),
            V3_WITHIN_FAMILIES[0]: (0.50, 0.26, 0.70, 0.35, 0.04),
            V3_WITHIN_FAMILIES[1]: (0.48, 0.28, 0.65, 0.40, -0.02),
        }
        intercept, strength, a, b, interaction = params[family]
        q = intercept + strength * (a * u - b * v) + interaction * u * v
    elif family == V3_OOD_FAMILIES[0]:
        # Nonlinear signed cubic saddle: different from affine, old product,
        # parity, thresholds, Gaussian basins and periodic ridge objectives.
        q = 0.50 + 0.34 * tanh(2.2 * (u**3 - 0.70 * v**3 + 0.60 * u * v))
    elif family == V3_OOD_FAMILIES[1]:
        # Rotated L1-like corridor with a triangular response and no peaks
        # at Gaussian centers; outside TRAIN's monotone coordinate objective.
        corridor = abs(0.70 * x + 0.30 * y - 0.55) / 0.55
        q = 0.15 + 0.68 * max(0.0, 1.0 - corridor) ** 1.5
    elif family == V3_OOD_FAMILIES[2]:
        # Superlinear disagreement objective, explicitly nonmonotone in x,y.
        q = 0.16 + 0.70 * abs(x - y) ** 1.7
    else:
        raise ValueError("unknown v3 quality family")
    return _unit(max(0.0, min(1.0, q)), "v3 synthetic quality")


def _rule_sha(family: str) -> str:
    if type(family) is not str or family not in V3_ALL_FAMILIES:
        raise ValueError("unknown v3 source-visible rule")
    source_digest = sha256(inspect.getsource(_v3_quality).encode("utf-8")).hexdigest()
    return _hex(V3_VERSION, family, source_digest)


class _V3PayoffArena(_NovelPayoffArena):
    """Worker-only unique schedule and chosen-only source-local oracle."""

    @property
    def manifest(self) -> V3GeneratorManifest:
        train = tuple(_rule_sha(family) for family in V3_TRAIN_FAMILIES)
        heldout = tuple(_rule_sha(family) for family in V3_EVAL_FAMILIES)
        return V3GeneratorManifest(
            train_families=V3_TRAIN_FAMILIES,
            heldout_families=V3_EVAL_FAMILIES,
            primary_within_families=V3_WITHIN_FAMILIES,
            secondary_ood_families=V3_OOD_FAMILIES,
            train_generator_sha256=train,
            heldout_generator_sha256=heldout,
            suite_sha256=_hex(V3_VERSION, train, heldout, "unfiltered"),
        )

    def issue(self, *, partition: str, family_id: str,
              pair_index: int, side: int) -> TaskOffer:
        if self._sealed:
            raise ValueError("v3 outcome arena already sealed")
        if ((partition == "TRAIN" and family_id not in V3_TRAIN_FAMILIES) or
            (partition == "EVAL" and family_id not in V3_EVAL_FAMILIES) or
            partition not in ("TRAIN", "EVAL") or
            type(pair_index) is not int or not 0 <= pair_index <= 4095 or
            type(side) is not int or side not in (0, 1)):
            raise ValueError("v3 outcome schedule or partition invalid")
        # Critical: NO payoff-dependent rejection/sampling. Coordinates are
        # determined by seed+partition+family+index before any oracle scoring.
        digest = bytes.fromhex(_hex(V3_VERSION, self._seed, partition,
                                    family_id, pair_index, "unfiltered"))
        values = tuple((int.from_bytes(digest[i:i + 4], "big") + 0.5) / 2**32
                       for i in (0, 4, 8, 12))
        a = (0.05 + 0.90 * values[0], 0.05 + 0.90 * values[1])
        b = (0.05 + 0.90 * values[2], 0.05 + 0.90 * values[3])
        offer = TaskOffer(
            task_id=_hex("task", V3_VERSION, self._seed, partition,
                         family_id, pair_index, side),
            episode_id=_hex("episode", V3_VERSION, self._seed, partition,
                            family_id, pair_index, side),
            partition=partition, family_id=family_id,
            pair_index=pair_index, side=side,
            nonce=_hex("nonce", V3_VERSION, self._seed, partition,
                       family_id, pair_index, side),
            features_a=b if side else a,
            features_b=a if side else b,
        )
        prior = self._issued.setdefault(offer.task_id, replace(offer))
        if prior != offer:
            raise RuntimeError("v3 task identity changed")
        return replace(prior)

    def resolve(self, ticket: CommitTicket) -> OutcomeReceipt:
        if type(ticket) is not CommitTicket:
            raise TypeError("exact v3 commit ticket required")
        fixed = replace(ticket)
        stored = self._committed.get(fixed.task_id)
        if (self._sealed or stored is None or stored != fixed or
            fixed.task_id in self._resolved):
            raise ValueError("v3 ticket unknown, altered or already resolved")
        offer = self._issued[fixed.task_id]
        quality = _v3_quality(
            offer.family_id,
            offer.features_a if fixed.action == "A" else offer.features_b,
        )
        rule = _rule_sha(offer.family_id)
        receipt = OutcomeReceipt(
            task_id=offer.task_id, episode_id=offer.episode_id,
            partition=offer.partition, family_id=offer.family_id,
            action=fixed.action, model_id=fixed.model_id,
            runtime_instance_id=fixed.runtime_instance_id,
            measured_quality=quality, decision_digest=fixed.decision_digest,
            evidence_id=_hex("outcome", V3_VERSION, fixed.commit_id, quality, rule),
            evaluator_id=self._evaluator_id, rubric_digest=rule,
            evidence_class=V3_EVIDENCE_CLASS,
        )
        self._resolved.add(fixed.task_id)
        return receipt

    def posthoc_evaluate(self, offer: TaskOffer) -> PosthocOutcome:
        fixed = replace(offer)
        if not self._sealed:
            raise PermissionError("v3 posthoc unavailable before complete seal")
        if (fixed.partition != "EVAL" or
            self._issued.get(fixed.task_id) != fixed or
            fixed.task_id not in self._resolved):
            raise PermissionError("v3 posthoc limited to resolved EVAL tasks")
        chosen_action = self._committed[fixed.task_id].action
        a = _v3_quality(fixed.family_id, fixed.features_a)
        b = _v3_quality(fixed.family_id, fixed.features_b)
        chosen, unchosen = (a, b) if chosen_action == "A" else (b, a)
        return PosthocOutcome(
            task_id=fixed.task_id, episode_id=fixed.episode_id,
            family_id=fixed.family_id, chosen_action=chosen_action,
            chosen_quality=chosen, unchosen_quality=unchosen,
            quality_a=a, quality_b=b,
            oracle_action="A" if a > b else "B", oracle_quality=max(a, b),
            regret=max(a, b) - chosen,
            generator_sha256=_rule_sha(fixed.family_id),
            evidence_class=V3_EVIDENCE_CLASS,
        )


class _V3WorkerStudy(_WorkerStudy):
    """Reuse audited v2 one-use/schedule/barrier logic with new suite state."""

    def __init__(self, *, seed: int, evaluator_id: str, arms: tuple[str, ...],
                 training_pairs_per_family: int,
                 heldout_pairs_per_generator: int):
        self.arms = _validated_v3(seed, evaluator_id, arms,
                                  training_pairs_per_family,
                                  heldout_pairs_per_generator)
        self.suite = "novel_v3"
        self._arenas = {
            arm: _V3PayoffArena(seed=seed, evaluator_id=evaluator_id)
            for arm in (TRAIN_ARM,) + self.arms
        }
        self.manifest = self._arenas[TRAIN_ARM].manifest
        self._offers: dict[tuple[str, str, int, int], TaskOffer] = {}
        self._offer_by_task: dict[tuple[str, str], tuple[str, str, int, int]] = {}
        self._client_issued: set[tuple[str, str, int, int]] = set()
        self._tickets: dict[tuple[str, str, int, int], CommitTicket] = {}
        self._resolved: set[tuple[str, str, int, int]] = set()
        self._decisions_sealed = False
        self._sealed = False
        for family in V3_TRAIN_FAMILIES:
            self._add(TRAIN_ARM, "TRAIN", family, training_pairs_per_family)
        for arm in self.arms:
            for family in V3_EVAL_FAMILIES:
                self._add(arm, "EVAL", family, heldout_pairs_per_generator)


def _v3_manifest_record(raw: object) -> V3GeneratorManifest:
    if type(raw) is not dict or set(raw) != {field.name for field in fields(V3GeneratorManifest)}:
        raise ValueError("malformed v3 generator manifest")
    data = dict(raw)
    for name in (
        "train_families", "heldout_families", "primary_within_families",
        "secondary_ood_families", "train_generator_sha256",
        "heldout_generator_sha256",
    ):
        if type(data[name]) is not list:
            raise ValueError("malformed v3 tuple field")
        data[name] = tuple(data[name])
    result = V3GeneratorManifest(**data)
    expected = _V3PayoffArena(seed=0, evaluator_id="schema-check").manifest
    if result != expected:
        raise ValueError("v3 generator manifest does not match immutable source")
    return result


def _worker_main() -> int:
    study: _V3WorkerStudy | None = None
    session: str | None = None
    expected_id = 0
    while True:
        line = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            return 0
        if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES or not line.endswith("\n"):
            return 2
        request: dict[str, Any] = {}
        try:
            request = _strict_json(line)
            if set(request) != {"protocol", "id", "session", "command", "body"}:
                raise ValueError("invalid v3 IPC schema")
            if (request["protocol"] != PROTOCOL or
                type(request["id"]) is not int or request["id"] != expected_id or
                type(request["command"]) is not str or
                type(request["body"]) is not dict):
                raise ValueError("v3 IPC version/sequence mismatch")
            if study is None and (request["session"] is not None or
                                  request["command"] != "initialize"):
                raise ValueError("initialize v3 evaluator first")
            if study is not None and request["session"] != session:
                raise ValueError("v3 evaluator session mismatch")
        except (ValueError, TypeError, KeyError, UnicodeError):
            return 2
        try:
            expected_id += 1
            command = request["command"]
            if study is None:
                body = request["body"]
                if set(body) != {
                    "seed", "evaluator_id", "arms", "training_pairs_per_family",
                    "heldout_pairs_per_generator", "suite", "nonce",
                }:
                    raise ValueError("v3 initializer has unexpected fields")
                if body["suite"] != "novel_v3":
                    raise ValueError("v3 worker accepts only its isolated suite")
                nonce = body["nonce"]
                if type(nonce) is not str or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
                    raise ValueError("invalid v3 nonce")
                arms = _validated_v3(
                    body["seed"], body["evaluator_id"], body["arms"],
                    body["training_pairs_per_family"],
                    body["heldout_pairs_per_generator"],
                )
                study = _V3WorkerStudy(
                    seed=body["seed"], evaluator_id=body["evaluator_id"],
                    arms=arms, training_pairs_per_family=body["training_pairs_per_family"],
                    heldout_pairs_per_generator=body["heldout_pairs_per_generator"],
                )
                session = sha256((nonce + token_hex(16) + str(os.getpid())).encode()).hexdigest()
                result: Any = {
                    "pid": os.getpid(), "session": session,
                    "manifest": asdict(study.manifest),
                    "evidence_class": EVIDENCE_CLASS,
                }
            elif command == "shutdown":
                if request["body"]:
                    raise ValueError("shutdown body must be empty")
                result = {"closed": True}
            else:
                result = _worker_call(study, command, request["body"])
            payload = asdict(result) if is_dataclass(result) else result
            if command == "resolve":
                payload = {"receipt": payload, "provenance": {
                    "worker_pid": os.getpid(), "session": session,
                    "protocol": PROTOCOL,
                    "receipt_sha256": _hex("ipc-receipt-v1", os.getpid(), session,
                                           _checked_json(payload)),
                }}
            reply = {"protocol": PROTOCOL, "id": request["id"],
                     "session": session, "ok": True, "result": payload}
        except (ValueError, TypeError, KeyError, PermissionError, RuntimeError) as exc:
            reply = {"protocol": PROTOCOL, "id": expected_id - 1,
                     "session": session, "ok": False,
                     "error_type": type(exc).__name__, "error": str(exc)[:512]}
        try:
            print(_checked_json(reply), flush=True)
        except (OSError, ValueError):
            return 2
        if study is not None and request.get("command") == "shutdown" and reply.get("ok"):
            return 0


class V3OutcomeEvaluator(IsolatedOutcomeEvaluator):
    """Typed v3 child process; inherits audited bounded IPC methods only."""

    def __init__(self, *, seed: int, evaluator_id: str,
                 arms: tuple[str, ...] = DEFAULT_V3_ARMS,
                 training_pairs_per_family: int = 8,
                 heldout_pairs_per_generator: int = 4,
                 timeout_seconds: float = 10.0):
        _validated_v3(seed, evaluator_id, arms,
                      training_pairs_per_family, heldout_pairs_per_generator)
        if (type(timeout_seconds) not in (int, float) or
            not 0.1 <= float(timeout_seconds) <= 60.0):
            raise ValueError("v3 timeout_seconds must be within [0.1,60]")
        self._timeout = float(timeout_seconds)
        self._lock = Lock()
        self._responses: Queue[str | None] = Queue()
        self._session: str | None = None
        self._counter = 0
        self._closed = False
        self._worker_pid: int | None = None
        self._manifest: V3GeneratorManifest | None = None
        self._suite = "novel_v3"
        root = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR")}
        env["PYTHONPATH"] = str(root)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [sys.executable, "-B", "-m",
             "pleasure_experiment.isolated_outcome_evaluator_v3", "--worker"],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            close_fds=True,
        )
        self._reader = Thread(target=self._read_responses, daemon=True)
        self._reader.start()
        try:
            initial = self._request("initialize", {
                "seed": seed, "evaluator_id": evaluator_id,
                "arms": list(arms),
                "training_pairs_per_family": training_pairs_per_family,
                "heldout_pairs_per_generator": heldout_pairs_per_generator,
                "suite": "novel_v3", "nonce": token_hex(16),
            }, initialization=True)
            if (type(initial) is not dict or
                set(initial) != {"pid", "session", "manifest", "evidence_class"} or
                initial["evidence_class"] != EVIDENCE_CLASS or
                type(initial["pid"]) is not int or initial["pid"] <= 0 or
                type(initial["session"]) is not str or
                _SHA.fullmatch(initial["session"]) is None):
                raise RuntimeError("v3 evaluator handshake malformed")
            if initial["pid"] == os.getpid() or initial["pid"] != self._process.pid:
                raise RuntimeError("v3 evaluator is not distinct spawned OS process")
            self._worker_pid = initial["pid"]
            self._session = initial["session"]
            self._manifest = _v3_manifest_record(initial["manifest"])
        except Exception:
            self.close(force=True)
            raise

    def _read_responses(self) -> None:
        """Reject malformed/duplicate-key replies even before v2 client parse.

        This transport guard does not authenticate a malicious same-user OS
        process. It prevents a noncanonical reply from reaching the inherited
        parser's last-wins JSON behavior.
        """
        stream = self._process.stdout
        if stream is None:
            self._responses.put(None)
            return
        try:
            while True:
                line = stream.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                try:
                    _strict_json(line)
                except (UnicodeError, ValueError, json.JSONDecodeError):
                    break
                self._responses.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._responses.put(None)

    @property
    def manifest(self) -> V3GeneratorManifest:
        if self._manifest is None:
            raise RuntimeError("v3 evaluator manifest unavailable")
        return self._manifest

    @property
    def train_families(self) -> tuple[str, ...]:
        return self.manifest.train_families

    @property
    def eval_families(self) -> tuple[str, ...]:
        return self.manifest.heldout_families

    @property
    def primary_within_families(self) -> tuple[str, ...]:
        return self.manifest.primary_within_families

    @property
    def primary_ood_families(self) -> tuple[str, ...]:
        return self.manifest.secondary_ood_families

    @property
    def provenance(self) -> dict[str, object]:
        return {"worker_pid": self.worker_pid, "session": self._session,
                "protocol": PROTOCOL, "suite": "novel_v3",
                "evidence_class": EVIDENCE_CLASS,
                "generator_evidence_class": V3_EVIDENCE_CLASS}


def seeded_derangement(seed: int, count: int) -> tuple[int, ...]:
    """Uniform rejection-sampled seeded derangement, not fixed cyclic offsets.

    This is a *synthetic* negative-control donor map, never an outcome receipt.
    Selection uses a separate seed and no label magnitudes. Random draws are
    reproducible, and every target has a distinct donor and no fixed point.
    """
    import random
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("shuffle seed must be uint32")
    if type(count) is not int or not 3 <= count <= 256:
        raise ValueError("derangement count must be in [3,256]")
    rng = random.Random(seed ^ 0x6C5EED9A)
    original = tuple(range(count))
    for _ in range(8192):
        donor = list(original)
        rng.shuffle(donor)
        if all(i != donor[i] for i in original):
            return tuple(donor)
    raise RuntimeError("seeded derangement rejection budget exhausted")


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    raise SystemExit(_worker_main())
