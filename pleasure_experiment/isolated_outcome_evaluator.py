"""Offline JSON-line, separate-process custody for synthetic payoff outcomes.

One child process owns every per-arm WithheldPayoffArena for one seed. Its
parent receives offers and *only chosen-action* reward receipts until **all**
predeclared TRAIN and EVAL commitments across **all** arms have resolved.
Only then can ``finalize_evaluation`` unlock posthoc counterfactual scores.

The API does not transfer Python objects through a serializer, run eval(),
open sockets, or execute any caller-supplied code. Pipes carry bounded JSON.
The process boundary prevents accidental direct access to oracle instances,
not deliberate inspection by the same OS user. The generator code and seed
remain source-local; no external evaluator, secure enclave, real-world reward
authority, or subjective feeling is authenticated here.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass, replace
from hashlib import sha256
import inspect
import json
from math import cos, exp, pi, sin
import os
from pathlib import Path
from queue import Empty, Queue
import re
from secrets import token_hex
import subprocess
import sys
from threading import Lock, Thread
from typing import Any

from pleasure_experiment.outcome_arena import (
    CommitTicket, OutcomeReceipt, TaskOffer, _digest_hex, _hex, _identifier, _unit,
)
from pleasure_experiment.withheld_payoffs import (
    EVAL_FAMILIES as LEGACY_EVAL_FAMILIES,
    TRAIN_FAMILY as LEGACY_TRAIN_FAMILY,
    GeneratorManifest, PosthocOutcome,
)


PROTOCOL = "pleasure-source-local-evaluator-ipc-v1"
EVIDENCE_CLASS = "SOURCE_LOCAL_SEPARATE_PROCESS_EVALUATOR_UNAUTHENTICATED"
TRAIN_ARM = "shared_train"
DEFAULT_ARMS = (
    "trained_on", "trained_off", "trained_inverted", "shuffled_on", "frozen_on",
)
NOVEL_TRAIN_FAMILY = "sealed_train_affine_v2"
# Opaque public EVAL family labels avoid disclosing the structural rule class
# in an offer before the study seal. This is protocol hygiene, not secrecy:
# same-user inspection of this module still reveals every generator.
NOVEL_IN_DISTRIBUTION_FAMILY = "sealed_eval_00_v2"
NOVEL_OOD_FAMILIES = (
    "sealed_eval_01_v2",
    "sealed_eval_02_v2",
    "sealed_eval_03_v2",
)
NOVEL_EVAL_FAMILIES = (NOVEL_IN_DISTRIBUTION_FAMILY,) + NOVEL_OOD_FAMILIES
SUITES = ("novel_v2", "legacy_v1")
NOVEL_GENERATOR_VERSION = "PLEASURE_SEALED_SYNTHETIC_OOD_GENERATORS_V2"
NOVEL_EVIDENCE_CLASS = "SOURCE_LOCAL_ISOLATED_UNSEEN_GENERATOR_V2_UNAUTHENTICATED"
MAX_MESSAGE_BYTES = 16 * 1024
MAX_ARMS = 8
MAX_TRAIN_PAIRS = 64
MAX_EVAL_PAIRS_PER_GENERATOR = 64
_LABEL = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _checked_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False)
    if not 1 <= len(encoded.encode("ascii")) < MAX_MESSAGE_BYTES:
        raise ValueError("evaluator IPC payload exceeds byte limit")
    return encoded


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric constant {value}")


def _parsed_json(line: str) -> dict[str, Any]:
    if not line or len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("evaluator IPC line exceeds byte limit")
    decoded = json.loads(line, parse_constant=_reject_json_constant)
    if type(decoded) is not dict:
        raise ValueError("evaluator IPC message must be an object")
    return decoded


def _model_record(kind: type, raw: object) -> Any:
    if type(raw) is not dict or set(raw) != {field.name for field in fields(kind)}:
        raise ValueError(f"invalid {kind.__name__} IPC record fields")
    data = dict(raw)
    if kind is TaskOffer:
        for name in ("features_a", "features_b"):
            if type(data[name]) is not list or len(data[name]) != 2:
                raise ValueError("invalid evaluator IPC offer features")
            data[name] = tuple(data[name])
    elif kind is GeneratorManifest:
        for name in ("heldout_families", "heldout_generator_sha256"):
            if type(data[name]) is not list:
                raise ValueError("invalid evaluator generator manifest")
            data[name] = tuple(data[name])
    return kind(**data)


def _validated_study(seed: object, evaluator_id: object, arms: object,
                     training_pairs: object, heldout_pairs: object,
                     suite: object) -> tuple[str, ...]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("evaluator seed must be uint32")
    if type(evaluator_id) is not str or not 1 <= len(evaluator_id) <= 128:
        raise ValueError("bounded evaluator label required")
    if (type(arms) not in (tuple, list) or not 1 <= len(arms) <= MAX_ARMS or
        any(type(arm) is not str or _LABEL.fullmatch(arm) is None or
            arm == TRAIN_ARM for arm in arms) or len(set(arms)) != len(arms)):
        raise ValueError("arms must be distinct bounded labels")
    if type(training_pairs) is not int or not 1 <= training_pairs <= MAX_TRAIN_PAIRS:
        raise ValueError("training_pairs outside bounded schedule")
    if (type(heldout_pairs) is not int or
        not 1 <= heldout_pairs <= MAX_EVAL_PAIRS_PER_GENERATOR):
        raise ValueError("heldout_pairs_per_generator outside bounded schedule")
    if type(suite) is not str or suite not in SUITES:
        raise ValueError("unsupported versioned evaluator generator suite")
    return tuple(arms)


def _novel_quality(family: str, features: tuple[float, float]) -> float:
    """Fresh versioned synthetic payoff rules, distinct from prior PR #22.

    These definitions are intentionally worker-owned, not exported through
    learner IPC. They are still visible to a same-user source-code reader.
    """
    x, y = features
    if family in (NOVEL_TRAIN_FAMILY, NOVEL_IN_DISTRIBUTION_FAMILY):
        q = 0.50 + 0.27 * (0.72 * (2 * x - 1) - 0.38 * (2 * y - 1))
    elif family == NOVEL_OOD_FAMILIES[0]:
        # Two-feature parity/checkerboard: neither coordinate's monotone
        # reward gradient suffices to identify a good region.
        q = 0.18 + 0.63 * (int(x > 0.52) ^ int(y > 0.47)) + 0.11 * (x - y)
    elif family == NOVEL_OOD_FAMILIES[1]:
        # Two separated Gaussian optima; not an affine, radial single-basin,
        # pairwise multiplication or one-dimensional threshold rule.
        q = (0.17 + 0.54 * exp(-10 * ((x - 0.25) ** 2 + (y - 0.75) ** 2))
             + 0.43 * exp(-12 * ((x - 0.77) ** 2 + (y - 0.30) ** 2)))
    elif family == NOVEL_OOD_FAMILIES[2]:
        # Periodic cross-coordinate ridges with multiple preference reversals.
        q = 0.48 + 0.30 * sin(3 * pi * x + 4 * pi * y) * cos(2 * pi * y)
    else:
        raise ValueError("unknown novel outcome generator")
    return _unit(max(0.0, min(1.0, q)), "novel quality")


def _novel_rule_digest(family: str) -> str:
    if family not in (NOVEL_TRAIN_FAMILY,) + NOVEL_EVAL_FAMILIES:
        raise ValueError("unknown novel generator family")
    # Pin the implementation as shipped, not merely an easily forgotten
    # manually maintained version string. This is source integrity only;
    # neither this digest nor the worker is externally attested.
    return _hex(NOVEL_GENERATOR_VERSION, family,
                sha256(inspect.getsource(_novel_quality).encode("utf-8")).hexdigest())


class _NovelPayoffArena:
    """Worker-only one-use arena for a new, source-local payoff family set."""

    def __init__(self, *, seed: int, evaluator_id: str):
        self._seed = seed
        self._evaluator_id = evaluator_id
        self._issued: dict[str, TaskOffer] = {}
        self._committed: dict[str, CommitTicket] = {}
        self._resolved: set[str] = set()
        self._sealed = False

    @property
    def manifest(self) -> GeneratorManifest:
        train_digest = _novel_rule_digest(NOVEL_TRAIN_FAMILY)
        eval_digests = tuple(_novel_rule_digest(family) for family in NOVEL_EVAL_FAMILIES)
        return GeneratorManifest(
            train_family=NOVEL_TRAIN_FAMILY,
            heldout_families=NOVEL_EVAL_FAMILIES,
            train_generator_sha256=train_digest,
            heldout_generator_sha256=eval_digests,
            suite_sha256=_hex(NOVEL_GENERATOR_VERSION, train_digest, eval_digests),
            evidence_class=NOVEL_EVIDENCE_CLASS,
        )

    def issue(self, *, partition: str, family_id: str,
              pair_index: int, side: int) -> TaskOffer:
        if self._sealed:
            raise ValueError("novel outcome arena sealed")
        if ((partition == "TRAIN" and family_id != NOVEL_TRAIN_FAMILY) or
            (partition == "EVAL" and family_id not in NOVEL_EVAL_FAMILIES) or
            partition not in ("TRAIN", "EVAL") or
            type(pair_index) is not int or not 0 <= pair_index <= 4095 or
            type(side) is not int or side not in (0, 1)):
            raise ValueError("novel source-local generator partition/schedule invalid")
        for attempt in range(128):
            digest = bytes.fromhex(_hex(
                NOVEL_GENERATOR_VERSION, self._seed, partition,
                family_id, pair_index, attempt,
            ))
            values = tuple(
                (int.from_bytes(digest[i:i + 4], "big") + 0.5) / 2**32
                for i in (0, 4, 8, 12)
            )
            a = (0.05 + 0.90 * values[0], 0.05 + 0.90 * values[1])
            b = (0.05 + 0.90 * values[2], 0.05 + 0.90 * values[3])
            if abs(_novel_quality(family_id, a) - _novel_quality(family_id, b)) > 0.02:
                break
        else:
            raise RuntimeError("could not issue distinguishable novel offers")
        offer = TaskOffer(
            task_id=_hex("task", NOVEL_GENERATOR_VERSION, self._seed,
                         partition, family_id, pair_index, side),
            episode_id=_hex("episode", NOVEL_GENERATOR_VERSION, self._seed,
                            partition, family_id, pair_index, side),
            partition=partition, family_id=family_id, pair_index=pair_index,
            side=side, nonce=_hex("nonce", NOVEL_GENERATOR_VERSION, self._seed,
                                  partition, family_id, pair_index, side),
            features_a=b if side else a, features_b=a if side else b,
        )
        prior = self._issued.setdefault(offer.task_id, replace(offer))
        if prior != offer:
            raise RuntimeError("novel task schedule diverged")
        return replace(prior)

    def commit(self, offer: TaskOffer, action: str, decision_digest: str,
               *, model_id: str, runtime_instance_id: str) -> CommitTicket:
        if type(offer) is not TaskOffer:
            raise TypeError("exact task offer required")
        if type(action) is not str or action not in ("A", "B"):
            raise ValueError("action must be A or B")
        _digest_hex(decision_digest, "decision_digest")
        _identifier(model_id, "model_id")
        _identifier(runtime_instance_id, "runtime_instance_id")
        fixed = replace(offer)
        if (self._sealed or self._issued.get(fixed.task_id) != fixed or
            fixed.task_id in self._committed):
            raise ValueError("novel task unknown, altered or already committed")
        ticket = CommitTicket(
            task_id=fixed.task_id, episode_id=fixed.episode_id, action=action,
            model_id=model_id, runtime_instance_id=runtime_instance_id,
            decision_digest=decision_digest,
            commit_id=_hex("commit", fixed.task_id, action, model_id,
                           runtime_instance_id, decision_digest, fixed.nonce),
        )
        self._committed[fixed.task_id] = replace(ticket)
        return replace(ticket)

    def resolve(self, ticket: CommitTicket) -> OutcomeReceipt:
        if type(ticket) is not CommitTicket:
            raise TypeError("exact commit ticket required")
        fixed = replace(ticket)
        stored = self._committed.get(fixed.task_id)
        if (self._sealed or stored is None or stored != fixed or
            fixed.task_id in self._resolved):
            raise ValueError("novel ticket unknown, altered or already resolved")
        offer = self._issued[fixed.task_id]
        quality = _novel_quality(
            offer.family_id,
            offer.features_a if fixed.action == "A" else offer.features_b,
        )
        rule_digest = _novel_rule_digest(offer.family_id)
        receipt = OutcomeReceipt(
            task_id=offer.task_id, episode_id=offer.episode_id,
            partition=offer.partition, family_id=offer.family_id,
            action=fixed.action, model_id=fixed.model_id,
            runtime_instance_id=fixed.runtime_instance_id,
            measured_quality=quality, decision_digest=fixed.decision_digest,
            evidence_id=_hex("outcome", fixed.commit_id, quality, rule_digest),
            evaluator_id=self._evaluator_id, rubric_digest=rule_digest,
            evidence_class=NOVEL_EVIDENCE_CLASS,
        )
        self._resolved.add(fixed.task_id)
        return receipt

    def finalize_evaluation(self) -> None:
        if self._sealed or not any(o.partition == "EVAL" for o in self._issued.values()):
            raise ValueError("novel arena already sealed or has no EVAL")
        if set(self._issued) != self._resolved:
            raise ValueError("novel arena cannot seal unresolved tasks")
        self._sealed = True

    def posthoc_evaluate(self, offer: TaskOffer) -> PosthocOutcome:
        fixed = replace(offer)
        if not self._sealed:
            raise PermissionError("novel posthoc requires seal")
        if (fixed.partition != "EVAL" or
            self._issued.get(fixed.task_id) != fixed or
            fixed.task_id not in self._resolved):
            raise PermissionError("novel posthoc is only for resolved EVAL offers")
        chosen_action = self._committed[fixed.task_id].action
        a = _novel_quality(fixed.family_id, fixed.features_a)
        b = _novel_quality(fixed.family_id, fixed.features_b)
        chosen, unchosen = (a, b) if chosen_action == "A" else (b, a)
        return PosthocOutcome(
            task_id=fixed.task_id, episode_id=fixed.episode_id,
            family_id=fixed.family_id, chosen_action=chosen_action,
            chosen_quality=chosen, unchosen_quality=unchosen,
            quality_a=a, quality_b=b,
            oracle_action="A" if a > b else "B", oracle_quality=max(a, b),
            regret=max(a, b) - chosen,
            generator_sha256=_novel_rule_digest(fixed.family_id),
            evidence_class=NOVEL_EVIDENCE_CLASS,
        )


class _WorkerStudy:
    """Only instantiated in the child-process command handler."""

    def __init__(self, *, seed: int, evaluator_id: str, arms: tuple[str, ...],
                 training_pairs: int, heldout_pairs_per_generator: int,
                 suite: str):
        from pleasure_experiment.withheld_payoffs import WithheldPayoffArena

        self.arms = _validated_study(
            seed, evaluator_id, arms, training_pairs,
            heldout_pairs_per_generator, suite,
        )
        self.suite = suite
        arena_class = _NovelPayoffArena if suite == "novel_v2" else WithheldPayoffArena
        train_family = (NOVEL_TRAIN_FAMILY if suite == "novel_v2"
                        else LEGACY_TRAIN_FAMILY)
        eval_families = (NOVEL_EVAL_FAMILIES if suite == "novel_v2"
                         else LEGACY_EVAL_FAMILIES)
        self._arenas = {arm: arena_class(seed=seed, evaluator_id=evaluator_id)
                        for arm in (TRAIN_ARM,) + self.arms}
        self.manifest = self._arenas[TRAIN_ARM].manifest
        self._offers: dict[tuple[str, str, int, int], TaskOffer] = {}
        self._offer_by_task: dict[tuple[str, str], tuple[str, str, int, int]] = {}
        self._client_issued: set[tuple[str, str, int, int]] = set()
        self._tickets: dict[tuple[str, str, int, int], CommitTicket] = {}
        self._resolved: set[tuple[str, str, int, int]] = set()
        self._decisions_sealed = False
        self._sealed = False
        self._add(TRAIN_ARM, "TRAIN", train_family, training_pairs)
        for arm in self.arms:
            for family in eval_families:
                self._add(arm, "EVAL", family, heldout_pairs_per_generator)

    def _add(self, arm: str, partition: str, family_id: str, pairs: int) -> None:
        arena = self._arenas[arm]
        for pair_index in range(pairs):
            offer = arena.issue(partition=partition, family_id=family_id,
                                pair_index=pair_index, side=0)
            key = (arm, family_id, pair_index, 0)
            self._offers[key] = offer
            self._offer_by_task[(arm, offer.task_id)] = key

    @staticmethod
    def _key(arm: str, offer: TaskOffer) -> tuple[str, str, int, int]:
        return (arm, offer.family_id, offer.pair_index, offer.side)

    def issue(self, *, arm: str, family_id: str, pair_index: int, side: int,
              partition: str) -> TaskOffer:
        key = (arm, family_id, pair_index, side)
        offer = self._offers.get(key)
        if offer is None or offer.partition != partition or key in self._client_issued:
            raise ValueError("unknown, wrong-partition, or already issued evaluator offer")
        if self._sealed or self._decisions_sealed:
            raise ValueError("evaluation is sealed")
        self._client_issued.add(key)
        return offer

    def commit(self, *, arm: str, offer: TaskOffer, action: str,
               decision_digest: str, model_id: str,
               runtime_instance_id: str) -> CommitTicket:
        key = self._key(arm, offer)
        if (self._sealed or self._decisions_sealed or key not in self._client_issued or
            self._offers.get(key) != offer or key in self._tickets):
            raise ValueError("unissued, altered, replayed, or sealed evaluator offer")
        ticket = self._arenas[arm].commit(
            offer, action, decision_digest, model_id=model_id,
            runtime_instance_id=runtime_instance_id,
        )
        self._tickets[key] = ticket
        return ticket

    def resolve(self, *, arm: str, ticket: CommitTicket) -> OutcomeReceipt:
        key = self._offer_by_task.get((arm, ticket.task_id))
        if (self._sealed or key is None or key in self._resolved or
            self._tickets.get(key) != ticket):
            raise ValueError("unknown, replayed, altered, or sealed evaluator ticket")
        if key[0] != TRAIN_ARM and not self._decisions_sealed:
            raise PermissionError("EVAL quality is withheld until all arm choices are sealed")
        outcome = self._arenas[arm].resolve(ticket)
        self._resolved.add(key)
        return outcome

    def seal_decisions(self) -> dict[str, object]:
        """Freeze *every* arm's choices before returning any EVAL reward.

        Otherwise two different actions chosen by matched arms could reveal
        both counterfactuals before the other arms had committed decisions.
        """
        if self._decisions_sealed or self._sealed:
            raise ValueError("EVAL choices already sealed")
        train_keys = {key for key in self._offers if key[0] == TRAIN_ARM}
        eval_keys = set(self._offers) - train_keys
        if not train_keys.issubset(self._resolved) or not eval_keys.issubset(self._tickets):
            raise ValueError("all TRAIN receipts and all EVAL decisions required before decision seal")
        self._decisions_sealed = True
        return {"decisions_sealed": True, "eval_commits": len(eval_keys),
                "train_resolved": len(train_keys)}

    def finalize_evaluation(self) -> dict[str, object]:
        if self._sealed:
            raise ValueError("evaluator already sealed")
        if not self._decisions_sealed or set(self._offers) != self._resolved:
            raise ValueError("all predeclared TRAIN and EVAL arm tasks must resolve before seal")
        for arm in self.arms:
            self._arenas[arm].finalize_evaluation()
        self._sealed = True
        return {"sealed": True, "all_arms_committed": True,
                "resolved_tasks": len(self._resolved), "arm_count": len(self.arms)}

    def posthoc_evaluate(self, *, arm: str, offer: TaskOffer) -> PosthocOutcome:
        if not self._sealed or arm not in self.arms:
            raise PermissionError("posthoc requires whole-study seal and EVAL arm")
        key = self._key(arm, offer)
        if self._offers.get(key) != offer or key not in self._resolved:
            raise ValueError("posthoc offer is not a matched resolved EVAL task")
        return self._arenas[arm].posthoc_evaluate(offer)


def _worker_call(study: _WorkerStudy, command: str, body: dict[str, Any]) -> object:
    if command == "issue":
        if set(body) != {"arm", "partition", "family_id", "pair_index", "side"}:
            raise ValueError("invalid issue payload")
        return study.issue(**body)
    if command == "commit":
        if set(body) != {"arm", "offer", "action", "decision_digest",
                         "model_id", "runtime_instance_id"}:
            raise ValueError("invalid commit payload")
        body = dict(body)
        body["offer"] = _model_record(TaskOffer, body["offer"])
        return study.commit(**body)
    if command == "resolve":
        if set(body) != {"arm", "ticket"}:
            raise ValueError("invalid resolve payload")
        return study.resolve(arm=body["arm"],
                             ticket=_model_record(CommitTicket, body["ticket"]))
    if command == "finalize_evaluation":
        if body:
            raise ValueError("seal payload must be empty")
        return study.finalize_evaluation()
    if command == "seal_decisions":
        if body:
            raise ValueError("decision seal payload must be empty")
        return study.seal_decisions()
    if command == "posthoc_evaluate":
        if set(body) != {"arm", "offer"}:
            raise ValueError("invalid posthoc payload")
        return study.posthoc_evaluate(
            arm=body["arm"], offer=_model_record(TaskOffer, body["offer"]),
        )
    raise ValueError("unknown evaluator command")


def _worker_main() -> int:
    study: _WorkerStudy | None = None
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
            request = _parsed_json(line)
            if set(request) != {"protocol", "id", "session", "command", "body"}:
                raise ValueError("invalid evaluator request schema")
            if (request["protocol"] != PROTOCOL or
                type(request["id"]) is not int or request["id"] != expected_id or
                type(request["command"]) is not str or
                type(request["body"]) is not dict):
                raise ValueError("protocol, ID, command or payload mismatch")
            if study is None and (request["session"] is not None or
                                  request["command"] != "initialize"):
                raise ValueError("evaluator must initialize first")
            if study is not None and request["session"] != session:
                raise ValueError("evaluator session mismatch")
        except (ValueError, TypeError, KeyError, UnicodeError):
            # Protocol faults kill the worker. Recoverable invalid choices are
            # handled below by a structured application-level rejection.
            return 2
        try:
            expected_id += 1
            command = request["command"]
            if study is None:
                body = request["body"]
                if set(body) != {"seed", "evaluator_id", "arms",
                                 "training_pairs", "heldout_pairs_per_generator",
                                 "suite", "nonce"}:
                    raise ValueError("invalid evaluator initialization schema")
                nonce = body.pop("nonce")
                if type(nonce) is not str or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
                    raise ValueError("invalid initialization nonce")
                arms = _validated_study(
                    body["seed"], body["evaluator_id"], body["arms"],
                    body["training_pairs"], body["heldout_pairs_per_generator"],
                    body["suite"],
                )
                study = _WorkerStudy(seed=body["seed"],
                                     evaluator_id=body["evaluator_id"], arms=arms,
                                     training_pairs=body["training_pairs"],
                                     heldout_pairs_per_generator=body["heldout_pairs_per_generator"],
                                     suite=body["suite"])
                session = sha256((nonce + token_hex(16) + str(os.getpid())).encode()).hexdigest()
                result = {"pid": os.getpid(), "session": session,
                          "manifest": asdict(study.manifest),
                          "evidence_class": EVIDENCE_CLASS}
            else:
                if command == "shutdown":
                    if request["body"]:
                        raise ValueError("shutdown payload must be empty")
                    result = {"closed": True}
                else:
                    result = _worker_call(study, command, request["body"])
            payload = asdict(result) if is_dataclass(result) else result
            if command == "resolve":
                # Self-attested, auditable OS-process/session provenance. Do
                # not mistake it for a third-party signature or host evidence.
                provenance = {
                    "worker_pid": os.getpid(), "session": session,
                    "protocol": PROTOCOL,
                    "receipt_sha256": _hex("ipc-receipt-v1", os.getpid(),
                                           session, _checked_json(payload)),
                }
                payload = {"receipt": payload, "provenance": provenance}
            reply = {"protocol": PROTOCOL, "id": request["id"],
                     "session": session, "ok": True, "result": payload}
        except (ValueError, TypeError, KeyError, PermissionError, RuntimeError) as exc:
            reply = {"protocol": PROTOCOL, "id": expected_id - 1,
                     "session": session, "ok": False, "error_type": type(exc).__name__,
                     "error": str(exc)[:512]}
        try:
            print(_checked_json(reply), flush=True)
        except (ValueError, OSError):
            return 2
        if study is not None and request.get("command") == "shutdown" and reply.get("ok"):
            return 0
    return 0


class IsolatedOutcomeEvaluator:
    """Typed client for a single seeded worker; use a context manager."""

    def __init__(self, *, seed: int, evaluator_id: str,
                 arms: tuple[str, ...] = DEFAULT_ARMS,
                 training_pairs: int = 8, heldout_pairs_per_generator: int = 4,
                 suite: str = "novel_v2",
                 timeout_seconds: float = 10.0):
        _validated_study(seed, evaluator_id, arms,
                         training_pairs, heldout_pairs_per_generator, suite)
        if (type(timeout_seconds) not in (int, float) or
            not 0.1 <= float(timeout_seconds) <= 60.0):
            raise ValueError("timeout_seconds must be within [0.1,60]")
        self._timeout = float(timeout_seconds)
        self._lock = Lock()
        self._responses: Queue[str | None] = Queue()
        self._session: str | None = None
        self._counter = 0
        self._closed = False
        self._worker_pid: int | None = None
        self._manifest: GeneratorManifest | None = None
        self._suite = suite
        root = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR")}
        env["PYTHONPATH"] = str(root)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [sys.executable, "-B", "-m", "pleasure_experiment.isolated_outcome_evaluator", "--worker"],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            close_fds=True,
        )
        self._reader = Thread(target=self._read_responses, daemon=True)
        self._reader.start()
        try:
            initial = self._request("initialize", {
                "seed": seed, "evaluator_id": evaluator_id, "arms": list(arms),
                "training_pairs": training_pairs,
                "heldout_pairs_per_generator": heldout_pairs_per_generator,
                "suite": suite,
                "nonce": token_hex(16),
            }, initialization=True)
            if (type(initial) is not dict or
                set(initial) != {"pid", "session", "manifest", "evidence_class"} or
                initial["evidence_class"] != EVIDENCE_CLASS or
                type(initial["pid"]) is not int or initial["pid"] <= 0 or
                type(initial["session"]) is not str or
                _SHA.fullmatch(initial["session"]) is None):
                raise RuntimeError("evaluator handshake malformed")
            if initial["pid"] == os.getpid() or initial["pid"] != self._process.pid:
                raise RuntimeError("evaluator must attest the spawned child PID")
            self._worker_pid = initial["pid"]
            self._session = initial["session"]
            self._manifest = _model_record(GeneratorManifest, initial["manifest"])
        except Exception:
            self.close(force=True)
            raise

    @property
    def worker_pid(self) -> int:
        if self._worker_pid is None:
            raise RuntimeError("evaluator worker identity unavailable")
        return self._worker_pid

    @property
    def manifest(self) -> GeneratorManifest:
        if self._manifest is None:
            raise RuntimeError("evaluator manifest unavailable")
        return self._manifest

    @property
    def eval_families(self) -> tuple[str, ...]:
        return self.manifest.heldout_families

    @property
    def primary_ood_families(self) -> tuple[str, ...]:
        return (NOVEL_OOD_FAMILIES if self._suite == "novel_v2"
                else LEGACY_EVAL_FAMILIES)

    @property
    def provenance(self) -> dict[str, object]:
        """Untrusted-but-auditable local worker identity for study metadata."""
        if self._session is None or self._worker_pid is None:
            raise RuntimeError("evaluator handshake unavailable")
        return {"worker_pid": self._worker_pid, "session": self._session,
                "protocol": PROTOCOL, "suite": self._suite,
                "evidence_class": EVIDENCE_CLASS}

    def _read_responses(self) -> None:
        stream = self._process.stdout
        if stream is None:
            self._responses.put(None)
            return
        try:
            while True:
                line = stream.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                self._responses.put(line)
                if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
                    break
        except (OSError, ValueError):
            pass
        finally:
            self._responses.put(None)

    def _request(self, command: str, body: dict[str, object], *,
                 initialization: bool = False) -> object:
        with self._lock:
            if self._closed or self._process.poll() is not None:
                self.close(force=True)
                raise RuntimeError("evaluator process is unavailable")
            request_id = self._counter
            message = {"protocol": PROTOCOL, "id": request_id,
                       "session": None if initialization else self._session,
                       "command": command, "body": body}
            encoded = _checked_json(message)
            try:
                if self._process.stdin is None:
                    raise OSError("evaluator IPC input is unavailable")
                self._process.stdin.write(encoded + "\n")
                self._process.stdin.flush()
                response_line = self._responses.get(timeout=self._timeout)
                if response_line is None:
                    raise OSError("evaluator response stream closed")
                try:
                    response = _parsed_json(response_line)
                except (ValueError, TypeError) as exc:
                    self.close(force=True)
                    raise RuntimeError("invalid evaluator response JSON") from exc
                self._counter += 1
                common = {"protocol", "id", "session", "ok"}
                if (not common.issubset(response) or
                    response["protocol"] != PROTOCOL or
                    response["id"] != request_id or
                    type(response["ok"]) is not bool or
                    (not initialization and response["session"] != self._session)):
                    self.close(force=True)
                    raise RuntimeError("evaluator response identity mismatch")
                if (initialization and response["ok"] and
                    (type(response.get("result")) is not dict or
                     response["session"] != response["result"].get("session"))):
                    self.close(force=True)
                    raise RuntimeError("evaluator handshake session mismatch")
                if response["ok"]:
                    if set(response) != common | {"result"}:
                        self.close(force=True)
                        raise RuntimeError("unexpected evaluator success envelope")
                    return response["result"]
                if set(response) != common | {"error_type", "error"}:
                    self.close(force=True)
                    raise RuntimeError("unexpected evaluator error envelope")
                message = str(response["error"])[:512]
                error_kind = response["error_type"]
                if type(error_kind) is not str or error_kind not in (
                    "PermissionError", "ValueError", "TypeError", "RuntimeError"
                ) or type(response["error"]) is not str:
                    self.close(force=True)
                    raise RuntimeError("invalid evaluator error type or message")
                if error_kind == "PermissionError":
                    raise PermissionError(message)
                if error_kind == "ValueError":
                    raise ValueError(message)
                if error_kind == "TypeError":
                    raise TypeError(message)
                raise RuntimeError(message)
            except PermissionError:
                # PermissionError inherits OSError. This is an intentional,
                # recoverable worker denial, *not* an IPC failure.
                raise
            except (Empty, BrokenPipeError, OSError, UnicodeError,
                    json.JSONDecodeError, EOFError) as exc:
                self.close(force=True)
                raise RuntimeError("evaluator IPC timed out, died, or returned invalid bytes") from exc

    def issue(self, *, partition: str, arm: str, family_id: str,
              pair_index: int, side: int = 0) -> TaskOffer:
        return _model_record(TaskOffer, self._request("issue", {
            "partition": partition, "arm": arm, "family_id": family_id,
            "pair_index": pair_index, "side": side,
        }))

    def commit(self, offer: TaskOffer, action: str, decision_digest: str,
               *, arm: str, model_id: str,
               runtime_instance_id: str) -> CommitTicket:
        if type(offer) is not TaskOffer:
            raise TypeError("exact task offer required")
        return _model_record(CommitTicket, self._request("commit", {
            "arm": arm, "offer": asdict(offer), "action": action,
            "decision_digest": decision_digest, "model_id": model_id,
            "runtime_instance_id": runtime_instance_id,
        }))

    def resolve(self, ticket: CommitTicket, *, arm: str) -> OutcomeReceipt:
        if type(ticket) is not CommitTicket:
            raise TypeError("exact commit ticket required")
        envelope = self._request("resolve", {
            "arm": arm, "ticket": asdict(ticket),
        })
        if (type(envelope) is not dict or
            set(envelope) != {"receipt", "provenance"} or
            type(envelope["provenance"]) is not dict):
            self.close(force=True)
            raise RuntimeError("evaluator receipt provenance envelope malformed")
        proof = envelope["provenance"]
        if (set(proof) != {"worker_pid", "session", "protocol", "receipt_sha256"} or
            type(proof["worker_pid"]) is not int or
            proof["worker_pid"] != self.worker_pid or
            proof["session"] != self._session or proof["protocol"] != PROTOCOL or
            type(proof["receipt_sha256"]) is not str or
            proof["receipt_sha256"] != _hex("ipc-receipt-v1", self.worker_pid,
                                             self._session, _checked_json(envelope["receipt"]))):
            self.close(force=True)
            raise RuntimeError("evaluator receipt source/session mismatch")
        try:
            return _model_record(OutcomeReceipt, envelope["receipt"])
        except (TypeError, ValueError) as exc:
            self.close(force=True)
            raise RuntimeError("evaluator receipt data malformed") from exc

    def finalize_evaluation(self) -> dict[str, object]:
        result = self._request("finalize_evaluation", {})
        if type(result) is not dict or result.get("sealed") is not True or result.get("all_arms_committed") is not True:
            raise RuntimeError("evaluator did not confirm whole-study seal")
        return result

    def seal_decisions(self) -> dict[str, object]:
        result = self._request("seal_decisions", {})
        if type(result) is not dict or result.get("decisions_sealed") is not True:
            raise RuntimeError("evaluator did not confirm decision barrier")
        return result

    def posthoc_evaluate(self, offer: TaskOffer, *, arm: str) -> PosthocOutcome:
        if type(offer) is not TaskOffer:
            raise TypeError("exact EVAL offer required")
        return _model_record(PosthocOutcome, self._request("posthoc_evaluate", {
            "arm": arm, "offer": asdict(offer),
        }))

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if force:
            if process.poll() is None:
                process.kill()
        else:
            if process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                    process.wait(timeout=min(self._timeout, 2.0))
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self) -> "IsolatedOutcomeEvaluator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    raise SystemExit(_worker_main())
