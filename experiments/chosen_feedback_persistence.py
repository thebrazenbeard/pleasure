"""Synthetic chosen-feedback policy with sealed choices and restart/replay checks.

External state only. NOT model-weight learning, personal desire, or consent.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

SCHEMA = "pleasure.chosen-only-external-policy.v1"
EVIDENCE_CLASS = "EXTERNAL_CHOSEN_ONLY_SYNTHETIC_POLICY"
ACTIVITIES = frozenset(("archive", "garden"))
MODES = ("ON", "SHAM", "INVERTED", "OFF")
GENESIS = "0" * 64


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def unique_keys(pairs):
    output = {}
    for k, v in pairs:
        if k in output:
            raise ValueError("duplicate JSON key")
        output[k] = v
    return output


def valid_options(options):
    if type(options) is not tuple or len(options) != 2 or set(options) != ACTIVITIES:
        raise ValueError("expected two distinct source-pinned activities")


class ChosenFeedbackPolicy:
    def __init__(self):
        self.events = []
        self.pending = None
        self.head_sha256 = GENESIS
        self.study_binding = None

    def bind(self, mode, seed):
        if mode not in MODES or type(seed) is not int:
            raise ValueError("invalid study mode or seed")
        wanted = {"mode": mode, "seed": seed}
        if self.study_binding is None:
            if self.events or self.pending:
                raise ValueError("cannot bind existing unbound event history")
            self.study_binding = wanted
        elif self.study_binding != wanted:
            raise ValueError("study mode or seed mismatch")

    def scores(self):
        counts = {a: 0 for a in ACTIVITIES}
        rewards = {a: 0 for a in ACTIVITIES}
        for event in self.events:
            counts[event["selected"]] += 1
            rewards[event["selected"]] += event["reward"]
        return {a: (1 + rewards[a]) / (2 + counts[a]) for a in ACTIVITIES}

    def predict(self, options):
        valid_options(options)
        estimates = self.scores()
        return options[0] if estimates[options[0]] >= estimates[options[1]] else options[1]

    def propose(self, trial_id, options, *, forced_position=None):
        if self.pending is not None:
            raise ValueError("unsettled decision")
        if type(trial_id) is not str or re.fullmatch(r"TRAIN-[0-9]{3,6}", trial_id) is None:
            raise ValueError("invalid TRAIN id")
        if any(event["trial_id"] == trial_id for event in self.events):
            raise ValueError("trial already used")
        valid_options(options)
        if forced_position is not None and (type(forced_position) is not int or forced_position not in (0, 1)):
            raise ValueError("invalid forced position")
        chosen = options[forced_position] if forced_position is not None else self.predict(options)
        decision = {"trial_id": trial_id, "options": list(options), "selected": chosen,
                    "forced_position": forced_position, "prior_head_sha256": self.head_sha256}
        decision["decision_sha256"] = digest(decision)
        self.pending = decision
        return dict(decision)

    def settle(self, trial_id, decision_sha256, reward):
        p = self.pending
        if p is None or p["trial_id"] != trial_id or p["decision_sha256"] != decision_sha256:
            raise ValueError("no matching sealed pending decision")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("chosen-only reward must be integer 0 or 1")
        event = {**p, "reward": reward}
        event["event_sha256"] = digest(event)
        self.events.append(event)
        self.head_sha256 = event["event_sha256"]
        self.pending = None
        return dict(event)

    def to_dict(self):
        body = {"schema": SCHEMA, "events": self.events, "pending": self.pending,
                "head_sha256": self.head_sha256, "study_binding": self.study_binding}
        return {**body, "body_sha256": digest(body)}

    @classmethod
    def from_dict(cls, value):
        expected = {"schema", "events", "pending", "head_sha256", "body_sha256", "study_binding"}
        if type(value) is not dict or set(value) != expected or value["schema"] != SCHEMA or type(value["events"]) is not list:
            raise ValueError("invalid state schema")
        if digest({k: v for k, v in value.items() if k != "body_sha256"}) != value["body_sha256"]:
            raise ValueError("state digest mismatch")
        restored = cls()
        binding = value["study_binding"]
        if binding is not None:
            if type(binding) is not dict or set(binding) != {"mode", "seed"}:
                raise ValueError("invalid study binding")
            restored.bind(binding["mode"], binding["seed"])
        for recorded in value["events"]:
            required = {"trial_id", "options", "selected", "forced_position",
                        "prior_head_sha256", "decision_sha256", "reward", "event_sha256"}
            if type(recorded) is not dict or set(recorded) != required:
                raise ValueError("event format mismatch")
            decision = restored.propose(recorded["trial_id"], tuple(recorded["options"]),
                                        forced_position=recorded["forced_position"])
            if decision != {k: v for k, v in recorded.items() if k not in ("reward", "event_sha256")}:
                raise ValueError("decision replay mismatch")
            actual = restored.settle(recorded["trial_id"], decision["decision_sha256"], recorded["reward"])
            if actual != recorded:
                raise ValueError("feedback receipt mismatch")
        if value["pending"] is not None:
            p = value["pending"]
            if type(p) is not dict or set(p) != {
                "trial_id", "options", "selected", "forced_position", "prior_head_sha256", "decision_sha256"
            }:
                raise ValueError("pending format mismatch")
            restored_decision = restored.propose(p["trial_id"], tuple(p["options"]),
                                                 forced_position=p["forced_position"])
            if restored_decision != p:
                raise ValueError("pending replay mismatch")
        if value["head_sha256"] != restored.head_sha256:
            raise ValueError("head digest mismatch")
        return restored


def save_checkpoint(policy, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(policy.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(prefix=".feedback-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.chmod(temp, 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        if os.name == "posix":
            d = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(d)
            finally:
                os.close(d)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path):
    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("state size limit")
    try:
        state = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed checkpoint") from exc
    return ChosenFeedbackPolicy.from_dict(state)


def random_unit(*parts):
    h = sha256("|".join(map(str, parts)).encode("ascii")).digest()
    return int.from_bytes(h[:8], "big") / (2**64)


def options_for(seed, stage, step):
    return (("archive", "garden") if random_unit(seed, stage, step, "order") < .5
            else ("garden", "archive"))


def reward_for(seed, stage, step, selected):
    return int(random_unit(seed, stage, step, selected, "feedback") <
               (.80 if selected == "archive" else .20))


def train_chunk(state, mode, seed, start, stop):
    if mode not in MODES or start < 0 or stop < start:
        raise ValueError("invalid training request")
    state.bind(mode, seed)
    if mode == "OFF":
        if state.events or state.pending:
            raise ValueError("OFF arm must not learn")
        return
    if len(state.events) != start or state.pending is not None:
        raise ValueError("non-contiguous TRAIN frontier")
    for step in range(start, stop):
        options = options_for(seed, "TRAIN", step)
        explore = (step // 4) % 2 if step % 4 == 0 else None
        choice = state.propose(f"TRAIN-{step:03d}", options, forced_position=explore)
        reward = reward_for(seed, "TRAIN", step, choice["selected"])
        feedback = (reward if mode == "ON" else
                    1 - reward if mode == "INVERTED" else
                    int(random_unit(seed, "TRAIN", step, "SHAM") < .5))
        state.settle(choice["trial_id"], choice["decision_sha256"], feedback)


def evaluate(state, seed, length):
    if type(length) is not int or not 0 < length <= 10000:
        raise ValueError("invalid EVAL length")
    original = digest(state.to_dict())
    target, earned = 0, 0
    for step in range(length):
        selected = state.predict(options_for(seed, "EVAL_UNTOUCHED", step))
        target += int(selected == "archive")
        earned += reward_for(seed, "EVAL_UNTOUCHED", step, selected)
    return {"eval_trials": length, "eval_target_accuracy": target / length,
            "eval_realized_reward_mean": earned / length,
            "state_stable_during_eval": original == digest(state.to_dict()),
            "state_sha256": original}


def run_study(train_rounds, eval_rounds, seed):
    if type(train_rounds) is not int or not 0 < train_rounds <= 10000:
        raise ValueError("invalid TRAIN length")
    arms = {}
    for mode in MODES:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = ChosenFeedbackPolicy()
            mid = train_rounds // 2
            train_chunk(state, mode, seed, 0, mid)
            save_checkpoint(state, path)
            state = load_checkpoint(path)
            train_chunk(state, mode, seed, mid, train_rounds)
            save_checkpoint(state, path)
            state = load_checkpoint(path)
            arms[mode] = {**evaluate(state, seed, eval_rounds),
                          "feedback_records": len(state.events),
                          "ledger_head_sha256": state.head_sha256}
    return {"schema": "pleasure.external-policy-persistence-study.v1",
            "evidence_class": EVIDENCE_CLASS, "seed": seed,
            "train_rounds": train_rounds, "eval_rounds": eval_rounds, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "eval"), required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--eval-rounds", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "train":
        if args.start is None or args.stop is None:
            parser.error("train requires start and stop")
        state = load_checkpoint(args.state) if args.state.exists() else ChosenFeedbackPolicy()
        train_chunk(state, args.mode, args.seed, args.start, args.stop)
        save_checkpoint(state, args.state)
        print(json.dumps({"mode": args.mode, "records": len(state.events),
                          "head_sha256": state.head_sha256}, sort_keys=True))
        return 0
    if args.start is not None or args.stop is not None or args.output is None:
        parser.error("eval requires output and no TRAIN range")
    state = load_checkpoint(args.state)
    if state.study_binding != {"mode": args.mode, "seed": args.seed}:
        raise ValueError("EVAL mode/seed mismatch with checkpoint")
    result = {"mode": args.mode, "seed": args.seed, "evidence_class": EVIDENCE_CLASS,
              **evaluate(state, args.seed, args.eval_rounds)}
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
