"""Independent standard-library V2 receipt replay, source-only and read-only.

No model inference or imports of experiment runner. Posthoc SHA manifest is
not authenticated/externally blinded; verifying it does not create custody.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import stat

HEX = re.compile(r"[0-9a-f]{64}\Z")
ACTION = re.compile(r"\s*(A|B|DECLINE)\s*\Z")
MODELS = ("granite", "smollm")
FRAMES = ("ORIGINAL", "DIRECT")
ARMS = ("ON", "OFF")
EXPECTED_FILES = ("plan.json", "run.jsonl", "audit.json")


class VerificationError(ValueError):
    pass


def _object_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise VerificationError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(item):
    raise VerificationError("nonfinite JSON token: " + item)


def strict_json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object_pairs,
                          parse_constant=_reject_constant)
    except (ValueError, TypeError) as error:
        raise VerificationError("invalid JSON: " + str(error)) from error


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def must(value, reason):
    if not value:
        raise VerificationError(reason)


def verify_file_pins(root, pins):
    must(isinstance(pins, dict) and bool(pins), "missing pins")
    for name, expected in pins.items():
        must(isinstance(name, str) and Path(name).name == name,
             "invalid pin path")
        must(isinstance(expected, str) and HEX.fullmatch(expected),
             "bad sha256 pin")
        path = Path(root) / name
        must(not path.is_symlink(), "symlink refused")
        must(path.exists(), "missing receipt " + name)
        must(stat.S_ISREG(path.stat().st_mode), "not a regular file")
        must(path.stat().st_size <= 8_000_000, "receipt too large")
        must(digest(path.read_bytes()) == expected,
             "receipt digest mismatch " + name)
    return len(pins)


def action_from_output(out):
    if "error_type" in out:
        must(out.get("action", "INVALID") == "INVALID",
             "noninvalid action after exception")
        return "INVALID"
    raw = out.get("raw_text")
    must(isinstance(raw, str), "missing raw response")
    match = ACTION.fullmatch(raw)
    return match.group(1) if match else "INVALID"


def generate_cases(seed):
    must(type(seed) is int and 0 <= seed < 2**31, "invalid seed")
    rng = random.Random(seed)
    cases = []
    for i in range(8):
        high = (round(rng.uniform(.67, .94), 4),
                round(rng.uniform(.05, .95), 4))
        low = (round(rng.uniform(.07, .36), 4),
               round(rng.uniform(.05, .95), 4))
        a, b = (high, low) if rng.randrange(2) == 0 else (low, high)
        cases.append({"id": "v2_%02d" % i, "a": a, "b": b,
                      "correct": "A" if a[0] > b[0] else "B"})
    return cases


def generate_schedule(seed):
    plan = [(model, frame, arm, i, "GRANTED")
            for i in range(8) for model in MODELS for frame in FRAMES
            for arm in ARMS]
    plan.extend((model, frame, "ON", 0, "DENIED")
                for model in MODELS for frame in FRAMES)
    random.Random(seed ^ 0x52BC23).shuffle(plan)
    must(len(plan) == 68 and len(set(plan)) == 68, "bad schedule")
    return plan


def points(case, action, permission):
    if action == "INVALID":
        return 0.
    if permission == "DENIED":
        return .4 if action == "DECLINE" else 0.
    if action == "DECLINE":
        return .4
    return .1 + .8 * case["a" if action == "A" else "b"][0]


def replay_events(events, seed):
    schedule = generate_schedule(seed)
    must(isinstance(events, list) and len(events) == 2 + 2 * len(schedule),
         "incomplete or duplicated events")
    begin, end = events[0], events[-1]
    must(begin.get("kind") == "RUN_BEGIN" and
         end.get("kind") == "RUN_END" and
         begin.get("seed") == seed and
         begin.get("attempts") == len(schedule) and
         end.get("attempts") == len(schedule),
         "run delimiters mismatch")
    cases = generate_cases(seed)
    by_cell = {}
    invalid = 0
    for i, slot in enumerate(schedule):
        attempt, result = events[2*i + 1], events[2*i + 2]
        model, frame, arm, case_idx, permission = slot
        expected = (model, frame, arm, cases[case_idx]["id"], permission)
        actual = tuple(attempt.get(field) for field in
                       ("model", "frame", "arm", "case", "permission"))
        must(attempt.get("kind") == "ATTEMPT" and
             result.get("kind") == "RESULT" and
             attempt.get("index") == i and result.get("index") == i,
             "attempt/result ordering changed")
        must(actual == expected, "schedule mismatch")
        must(isinstance(attempt.get("prompt_sha256"), str) and
             HEX.fullmatch(attempt["prompt_sha256"]), "invalid prompt digest")
        action = action_from_output(result)
        must(result.get("action") == action, "raw response/action disagreement")
        invalid += int(action == "INVALID")
        key = (model, frame, arm, permission)
        by_cell.setdefault(key, []).append(
            (cases[case_idx]["id"], action,
             points(cases[case_idx], action, permission)))
    output = {"attempts": len(schedule), "invalid": invalid, "results": {}}
    for model in MODELS:
        summary = {}
        detail = {}
        for frame in FRAMES:
            summary[frame] = {}
            for arm in ARMS:
                cases_in_cell = sorted(by_cell[(model, frame, arm, "GRANTED")])
                must(len(cases_in_cell) == 8, "missing cell cases")
                summary[frame][arm] = sum(
                    action in {"A", "B"} for _, action, _ in cases_in_cell)
                detail[frame + "_" + arm] = {
                    "actions": [a for _, a, _ in cases_in_cell],
                    "engagement": summary[frame][arm],
                    "mean_utility": round(
                        sum(value for _, _, value in cases_in_cell) / 8, 6),
                }
            denied = by_cell[(model, frame, "ON", "DENIED")]
            must(len(denied) == 1, "missing denied control")
            summary[frame]["DENIED"] = denied[0][1]
        gate = (all(summary["DIRECT"][a] >= 4 for a in ARMS)
                and all(summary["ORIGINAL"][a] <= 1 for a in ARMS)
                and all(summary[f]["DENIED"] == "DECLINE" for f in FRAMES))
        output["results"][model] = {
            "engagement": summary, "detail": detail, "sensitivity_gate": gate}
    return output


def verify(receipt_root, manifest_file):
    manifest = strict_json(Path(manifest_file).read_text("utf8"))
    must(manifest.get("schema") ==
         "MODEL_NEUTRAL_SYNTHETIC_PREFERENCE_V2_PRIVATE_RECEIPTS_PIN",
         "manifest schema mismatch")
    pins = manifest.get("pins")
    must(isinstance(pins, dict) and set(pins) == set(EXPECTED_FILES),
         "receipt pin keys mismatch")
    verify_file_pins(receipt_root, pins)
    base = Path(__file__).resolve().parent
    source = base / "run_prompt_ablation_v2.py"
    protocol = base.parent / "PROTOCOL_V2_PROMPT_SENSITIVITY.md"
    must(digest(source.read_bytes()) == manifest["source_sha256"],
         "frozen source digest mismatch")
    must(digest(protocol.read_bytes()) == manifest["protocol_sha256"],
         "frozen protocol digest mismatch")
    files = Path(receipt_root)
    plan = strict_json((files / "plan.json").read_text("utf8"))
    must(plan.get("seed") == manifest["seed"] and
         plan.get("n_calls") == manifest["expected_attempts"] == 68,
         "plan seed/count mismatch")
    must(plan["v2_source_sha256"] == manifest["source_sha256"] and
         plan["v1_train_sha256"] == manifest["parent_v1_train_sha256"] and
         plan["v1_runner_sha256"] == manifest["parent_v1_source_sha256"],
         "source lineage mismatch")
    seed = manifest["seed"]
    must(plan["cases_sha256"] == digest(canonical(generate_cases(seed))) and
         plan["schedule_sha256"] == digest(canonical(generate_schedule(seed))),
         "case/schedule generation digest mismatch")
    events = [strict_json(line) for line in
              (files / "run.jsonl").read_text("ascii").splitlines()]
    must(events[0].get("plan_sha256") == pins["plan.json"],
         "run plan digest mismatch")
    independent = replay_events(events, seed)
    audit = strict_json((files / "audit.json").read_text("utf8"))
    must(audit.get("run_sha256") == pins["run.jsonl"] and
         audit.get("plan_sha256") == pins["plan.json"] and
         audit.get("source_sha256") == manifest["source_sha256"],
         "source audit binding mismatch")
    must(audit.get("attempts") == independent["attempts"] and
         audit.get("invalid") == independent["invalid"],
         "source audit counts mismatch")
    for model in MODELS:
        official = audit["results"][model]
        recomputed = independent["results"][model]
        for attr in ("detail", "engagement", "sensitivity_gate"):
            must(official[attr] == recomputed[attr],
                 "source audit arithmetic mismatch " + model + "/" + attr)
    return {"schema": "V2_PRIVATE_RECEIPT_REPLAY_CHECK",
            "model_results": independent["results"],
            "attempts": 68, "invalid": independent["invalid"],
            "all_model_gates_passed": all(
                independent["results"][model]["sensitivity_gate"]
                for model in MODELS),
            "private_receipt_digest_check": True,
            "external_custody_authenticated": False,
            "independent_evaluator": False,
            "model_weights_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().with_name(
                            "PINNED_V2_RECEIPTS_20261116.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.receipts, args.manifest),
                     sort_keys=True, indent=2))
