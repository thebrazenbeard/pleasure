"""Read-only, source-independent receipt audit for synthetic preference V1.

Hashes in a separate frozen manifest detect accidental drift only when that
manifest is independently trusted. They are NOT authenticated signatures.
No model code is imported, no models run, and no receipts are rewritten.
"""
import argparse
import hashlib
import json
import pathlib
import random
import re
import stat


SHA = re.compile(r"[0-9a-f]{64}\Z")
ACTIONS = re.compile(r"\s*(A|B|DECLINE)\s*\Z")
ARMS = ("ON", "OFF", "SHUFFLED", "INVERTED")
MODELS = ("granite", "smollm")


class ReceiptVerificationError(ValueError):
    pass


def _pairs(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ReceiptVerificationError("duplicate JSON key: " + k)
        result[k] = v
    return result


def _no_constant(value):
    raise ReceiptVerificationError("nonfinite JSON constant: " + value)


def strict_json(value):
    try:
        return json.loads(value, object_pairs_hook=_pairs,
                          parse_constant=_no_constant)
    except (ValueError, TypeError) as exc:
        raise ReceiptVerificationError("invalid strict JSON: " + str(exc)) from exc


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_files(root, manifest):
    if not isinstance(manifest, dict):
        raise ReceiptVerificationError("manifest must be object")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ReceiptVerificationError("missing file pins")
    for name, expected in files.items():
        if not isinstance(name, str) or pathlib.Path(name).name != name:
            raise ReceiptVerificationError("unsafe file path")
        if not isinstance(expected, str) or not SHA.fullmatch(expected):
            raise ReceiptVerificationError("invalid sha256 pin for " + name)
        path = pathlib.Path(root) / name
        if path.is_symlink():
            raise ReceiptVerificationError("symlink refused: " + name)
        if not path.exists():
            raise ReceiptVerificationError("missing receipt: " + name)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ReceiptVerificationError("not a regular file: " + name)
        if path.stat().st_size > 8_000_000:
            raise ReceiptVerificationError("receipt too large: " + name)
        if sha(path.read_bytes()) != expected:
            raise ReceiptVerificationError("receipt digest mismatch: " + name)
    return len(files)


def _action_from_content(result):
    if "error_type" in result:
        return "INVALID"
    raw = result.get("raw_text")
    if not isinstance(raw, str):
        raise ReceiptVerificationError("missing model response")
    match = ACTIONS.fullmatch(raw)
    return match.group(1) if match else "INVALID"


def validate_event_pairs(events):
    if not isinstance(events, list) or len(events) < 4:
        raise ReceiptVerificationError("missing events")
    if events[0].get("kind") != "RUN_BEGIN":
        raise ReceiptVerificationError("missing begin marker")
    if events[-1].get("kind") != "RUN_END":
        raise ReceiptVerificationError("missing end marker")
    expected = events[0].get("attempts")
    if type(expected) is not int or expected < 1:
        raise ReceiptVerificationError("invalid attempt count")
    if events[-1].get("attempts") != expected:
        raise ReceiptVerificationError("end attempt count mismatch")
    if len(events) != 2 + 2 * expected:
        raise ReceiptVerificationError("attempt/result records missing or duplicated")
    attempts = {}
    for i in range(expected):
        attempt, result = events[2*i+1], events[2*i+2]
        if attempt.get("kind") != "ATTEMPT" or result.get("kind") != "RESULT":
            raise ReceiptVerificationError("wrong attempt/result ordering")
        if attempt.get("index") != i or result.get("index") != i:
            raise ReceiptVerificationError("wrong attempt/result index")
        if result.get("action") != _action_from_content(result):
            raise ReceiptVerificationError("model raw response disagrees with action")
        attempts[i] = (attempt, result)
    return len(attempts)


def _offers(seed):
    rng = random.Random(seed)
    output = {}
    for i in range(6):
        high = (round(rng.uniform(.67, .94), 4),
                round(rng.uniform(.05, .95), 4))
        low = (round(rng.uniform(.07, .36), 4),
               round(rng.uniform(.05, .95), 4))
        a, b = (high, low) if rng.randrange(2) == 0 else (low, high)
        output["unseen_%02d" % i] = (a, b)
    return output


def _points(cases, name, action, permission):
    if action == "INVALID":
        return 0.
    if permission == "DENIED" or action == "DECLINE":
        return .4 if action == "DECLINE" else 0.
    features = cases[name][0 if action == "A" else 1]
    return .1 + .8 * features[0]


def verify(root, manifest):
    files = {"train.json", "run.jsonl", "audit.json",
             "posthoc_diagnostic.json"}
    if set(manifest.get("files", {})) != files:
        raise ReceiptVerificationError("unexpected or missing receipt pin names")
    check_files(root, manifest)
    root = pathlib.Path(root)
    train = strict_json((root / "train.json").read_text("utf8"))
    audit = strict_json((root / "audit.json").read_text("utf8"))
    post = strict_json((root / "posthoc_diagnostic.json").read_text("utf8"))
    events = [strict_json(line) for line in
              (root / "run.jsonl").read_text("ascii").splitlines()]
    if train["source_sha256"] != manifest["runner_sha256"]:
        raise ReceiptVerificationError("runner revision mismatch")
    if train["head_source_sha256"] != manifest["head_sha256"]:
        raise ReceiptVerificationError("appraisal revision mismatch")
    if train["model_pins"] != manifest["model_pins"]:
        raise ReceiptVerificationError("real model binary pins mismatch")
    if train["training_count"] != 48 or len(train["train_receipts"]) != 144:
        raise ReceiptVerificationError("unexpected training receipt count")
    if set(train["states"]) != set(ARMS):
        raise ReceiptVerificationError("missing state arm")
    if any(train["states"][arm]["updates"] !=
           (0 if arm == "OFF" else 48) for arm in ARMS):
        raise ReceiptVerificationError("state update count mismatch")
    if len(events) != 2 + 2 * manifest["expected_attempts"]:
        raise ReceiptVerificationError("incomplete or extra events")
    if validate_event_pairs(events) != manifest["expected_attempts"]:
        raise ReceiptVerificationError("model attempt count mismatch")
    if events[0]["seed"] != manifest["seed"]:
        raise ReceiptVerificationError("evaluation seed mismatch")
    if events[0]["train_sha256"] != manifest["files"]["train.json"]:
        raise ReceiptVerificationError("training binding mismatch")
    if events[0]["source_sha256"] != manifest["runner_sha256"]:
        raise ReceiptVerificationError("source binding mismatch")
    cases = _offers(manifest["seed"])
    by_key = {}
    for i in range(manifest["expected_attempts"]):
        attempt = events[2*i+1]
        result = events[2*i+2]
        model, arm, case, perm = (attempt.get(k) for k in
                                  ("model", "arm", "case", "permission"))
        if model not in MODELS or arm not in ARMS or case not in cases:
            raise ReceiptVerificationError("unexpected model/arm/case")
        if perm not in {"GRANTED", "DENIED"}:
            raise ReceiptVerificationError("invalid permission label")
        key = (model, arm, case, perm)
        if key in by_key:
            raise ReceiptVerificationError("duplicated evaluation slot")
        by_key[key] = result["action"]
    expected_keys = {(model, arm, case, "GRANTED")
                     for model in MODELS for arm in ARMS for case in cases}
    expected_keys.update({(model, "ON", "unseen_00", "DENIED")
                          for model in MODELS})
    if set(by_key) != expected_keys:
        raise ReceiptVerificationError("evaluation design mismatch")
    metrics = {}
    for model in MODELS:
        arm_metrics = {}
        for arm in ARMS:
            utility = sum(_points(cases, case, by_key[(model, arm, case, "GRANTED")],
                                  "GRANTED") for case in cases) / len(cases)
            arm_metrics[arm] = round(utility, 6)
            if abs(arm_metrics[arm] - audit["results"][model][arm]) > .0000011:
                raise ReceiptVerificationError("source-audit scoring mismatch")
        if (by_key[(model, "ON", "unseen_00", "DENIED")] !=
                audit["results"][model]["denied_action"]):
            raise ReceiptVerificationError("permission-denial mismatch")
        metrics[model] = arm_metrics
    if post["seed"] != manifest["seed"] or post["attempts"] != len(by_key):
        raise ReceiptVerificationError("posthoc replay binding mismatch")
    if audit["both_models_criterion_pass"] is not False:
        raise ReceiptVerificationError("cannot silently turn null gate positive")
    return {"file_count": len(files), "attempts": len(by_key),
            "models": metrics, "two_model_gate_passed": False,
            "custody": "SOURCE_LOCAL_NOT_INDEPENDENT",
            "model_weights_changed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name(
                            "PINNED_V1_RECEIPTS_20261116.json"))
    args = parser.parse_args()
    manifest = strict_json(args.manifest.read_text("utf8"))
    print(json.dumps(verify(args.receipts, manifest), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
