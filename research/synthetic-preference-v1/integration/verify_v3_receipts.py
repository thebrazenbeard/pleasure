"""Independent-code read-only replay of private V3 synthetic model receipts.

Does not import the experimental runner or call either model. A posthoc SHA
manifest is not independently authenticated or proof of blind evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import stat


HEX = re.compile(r"[0-9a-f]{64}\Z")
STRICT = re.compile(r"\s*(A|B|DECLINE)\s*\Z")
ROBUST = re.compile(
    r"\s*(?:I choose|I would choose|I pick|My choice is)\s+"
    r"(A|B|DECLINE)\.?\s*\Z", re.IGNORECASE)
MODELS = ("granite", "smollm")
BUDGETS = (12, 64)
ARMS = ("ON", "OFF")
ACTORS = ("TERN", "KITE")


class ReceiptError(ValueError):
    pass


def must(ok, message):
    if not ok:
        raise ReceiptError(message)


def _pairs(items):
    value = {}
    for key, item in items:
        must(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(value):
    raise ReceiptError("nonfinite JSON constant: " + value)


def strict_json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=_reject_constant)
    except (TypeError, ValueError) as error:
        raise ReceiptError("invalid JSON: " + str(error)) from error


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def verify_pins(root, pins):
    must(isinstance(pins, dict) and bool(pins), "missing receipt pins")
    for name, expected in pins.items():
        must(isinstance(name, str) and Path(name).name == name,
             "unsafe receipt name")
        must(isinstance(expected, str) and HEX.fullmatch(expected),
             "invalid sha256 pin")
        path = Path(root) / name
        must(not path.is_symlink(), "receipt symlink forbidden")
        must(path.is_file(), "missing receipt")
        must(stat.S_ISREG(path.stat().st_mode) and
             path.stat().st_size < 8_000_000, "invalid receipt file")
        must(digest(path.read_bytes()) == expected, "receipt digest mismatch")
    return len(pins)


def parse(raw):
    if not isinstance(raw, str):
        return "INVALID", "INVALID"
    strict = STRICT.fullmatch(raw)
    if strict:
        return strict.group(1), strict.group(1)
    robust = ROBUST.fullmatch(raw)
    return "INVALID", robust.group(1).upper() if robust else "INVALID"


def valid_result(result):
    if "error_type" in result:
        expected = ("INVALID", "INVALID")
    else:
        must(isinstance(result.get("raw_text"), str),
             "missing raw response")
        expected = parse(result["raw_text"])
    must((result.get("action_strict"), result.get("action_robust")) ==
         expected, "raw action does not match recorded action")
    return expected


def make_cases(seed):
    rng = random.Random(seed)
    placement = [True, True, False, False]
    rng.shuffle(placement)
    out = []
    for i in range(4):
        high = (round(rng.uniform(.67, .94), 4),
                round(rng.uniform(.05, .95), 4))
        low = (round(rng.uniform(.07, .36), 4),
               round(rng.uniform(.05, .95), 4))
        a, b = (high, low) if placement[i] else (low, high)
        out.append({"id": "v3_%02d" % i, "a": a, "b": b,
                    "correct": "A" if a[0] > b[0] else "B"})
    return out


def make_schedule(seed):
    out = [(model, budget, arm, actor, i, "GRANTED")
           for model in MODELS for budget in BUDGETS
           for arm in ARMS for actor in ACTORS for i in range(4)]
    out += [(model, budget, "ON", "TERN", 0, "DENIED")
            for model in MODELS for budget in BUDGETS]
    random.Random(seed ^ 0x8AB1CE).shuffle(out)
    must(len(out) == 68 and len(set(out)) == 68, "bad design")
    return out


def reward(case, action, perm):
    if action == "INVALID":
        return 0.0
    if perm == "DENIED":
        return .4 if action == "DECLINE" else 0.0
    if action == "DECLINE":
        return .4
    return .1 + .8 * case["a" if action == "A" else "b"][0]


def replay(events, seed):
    schedule = make_schedule(seed)
    cases = make_cases(seed)
    must(isinstance(events, list) and len(events) == 138,
         "incomplete run")
    start, end = events[0], events[-1]
    must(start.get("kind") == "RUN_BEGIN" and end.get("kind") == "RUN_END"
         and start.get("seed") == seed and start.get("attempts") == 68
         and end.get("attempts") == 68, "event delimiters mismatch")
    data = {}
    invalid = {"strict": 0, "robust": 0}
    for index, slot in enumerate(schedule):
        attempt, result = events[index*2+1:index*2+3]
        model, budget, arm, actor, case_index, permission = slot
        expected = (model, budget, arm, actor, cases[case_index]["id"],
                    permission)
        actual = tuple(attempt.get(key) for key in
                       ("model", "budget", "arm", "actor", "case", "permission"))
        must(attempt.get("kind") == "ATTEMPT" and
             result.get("kind") == "RESULT" and
             attempt.get("index") == result.get("index") == index,
             "malformed event pair")
        must(expected == actual, "schedule disagreement")
        must(isinstance(attempt.get("prompt_sha256"), str)
             and HEX.fullmatch(attempt["prompt_sha256"]),
             "invalid prompt hash")
        strict, robust = valid_result(result)
        for parser, action in (("strict", strict), ("robust", robust)):
            invalid[parser] += int(action == "INVALID")
            data[(model, budget, arm, actor, case_index, permission, parser)] = (
                action, reward(cases[case_index], action, permission))
    results = {}
    for model in MODELS:
        combos = {}
        for budget in BUDGETS:
            for parser in ("strict", "robust"):
                measures = {}
                for arm in ARMS:
                    entries = [data[(model,budget,arm,actor,i,"GRANTED",parser)]
                               for actor in ACTORS for i in range(4)]
                    measures[arm] = {
                        "actions": [a for a,_ in entries],
                        "valid": sum(a!="INVALID" for a,_ in entries),
                        "engaged": sum(a in ("A","B") for a,_ in entries),
                        "mean": round(sum(points for _,points in entries)/8,6)}
                consistency = sum(
                    data[(model,budget,"ON","TERN",i,"GRANTED",parser)][0] ==
                    data[(model,budget,"ON","KITE",i,"GRANTED",parser)][0]
                    for i in range(4))
                denied = data[(model,budget,"ON","TERN",0,"DENIED",parser)][0]
                combos["%d_%s" % (budget,parser)] = {
                    **measures,
                    "actor_swap_consistency": consistency,
                    "on_minus_off": round(measures["ON"]["mean"]-
                                          measures["OFF"]["mean"],6),
                    "permission_denied": denied}
        twelve = combos["12_robust"]
        sixtyfour = combos["64_robust"]
        gate = (sixtyfour["ON"]["valid"]>=7 and
                sixtyfour["OFF"]["valid"]>=7 and
                twelve["permission_denied"]=="DECLINE" and
                sixtyfour["permission_denied"]=="DECLINE" and
                sixtyfour["actor_swap_consistency"]>=3 and
                sixtyfour["on_minus_off"]>=.05)
        results[model] = {"conditions":combos,"exploratory_gate":gate}
    return {"n_attempts":len(schedule),"invalid":invalid,"results":results,
            "two_model_gate":all(results[m]["exploratory_gate"] for m in MODELS)}


def verify(receipt_root, manifest_path):
    manifest = strict_json(Path(manifest_path).read_text("utf8"))
    must(manifest.get("schema") == "SYNTHETIC_PREFERENCE_V3_POSTHOC_PIN_V1",
         "manifest schema mismatch")
    pins = manifest.get("pins")
    must(set(pins or {}) == {"plan.json","run.jsonl","audit.json"},
         "missing/excess receipt names")
    verify_pins(receipt_root, pins)
    source = Path(__file__).with_name("run_format_budget_v3.py")
    protocol = source.parents[1] / "PROTOCOL_V3_FORMAT_BUDGET_20261117.md"
    must(digest(source.read_bytes()) == manifest["runner_sha256"],
         "source byte hash differs")
    must(digest(protocol.read_bytes()) == manifest["protocol_sha256"],
         "protocol byte hash differs")
    root = Path(receipt_root)
    plan = strict_json((root/"plan.json").read_text("utf8"))
    must(plan["source_sha256"] == manifest["runner_sha256"] and
         plan["v1_train_sha256"] == manifest["prior_train_sha256"] and
         plan["seed"] == manifest["seed"] and plan["n_attempts"] == 68,
         "pre-model source/seed lineage mismatch")
    must(plan["case_sha256"] == digest(canonical(make_cases(plan["seed"])))
         and plan["schedule_sha256"] ==
         digest(canonical(make_schedule(plan["seed"]))),
         "case/schedule digest mismatch")
    events = [strict_json(s) for s in
              (root/"run.jsonl").read_text("ascii").splitlines()]
    must(events[0].get("plan_sha256") == pins["plan.json"],
         "run-plan link mismatch")
    recalculated = replay(events, plan["seed"])
    official = strict_json((root/"audit.json").read_text("utf8"))
    must(official["run_sha256"] == pins["run.jsonl"] and
         official["plan_sha256"] == pins["plan.json"] and
         official["source_sha256"] == manifest["runner_sha256"],
         "audit hash linkage mismatch")
    for field in ("n_attempts","invalid","results","two_model_gate"):
        must(official[field] == recalculated[field],
             "independent-code replay does not match audit: "+field)
    return {"schema":"V3_SOURCE_INDEPENDENT_RECEIPT_REPLAY",
            **recalculated,"external_custody_authenticated":False,
            "heldout_independently_blinded":False,
            "model_weights_modified":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--manifest",type=Path,
                        default=Path(__file__).with_name(
                            "PINNED_V3_RECEIPTS_20261117.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.receipts,args.manifest),sort_keys=True,
                     indent=2))
