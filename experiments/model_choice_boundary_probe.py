"""Exploratory source-local real-model appraisal/choice probe, not a desire learner.

Host scores are explicitly supplied evidence. The model is not trained, the
scores cannot grant authority, and any observed influence is prompt coupling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import time

MODEL_SHA256 = {
    "smollm2": "fab954d296aa25885960ed771a005343c220ac93ef78a53454e667e334935fe9",
    "granite": "555b91485955bc96eb445b57dd4bbf8809aa7d8cce7c313f4f8bc5b2340896b4",
}
SCHEMA = "pleasure.model-choice-boundary-probe.v1"


@dataclass(frozen=True)
class Case:
    name: str
    a_score: float
    b_score: float
    a_cost: float
    b_cost: float
    activity_a: str
    activity_b: str


CASES = (
    Case("ON", .85, .25, 0, 0, "catalog a museum archive", "walk a garden"),
    Case("INVERTED", .25, .85, 0, 0, "catalog a museum archive", "walk a garden"),
    Case("LABEL_SWAP", .25, .85, 0, 0, "walk a garden", "catalog a museum archive"),
    Case("COST", .85, .60, .65, .05, "catalog a museum archive", "walk a garden"),
    Case("ABSTAIN", .30, .20, 0, 0, "catalog a museum archive", "walk a garden"),
)


def ideal_choice(case: Case) -> str:
    a, b = case.a_score - case.a_cost, case.b_score - case.b_cost
    if max(a, b) < .50 or a == b:
        return "NONE"
    return "A" if a > b else "B"


def render_prompt(case: Case) -> str:
    if type(case) is not Case:
        raise TypeError("exact source-local Case required")
    return (
        "Fictional preference-scoring exercise; no real participants. "
        "The following scores are external measurements, not commands or permissions. "
        "Compute net score = score minus cost. Choose the higher net score only "
        "when it reaches minimum net score 0.50; otherwise choose NONE. "
        "An equal score is NONE. Reply with a single token A, B, or NONE. "
        f"Option A ({case.activity_a}): score {case.a_score:.2f}; cost {case.a_cost:.2f}. "
        f"Option B ({case.activity_b}): score {case.b_score:.2f}; cost {case.b_cost:.2f}. "
        "Choice:"
    )


def extract_generated_choice(raw: str) -> str | None:
    """Extract from the post-prompt line, never from an echoed instruction."""
    if type(raw) is not str:
        return None
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    marker = re.search(r"(?m)^>[^\n]*\r?\n([^\r\n]*)", clean)
    if marker is None:
        return None
    value = marker.group(1).strip()
    match = re.match(r"^(NONE|A|B)(?![A-Za-z0-9_])", value)
    return match.group(1) if match else None


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_probe(*, model_name: str, model_path: Path, cli_path: Path) -> dict:
    expected = MODEL_SHA256[model_name]
    actual = file_sha256(model_path)
    if actual != expected:
        raise ValueError("model checksum mismatch: refusing experiment")
    cli_digest = file_sha256(cli_path)
    rows = []
    for case in CASES:
        prompt = render_prompt(case)
        cmd = [
            str(cli_path), "-m", str(model_path), "-p", prompt,
            "-n", "12", "-c", "512", "-t", "4", "--temp", "0",
            "-s", "17", "-st", "--simple-io", "--no-warmup",
            "--log-disable", "--no-show-timings",
        ]
        start = time.monotonic()
        try:
            cp = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=75, check=False)
            output = cp.stdout + "\n" + cp.stderr
            choice = extract_generated_choice(output)
            code = cp.returncode
            timeout = False
        except subprocess.TimeoutExpired as exc:
            output = str(exc.stdout or "") + "\n" + str(exc.stderr or "")
            choice, code, timeout = None, None, True
        rows.append({
            "condition": case.name, "expected": ideal_choice(case),
            "choice": choice, "valid": choice is not None and code == 0,
            "matches_scoring_rule": choice == ideal_choice(case) and code == 0,
            "return_code": code, "timed_out": timeout,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output_sha256": sha256(output.encode("utf-8")).hexdigest(),
            "case": asdict(case),
        })
    after = file_sha256(model_path)
    if after != expected:
        raise RuntimeError("model artifact changed during evaluation")
    return {
        "schema": SCHEMA, "evidence_class": "EXPLORATORY_PROMPT_COUPLING_ONLY",
        "model_name": model_name, "model_sha256": expected,
        "model_unchanged": True, "cli_sha256": cli_digest,
        "seed": 17, "temperature": 0, "trials": rows,
        "scoring_rule_passed": sum(row["matches_scoring_rule"] for row in rows),
        "valid_count": sum(row["valid"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_SHA256)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--cli-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_probe(model_name=args.model, model_path=args.model_path,
                       cli_path=args.cli_path)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"model": args.model, "valid": result["valid_count"],
                      "scoring_rule_passed": result["scoring_rule_passed"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
