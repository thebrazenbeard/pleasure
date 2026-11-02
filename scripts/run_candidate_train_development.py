"""TRAIN-only source-local development comparison for numeric candidate appraisal.

Not a v4 confirmatory EVAL: the development payoff is known source code,
there is no external custodian, and the model's open weights are untouched.
Runs matched chosen-only training with shuffled and frozen controls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import fsum
import random

from pleasure_experiment.candidate_value_head import CandidateValueHead

TRAIN_ONLY_EVIDENCE = "SOURCE_LOCAL_V4_CANDIDATE_TRAIN_DEVELOPMENT_NOT_CONFIRMATORY"
ARMS = ("genuine_on", "genuine_off", "genuine_inverted",
        "shuffled_on", "frozen_on")
DEFAULT_SEEDS = (17, 29, 43, 59, 71, 83, 97, 109)
EPSILON = 0.4
GAIN = 12.0


def _sha(data: object) -> str:
    return sha256(json.dumps(data, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _candidate(rng: random.Random) -> tuple[float, float]:
    return (round(.05 + .90*rng.random(), 6),
            round(.05 + .90*rng.random(), 6))


def _development_quality(features: tuple[float, float]) -> float:
    """Explicit source-visible affine development oracle; NEVER v4 EVAL."""
    x, y = features
    return .5 + .32*(.75*(2*x-1) - .25*(2*y-1))


@dataclass(frozen=True)
class ChosenTrainRow:
    episode_id: str
    features_a: tuple[float, float]
    features_b: tuple[float, float]
    chosen_action: str
    propensity: float
    random_draw: float
    chosen_quality: float
    evidence_id: str


def development_training_choices(*, seed: int, count: int) -> tuple[ChosenTrainRow, ...]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("development seed must be bounded uint32")
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("development TRAIN count must be 1..256")
    rng = random.Random(seed ^ 0x4A2E7711)
    rows = []
    for i in range(count):
        a, b = _candidate(rng), _candidate(rng)
        draw = rng.random()
        choice = "A" if draw < .5 else "B"
        # All behavior decisions are sampled and identified before the
        # corresponding *chosen-only* source-local quality is observed.
        episode = f"v4-dev-train:{seed}:{i}"
        pre_feedback = (episode, a, b, .5, .5, draw, choice)
        decision_sha = _sha(("pre-feedback-v4-dev", pre_feedback))
        observed = _development_quality(a if choice == "A" else b)
        evidence = _sha(("chosen-dev-receipt", decision_sha, observed))
        rows.append(ChosenTrainRow(episode, a, b, choice, .5,
                                   draw, observed, evidence))
    return tuple(rows)


def _derangement(size: int, seed: int) -> tuple[int, ...]:
    if size < 2:
        raise ValueError("shuffled feedback requires two or more observations")
    rng = random.Random(seed ^ 0x5157EEDD)
    for _ in range(10000):
        indexes = list(range(size))
        rng.shuffle(indexes)
        if all(i != index for i, index in enumerate(indexes)):
            return tuple(indexes)
    raise RuntimeError("deranged negative control generation failed")


def _diagnostic_offers(seed: int, count: int):
    rng = random.Random(seed ^ 0x6B10A4B3)
    return tuple((_candidate(rng), _candidate(rng)) for _ in range(count))


def _run_seed(seed: int, train_count: int, diagnostic_count: int) -> dict:
    train = development_training_choices(seed=seed, count=train_count)
    wrong_labels = _derangement(train_count, seed)
    true_head = CandidateValueHead(
        allow_training=True, learning_rate=1.0, max_updates=train_count)
    wrong_head = CandidateValueHead(
        allow_training=True, learning_rate=1.0, max_updates=train_count)
    frozen = CandidateValueHead(allow_training=False)
    feedback_trace = []
    for index, row in enumerate(train):
        p = true_head.prepare_train(row.episode_id,
                                    row.features_a, row.features_b)
        committed = true_head.commit_train(p, row.chosen_action, row.propensity)
        update = true_head.learn_chosen(
            committed, row.evidence_id, row.chosen_quality)

        other = wrong_head.prepare_train(
            row.episode_id, row.features_a, row.features_b)
        shuffled_commit = wrong_head.commit_train(
            other, row.chosen_action, row.propensity)
        donor = train[wrong_labels[index]]
        # Wrong labels are *synthetic targets*, not forged genuine receipts.
        wrong_id = _sha(("shuffled-synthetic-target", row.evidence_id, donor.evidence_id))
        wrong_head.learn_chosen(
            shuffled_commit, wrong_id, donor.chosen_quality)
        feedback_trace.append({
            **asdict(row), "wrong_label_donor_episode": donor.episode_id,
            "pre_feedback_decision_digest": committed.decision_digest,
            "genuine_update_index": update.update_index,
        })
    state = true_head.export_state()
    reconstructed = CandidateValueHead.from_state(state, allow_training=False,
                                                    min_updates=train_count)
    if any(
        reconstructed.predict(row.features_a, row.features_b) !=
        true_head.predict(row.features_a, row.features_b)
        for row in train
    ):
        raise RuntimeError("local export/import altered candidate inference")

    diagnostic = _diagnostic_offers(seed, diagnostic_count)
    pending_decisions = []
    for i, (a, b) in enumerate(diagnostic):
        # All arms commit their probabilities *before* any diagnostic oracle.
        predictions = {
            "genuine_on": true_head.predict(a, b, gain=GAIN),
            "genuine_off": true_head.predict(a, b, gain=0.0),
            "genuine_inverted": true_head.predict(a, b, gain=-GAIN),
            "shuffled_on": wrong_head.predict(a, b, gain=GAIN),
            "frozen_on": frozen.predict(a, b, gain=GAIN),
        }
        pending_decisions.append((i, a, b, predictions))
    decision_seal = _sha([
        (i, a, b, {arm: p.prob_a for arm, p in predictions.items()})
        for i, a, b, predictions in pending_decisions
    ])
    qualities: dict[str, list[float]] = {name: [] for name in ARMS}
    sign_agreements = []
    diagnostic_rows = []
    for index, a, b, predictions in pending_decisions:
        qa, qb = _development_quality(a), _development_quality(b)
        sign_agreements.append(
            (predictions["genuine_on"].quality_a -
             predictions["genuine_on"].quality_b) * (qa - qb) > 0)
        actions = {}
        for arm, estimate in predictions.items():
            actual_prob_a = (1-EPSILON)*estimate.prob_a + EPSILON*.5
            value = actual_prob_a*qa + (1.0-actual_prob_a)*qb
            qualities[arm].append(value)
            actions[arm] = {
                "raw_probability_a_before_exploration": estimate.prob_a,
                "probability_a_pre_outcome": actual_prob_a,
                "expected_quality_posthoc": value,
            }
        diagnostic_rows.append({
            "diagnostic_id": f"v4-dev-diagnostic:{seed}:{index}",
            "features_a": a, "features_b": b,
            "quality_a_posthoc": qa, "quality_b_posthoc": qb,
            "arms": actions,
        })
    means = {arm: fsum(vals)/len(vals) for arm, vals in qualities.items()}
    return {
        "seed": seed, "source_local_chosen_receipt_count": len(train),
        "genuine_updates": true_head.updates,
        "shuffled_updates": wrong_head.updates,
        "frozen_updates": frozen.updates,
        "shuffled_labels_deranged": all(i != x for i, x in enumerate(wrong_labels)),
        "wrong_label_permutation": wrong_labels,
        "train_trace_sha256": _sha(feedback_trace),
        "train_trace": feedback_trace,
        "diagnostic_decision_seal": decision_seal,
        "diagnostic_offer_count": diagnostic_count,
        "diagnostic_rows": diagnostic_rows,
        "head_state": state, "shuffled_state": wrong_head.export_state(),
        "reloaded_prediction_exact": True,
        "train_episodes_resolved_before_diagnostic": True,
        "development_expected_quality": means,
        "preference_sign_accuracy_development": sum(sign_agreements)/len(sign_agreements),
    }


def run_candidate_development(*, seeds=DEFAULT_SEEDS, train_count: int = 96,
                              diagnostic_count: int = 64) -> dict:
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= 32 or
        any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)):
        raise ValueError("unique bounded development seeds required")
    if (type(train_count) is not int or not 2 <= train_count <= 256 or
        type(diagnostic_count) is not int or not 1 <= diagnostic_count <= 256):
        raise ValueError("development counts must be bounded")
    reports = [_run_seed(seed, train_count, diagnostic_count) for seed in seeds]
    means = {arm: fsum(r["development_expected_quality"][arm]
                       for r in reports)/len(reports) for arm in ARMS}
    return {
        "evidence_class": TRAIN_ONLY_EVIDENCE,
        "training_feedback_source": "chosen_only",
        "diagnostic_status": "POSTHOC_DEVELOPMENT_NOT_CONFIRMATORY",
        "external_blind_custody": False,
        "pretrained_model_modified": False,
        "phenomenology_established": False,
        "arms": ARMS, "seed_count": len(seeds),
        "train_count_per_seed": train_count,
        "diagnostic_count_per_seed": diagnostic_count,
        "seed_reports": reports,
        "mean_expected_quality": means,
        "development_contrast_on_minus_frozen":
            means["genuine_on"]-means["frozen_on"],
        "development_contrast_on_minus_shuffled":
            means["genuine_on"]-means["shuffled_on"],
        "development_contrast_on_minus_off":
            means["genuine_on"]-means["genuine_off"],
        "interpretation": "Source-visible synthetic development positive control, "
            "not independent evidence of heldout utility or felt pleasure.",
    }


def main() -> None:
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", help="new path for development JSON, never overwrite")
    args = p.parse_args()
    result = run_candidate_development()
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.report:
        path = Path(args.report)
        if not path.is_absolute() or ".." in path.parts or not path.parent.is_dir():
            raise ValueError("existing parent and absolute report path required")
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(payload)
    print(json.dumps({
        "evidence_class": result["evidence_class"],
        "seed_count": result["seed_count"],
        "mean_expected_quality": result["mean_expected_quality"],
        "on_minus_frozen": result["development_contrast_on_minus_frozen"],
        "on_minus_shuffled": result["development_contrast_on_minus_shuffled"],
        "report_path": args.report,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
