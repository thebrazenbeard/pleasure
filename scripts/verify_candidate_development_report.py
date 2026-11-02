#!/usr/bin/env python3
"""Independent read-only reconstruction of source-local v4 development evidence.

Uses stdlib only. Does NOT import the candidate head or development runner,
invoke reward generators, train a real model, or mutate a source artifact.
Hash/consistency verifies author-controlled data, not external authenticity.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import exp, fsum, isclose, isfinite, sqrt
from pathlib import Path
import random
import re

SHA = re.compile(r"[a-f0-9]{64}\Z")
ARMS = ("genuine_on", "genuine_off", "genuine_inverted",
        "shuffled_on", "frozen_on")
EVIDENCE = "SOURCE_LOCAL_V4_CANDIDATE_TRAIN_DEVELOPMENT_NOT_CONFIRMATORY"
HEAD_SCHEMA = "numeric-candidate-xy-shared-sigmoid-v4"
HEAD_EVIDENCE = "SOURCE_LOCAL_CHOSEN_FEEDBACK_CANDIDATE_V4_DEVELOPMENT"
MAX_REPORT_BYTES = 8 * 1024 * 1024


def _require(ok, why):
    if not ok:
        raise ValueError("invalid candidate development report: " + why)


def _sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _equal(actual, expected, why, tol=1e-11):
    _require(type(actual) in (int, float) and
             isfinite(actual) and isfinite(expected) and
             isclose(actual, expected, rel_tol=0.0, abs_tol=tol), why)


def _features(rng):
    return tuple(round(.05 + .90 * rng.random(), 6) for _ in range(2))


def _value(features):
    x, y = features
    return .5 + .32*(.75*(2*x-1) - .25*(2*y-1))


def _basis(features):
    return (1.0, 2*features[0]-1, 2*features[1]-1)


def _sigmoid(x):
    if x >= 0:
        return 1/(1+exp(-x))
    r = exp(x)
    return r/(1+r)


def _quality(weights, features):
    return _sigmoid(fsum(w*x for w,x in zip(weights,_basis(features))))


def _head_digest(weights):
    return _sha((HEAD_SCHEMA, weights))


def _update(weights, features, observed, propensity=.5):
    prediction = _quality(weights, features)
    common = (prediction-observed)*prediction*(1-prediction)*min(5,1/propensity)
    gradient = tuple(common*x for x in _basis(features))
    norm = sqrt(fsum(g*g for g in gradient))
    scale = min(1, 1/max(norm,1e-12))
    return tuple(max(-8.0,min(8.0,w-scale*g))
                 for w,g in zip(weights,gradient))


def _derangement(n, seed):
    rng=random.Random(seed ^ 0x5157EEDD)
    for _ in range(10000):
        order=list(range(n))
        rng.shuffle(order)
        if all(i!=j for i,j in enumerate(order)):
            return order
    raise ValueError("derangement unavailable")


def _check_state(record, weights, episodes, evidence, train_count, why):
    _require(type(record) is dict, why+" must be state dict")
    _require(record.get("format")=="CANDIDATE_VALUE_HEAD_V4_JSON_V1" and
             record.get("feature_schema")==HEAD_SCHEMA and
             record.get("evidence_class")==HEAD_EVIDENCE, why+" wrong schema")
    payload={key:value for key,value in record.items() if key!="state_digest"}
    _require(record.get("state_digest")==_sha(payload),why+" state digest")
    for key,expected in (("updates",train_count),("max_updates",train_count)):
        _require(type(record.get(key)) is int and record[key]==expected,
                 why+" update count")
    for key,expected in (("learning_rate",1.),("max_gradient_l2",1.)):
        _equal(record.get(key),expected,why+" optimizer "+key)
    values=record.get("weights")
    _require(type(values) is list and len(values)==3,why+" weights length")
    for a,b in zip(values,weights):
        _equal(a,b,why+" recomputed weight",2e-11)
    _require(record.get("seen_episode_ids")==sorted(episodes) and
             record.get("seen_evidence_ids")==sorted(evidence),
             why+" consumed receipt ledger")


def _verify_seed(report, train_count, diagnostic_count):
    seed=report.get("seed")
    _require(type(seed) is int and 0<=seed<2**32,"seed value")
    rows=report.get("train_trace")
    _require(type(rows) is list and len(rows)==train_count,"TRAIN trace count")
    _require(report.get("source_local_chosen_receipt_count")==train_count and
             report.get("genuine_updates")==train_count and
             report.get("shuffled_updates")==train_count and
             report.get("frozen_updates")==0,"TRAIN arm update count")
    _require(report.get("train_episodes_resolved_before_diagnostic") is True and
             report.get("reloaded_prediction_exact") is True,"TRAIN ledger gate")
    rng=random.Random(seed ^ 0x4A2E7711)
    genuine=(0.,0.,0.)
    wrong=(0.,0.,0.)
    expected_rows=[]
    for i in range(train_count):
        candidate_a=_features(rng)
        candidate_b=_features(rng)
        draw=rng.random()
        choice="A" if draw < .5 else "B"
        episode=f"v4-dev-train:{seed}:{i}"
        actual=_value(candidate_a if choice=="A" else candidate_b)
        decision=_sha(("pre-feedback-v4-dev",(
            episode,candidate_a,candidate_b,.5,.5,draw,choice)))
        evidence=_sha(("chosen-dev-receipt",decision,actual))
        pre_commit=_sha(("candidate-v4-commit",episode,
                         _head_digest(genuine),candidate_a,candidate_b,choice,.5))
        recorded=rows[i]
        _require(recorded.get("episode_id")==episode and
                 recorded.get("chosen_action")==choice and
                 recorded.get("features_a")==list(candidate_a) and
                 recorded.get("features_b")==list(candidate_b) and
                 recorded.get("evidence_id")==evidence and
                 recorded.get("pre_feedback_decision_digest")==pre_commit and
                 recorded.get("genuine_update_index")==i+1,
                 "TRAIN commitment mismatch")
        _equal(recorded.get("propensity"),.5,"TRAIN propensity",0)
        _equal(recorded.get("random_draw"),draw,"TRAIN draw",0)
        _equal(recorded.get("chosen_quality"),actual,"TRAIN chosen reward")
        expected_rows.append({
            "episode_id":episode,"features_a":candidate_a,
            "features_b":candidate_b,"chosen_action":choice,
            "propensity":.5,"random_draw":draw,
            "chosen_quality":actual,"evidence_id":evidence,
            "pre_feedback_decision_digest":pre_commit,
            "genuine_update_index":i+1,
        })
        genuine=_update(genuine,candidate_a if choice=="A" else candidate_b,actual)
    order=_derangement(train_count,seed)
    _require(report.get("wrong_label_permutation")==order and
             report.get("shuffled_labels_deranged") is True,
             "shuffled derangement")
    for i,row in enumerate(expected_rows):
        donor=expected_rows[order[i]]
        _require(rows[i].get("wrong_label_donor_episode")==donor["episode_id"],
                 "shuffled donor mapping")
        row["wrong_label_donor_episode"]=donor["episode_id"]
        chosen=row["features_a"] if row["chosen_action"]=="A" else row["features_b"]
        wrong=_update(wrong,chosen,donor["chosen_quality"])
    _require(report.get("train_trace_sha256")==_sha(expected_rows),
             "complete TRAIN trace SHA")
    episode_ledger=[r["episode_id"] for r in expected_rows]
    genuine_ledger=[r["evidence_id"] for r in expected_rows]
    wrong_ledger=[_sha(("shuffled-synthetic-target",r["evidence_id"],
                        expected_rows[order[i]]["evidence_id"]))
                  for i,r in enumerate(expected_rows)]
    _check_state(report["head_state"],genuine,episode_ledger,
                 genuine_ledger,train_count,"genuine")
    _check_state(report["shuffled_state"],wrong,episode_ledger,
                 wrong_ledger,train_count,"shuffled")
    diagnostics=report.get("diagnostic_rows")
    _require(type(diagnostics) is list and
             len(diagnostics)==diagnostic_count and
             report.get("diagnostic_offer_count")==diagnostic_count,
             "diagnostic offer count")
    test_rng=random.Random(seed ^ 0x6B10A4B3)
    expected_arms={arm:[] for arm in ARMS}
    exact_seal=[]
    signs=[]
    for i in range(diagnostic_count):
        a,b=_features(test_rng),_features(test_rng)
        row=diagnostics[i]
        _require(row.get("diagnostic_id")==f"v4-dev-diagnostic:{seed}:{i}" and
                 row.get("features_a")==list(a) and
                 row.get("features_b")==list(b), "diagnostic feature selection")
        true_a,true_b=_value(a),_value(b)
        _equal(row.get("quality_a_posthoc"),true_a,"posthoc A reward")
        _equal(row.get("quality_b_posthoc"),true_b,"posthoc B reward")
        qa,qb=_quality(genuine,a),_quality(genuine,b)
        sa,sb=_quality(wrong,a),_quality(wrong,b)
        signs.append((qa-qb)*(true_a-true_b)>0)
        raw={
            "genuine_on":_sigmoid(12*(qa-qb)),
            "genuine_off":.5,
            "genuine_inverted":_sigmoid(-12*(qa-qb)),
            "shuffled_on":_sigmoid(12*(sa-sb)),
            "frozen_on":.5,
        }
        _require(type(row.get("arms")) is dict and
                 set(row["arms"])==set(ARMS),"diagnostic arm identity")
        exact_seal.append((i,a,b,raw))
        for arm in ARMS:
            actual_raw=raw[arm]
            mixed=.6*actual_raw+.2
            expected_reward=mixed*true_a+(1-mixed)*true_b
            rr=row["arms"][arm]
            _equal(rr.get("raw_probability_a_before_exploration"),actual_raw,
                   "raw action probability")
            _equal(rr.get("probability_a_pre_outcome"),mixed,
                   "exploration policy")
            _equal(rr.get("expected_quality_posthoc"),expected_reward,
                   "posthoc expected quality")
            expected_arms[arm].append(expected_reward)
    _require(report.get("diagnostic_decision_seal")==_sha(exact_seal),
             "pre-outcome global decision seal")
    means={name:fsum(vals)/len(vals) for name,vals in expected_arms.items()}
    _require(type(report.get("development_expected_quality")) is dict and
             set(report["development_expected_quality"])==set(ARMS),
             "seed summary arms")
    for arm in ARMS:
        _equal(report["development_expected_quality"][arm],means[arm],
               "seed expected quality")
    _equal(report.get("preference_sign_accuracy_development"),
           sum(signs)/len(signs),"seed preference accuracy")
    return means


def audit_report(report):
    _require(type(report) is dict and
             report.get("evidence_class")==EVIDENCE and
             report.get("training_feedback_source")=="chosen_only" and
             report.get("diagnostic_status")=="POSTHOC_DEVELOPMENT_NOT_CONFIRMATORY",
             "not a source-local development study")
    _require(report.get("external_blind_custody") is False and
             report.get("pretrained_model_modified") is False and
             report.get("phenomenology_established") is False,
             "forbidden evidentiary promotion")
    _require(report.get("arms")==list(ARMS),"arm schedule")
    n=report.get("seed_count")
    tr,diag=report.get("train_count_per_seed"),report.get("diagnostic_count_per_seed")
    _require(type(n) is int and 1<=n<=32 and
             type(tr) is int and 2<=tr<=256 and
             type(diag) is int and 1<=diag<=256,
             "count bounds")
    reports=report.get("seed_reports")
    _require(type(reports) is list and len(reports)==n and
             len({x.get("seed") for x in reports})==n,"seed schedule")
    averages={arm:[] for arm in ARMS}
    for seed in reports:
        means=_verify_seed(seed,tr,diag)
        for arm in ARMS:
            averages[arm].append(means[arm])
    result={arm:fsum(v)/len(v) for arm,v in averages.items()}
    _require(type(report.get("mean_expected_quality")) is dict and
             set(report["mean_expected_quality"])==set(ARMS),
             "aggregate arm names")
    for arm in ARMS:
        _equal(report["mean_expected_quality"][arm],result[arm],
               "aggregate expected quality")
    contrasts={
        "development_contrast_on_minus_frozen":
            result["genuine_on"]-result["frozen_on"],
        "development_contrast_on_minus_shuffled":
            result["genuine_on"]-result["shuffled_on"],
        "development_contrast_on_minus_off":
            result["genuine_on"]-result["genuine_off"],
    }
    for name,expected in contrasts.items():
        _equal(report.get(name),expected,"aggregate contrast "+name)
    return {
        "status":"verified_source_local",
        "evidence_class":EVIDENCE,
        "seed_count":n,"train_receipts":n*tr,
        "diagnostic_offers":n*diag,"candidate_controls_checked":len(ARMS),
        "recomputed_mean_expected_quality":result,
        "recomputed_contrasts":contrasts,
        "limitation":"author-controlled source-local evidence, not authenticated blind custody",
    }


def _reject_duplicates(pairs):
    out={}
    for key,value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: "+key)
        out[key]=value
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path)
    p.add_argument("--expected-sha256",required=True)
    args=p.parse_args()
    expected=args.expected_sha256
    _require(type(expected) is str and SHA.fullmatch(expected) is not None,
             "expected digest must be SHA256")
    path=args.report.absolute()
    _require(".." not in path.parts and
             all(not item.is_symlink() for item in (path,)+tuple(path.parents))
             and path.is_file() and 0<path.stat().st_size<=MAX_REPORT_BYTES,
             "safe bounded complete report required")
    raw=path.read_bytes()
    _require(sha256(raw).hexdigest()==expected,"report-byte SHA mismatch")
    document=json.loads(raw,object_pairs_hook=_reject_duplicates,
                        parse_constant=lambda v: (_ for _ in ()).throw(
                            ValueError("nonfinite JSON constant "+v)))
    checked=audit_report(document)
    print(json.dumps(checked,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
