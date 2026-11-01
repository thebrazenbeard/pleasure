"""Contract and hostile replay tests for source-only online evidence chains."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unittest

from pleasure_experiment.online_learning_receipts import (
    Aborted, ActionCommitted, CAPABILITY_TIER, EVIDENCE_CLASS,
    OnlineEvidenceJournal, OnlineEvidenceReceipt, OptimizerReported,
    OutcomeObserved, canonical_json, classify_digests, receipt_sha256,
)
from pleasure_experiment.outcome_arena import OutcomeArena


ROOT = Path(__file__).resolve().parents[1]


def h(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def source_cycle(
    *,
    pair_index: int = 0,
    partition: str = "TRAIN",
    previous: str | None = None,
    sequence: int = 1,
    start_ms: int = 1000,
    adapter_before: str | None = None,
):
    """Build a real deterministic arena decision and source-only journal events."""
    arena = OutcomeArena(seed=101, evaluator_id="source-lab-evaluator")
    offer = arena.issue(partition=partition, family_id="goal-numeric", pair_index=pair_index, side=0)
    decision = h(f"prediction-before-outcome-{partition}-{pair_index}")
    ticket = arena.commit(offer, "A", decision_digest=decision,
                          model_id="tiny-source-surrogate", runtime_instance_id="instance-1")
    common = dict(
        session_id="source-session-1",
        session_epoch_id="source-epoch-1",
        model_id=ticket.model_id,
        model_revision="tiny-revision-1",
        model_binding_sha256=h("immutable-model-binding"),
        runtime_instance_id=ticket.runtime_instance_id,
        goal_instance_id=offer.family_id,
        task_id=offer.task_id,
        episode_id=offer.episode_id,
        decision_digest=ticket.decision_digest,
        action_choice=ticket.action,
        action_commit_id=ticket.commit_id,
        offer_nonce_sha256=offer.nonce,
    )
    action = OnlineEvidenceReceipt(
        **common,
        kind="ACTION_COMMITTED",
        event_id=ticket.commit_id,
        sequence=sequence,
        previous_event_sha256=previous,
        recorded_at_ms=start_ms,
        payload=ActionCommitted(
            partition=partition,
            goal_definition_sha256=h("numeric goal definition"),
            adapter_before_sha256=adapter_before or h("base adapter"),
            predicted_quality=0.5,
            execution_receipt_claim_sha256=None,
            train_manifest_sha256=h("locked train partition"),
            holdout_manifest_sha256=h("locked protected eval partition"),
        ),
    )
    outcome = arena.resolve(ticket)
    observed = OnlineEvidenceReceipt(
        **common,
        kind="OUTCOME_OBSERVED",
        event_id=outcome.evidence_id,
        sequence=sequence + 1,
        previous_event_sha256=receipt_sha256(action),
        recorded_at_ms=start_ms + 1,
        payload=OutcomeObserved(
            action_receipt_sha256=receipt_sha256(action),
            evidence_id=outcome.evidence_id,
            evaluator_id=outcome.evaluator_id,
            rubric_sha256=outcome.rubric_digest,
            outcome_evidence_sha256=h(json.dumps(asdict(outcome), sort_keys=True)),
            custody_claim_sha256=None,
            measured_quality=outcome.measured_quality,
        ),
    )
    return action, observed


def optimizer(
    action: OnlineEvidenceReceipt,
    observed: OnlineEvidenceReceipt,
    *,
    arm: str = "correct_outcome",
    steps: int = 1,
    after: str | None = None,
) -> OnlineEvidenceReceipt:
    before = action.payload.adapter_before_sha256
    after = after or h("trained adapter")
    return replace(
        action,
        kind="OPTIMIZER_REPORTED",
        event_id=h(f"optimizer-event-{action.task_id}"),
        sequence=observed.sequence + 1,
        previous_event_sha256=receipt_sha256(observed),
        recorded_at_ms=observed.recorded_at_ms + 1,
        payload=OptimizerReported(
            action_receipt_sha256=receipt_sha256(action),
            outcome_receipt_sha256=receipt_sha256(observed),
            arm=arm,
            train_example_manifest_sha256=action.payload.train_manifest_sha256,
            holdout_exclusion_manifest_sha256=action.payload.holdout_manifest_sha256,
            declared_optimizer_steps=steps,
            gradient_summary_claim_sha256=h("unverified numeric gradient") if steps else None,
            trainer_receipt_claim_sha256=h("unverified trainer") if steps else None,
            adapter_before_sha256=before,
            adapter_after_sha256=after,
            checkpoint_claim_sha256=None,
            weight_change_evidence=classify_digests(before, after),
        ),
    )


def abort(action: OnlineEvidenceReceipt, observed: OnlineEvidenceReceipt | None = None,
          *, restore_status: str = "NO_OPTIMIZER_STARTED") -> OnlineEvidenceReceipt:
    previous = observed or action
    return replace(
        action,
        kind="ABORTED",
        event_id=h(f"abort-event-{action.task_id}"),
        sequence=previous.sequence + 1,
        previous_event_sha256=receipt_sha256(previous),
        recorded_at_ms=previous.recorded_at_ms + 1,
        payload=Aborted(
            action_receipt_sha256=receipt_sha256(action),
            outcome_receipt_sha256=receipt_sha256(observed) if observed else None,
            reason="POLICY_DENIED",
            restore_status=restore_status,
            restore_receipt_claim_sha256=h("unverified restore") if restore_status != "NO_OPTIMIZER_STARTED" else None,
        ),
    )


def journal() -> OnlineEvidenceJournal:
    return OnlineEvidenceJournal(
        session_id="source-session-1",
        session_epoch_id="source-epoch-1",
        model_binding_sha256=h("immutable-model-binding"),
    )


class OnlineLearningReceiptTests(unittest.TestCase):
    def test_real_arena_commit_then_independent_source_outcome_then_reported_optimizer(self):
        action, observed = source_cycle()
        j = journal()
        action_hash = j.append(action)
        self.assertEqual(action_hash, receipt_sha256(action))
        self.assertEqual(j.append(observed), receipt_sha256(observed))
        trained = optimizer(action, observed)
        self.assertEqual(j.append(trained), receipt_sha256(trained))
        self.assertEqual(trained.payload.weight_change_evidence, "DIGESTS_DIFFER_UNVERIFIED")
        self.assertEqual(trained.capability_tier, CAPABILITY_TIER)
        self.assertEqual(trained.phenomenology, "UNRESOLVED")
        self.assertEqual(observed.payload.evidence_class, EVIDENCE_CLASS)
        self.assertEqual(j.next_sequence, 4)
        self.assertEqual(len(j.receipts), 3)

    def test_digest_chain_export_import_reconstructs_exact_replay_state(self):
        action, observed = source_cycle()
        j = journal()
        for event in (action, observed, optimizer(action, observed)):
            j.append(event)
        exported = j.export_json()
        restored = OnlineEvidenceJournal.import_json(exported)
        self.assertEqual(restored.export_json(), exported)
        self.assertEqual(restored.head_sha256, j.head_sha256)
        with self.assertRaisesRegex(ValueError, "nonmonotonic|replay"):
            restored.append(observed)
        # New session epochs are NOT a durable host replay prohibition.
        self.assertIn("SOURCE_ONLY_UNAUTHENTICATED", exported)
        self.assertNotIn("private_transcript", exported)

    def test_reordered_future_outcome_and_duplicate_event_are_rejected(self):
        action, observed = source_cycle()
        j = journal()
        with self.assertRaisesRegex(ValueError, "nonmonotonic|replay"):
            j.append(observed)
        j.append(action)
        with self.assertRaisesRegex(ValueError, "out-of-order|future"):
            j.append(replace(observed, recorded_at_ms=action.recorded_at_ms))
        j.append(observed)
        with self.assertRaisesRegex(ValueError, "nonmonotonic|replay"):
            j.append(observed)
        with self.assertRaisesRegex(ValueError, "previous action cycle not finalized"):
            new_action, _ = source_cycle(pair_index=1, sequence=3, previous=j.head_sha256,
                                         start_ms=observed.recorded_at_ms + 1)
            j.append(new_action)

    def test_cross_session_model_and_exact_identity_substitution_rejected(self):
        action, observed = source_cycle()
        j = journal()
        with self.assertRaisesRegex(ValueError, "cross-session|cross-model"):
            j.append(replace(action, session_epoch_id="wrong-epoch"))
        j.append(action)
        for key, val in (
            ("model_id", "wrong-model"),
            ("model_revision", "wrong-revision"),
            ("model_binding_sha256", h("wrong-model")),
            ("runtime_instance_id", "other-instance"),
            ("goal_instance_id", "other-goal"),
            ("task_id", h("other-task")),
            ("episode_id", h("other-episode")),
            ("decision_digest", h("future-prediction")),
            ("action_choice", "B"),
            ("action_commit_id", h("other-action-commit")),
            ("offer_nonce_sha256", h("other-nonce")),
        ):
            with self.subTest(field=key):
                with self.assertRaisesRegex(ValueError, "cross-action|cross-model"):
                    j.append(replace(observed, **{key: val}))
        j.append(observed)

    def test_action_and_outcome_receipt_digest_swaps_fail_closed(self):
        action, observed = source_cycle()
        j = journal()
        j.append(action)
        bad_outcome = replace(observed, payload=replace(
            observed.payload, action_receipt_sha256=h("wrong-action")))
        with self.assertRaisesRegex(ValueError, "outcome/action receipt"):
            j.append(bad_outcome)
        j.append(observed)
        trained = optimizer(action, observed)
        with self.assertRaisesRegex(ValueError, "optimizer/action/outcome"):
            j.append(replace(trained, payload=replace(trained.payload, outcome_receipt_sha256=h("wrong"))))
        with self.assertRaisesRegex(ValueError, "adapter-before digest mismatch"):
            j.append(replace(trained, payload=replace(trained.payload, adapter_before_sha256=h("wrong"),
                                                       weight_change_evidence="DIGESTS_DIFFER_UNVERIFIED")))
        j.append(trained)

    def test_train_holdout_manifest_and_eval_partition_lock(self):
        action, observed = source_cycle()
        with self.assertRaisesRegex(ValueError, "must not be identical"):
            replace(action.payload, holdout_manifest_sha256=action.payload.train_manifest_sha256)
        j = journal()
        j.append(action)
        j.append(observed)
        report = optimizer(action, observed)
        with self.assertRaisesRegex(ValueError, "locked holdout manifest"):
            j.append(replace(report, payload=replace(report.payload,
                                                     holdout_exclusion_manifest_sha256=h("other holdout"))))
        with self.assertRaisesRegex(ValueError, "locked train manifest"):
            j.append(replace(report, payload=replace(report.payload,
                                                     train_example_manifest_sha256=h("other train"))))

        eval_action, eval_observed = source_cycle(partition="EVAL")
        e = journal()
        e.append(eval_action)
        e.append(eval_observed)
        with self.assertRaisesRegex(ValueError, "protected EVAL"):
            e.append(optimizer(eval_action, eval_observed))
        unchanged = optimizer(eval_action, eval_observed, arm="frozen", steps=0,
                              after=eval_action.payload.adapter_before_sha256)
        e.append(unchanged)

    def test_frozen_zero_and_no_step_control_guards(self):
        action, observed = source_cycle()
        with self.assertRaisesRegex(ValueError, "frozen control"):
            optimizer(action, observed, arm="frozen", steps=1)
        with self.assertRaisesRegex(ValueError, "control or no-step"):
            optimizer(action, observed, arm="zero", steps=1)
        with self.assertRaisesRegex(ValueError, "control or no-step"):
            optimizer(action, observed, arm="correct_outcome", steps=0)
        unchanged = optimizer(action, observed, arm="zero", steps=1,
                              after=action.payload.adapter_before_sha256)
        self.assertEqual(unchanged.payload.weight_change_evidence, "DIGESTS_IDENTICAL")
        j = journal()
        j.append(action)
        j.append(observed)
        j.append(unchanged)

    def test_abort_before_or_after_outcome_does_not_claim_optimizer_and_recovery_gate(self):
        action, observed = source_cycle()
        j = journal()
        j.append(action)
        j.append(abort(action))
        next_action, next_observed = source_cycle(pair_index=1, previous=j.head_sha256,
                                                  sequence=3, start_ms=1002)
        j.append(next_action)
        j.append(next_observed)
        j.append(abort(next_action, next_observed, restore_status="HOST_RESTORE_UNVERIFIED"))
        with self.assertRaisesRegex(RuntimeError, "blocked by unverified restore"):
            next_after_abort, _ = source_cycle(pair_index=2, sequence=6,
                                               previous=j.head_sha256, start_ms=1005)
            j.append(next_after_abort)
        self.assertEqual(len(j.receipts), 5)

    def test_failed_optimizer_cannot_claim_no_optimizer_and_continue(self):
        action, observed = source_cycle()
        with self.assertRaisesRegex(ValueError, "failed trainer requires unverified host restore"):
            Aborted(
                action_receipt_sha256=receipt_sha256(action),
                outcome_receipt_sha256=receipt_sha256(observed),
                reason="TRAINER_FAILED",
                restore_status="NO_OPTIMIZER_STARTED",
            )
        j = journal()
        j.append(action)
        j.append(observed)
        failed = replace(abort(action, observed, restore_status="HOST_RESTORE_UNVERIFIED"),
                         payload=Aborted(
                             action_receipt_sha256=receipt_sha256(action),
                             outcome_receipt_sha256=receipt_sha256(observed),
                             reason="TRAINER_FAILED",
                             restore_status="HOST_RESTORE_UNVERIFIED",
                         ))
        j.append(failed)
        with self.assertRaisesRegex(RuntimeError, "blocked by unverified restore"):
            next_action, _ = source_cycle(pair_index=1, sequence=4,
                                           previous=j.head_sha256, start_ms=1003)
            j.append(next_action)

    def test_next_action_requires_exact_prior_reported_adapter_digest(self):
        action, observed = source_cycle()
        j = journal()
        j.append(action)
        j.append(observed)
        reported = optimizer(action, observed)
        j.append(reported)
        next_action, _ = source_cycle(pair_index=1, sequence=4, previous=j.head_sha256,
                                      start_ms=1003)
        with self.assertRaisesRegex(ValueError, "adapter digest continuity"):
            j.append(next_action)
        j.append(replace(next_action, payload=replace(next_action.payload,
                                                      adapter_before_sha256=reported.payload.adapter_after_sha256)))

    def test_import_fails_on_corruption_unknown_fields_duplicate_keys_and_wrong_head(self):
        action, observed = source_cycle()
        j = journal()
        j.append(action)
        j.append(observed)
        raw = json.loads(j.export_json())
        raw["events"][0]["receipt"]["payload"]["raw_private_text"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            OnlineEvidenceJournal.import_json(json.dumps(raw))
        raw = json.loads(j.export_json())
        raw["events"][0]["receipt"]["goal_instance_id"] = "different-goal"
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            OnlineEvidenceJournal.import_json(json.dumps(raw))
        raw = json.loads(j.export_json())
        raw["head_sha256"] = h("forged")
        with self.assertRaisesRegex(ValueError, "head digest mismatch"):
            OnlineEvidenceJournal.import_json(json.dumps(raw))
        raw = json.loads(j.export_json())
        raw["events"].reverse()
        with self.assertRaisesRegex(ValueError, "nonmonotonic|replay"):
            OnlineEvidenceJournal.import_json(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            OnlineEvidenceJournal.import_json('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            OnlineEvidenceJournal.import_json('{"schema":NaN}')

    def test_nonfinite_bool_invalid_kind_and_no_private_fields(self):
        action, observed = source_cycle()
        for bad in (math.nan, math.inf, -math.inf, True, -0.1, 1.1):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "predicted_quality must be finite"):
                    replace(action.payload, predicted_quality=bad)
        with self.assertRaisesRegex(ValueError, "integer"):
            replace(action, sequence=True)
        with self.assertRaisesRegex(ValueError, "payload type"):
            replace(action, kind="OUTCOME_OBSERVED")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            OnlineEvidenceReceipt.from_dict({**asdict(action), "raw_transcript": "sensitive"})
        self.assertEqual(OnlineEvidenceReceipt.from_dict(json.loads(canonical_json(observed))), observed)

    def test_action_commit_cannot_claim_future_physical_execution(self):
        action, _ = source_cycle()
        with self.assertRaisesRegex(ValueError, "cannot claim future host execution"):
            replace(action.payload, execution_receipt_claim_sha256=h("imagined host execution"))
        schema = json.loads((ROOT / "schemas" / "pleasure-online-learning-evidence-v1.schema.json").read_text())
        self.assertEqual(schema["$defs"]["action"]["properties"]["execution_receipt_claim_sha256"]["type"],
                         "null")

    def test_journal_bindings_and_transitions_are_ordinary_python_read_only(self):
        j = journal()
        for field, value in (
            ("session_id", "other-session"),
            ("session_epoch_id", "other-epoch"),
            ("model_binding_sha256", h("other-model")),
            ("_last_adapter_sha256", h("forged-adapter")),
            ("_blocked", False),
            ("_action", None),
            ("_receipts", []),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(AttributeError, "not publicly assignable"):
                    setattr(j, field, value)
                with self.assertRaisesRegex(AttributeError, "not publicly deletable"):
                    delattr(j, field)
        self.assertEqual(j.next_sequence, 1)

    def test_post_append_caller_mutation_cannot_rewrite_snapshot_or_chain(self):
        action, observed = source_cycle()
        j = journal()
        original_hash = j.append(action)
        original_export = j.export_json()
        object.__setattr__(action, "model_id", "relabeled-model")
        object.__setattr__(action.payload, "predicted_quality", math.nan)
        self.assertEqual(j.receipts[0].model_id, "tiny-source-surrogate")
        self.assertEqual(j.receipts[0].payload.predicted_quality, 0.5)
        self.assertEqual(j.head_sha256, original_hash)
        self.assertEqual(j.export_json(), original_export)
        exposed_snapshot = j.receipts[0]
        object.__setattr__(exposed_snapshot, "model_id", "mutated-through-public-property")
        self.assertEqual(j.receipts[0].model_id, "tiny-source-surrogate")
        self.assertEqual(j.export_json(), original_export)
        j.append(observed)
        self.assertEqual(len(j.receipts), 2)

    def test_schema_closed_variant_shapes_and_identifier_parity(self):
        schema = json.loads((ROOT / "schemas" / "pleasure-online-learning-evidence-v1.schema.json").read_text())
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(len(schema["allOf"]), 4)
        for name in ("action", "outcome", "optimizer", "abort"):
            self.assertIs(schema["$defs"][name]["additionalProperties"], False)
        identifier = schema["$defs"]["identifier"]
        pattern = re.compile(identifier["pattern"])
        for candidate, valid in (
            ("model", True), ("model with space", True), (" name", False),
            ("name ", False), ("na\nme", False), ("na\x7fme", False),
            ("name\u3000", False), ("a" * 256, True), ("a" * 257, False),
        ):
            with self.subTest(candidate=repr(candidate)):
                matched = (identifier["minLength"] <= len(candidate) <= identifier["maxLength"]
                           and pattern.fullmatch(candidate) is not None)
                self.assertEqual(matched, valid)

    def test_parallel_same_sequence_only_one_append(self):
        action, observed = source_cycle()
        j = journal()
        j.append(action)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(
                lambda item: _attempt_append(j, item), (observed, observed)
            ))
        self.assertEqual(sum(isinstance(v, str) for v in outcomes), 1)
        self.assertEqual(sum(isinstance(v, ValueError) for v in outcomes), 1)
        self.assertEqual(j.next_sequence, 3)


def _attempt_append(journal: OnlineEvidenceJournal, item: OnlineEvidenceReceipt):
    try:
        return journal.append(item)
    except ValueError as exc:
        return exc


if __name__ == "__main__":
    unittest.main()
