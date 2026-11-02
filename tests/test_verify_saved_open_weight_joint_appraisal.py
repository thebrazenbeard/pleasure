"""Mocked fresh-worker, strict consent, and leakage tests for v3 continuation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import verify_saved_open_weight_joint_appraisal as verifier


class FreshJointReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "original-v3"
        self.source.mkdir()
        (self.source / "manifest.json").write_text('{"version":3}\n', encoding="utf-8")
        (self.source / "weights.safetensors").write_bytes(b"safe-tensor-fixture")
        self.destination = self.base / "continued-v3"
        self.args = verifier.build_parser().parse_args([
            "--snapshot", str(self.base / "snapshot"),
            "--head-dir", str(self.source),
            "--continuation-dir", str(self.destination),
            "--model-id", verifier.PINNED_MODEL_ID,
            "--revision", verifier.PINNED_REVISION,
            "--allow-reload", "--allow-continuation-training",
            "--allow-persistent-export",
        ])

    def test_all_three_explicit_permissions_required(self) -> None:
        for name in ("allow_reload", "allow_continuation_training",
                     "allow_persistent_export"):
            args = Namespace(**vars(self.args))
            setattr(args, name, False)
            with self.subTest(permission=name), self.assertRaises(PermissionError):
                verifier._validate(args)

    def test_new_reserved_training_seed_and_no_replace_enforced(self) -> None:
        for seed in (self.args.seed, 163):
            args = Namespace(**vars(self.args))
            args.continuation_seed = seed
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                verifier._validate(args)
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            verifier._validate(self.args)

    def test_wrong_model_or_unpinned_safetensors_rejected(self) -> None:
        args = Namespace(**vars(self.args))
        args.model_id = "some-other-model"
        with self.assertRaisesRegex(ValueError, "exact preregistered"):
            verifier._validate(args)
        wrong = SimpleNamespace(
            binding=SimpleNamespace(model_id=verifier.PINNED_MODEL_ID,
                                    model_revision=verifier.PINNED_REVISION),
            manifest=SimpleNamespace(files=(SimpleNamespace(
                name="model.safetensors", sha256="b" * 64,
            ),)),
        )
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            verifier._assert_pinned_snapshot(wrong)

    def test_same_or_nested_source_destination_rejected(self) -> None:
        for dest in (self.source, self.source / "nested"):
            args = Namespace(**vars(self.args))
            args.continuation_dir = str(dest)
            with self.subTest(destination=dest), self.assertRaises(ValueError):
                verifier._validate(args)

    def test_child_invocation_is_distinct_isolated_interpreter(self) -> None:
        argv = verifier._worker_command(self.args)
        self.assertEqual(argv[:3], [verifier.sys.executable, "-I", "-B"])
        self.assertIn("--_fresh-worker", argv)
        self.assertEqual(argv[argv.index("--_parent-pid") + 1], str(os.getpid()))
        for consent in ("--allow-reload", "--allow-continuation-training",
                        "--allow-persistent-export"):
            self.assertIn(consent, argv)

    def test_strict_child_report_accepts_only_verified_distinct_child(self) -> None:
        row = {
            "episode_id": "new-heldout-0", "input_ids_sha256": "f" * 64,
            "joint_sha256": "j" * 64, "head_sha256": "h" * 64,
            "lora_sha256": "l" * 64, "base_sha256": "b" * 64,
            "on_prob_a": .7, "on_prob_b": .3,
            "disabled_head_prob_a": .5, "disabled_head_prob_b": .5,
            "predicted_quality_a": .6, "predicted_quality_b": .4,
        }
        reference = {"rows": [{**row, "episode_id": f"new-heldout-{i}"}
                              for i in range(4)],
                     "joint_sha256": "j" * 64, "head_sha256": "h" * 64,
                     "lora_sha256": "l" * 64, "base_sha256": "b" * 64}
        report = {
            "fresh_process_pid": os.getpid() + 10000,
            "parent_process_pid": os.getpid(),
            "spawned_fresh_model_process": True,
            "tensor_import_verified": True,
            "source_artifact_unchanged": True,
            "continuation_training_saved": True,
            "heldout_outcomes_not_resolved": True,
            "distinct_evaluator_workers": True,
            "imported_joint_sha256": "j" * 64,
            "imported_head_sha256": "h" * 64,
            "imported_lora_sha256": "l" * 64,
            "frozen_base_sha256_unchanged": "b" * 64,
            "four_new_unresolved_heldout_offers": reference["rows"],
        }
        with (patch.object(verifier, "_parent_reference", return_value=reference),
              patch.object(verifier.subprocess, "run", return_value=SimpleNamespace(
                  returncode=0, stdout=json.dumps(report), stderr="",
              ))):
            result = verifier._fresh_process(self.args)
        self.assertTrue(result["parent_child_numeric_prediction_parity_verified"])
        self.assertEqual(result["parent_reference_four_unresolved_heldout_offers"],
                         reference["rows"])
        for bad in (
            {**report, "fresh_process_pid": os.getpid()},
            {**report, "source_artifact_unchanged": False},
            {**report, "continuation_training_saved": False},
            {**report, "heldout_outcomes_not_resolved": False},
            {**report, "distinct_evaluator_workers": False},
        ):
            with (self.subTest(bad=bad),
                  patch.object(verifier, "_parent_reference", return_value=reference),
                  patch.object(verifier.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0,
                    stdout=json.dumps(bad), stderr=""))):
                with self.assertRaises(RuntimeError):
                    verifier._fresh_process(self.args)

        wrong = {**report, "four_new_unresolved_heldout_offers": [
            {**reference["rows"][0], "on_prob_a": .4},
            *reference["rows"][1:],
        ]}
        with (patch.object(verifier, "_parent_reference", return_value=reference),
              patch.object(verifier.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(wrong), stderr=""))):
            with self.assertRaisesRegex(RuntimeError, "cross-process on_prob_a"):
                verifier._fresh_process(self.args)

    def test_corrupt_child_report_timeout_and_duplicate_key_rejected(self) -> None:
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', 'not-json'):
            with (self.subTest(raw=raw),
                  patch.object(verifier, "_parent_reference", return_value={}),
                  patch.object(verifier.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=raw, stderr=""))):
                with self.assertRaises(RuntimeError):
                    verifier._fresh_process(self.args)
        with (patch.object(verifier, "_parent_reference", return_value={}),
              patch.object(verifier.subprocess, "run",
                side_effect=subprocess.TimeoutExpired(cmd="fresh", timeout=900))):
            with self.assertRaises(subprocess.TimeoutExpired):
                verifier._fresh_process(self.args)

    def test_artifact_hash_rejects_extra_file_and_symlinks(self) -> None:
        self.assertRegex(verifier._artifact_sha256(self.source), r"^[0-9a-f]{64}$")
        (self.source / "extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            verifier._artifact_sha256(self.source)
        (self.source / "extra.json").unlink()
        symlink_source = self.base / "alias"
        symlink_source.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            verifier._artifact_sha256(symlink_source)

    def test_real_v3_worker_uses_shared_train_arm_for_chosen_receipt(self) -> None:
        # This accesses a NEW synthetic TRAIN receipt only. No EVAL outcome
        # is committed, resolved, or inspected by this test.
        with verifier.V3OutcomeEvaluator(
            seed=4097, evaluator_id=verifier.EVALUATOR,
            arms=(verifier.FRESH_TRAIN_ARM,),
            training_pairs_per_family=1, heldout_pairs_per_generator=1,
        ) as evaluator:
            offer = evaluator.issue(
                partition="TRAIN", arm=verifier.TRAIN_ARM,
                family_id=verifier.V3_TRAIN_FAMILIES[0], pair_index=0, side=0,
            )
            ticket = evaluator.commit(
                offer, "A", "d" * 64, arm=verifier.TRAIN_ARM,
                model_id=verifier.PINNED_MODEL_ID,
                runtime_instance_id="v3-test-train-only",
            )
            receipt = evaluator.resolve(ticket, arm=verifier.TRAIN_ARM)
            self.assertEqual(receipt.partition, "TRAIN")
            self.assertEqual(receipt.episode_id, offer.episode_id)
            self.assertEqual(receipt.action, "A")
            self.assertTrue(0 <= receipt.measured_quality <= 1)

    def test_mock_worker_never_resolves_eval_and_saves_new_train_only(self) -> None:
        self.args._fresh_worker = True
        self.args._parent_pid = os.getppid()
        binding = SimpleNamespace(
            model_id=verifier.PINNED_MODEL_ID,
            model_revision=verifier.PINNED_REVISION,
            runtime_instance_id=self.args.runtime_instance_id,
        )
        loaded = SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None,
                                  config=SimpleNamespace(hidden_size=8)),
            binding=binding,
            manifest=SimpleNamespace(snapshot_sha256="1" * 64, files=(
                SimpleNamespace(name="model.safetensors",
                                sha256=verifier.PINNED_SAFETENSORS_SHA256),
            )),
        )

        class FakeProbe:
            def __init__(self) -> None:
                self.base_digest = "b" * 64
                self.head_digest = "h" * 64
                self.lora_digest = "l" * 64
                self.joint_digest = "j" * 64
                self.ledger = {"updates": 4, "seen_evidence_ids": tuple("1" * 64 for _ in []),
                               "seen_episode_ids": ()}

            def export_training_state(self):
                # The imported mock starts at four real historical updates.
                return dict(self.ledger)

            def score(self, ids, *, gain, partition, defer_choice=False, episode_id):
                p = .7 if gain else .5
                return SimpleNamespace(
                    episode_id=episode_id, input_ids_sha256="f" * 64,
                    base_digest=self.base_digest, head_digest=self.head_digest,
                    lora_digest=self.lora_digest, joint_digest=self.joint_digest,
                    base_prob_a=.5, base_prob_b=.5,
                    prob_a=p, prob_b=1.0-p,
                    predicted_quality_a=.6, predicted_quality_b=.4,
                )

            def bind_choice(self, decision, action):
                return SimpleNamespace(chosen_action=action)

            def observe_chosen(self, chosen, *, measured_quality, evidence_id):
                self.head_digest = "H" * 64
                self.lora_digest = "L" * 64
                self.joint_digest = "J" * 64
                self.ledger["updates"] += 1
                self.ledger["seen_evidence_ids"] += (evidence_id,)
                self.ledger["seen_episode_ids"] += ("new-train-episode",)
                return SimpleNamespace(
                    update_index=self.ledger["updates"],
                    optimizer_step_executed=True, head_changed=True, lora_changed=True,
                )

        probe = FakeProbe()
        # Keep mocked state consistent with a four-step source ledger.
        probe.ledger["seen_evidence_ids"] = tuple(str(i) * 64 for i in range(1, 5))
        probe.ledger["seen_episode_ids"] = tuple(f"old-episode-{i}" for i in range(4))
        source_record = SimpleNamespace(
            binding=SimpleNamespace(runtime_instance_id="old-original-instance"),
            head_weights_sha256="h" * 64, lora_weights_sha256="l" * 64,
            joint_weights_sha256="j" * 64, frozen_base_sha256="b" * 64,
            updates_at_export=4,
        )
        new_record = SimpleNamespace(
            head_weights_sha256="H" * 64, lora_weights_sha256="L" * 64,
            joint_weights_sha256="J" * 64, frozen_base_sha256="b" * 64,
            updates_at_export=5, artifact_directory=str(self.destination),
            manifest_sha256="m" * 64,
        )
        events: list[tuple[str, str]] = []

        class FakeEvaluator:
            next_pid = 30000

            def __init__(self, **kwargs):
                self.seed = kwargs["seed"]
                self.worker_pid = FakeEvaluator.next_pid
                FakeEvaluator.next_pid += 1

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def issue(self, *, partition, arm, family_id, pair_index, side):
                if partition == "TRAIN":
                    self_case.assertEqual(arm, verifier.TRAIN_ARM)
                else:
                    self_case.assertEqual(arm, verifier.FRESH_TRAIN_ARM)
                events.append(("issue", partition))
                episode_id = (f"eval-{pair_index}" if partition == "EVAL"
                              else "new-train-episode")
                return SimpleNamespace(
                    partition=partition, episode_id=episode_id,
                    task_id="new-train-task" if partition == "TRAIN" else "heldout-task",
                    family_id=family_id,
                )

            def commit(self, offer, action, decision_digest, **kwargs):
                self_case.assertEqual(kwargs["arm"], verifier.TRAIN_ARM)
                events.append(("commit", offer.partition))
                if offer.partition != "TRAIN":
                    raise AssertionError("EVAL commit forbidden")
                return SimpleNamespace(offer=offer, action=action,
                                       decision_digest=decision_digest)

            def resolve(self, ticket, *, arm):
                self_case.assertEqual(arm, verifier.TRAIN_ARM)
                events.append(("resolve", ticket.offer.partition))
                if ticket.offer.partition != "TRAIN":
                    raise AssertionError("EVAL resolve forbidden")
                return SimpleNamespace(
                    partition="TRAIN", episode_id=ticket.offer.episode_id,
                    task_id=ticket.offer.task_id, family_id=ticket.offer.family_id,
                    action=ticket.action, decision_digest=ticket.decision_digest,
                    runtime_instance_id=binding.runtime_instance_id,
                    model_id=binding.model_id, evaluator_id=verifier.EVALUATOR,
                    measured_quality=.91, evidence_id="c" * 64,
                )

        def fake_load(*_, **__):
            return source_record

        def fake_hash(path):
            return "s" * 64 if Path(path) == self.source else "t" * 64

        self_case = self
        with (patch.object(verifier, "_artifact_sha256", side_effect=fake_hash),
              patch.object(verifier, "load_verified_local_causal_model", return_value=loaded),
              patch.object(verifier, "prepare_explicit_lora", return_value=object()),
              patch.object(verifier, "TorchJointLoRAAppraisal", return_value=probe),
              patch.object(verifier, "load_versioned_joint_appraisal", side_effect=fake_load),
              patch.object(verifier, "save_versioned_joint_appraisal", return_value=new_record) as save,
              patch.object(verifier, "V3OutcomeEvaluator", FakeEvaluator),
              patch.object(verifier, "_encode_offer", return_value=(42,))):
            report = verifier._run_worker(self.args)
        self.assertTrue(report["heldout_outcomes_not_resolved"])
        self.assertTrue(report["continuation_training_saved"])
        self.assertEqual(report["updates_before"], 4)
        self.assertEqual(report["updates_after"], 5)
        self.assertEqual(len(report["four_new_unresolved_heldout_offers"]), 4)
        self.assertEqual(events.count(("issue", "EVAL")), 4)
        self.assertEqual(events.count(("commit", "TRAIN")), 1)
        self.assertEqual(events.count(("resolve", "TRAIN")), 1)
        self.assertFalse(any(kind in ("resolve", "commit") and part == "EVAL"
                             for kind, part in events))
        self.assertTrue(save.call_args.kwargs["allow_persistent_export"])
        self.assertEqual(save.call_args.kwargs["destination_directory"], str(self.destination))


if __name__ == "__main__":
    unittest.main()
