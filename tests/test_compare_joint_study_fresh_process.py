"""Post-COMPLETE original-study verification without accessing v3 outcomes."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import compare_joint_study_fresh_process as verify


def _sha(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _row(index: int) -> dict[str, object]:
    return {
        "episode_id": _sha(f"episode-{index}"),
        "family_id": verify.FAMILY,
        "decision_joint_sha256": _sha("joint"),
        "decision_head_sha256": _sha("head"),
        "decision_lora_sha256": _sha("lora"),
        "prediction_a": .51, "prediction_b": .49,
        "chosen_quality": 0.98,  # MUST NOT be forwarded to subprocess.
        "quality_a_posthoc": 0.99,  # MUST NOT be forwarded.
        "sampled_pre_feedback": {
            "seed": verify.SEED ^ verify.BEHAVIOR_RNG_XOR,
            "draw_index": index,
            "partition": "EVAL", "policy_id": verify.ARM,
            "episode_id": _sha(f"episode-{index}"),
            "input_ids_sha256": _sha(f"tokens-{index}"),
            "prob_a": .6, "prob_b": .4, "exploration_rate": .4,
            "behavior_prob_a": .56, "behavior_prob_b": .44,
            "random_draw": .5, "chosen_action": "A", "chosen_propensity": .56,
        },
    }


def _fixture(study: Path) -> dict[str, object]:
    seed_reports = []
    for seed in verify.SEEDS:
        source = {
            "seed": seed,
            "saved_joint_head_directory": str(study / f"seed-{seed}-joint-lora-head-v1"),
            "trained_joint_sha256": _sha("joint"),
            "trained_head_sha256": _sha("head"),
            "trained_lora_sha256": _sha("lora"),
            "saved_joint_manifest_sha256": _sha("adapter-manifest"),
            "source_local_generator_manifest": {"suite_sha256": _sha("generator")},
            "eval_arms": {"joint_on": {"rows": [_row(i) for i in range(20)]}},
        }
        seed_reports.append(source)
    (study / "seed-163-joint-lora-head-v1").mkdir()
    return {
        "artifact_run_directory": str(study),
        "evidence_class": "SOURCE_LOCAL_JOINT_LORA_SEALED_OUTCOME_V3_UNAUTHENTICATED",
        "model_id": verify.MODEL_ID, "model_revision": verify.REVISION,
        "pretrained_model_sha256": _sha("pretrained"),
        "pretrained_model_unchanged": True,
        "predeclared_protocol": {
            "seeds": list(verify.SEEDS),
            "training_pairs_per_family": 8,
            "heldout_pairs_per_generator": 4,
            "exploration_rate": .4, "appraisal_gain": 12.,
        },
        "seed_reports": seed_reports,
    }


def _write_complete(study: Path, report: dict[str, object]) -> str:
    data = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (study / "study-report.json").write_bytes(data)
    digest = sha256(data).hexdigest()
    (study / "COMPLETE.sha256").write_text(digest + "  study-report.json\n", encoding="ascii")
    return digest


class OriginalStudyParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.study = self.base / "study"
        self.study.mkdir()
        self.snapshot = self.base / "snapshot"
        self.snapshot.mkdir()
        self.out = self.base / "reproduction.json"
        self.fixture = _fixture(self.study)
        self.sha = _write_complete(self.study, self.fixture)
        self.args = verify.build_parser().parse_args([
            "--study-dir", str(self.study), "--snapshot", str(self.snapshot),
            "--output", str(self.out), "--allow-reload", "--allow-evidence-export",
        ])

    def test_complete_checksum_and_frozen_identity_are_mandatory(self) -> None:
        report, digest = verify._completed_report(self.study)
        self.assertEqual(digest, self.sha)
        self.assertEqual(report, self.fixture)
        with mock.patch.object(verify.subprocess, "run") as child:
            (self.study / "COMPLETE.sha256").unlink()
            with self.assertRaises(ValueError):
                verify._run_parent(self.args)
            child.assert_not_called()
        _write_complete(self.study, self.fixture)
        (self.study / "study-report.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify._completed_report(self.study)

    def test_original_rows_are_pre_feedback_allowlist_only(self) -> None:
        payload = verify._expected_input(self.fixture, self.study, self.sha)
        self.assertEqual(len(payload["expected_rows"]), 4)
        self.assertEqual(payload["expected_joint_sha256"], _sha("joint"))
        self.assertEqual(payload["expected_pretrained_sha256"], _sha("pretrained"))
        self.assertNotIn("chosen_quality", json.dumps(payload))
        self.assertNotIn("quality_a_posthoc", json.dumps(payload))
        self.assertEqual(payload["expected_rows"][0]["input_ids_sha256"], _sha("tokens-0"))

    def test_missing_joint_control_or_wrong_first_family_fails_closed(self) -> None:
        original = self.fixture["seed_reports"][0]["eval_arms"]["joint_on"]["rows"][0]
        original["family_id"] = "eval_ood_00_v3"
        with self.assertRaisesRegex(ValueError, "latent00"):
            verify._expected_input(self.fixture, self.study, self.sha)
        original["family_id"] = verify.FAMILY
        self.fixture["seed_reports"][0]["trained_lora_sha256"] = _sha("altered")
        with self.assertRaisesRegex(ValueError, "evaluation tensors"):
            verify._expected_input(self.fixture, self.study, self.sha)

    def _child(self, command: list[str], *, input: str, **_kw: object) -> object:
        self.assertEqual(command[:3], [verify.sys.executable, "-I", "-B"])
        ref = verify._strict_json(input)
        self.assertNotIn("chosen_quality", input)
        self.assertNotIn("quality_a_posthoc", input)
        response = {
            "fresh_process_pid": os.getpid() + 10000,
            "parent_process_pid": os.getpid(),
            "all_original_study_probabilities_and_predictions_match": True,
            "no_eval_outcomes_requested": True,
            "no_training_performed": True,
            "completed_study_report_sha256": ref["report_sha256"],
            "imported_joint_sha256": ref["expected_joint_sha256"],
            "imported_lora_sha256": ref["expected_lora_sha256"],
            "imported_head_sha256": ref["expected_head_sha256"],
            "four_original_study_decisions_reproduced": ref["expected_rows"],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

    def test_new_python_process_and_no_overwrite_evidence(self) -> None:
        with mock.patch.object(verify.subprocess, "run", side_effect=self._child) as child:
            result = verify._run_parent(self.args)
        child.assert_called_once()
        self.assertEqual(result["original_study_report_sha256"], self.sha)
        self.assertEqual(result["reproduction"]["fresh_process_pid"], os.getpid() + 10000)
        self.assertEqual(json.loads(self.out.read_text()), result)
        with mock.patch.object(verify.subprocess, "run") as again:
            with self.assertRaisesRegex(ValueError, "output"):
                verify._run_parent(self.args)
            again.assert_not_called()

    def test_child_mismatched_probability_rejected_without_evidence_file(self) -> None:
        def altered(command: list[str], *, input: str, **kwargs: object) -> object:
            raw = self._child(command, input=input, **kwargs)
            value = json.loads(raw.stdout)
            value["four_original_study_decisions_reproduced"][1]["prob_a"] = 0.33
            return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")
        with mock.patch.object(verify.subprocess, "run", side_effect=altered):
            with self.assertRaisesRegex(RuntimeError, "record"):
                verify._run_parent(self.args)
        self.assertFalse(self.out.exists())

    def test_strict_json_rejects_duplicate_keys_and_nonfinite(self) -> None:
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '[]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                verify._strict_json(raw)

    def test_numeric_tolerance_is_small_and_finite(self) -> None:
        self.assertTrue(verify._close(0.6, 0.600000001))
        self.assertFalse(verify._close(0.6, 0.61))
        self.assertFalse(verify._close(float("nan"), 0.6))
        self.assertFalse(verify._close("0.6", 0.6))

    def test_worker_issues_but_never_resolves_original_eval_offers(self) -> None:
        """Exercise the worker policy path without torch or an actual 135M model."""
        from pleasure_experiment.open_weight_stochastic import SeededActionSampler

        sampler = SeededActionSampler(seed=verify.SEED ^ verify.BEHAVIOR_RNG_XOR,
                                      exploration_rate=verify.EPSILON)
        rows = []
        for index in range(4):
            episode = _sha(f"episode-{index}")
            token_sha = _sha(f"tokens-{index}")
            trace = sampler.sample(.6, .4, input_ids_sha256=token_sha,
                                   partition="EVAL", episode_id=episode,
                                   policy_id=verify.ARM)
            rows.append({
                "episode_id": episode, "input_ids_sha256": token_sha,
                "joint_sha256": _sha("joint"), "head_sha256": _sha("head"),
                "lora_sha256": _sha("lora"),
                "prob_a": trace.prob_a, "prob_b": trace.prob_b,
                "behavior_prob_a": trace.behavior_prob_a,
                "behavior_prob_b": trace.behavior_prob_b,
                "prediction_a": .51, "prediction_b": .49,
                "random_draw": trace.random_draw,
                "chosen_action": trace.chosen_action,
                "chosen_propensity": trace.chosen_propensity,
            })
        payload = verify._expected_input(self.fixture, self.study, self.sha)
        payload["expected_rows"] = rows

        class FakeModel:
            config = SimpleNamespace(hidden_size=32)

            def eval(self) -> None:
                return None

            def named_parameters(self):
                return iter(())

            def named_buffers(self):
                return iter(())

        loaded = SimpleNamespace(
            model=FakeModel(),
            manifest=SimpleNamespace(files=(SimpleNamespace(
                name="model.safetensors", sha256=verify.WEIGHTS_SHA256,
            ),)),
            tokenizer=SimpleNamespace(encode=lambda *_args, **_kw: (2, 3)),
            action_token_ids=(10, 11), action_suffixes=(" A", " B"),
            binding=SimpleNamespace(runtime_instance_id="fresh"),
        )
        artifact = SimpleNamespace(
            binding=SimpleNamespace(runtime_instance_id="original"),
            frozen_base_sha256=_sha("peft-base"),
            manifest_sha256=_sha("adapter-manifest"),
        )

        class FakeProbe:
            updates = 24
            base_digest = _sha("peft-base")
            head_digest = _sha("head")
            lora_digest = _sha("lora")
            joint_digest = _sha("joint")

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def score(self, ids, *, gain, partition, episode_id):
                self.outer.assertEqual(partition, "EVAL")
                self.outer.assertEqual(gain, verify.GAIN)
                index = next(i for i in range(4)
                             if episode_id == _sha(f"episode-{i}"))
                return SimpleNamespace(
                    input_ids_sha256=_sha(f"tokens-{index}"),
                    joint_digest=_sha("joint"), head_digest=_sha("head"),
                    lora_digest=_sha("lora"), prob_a=.6, prob_b=.4,
                    predicted_quality_a=.51, predicted_quality_b=.49,
                )

        FakeProbe.outer = self
        seen = []

        class FakeEvaluator:
            worker_pid = 43212
            manifest = SimpleNamespace(
                primary_within_families=verify.V3_WITHIN_FAMILIES,
                gap_filtering=False,
                suite_sha256=_sha("generator"),
            )

            def __init__(self, **kwargs) -> None:
                self_args = kwargs
                seen.append(("init", self_args))

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def issue(self, **kwargs):
                seen.append(("issue", kwargs))
                return SimpleNamespace(episode_id=_sha(f"episode-{kwargs['pair_index']}"))

            def commit(self, **_kwargs):
                raise AssertionError("EVAL commit must not occur in parity verifier")

            def resolve(self, **_kwargs):
                raise AssertionError("EVAL resolve must not occur in parity verifier")

            def posthoc_evaluate(self, **_kwargs):
                raise AssertionError("posthoc payoff query forbidden")

        imports = lambda name: (SimpleNamespace(set_num_threads=lambda *_: None)
                                if name == "torch" else None)
        with (mock.patch.object(verify.importlib, "import_module", side_effect=imports),
              mock.patch.object(verify, "load_verified_local_causal_model", return_value=loaded),
              mock.patch.object(verify, "_named_digest", return_value=_sha("pretrained")),
              mock.patch.object(verify, "prepare_explicit_lora", return_value=object()),
              mock.patch.object(verify, "TorchJointLoRAAppraisal", FakeProbe),
              mock.patch.object(verify, "load_versioned_joint_appraisal", return_value=artifact),
              mock.patch.object(verify, "V3OutcomeEvaluator", FakeEvaluator),
              mock.patch.object(verify, "public_prompt", return_value="stub prompt"),
              mock.patch.object(verify, "validate_single_token_actions", return_value=(10, 11))):
            result = verify._score_from_fresh_model("/unused", payload, 1024**3)
        self.assertTrue(result["no_eval_outcomes_requested"])
        self.assertEqual(len(result["four_original_study_decisions_reproduced"]), 4)
        self.assertEqual(len([name for name, _ in seen if name == "issue"]), 4)
        self.assertFalse(any(name in ("commit", "resolve") for name, _ in seen))


if __name__ == "__main__":
    unittest.main()
