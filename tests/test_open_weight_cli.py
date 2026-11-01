"""Entry-point and fail-closed consent checks without downloading or loading models."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from unittest import TestCase

from scripts.run_open_weight_local_trial import build_parser, public_prompt, run
from pleasure_experiment.outcome_arena import OutcomeArena


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_open_weight_local_trial.py"


class LocalOpenWeightCLIContractTests(TestCase):
    def test_help_works_as_module_and_direct_script(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for args in (
            ("-m", "scripts.run_open_weight_local_trial", "--help"),
            (str(SCRIPT), "--help"),
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    (sys.executable, *args), cwd=ROOT, env=environment,
                    capture_output=True, text=True, check=False, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--allow-in-place-training", result.stdout)

    def test_consent_refusal_precedes_any_snapshot_access(self):
        args = build_parser().parse_args((
            "--snapshot", "/does-not-exist/never-read",
            "--model-id", "public-test",
            "--revision", "0" * 40,
        ))
        with self.assertRaisesRegex(PermissionError, "explicit|allow-in-place-training"):
            run(args)

    def test_revision_must_be_pinned_before_file_load(self):
        args = build_parser().parse_args((
            "--snapshot", "/does-not-exist/never-read",
            "--model-id", "public-test",
            "--revision", "main",
            "--allow-in-place-training",
        ))
        with self.assertRaisesRegex(ValueError, "immutable 40-hex"):
            run(args)

    def test_persistent_export_requires_separate_explicit_consent(self):
        base = (
            "--snapshot", "/does-not-exist/never-read",
            "--model-id", "public-test",
            "--revision", "0" * 40,
            "--allow-in-place-training",
        )
        for extras in (
            ("--save-adapter-dir", "/tmp/persist-does-not-exist"),
            ("--allow-persistent-export",),
        ):
            with self.subTest(extras=extras):
                with self.assertRaisesRegex(PermissionError, "both"):
                    run(build_parser().parse_args(base + extras))

    def test_persistent_export_outside_repository_only(self):
        args = build_parser().parse_args((
            "--snapshot", "/does-not-exist/never-read",
            "--model-id", "public-test",
            "--revision", "0" * 40,
            "--allow-in-place-training", "--allow-persistent-export",
            "--save-adapter-dir", str(ROOT / "accidental-adapter"),
        ))
        with self.assertRaisesRegex(ValueError, "outside the source repository"):
            run(args)

    def test_prompt_contains_only_public_features_and_fixed_action_suffix(self):
        offer = OutcomeArena(seed=11, evaluator_id="source-local-test-oracle").issue(
            partition="TRAIN", family_id="demo", pair_index=0, side=0
        )
        prompt = public_prompt(offer)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertIn("Option A features:", prompt)
        self.assertIn("Option B features:", prompt)
        self.assertNotIn(offer.nonce, prompt)
        self.assertNotIn("measured_quality", prompt)
