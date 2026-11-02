"""Safety-gate tests for optional offline benchmark and appraisal commands."""

from __future__ import annotations

from unittest import TestCase

from scripts.run_open_weight_appraisal_pilot import (
    build_parser as appraisal_parser,
    run as appraisal_run,
)
from scripts.run_open_weight_multiseed_benchmark import (
    build_parser as benchmark_parser,
    run as benchmark_run,
)


class ExtendedOpenWeightCLITests(TestCase):
    def _args(self, builder, *extra):
        return builder().parse_args([
            "--snapshot", "/tmp/absent-model",
            "--model-id", "test-offline-model",
            "--revision", "a" * 40,
            *extra,
        ])

    def test_benchmark_rejects_missing_consent_before_loader(self):
        with self.assertRaisesRegex(PermissionError, "allow-in-place-training"):
            benchmark_run(self._args(benchmark_parser))

    def test_benchmark_rejects_duplicate_seeds_before_loader(self):
        with self.assertRaisesRegex(ValueError, "unique uint32"):
            benchmark_run(self._args(
                benchmark_parser, "--allow-in-place-training", "--seeds", "1", "1",
            ))

    def test_appraisal_rejects_missing_consent_before_loader(self):
        with self.assertRaisesRegex(PermissionError, "allow-appraisal-training"):
            appraisal_run(self._args(appraisal_parser))

    def test_appraisal_rejects_invalid_counts_before_loader(self):
        with self.assertRaisesRegex(ValueError, "heldout-pairs"):
            appraisal_run(self._args(
                appraisal_parser, "--allow-appraisal-training", "--heldout-pairs", "0",
            ))
