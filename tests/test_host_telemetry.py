"""Host-only Tattler evidence tests; none assert model behavior or valence."""

import json
import math
import unittest
from hashlib import sha256

from pleasure_experiment.host_telemetry import (
    MAX_JSON_BYTES,
    MAX_MEMORY_KB,
    MAX_SAMPLES,
    parse_tattler_system_samples,
)


PARAMS = {
    "trial_id": "trial-001",
    "opaque_host_sha256": "a" * 64,
    "source_revision": "tattler-0.2.0-0004",
    "trial_start_utc": "2026-11-01T12:00:00Z",
    "trial_end_utc": "2026-11-01T12:00:20Z",
    "retrieved_at_utc": "2026-11-01T12:01:00Z",
}


def sample(at="2026-11-01T12:00:05Z", **changes):
    return {
        "schema_version": 1,
        "observed_at": at,
        "platform": "linux",
        "cpu_percent": 0.0,
        "io_wait_percent": 0.0,
        "mem_total_kb": 8000,
        "mem_available_kb": 4000,
        **changes,
    }


def parse(records, **params):
    return parse_tattler_system_samples(json.dumps(records), **(PARAMS | params))


class HostTelemetryTests(unittest.TestCase):
    def test_only_aggregate_allowlisted_host_metrics_survive(self):
        rows = [
            sample(
                "2026-11-01T12:00:05.123456789Z",
                cpu_percent=20.0,
                top_processes=[{"pid": 42, "name": "PRIVATE-PROCESS"}],
                network_endpoints=["PRIVATE-IP:1234"],
                host="PRIVATE-HOSTNAME",
            ),
            sample("2026-11-01T12:00:15Z", cpu_percent=40.0, mem_available_kb=2000),
        ]
        raw = json.dumps(rows).encode("utf-8")
        report = parse_tattler_system_samples(raw, **PARAMS)
        self.assertEqual(report["schema"], "PLEASURE_HOST_TELEMETRY_V1")
        self.assertEqual(report["evidence_class"], "HOST_TELEMETRY")
        self.assertEqual(report["qualification"], "SAMPLED_HOST_ONLY")
        self.assertEqual(report["model_internal_effect"], "UNKNOWN")
        self.assertEqual(report["phenomenology"], "UNRESOLVED")
        self.assertEqual(report["overlapping_sample_count"], 2)
        self.assertEqual(report["metrics"]["cpu_percent"]["mean_reported"], 30.0)
        self.assertEqual(report["metrics"]["mem_available_kb"]["min_reported"], 2000.0)
        self.assertEqual(report["max_observed_gap_seconds"], 9.876544)
        self.assertEqual(report["source_payload_sha256"], sha256(raw).hexdigest())
        rendered = json.dumps(report)
        for private in ("PRIVATE-PROCESS", "PRIVATE-IP", "PRIVATE-HOSTNAME", "pid", "top_processes"):
            self.assertNotIn(private, rendered)

    def test_unavailable_missing_and_reported_zero_remain_distinct(self):
        rows = [
            sample("2026-11-01T12:00:05Z", platform="windows", unavailable_metrics=["io_wait_percent"]),
            sample("2026-11-01T12:00:10Z", platform="windows", unavailable_metrics=["io_wait_percent"]),
        ]
        del rows[1]["mem_available_kb"]
        report = parse(rows)
        self.assertEqual(report["metrics"]["cpu_percent"]["mean_reported"], 0.0)
        self.assertEqual(report["metrics"]["cpu_percent"]["reported_samples"], 2)
        self.assertEqual(report["metrics"]["io_wait_percent"]["unavailable_samples"], 2)
        self.assertIsNone(report["metrics"]["io_wait_percent"]["mean_reported"])
        self.assertEqual(report["metrics"]["mem_available_kb"]["missing_samples"], 1)
        self.assertEqual(report["metrics"]["mem_available_kb"]["reported_samples"], 1)

    def test_no_overlapping_samples_is_unknown_not_zero_activity(self):
        report = parse([sample("2026-11-01T13:00:00Z")])
        self.assertEqual(report["qualification"], "UNKNOWN")
        self.assertEqual(report["source_sample_count"], 1)
        self.assertEqual(report["overlapping_sample_count"], 0)
        self.assertIsNone(report["first_sample_utc"])
        self.assertIsNone(report["max_observed_gap_seconds"])
        self.assertIsNone(report["metrics"]["cpu_percent"]["mean_reported"])
        empty = parse([])
        self.assertEqual(empty["qualification"], "UNKNOWN")

    def test_window_inclusive_boundaries_and_sorted_samples(self):
        report = parse([
            sample("2026-11-01T12:00:20Z"),
            sample("2026-11-01T12:00:00Z"),
        ])
        self.assertEqual(report["first_sample_utc"], "2026-11-01T12:00:00Z")
        self.assertEqual(report["last_sample_utc"], "2026-11-01T12:00:20Z")
        self.assertEqual(report["max_observed_gap_seconds"], 20.0)

    def test_invalid_schema_and_json_shapes_rejected(self):
        for value in ({}, ["not a sample"], [sample(schema_version=2)], [sample(schema_version=True)]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)
        with self.assertRaisesRegex(ValueError, "invalid Tattler system JSON"):
            parse_tattler_system_samples('[{"schema_version":1,"schema_version":1}]', **PARAMS)
        with self.assertRaisesRegex(ValueError, "invalid Tattler system JSON"):
            parse_tattler_system_samples("[NaN]", **PARAMS)
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            parse_tattler_system_samples(b"\xff", **PARAMS)

    def test_invalid_or_missing_metric_numbers_rejected(self):
        for bad in (math.inf, math.nan, True, -1, 101, "0"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                parse([sample(cpu_percent=bad)])
        with self.assertRaises(ValueError):
            parse([sample(mem_available_kb=-1)])
        with self.assertRaises(ValueError):
            parse([sample(mem_available_kb=9000)])
        with self.assertRaises(ValueError):
            parse([sample(mem_total_kb=2000.5)])
        with self.assertRaises(ValueError):
            parse([sample("2026-11-01T13:00:00Z", cpu_percent="invalid")])
        with self.assertRaises(ValueError):
            parse([sample(unavailable_metrics="io_wait_percent")])

    def test_enormous_memory_reports_cannot_overflow_aggregate(self):
        with self.assertRaisesRegex(ValueError, "outside reported range"):
            parse([sample(mem_total_kb=10**308, mem_available_kb=0)])
        with self.assertRaisesRegex(ValueError, "outside reported range"):
            parse([sample(mem_total_kb=MAX_MEMORY_KB + 1, mem_available_kb=0)])
        rows = [sample(mem_total_kb=MAX_MEMORY_KB, mem_available_kb=MAX_MEMORY_KB)] * MAX_SAMPLES
        report = parse(rows)
        self.assertEqual(report["metrics"]["mem_total_kb"]["mean_reported"], float(MAX_MEMORY_KB))
        json.dumps(report, allow_nan=False)  # Finite output, including the aggregate.

    def test_limits_and_timestamp_rules_are_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            parse_tattler_system_samples(b" " * (MAX_JSON_BYTES + 1), **PARAMS)
        with self.assertRaisesRegex(ValueError, "bounded array"):
            parse([sample()] * (MAX_SAMPLES + 1))
        with self.assertRaises(ValueError):
            parse([sample("2026-11-01T12:00:05")])
        with self.assertRaises(ValueError):
            parse([sample()], trial_start_utc="2026-11-01T13:00:00Z")
        with self.assertRaises(ValueError):
            parse([sample()], trial_end_utc="2026-11-01T13:00:01Z")
        with self.assertRaises(ValueError):
            parse([sample()], opaque_host_sha256="Lappy")
        with self.assertRaises(ValueError):
            parse([sample()], trial_id="trial\nsecret")


if __name__ == "__main__":
    unittest.main()
