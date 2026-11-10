"""Frozen-model feature sensitivity: pure and no-torch regression tests."""
from __future__ import annotations

from math import isclose, isfinite
from types import SimpleNamespace
from unittest import TestCase, mock

from scripts import probe_joint_action_feature_sensitivity as probe


class FeatureProbeTests(TestCase):
    def test_cases_are_train_like_and_payoff_free(self):
        cs = {case.case_id: case for case in probe.cases()}
        self.assertEqual(len(cs), 10)
        self.assertEqual(cs["baseline"].features_a, cs["baseline_repeat"].features_a)
        self.assertEqual(cs["baseline"].features_b, cs["baseline_repeat"].features_b)
        self.assertEqual(cs["baseline"].features_a, cs["swap_options"].features_b)
        self.assertEqual(cs["baseline"].features_b, cs["swap_options"].features_a)
        for case in cs.values():
            offer = probe._offer(case)
            self.assertEqual(offer.partition, "TRAIN")
            from scripts.run_open_weight_local_trial import public_prompt
            prompt = public_prompt(offer)
            self.assertNotIn("score:", prompt.lower())
            self.assertNotIn("payoff:", prompt.lower())

    def test_vector_metrics_and_expected_symmetry(self):
        self.assertAlmostEqual(probe._cosine((1., 0.), (0., 1.)), 0.)
        self.assertAlmostEqual(probe._cosine((1., 0.), (-1., 0.)), -1.)
        self.assertAlmostEqual(probe._relative_delta((3., 4.), (0., 0.)), 1.)
        self.assertEqual(probe._relative_delta((3., 4.), (3., 4.)), 0.)
        self.assertTrue(isfinite(probe._initial_head_gap((.6, .8))))
        self.assertTrue(isclose(probe._initial_head_gap((.6, .8)),
                                -probe._initial_head_gap((-.6, -.8)),
                                abs_tol=1e-14))

    def test_zero_or_nonfinite_vectors_rejected(self):
        for a, b in [((), ()), ((0., 0.), (1., 0.)),
                     ((1., float("nan")), (1., 1.)), ((1.,), (1., 2.))]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                probe._cosine(a, b)
        with self.assertRaises(ValueError):
            probe._relative_delta((0., 0.), (1., 1.))
        with self.assertRaises(ValueError):
            probe._offer(probe.Case("invalid", (float("nan"), .4), (.6, .3)))

    def test_read_only_probe_guards_identity_and_determinism(self):
        class Tokenizer:
            def encode(self, s, *, add_special_tokens):
                return [ord(c) % 100 for c in s]

        class Model:
            training = True
            config = SimpleNamespace(use_cache=True)
            def parameters(self):
                return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])
            def named_parameters(self):
                return iter(())
            def named_buffers(self):
                return iter(())
            def eval(self):
                self.training = False
            def train(self, value=True):
                self.training = value

        model = Model()
        loaded = SimpleNamespace(
            model=model, tokenizer=Tokenizer(), action_token_ids=(7, 8),
            action_suffixes=(" A", " B"),
            manifest=SimpleNamespace(files=[SimpleNamespace(
                name="model.safetensors", sha256=probe.SAFETENSORS_SHA)]),
        )
        def fake_feature(_model, _torch, ids, tokens):
            x = sum(ids) % 19
            delta = (1.0, x * 0.001)
            norm = (delta[0]**2 + delta[1]**2)**.5
            return delta, tuple(v/norm for v in delta), (.6, .4)

        with (mock.patch.object(probe, "_named_digest", return_value="same"),
              mock.patch.object(probe, "validate_single_token_actions",
                                return_value=(7, 8)),
              mock.patch.object(probe, "_model_feature", side_effect=fake_feature)):
            report = probe.run_probe(loaded, torch=object())
            self.assertEqual(report["evidence_class"], probe.EVIDENCE_CLASS)
            self.assertEqual(len(report["probes"]), 10)
            self.assertEqual(report["probes"]["baseline_repeat"]
                             ["relative_raw_delta_shift"], 0.0)
            self.assertTrue(model.training)
            self.assertTrue(model.config.use_cache)
            loaded.manifest.files[0].sha256 = "wrong"
            with self.assertRaisesRegex(ValueError, "pinned"):
                probe.run_probe(loaded, torch=object())

    def test_checksum_mutation_and_repeat_mismatch_fail_closed(self):
        class Tokenizer:
            def encode(self, prompt, *, add_special_tokens):
                return [ord(c) % 100 for c in prompt]

        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            training=False,
            parameters=lambda: iter([SimpleNamespace(
                device=SimpleNamespace(type="cpu"))]),
            named_parameters=lambda: iter(()),
            named_buffers=lambda: iter(()),
        )
        model.eval = lambda: None
        model.train = lambda x: None
        loaded = SimpleNamespace(
            model=model, tokenizer=Tokenizer(),
            action_token_ids=(7, 8), action_suffixes=(" A", " B"),
            manifest=SimpleNamespace(files=[SimpleNamespace(
                name="model.safetensors", sha256=probe.SAFETENSORS_SHA)]),
        )
        with (mock.patch.object(probe, "_named_digest",
                                side_effect=["sha1", "sha2"]),
              mock.patch.object(probe, "validate_single_token_actions",
                                return_value=(7, 8)),
              mock.patch.object(probe, "_model_feature",
                                return_value=((1., 0.), (1., 0.), (.6, .4)))):
            with self.assertRaisesRegex(RuntimeError, "modified"):
                probe.run_probe(loaded, torch=object())
