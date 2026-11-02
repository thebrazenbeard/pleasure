"""Bounded frozen-model hidden-feature reuse without EVAL/feedback leakage."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch unavailable")
class FrozenFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        import torch
        from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 3)
                self.out = torch.nn.Linear(3, 16)
                self.calls = 0
                with torch.no_grad():
                    self.embedding.weight.zero_()
                    self.embedding.weight[3] = torch.tensor([1.0, 0.0, 0.5])
                    self.embedding.weight[4] = torch.tensor([0.0, 1.0, -0.5])
                    self.out.weight.zero_()
                    self.out.bias.zero_()
                    self.out.bias[3] = 0.2

            def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
                self.calls += 1
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.out(hidden),
                    hidden_states=(hidden,) if output_hidden_states else None,
                )

        self.torch = torch
        self.model = TinyLM()
        self.cache = FrozenActionFeatureCache(
            self.model, action_token_ids=(3, 4),
            checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
            max_entries=2,
        )

    def test_cache_reuses_three_forwards_across_five_arm_scores(self):
        with self.cache.verified_epoch() as epoch:
            features = [epoch.get_features((1, 2), partition="EVAL", episode_id="heldout-1")
                        for _ in range(5)]
            self.assertEqual(self.model.calls, 3)
            self.assertEqual(self.cache.hits, 4)
            self.assertEqual(self.cache.misses, 1)
            self.assertEqual(tuple(features[0].logits.shape), (2,))
            self.assertEqual(tuple(features[0].feature_a.shape), (3,))
            self.assertTrue(self.torch.equal(features[0].feature_a,
                                             self.torch.tensor([1.0, 0.0, 0.5])))
            features[0].feature_a.zero_()
            self.assertTrue(self.torch.equal(features[1].feature_a,
                                             self.torch.tensor([1.0, 0.0, 0.5])))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            epoch.get_features((1, 2), partition="EVAL", episode_id="heldout-1")

    def test_epoch_hashes_model_only_at_bounded_entry_and_exit(self):
        with mock.patch.object(self.cache, "_current_model_sha",
                               wraps=self.cache._current_model_sha) as digest:
            with self.cache.verified_epoch() as epoch:
                for _ in range(20):
                    epoch.get_features((1, 2), partition="EVAL", episode_id="heldout-1")
            self.assertEqual(digest.call_count, 2)
        self.assertEqual(self.model.calls, 3)

    def test_direct_get_hashes_and_prevents_stale_state(self):
        first = self.cache.get_features((1,), partition="EVAL", episode_id="heldout")
        before = self.cache.frozen_model_sha256
        self.cache.get_features((1,), partition="EVAL", episode_id="heldout")
        self.assertEqual(self.model.calls, 3)
        with self.torch.no_grad():
            self.model.out.bias[3] += 0.2
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.cache.get_features((1,), partition="EVAL", episode_id="heldout")
        self.assertEqual(self.cache.cache_size, 0)
        self.assertEqual(self.cache.frozen_model_sha256, before)
        with self.torch.no_grad():
            self.model.out.bias[3] -= 0.2
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            self.cache.get_features((1,), partition="EVAL", episode_id="heldout")
        self.assertEqual(first.frozen_model_sha256, before)

    def test_train_and_eval_partitions_never_share_feature_keys_or_update_authority(self):
        heldout = self.cache.get_features((1,), partition="EVAL", episode_id="shared")
        with self.assertRaisesRegex(PermissionError, "TRAIN"):
            self.cache.chosen_training_feature(
                heldout, chosen_action="A", expected_episode_id="shared",
            )
        with self.assertRaisesRegex(ValueError, "not issued"):
            self.cache.chosen_training_feature(
                replace(heldout, partition="TRAIN"),
                chosen_action="A", expected_episode_id="shared",
            )
        train = self.cache.get_features((1,), partition="TRAIN", episode_id="shared")
        self.assertEqual(self.model.calls, 6)  # distinct partition = fresh forwards
        # Even with a genuine TRAIN cache hit, relabeling an EVAL-issued token
        # cannot authorize an update (distinct opaque issuance identities).
        with self.assertRaisesRegex(ValueError, "not issued"):
            self.cache.chosen_training_feature(
                replace(heldout, partition="TRAIN"),
                chosen_action="A", expected_episode_id="shared",
            )
        chosen = self.cache.chosen_training_feature(
            train, chosen_action="B", expected_episode_id="shared",
        )
        self.assertTrue(self.torch.equal(chosen, train.feature_b))
        chosen.zero_()
        self.assertFalse(self.torch.equal(chosen, train.feature_b))
        train.feature_b.zero_()
        self.assertTrue(self.torch.equal(
            self.cache.chosen_training_feature(
                train, chosen_action="B", expected_episode_id="shared",
            ), self.torch.tensor([0.0, 1.0, -0.5]),
        ))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.cache.chosen_training_feature(
                replace(train, tokenizer_sha256="c" * 64),
                chosen_action="B", expected_episode_id="shared",
            )
        with self.assertRaises(ValueError):
            self.cache.chosen_training_feature(
                train, chosen_action="B", expected_episode_id="wrong",
            )

    def test_verify_issued_checks_exact_tensor_bytes_and_all_identity_fields(self):
        eval_bundle = self.cache.get_features(
            (1, 2), partition="EVAL", episode_id="heldout-1",
        )
        trusted = self.cache.verify_issued(
            eval_bundle, input_ids=(1, 2), partition="EVAL", episode_id="heldout-1",
        )
        self.assertIsNot(trusted.feature_a, eval_bundle.feature_a)
        self.assertTrue(self.torch.equal(trusted.feature_a, eval_bundle.feature_a))
        with self.assertRaisesRegex(ValueError, "not issued"):
            self.cache.verify_issued(
                eval_bundle, input_ids=(1, 2), partition="TRAIN", episode_id="heldout-1",
            )
        train_bundle = self.cache.get_features(
            (1, 2), partition="TRAIN", episode_id="heldout-1",
        )
        with self.assertRaisesRegex(ValueError, "issuance"):
            self.cache.verify_issued(
                replace(eval_bundle, partition="TRAIN"),
                input_ids=(1, 2), partition="TRAIN", episode_id="heldout-1",
            )
        with self.assertRaisesRegex(ValueError, "issuance"):
            self.cache.verify_issued(
                replace(train_bundle, checkpoint_sha256="f" * 64),
                input_ids=(1, 2), partition="TRAIN", episode_id="heldout-1",
            )
        tampered = replace(train_bundle, feature_b=train_bundle.feature_b.clone())
        tampered.feature_b[0] += 0.02
        with self.assertRaisesRegex(ValueError, "feature_b was modified"):
            self.cache.verify_issued(
                tampered, input_ids=(1, 2), partition="TRAIN", episode_id="heldout-1",
            )
        with self.assertRaisesRegex(ValueError, "issuance"):
            self.cache.verify_issued(
                replace(train_bundle, issuance_id="0" * 48),
                input_ids=(1, 2), partition="TRAIN", episode_id="heldout-1",
            )
        with self.assertRaisesRegex(ValueError, "not issued"):
            self.cache.verify_issued(
                train_bundle, input_ids=(1,), partition="TRAIN", episode_id="heldout-1",
            )

    def test_verify_issued_inside_epoch_amortizes_model_hashes_and_fails_stale(self):
        with mock.patch.object(self.cache, "_current_model_sha",
                               wraps=self.cache._current_model_sha) as digest:
            with self.cache.verified_epoch() as epoch:
                bundle = epoch.get_features(
                    (1,), partition="EVAL", episode_id="heldout",
                )
                for _ in range(15):
                    self.cache.verify_issued(
                        bundle, input_ids=(1,), partition="EVAL", episode_id="heldout",
                    )
            self.assertEqual(digest.call_count, 2)
        with self.torch.no_grad():
            self.model.out.bias[3] += 0.3
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.cache.verify_issued(
                bundle, input_ids=(1,), partition="EVAL", episode_id="heldout",
            )

    def test_limit_invalidate_and_mode_restoration(self):
        self.model.train()
        self.cache.get_features((1,), partition="EVAL", episode_id="episode-a")
        self.cache.get_features((2,), partition="EVAL", episode_id="episode-b")
        self.cache.get_features((1, 2), partition="EVAL", episode_id="episode-c")
        self.assertEqual(self.cache.cache_size, 2)
        self.assertTrue(self.model.training)
        self.assertTrue(all(module.training for module in self.model.modules()))
        self.cache.invalidate()
        self.assertEqual(self.cache.cache_size, 0)
        self.cache.get_features((1,), partition="EVAL", episode_id="episode-a")
        self.assertEqual(self.model.calls, 12)

    def test_invalid_queries_and_nested_epoch_fail_closed(self):
        with self.assertRaises(ValueError):
            self.cache.get_features((1,), partition="TEST", episode_id="x")
        with self.assertRaises(ValueError):
            self.cache.get_features((1,), partition="EVAL", episode_id="")
        with self.assertRaises(ValueError):
            self.cache.get_features([1], partition="EVAL", episode_id="x")
        with self.cache.verified_epoch() as epoch:
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with self.cache.verified_epoch():
                    pass
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                self.cache.get_features((1,), partition="EVAL", episode_id="x")
            with self.assertRaisesRegex(RuntimeError, "epoch"):
                self.cache.invalidate()
            epoch.get_features((1,), partition="EVAL", episode_id="x")


class DependencyOptionalTests(unittest.TestCase):
    def test_source_import_does_not_require_torch(self):
        from pleasure_experiment.open_weight_feature_cache import EVIDENCE_CLASS
        self.assertIn("CACHE", EVIDENCE_CLASS)
