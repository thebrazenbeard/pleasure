"""Strict local-only model snapshot and opt-in LoRA interface tests.

No network, downloads, or actual checkpoint deserialization are performed.
The fake transformers/PEFT modules test arguments and guard boundaries only;
real pretrained-model loading requires separate runtime qualification.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json

from pleasure_experiment.open_weight_local_model import (
    LoadedLocalCausalModel,
    inspect_local_snapshot,
    load_verified_local_causal_model,
    prepare_explicit_lora,
    validate_single_token_actions,
)


class LocalSnapshotTests(TestCase):
    def setUp(self):
        self.scratch = TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name) / "snapshot"
        self.directory.mkdir()
        (self.directory / "config.json").write_text(
            '{"model_type":"llama","architectures":["LlamaForCausalLM"]}',
            encoding="utf-8",
        )
        (self.directory / "tokenizer_config.json").write_text(
            '{"tokenizer_class":"PreTrainedTokenizerFast"}', encoding="utf-8"
        )
        (self.directory / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
        (self.directory / "model.safetensors").write_bytes(b"fixture-not-real-safetensors")

    def test_actual_file_hashes_bind_weights_tokenizer_and_config(self):
        a = inspect_local_snapshot(self.directory)
        self.assertEqual(a.total_bytes, sum(p.stat().st_size for p in self.directory.iterdir()))
        self.assertEqual(len(a.files), 4)
        self.assertEqual(len(a.checkpoint_sha256), 64)
        self.assertEqual(len(a.tokenizer_sha256), 64)
        self.assertEqual(a.evidence_class, "LOCAL_FILES_SHA256_UNAUTHENTICATED")
        (self.directory / "tokenizer.json").write_text('{"version":"2.0"}', encoding="utf-8")
        b = inspect_local_snapshot(self.directory)
        self.assertEqual(a.checkpoint_sha256, b.checkpoint_sha256)
        self.assertNotEqual(a.tokenizer_sha256, b.tokenizer_sha256)
        self.assertNotEqual(a.snapshot_sha256, b.snapshot_sha256)
        (self.directory / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
        c = inspect_local_snapshot(self.directory)
        self.assertNotEqual(b.checkpoint_sha256, c.checkpoint_sha256)

    def test_requires_preexisting_absolute_snapshot(self):
        with self.assertRaises(ValueError):
            inspect_local_snapshot("relative/path")
        with self.assertRaises(ValueError):
            inspect_local_snapshot(self.directory / "absent")
        with self.assertRaises(ValueError):
            inspect_local_snapshot(self.directory, max_total_bytes=1)
        with self.assertRaises(ValueError):
            inspect_local_snapshot(self.directory, max_total_bytes=True)
        traversal = self.directory / ".." / self.directory.name
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            inspect_local_snapshot(traversal)

    def test_rejects_junction_like_path_components(self):
        # Windows can re-spell normal paths when resolving them, so the
        # security gate must detect redirecting entries directly. On POSIX
        # the synthetic predicate exercises the same cross-platform branch.
        with patch.object(Path, "is_junction", return_value=True, create=True):
            with self.assertRaisesRegex(ValueError, "symlink or junction"):
                inspect_local_snapshot(self.directory)

    def test_rejects_unsafe_files_nested_directories_and_symlinks(self):
        unsafe = self.directory / "pytorch_model.bin"
        unsafe.write_bytes(b"pickle")
        with self.assertRaisesRegex(ValueError, "unsupported snapshot entry"):
            inspect_local_snapshot(self.directory)
        unsafe.unlink()
        nested = self.directory / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(ValueError, "unsupported snapshot entry"):
            inspect_local_snapshot(self.directory)
        nested.rmdir()
        link = self.directory / "checkpoint.py"
        try:
            link.symlink_to(self.directory / "config.json")
        except (OSError, NotImplementedError):
            self.skipTest("platform does not support unprivileged symlinks")
        with self.assertRaisesRegex(ValueError, "unsupported snapshot entry"):
            inspect_local_snapshot(self.directory)

    def test_missing_manifest_inputs_rejected(self):
        (self.directory / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "safetensors checkpoint"):
            inspect_local_snapshot(self.directory)
        (self.directory / "model.safetensors").write_bytes(b"fixture")
        (self.directory / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer vocabulary"):
            inspect_local_snapshot(self.directory)

    def test_hugging_face_download_metadata_allowed_but_never_loaded_or_hashed(self):
        original = inspect_local_snapshot(self.directory)
        metadata = self.directory / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / "ignored.metadata").write_text("etag only", encoding="utf-8")
        with_sidecar = inspect_local_snapshot(self.directory)
        self.assertEqual(original, with_sidecar)
        # A sidecar symlink is never admitted, even though its content would
        # otherwise be ignored for manifest hashing.
        link = metadata / "linked.metadata"
        try:
            link.symlink_to(self.directory / "config.json")
        except (OSError, NotImplementedError):
            self.skipTest("platform does not support unprivileged symlinks")
        with self.assertRaisesRegex(ValueError, "unsafe Hugging Face metadata"):
            inspect_local_snapshot(self.directory)

    def test_remote_code_autoload_request_fails_closed(self):
        (self.directory / "config.json").write_text(
            '{"model_type":"llama","auto_map":{"AutoModelForCausalLM":"evil.py"}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "dynamic model"):
            inspect_local_snapshot(self.directory)

    def test_sharded_checkpoint_index_must_bind_all_shards(self):
        (self.directory / "model.safetensors").unlink()
        (self.directory / "model-00001-of-00002.safetensors").write_bytes(b"one")
        (self.directory / "model-00002-of-00002.safetensors").write_bytes(b"two")
        with self.assertRaisesRegex(ValueError, "sharded checkpoints require"):
            inspect_local_snapshot(self.directory)
        index = self.directory / "model.safetensors.index.json"
        index.write_text(json.dumps({
            "weight_map": {"a": "model-00001-of-00002.safetensors"}
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            inspect_local_snapshot(self.directory)
        index.write_text(json.dumps({"weight_map": {"a": []}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported weight mapping"):
            inspect_local_snapshot(self.directory)
        index.write_text(json.dumps({
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        }), encoding="utf-8")
        self.assertEqual(len(inspect_local_snapshot(self.directory).files), 6)

    def test_exact_single_token_continuation_validation(self):
        class Tokenizer:
            def encode(self, text, *, add_special_tokens):
                self.__class__.calls.append((text, add_special_tokens))
                return {
                    "Answer:": [10],
                    "Answer: A": [10, 11],
                    "Answer: B": [10, 12],
                }.get(text, [10, 14, 15])
        Tokenizer.calls = []
        self.assertEqual(
            validate_single_token_actions(Tokenizer(), action_prefix="Answer:"), (11, 12)
        )
        self.assertTrue(all(use is False for _, use in Tokenizer.calls))
        with self.assertRaisesRegex(ValueError, "single-token"):
            validate_single_token_actions(Tokenizer(), action_prefix="Other:")
        with self.assertRaises(ValueError):
            validate_single_token_actions(Tokenizer(), action_prefix="Answer:",
                                          action_suffixes=(" A", " A"))

    def test_offline_hf_loader_pins_exact_files_and_options(self):
        class Tokenizer:
            def encode(self, text, *, add_special_tokens):
                return {"Answer:": [10], "Answer: A": [10, 11],
                        "Answer: B": [10, 12]}[text]

        called = []

        def tokenizer_from_pretrained(root, **kwargs):
            called.append(("tokenizer", root, kwargs))
            return Tokenizer()

        def model_from_pretrained(root, **kwargs):
            called.append(("model", root, kwargs))
            return object()

        fake = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_from_pretrained),
        )
        with patch("pleasure_experiment.open_weight_local_model.importlib.import_module",
                   return_value=fake) as imp:
            loaded = load_verified_local_causal_model(
                self.directory, model_id="SmolLM2-local",
                model_revision="pinned-by-caller", runtime_instance_id="offline-001",
            )
        imp.assert_called_once_with("transformers")
        self.assertEqual(loaded.action_token_ids, (11, 12))
        self.assertEqual(loaded.binding.checkpoint_sha256, loaded.manifest.checkpoint_sha256)
        self.assertEqual(loaded.binding.tokenizer_sha256, loaded.manifest.tokenizer_sha256)
        self.assertFalse(loaded.externally_authenticated)
        self.assertEqual([entry[0] for entry in called], ["tokenizer", "model"])
        for _, root, kwargs in called:
            # Windows temporary directories can use 8.3 aliases (RUNNER~1),
            # while the verified loader passes the canonical resolved path.
            self.assertEqual(root, str(self.directory.resolve(strict=True)))
            self.assertTrue(kwargs["local_files_only"])
            self.assertIs(kwargs["trust_remote_code"], False)
        self.assertIs(called[1][2]["use_safetensors"], True)

    def test_detects_checkpoint_tamper_during_load(self):
        class Tokenizer:
            def encode(self, text, *, add_special_tokens):
                return {"Answer:": [10], "Answer: A": [10, 11],
                        "Answer: B": [10, 12]}[text]
        def mutating_loader(root, **kwargs):
            (self.directory / "model.safetensors").write_bytes(b"changed")
            return object()
        fake = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda root, **kwargs: Tokenizer()
            ),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=mutating_loader),
        )
        with patch("pleasure_experiment.open_weight_local_model.importlib.import_module",
                   return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "changed during local load"):
                load_verified_local_causal_model(
                    self.directory, model_id="local", model_revision="r1",
                    runtime_instance_id="instance-1",
                )

    def test_missing_transformers_does_not_try_to_download(self):
        with patch("pleasure_experiment.open_weight_local_model.importlib.import_module",
                   side_effect=ImportError("absent")):
            with self.assertRaisesRegex(RuntimeError, "transformers dependency"):
                load_verified_local_causal_model(
                    self.directory, model_id="local", model_revision="r1",
                    runtime_instance_id="instance-1",
                )

    def test_explicit_lora_preparation_never_runs_optimizer(self):
        class Param:
            requires_grad = True
            def numel(self):
                return 4
        class Model:
            def named_modules(self):
                return [("", self), ("blocks.0.q_proj", object())]
        class PeftModel:
            def named_parameters(self):
                return [("model.blocks.0.q_proj.lora_A.default.weight", Param()),
                        ("model.blocks.0.q_proj.lora_B.default.weight", Param())]
        configs = []
        def make_config(**kwargs):
            configs.append(kwargs)
            return object()
        fake_peft = SimpleNamespace(
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            LoraConfig=make_config,
            get_peft_model=lambda model, config: PeftModel(),
        )
        # LoRA dependency is imported only AFTER explicit target validation.
        manifest = inspect_local_snapshot(self.directory)
        from pleasure_experiment.open_weight_adapter import OpenWeightBinding
        loaded = LoadedLocalCausalModel(
            model=Model(), tokenizer=object(),
            binding=OpenWeightBinding("local", "r1", "run1",
                                      manifest.checkpoint_sha256,
                                      manifest.tokenizer_sha256),
            action_token_ids=(11, 12), manifest=manifest,
            action_prefix="Answer:", action_suffixes=(" A", " B"),
        )
        with patch("pleasure_experiment.open_weight_local_model.importlib.import_module",
                   return_value=fake_peft) as imp:
            prepared = prepare_explicit_lora(
                loaded, target_modules=("q_proj",), rank=2, alpha=4,
            )
        imp.assert_called_once_with("peft")
        self.assertEqual(len(prepared.trainable_parameter_names), 2)
        self.assertTrue(all("lora_" in name for name in prepared.trainable_parameter_names))
        self.assertEqual(prepared.action_token_ids, (11, 12))
        self.assertEqual(configs[0]["lora_dropout"], 0.0)
        self.assertEqual(configs[0]["bias"], "none")
        with self.assertRaises(ValueError):
            prepare_explicit_lora(loaded, target_modules=("not_a_real_layer",))
