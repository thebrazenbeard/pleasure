"""Opt-in pinned CPU feasibility witness, never run by routine unit discovery.

Set PLEASURE_RUN_REAL_CPU_LORA_SMOKE=1 to deliberately load an already-local,
hash-checked SmolLM2-135M checkpoint and perform ONE synthetic chosen-quality
SGD step. This is not a positive-valence or heldout utility experiment. It
does not download anything, write model artifacts, or access an EVAL oracle.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
import unittest

try:
    import resource  # Unix-only optional peak RSS diagnostic.
except ImportError:  # pragma: no cover - Windows has no resource module.
    resource = None


SNAPSHOT = Path("/tmp/pleasure-smollm2-safe-v3")
INTERPRETER_GATE = "PLEASURE_RUN_REAL_CPU_LORA_SMOKE"


@unittest.skipUnless(
    os.environ.get(INTERPRETER_GATE) == "1" and SNAPSHOT.is_dir(),
    "requires explicit local CPU LoRA smoke consent and verified snapshot",
)
class PinnedLoRARuntimeFeasibility(unittest.TestCase):
    def test_one_joint_chosen_quality_gradient_updates_lora_not_backbone(self):
        import torch
        from pleasure_experiment.open_weight_adapter import _named_digest
        from pleasure_experiment.open_weight_local_model import (
            load_verified_local_causal_model, prepare_explicit_lora,
        )

        torch.set_num_threads(2)
        started = time.perf_counter()
        loaded = load_verified_local_causal_model(
            SNAPSHOT, model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            model_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
            runtime_instance_id="isolated-cpu-joint-lora-feasibility",
        )
        loaded_time = time.perf_counter() - started
        prepared = prepare_explicit_lora(
            loaded, target_modules=("q_proj", "v_proj"), rank=2, alpha=4,
            max_trainable_parameters=1_000_000,
        )
        model = prepared.model
        model.eval()
        trainable_names = set(prepared.trainable_parameter_names)
        trainable = tuple(
            (name, p) for name, p in model.named_parameters()
            if name in trainable_names
        )
        frozen = tuple(
            (name, p) for name, p in model.named_parameters()
            if name not in trainable_names
        ) + tuple(("buffer:" + name, buf) for name, buf in model.named_buffers())
        count = sum(p.numel() for _, p in trainable)
        self.assertGreater(count, 0)
        self.assertLess(count, 1_000_000)
        self.assertEqual(len(trainable), len(trainable_names))
        base_sha = _named_digest(torch, frozen)
        lora_before = _named_digest(torch, trainable)
        prompt = (
            "Two independent options have numeric features. "
            "A has 0.14 and 0.82. B has 0.77 and 0.33. "
            "Choose the one with the more promising outcome. Answer:"
        )
        ids = tuple(loaded.tokenizer.encode(prompt, add_special_tokens=False))
        self.assertGreater(len(ids), 1)
        head = torch.nn.Linear(model.config.hidden_size, 1, dtype=torch.float32)
        # A zero-initialized quality head transmits no first-step gradient to
        # a LoRA representation. Deliberate small deterministic nonzero
        # initialization is necessary to test joint gradient flow.
        with torch.no_grad():
            head.weight.fill_(0.015 / math.sqrt(model.config.hidden_size))
            head.bias.zero_()
        optimizer = torch.optim.SGD(
            [p for _, p in trainable] + list(head.parameters()), lr=0.01,
        )
        gradient_started = time.perf_counter()
        choices = []
        for action_token in loaded.action_token_ids:
            batch = torch.tensor([ids + (action_token,)], dtype=torch.long)
            output = model(
                input_ids=batch, output_hidden_states=True, use_cache=False,
            )
            choices.append(output.hidden_states[-1][0, -1, :].float())
        difference = choices[0] - choices[1]
        relative = difference / difference.norm().clamp(min=1e-5)
        # The sole feedback is the committed synthetic A quality 0.8; no
        # unchosen quality, EVAL reward, or hidden generator is consulted.
        predicted_chosen_quality = torch.sigmoid(head(relative)).reshape(())
        loss = (predicted_chosen_quality - 0.8).square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonzero_lora_grads = sum(
            int(p.grad is not None and bool((p.grad != 0).any()))
            for _, p in trainable
        )
        self.assertGreater(nonzero_lora_grads, 0)
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable] + list(head.parameters()), 1.0,
        )
        optimizer.step()
        elapsed_step = time.perf_counter() - gradient_started
        changed_lora = _named_digest(torch, trainable)
        unchanged_backbone = _named_digest(torch, frozen)
        self.assertNotEqual(lora_before, changed_lora)
        self.assertEqual(base_sha, unchanged_backbone)
        self.assertTrue(all(p.grad is None for _, p in frozen))
        report = {
            "checkpoint_sha256": loaded.manifest.checkpoint_sha256,
            "snapshot_sha256": loaded.manifest.snapshot_sha256,
            "lora_rank": prepared.rank,
            "lora_alpha": prepared.alpha,
            "target_modules": ("q_proj", "v_proj"),
            "trainable_tensors": len(trainable),
            "trainable_parameters": count,
            "nonzero_lora_gradient_tensors": nonzero_lora_grads,
            "frozen_sha256": base_sha,
            "initial_lora_sha256": lora_before,
            "updated_lora_sha256": changed_lora,
            "loss_before_step": float(loss.detach()),
            "model_load_seconds": loaded_time,
            "candidate_forward_backward_and_step_seconds": elapsed_step,
            "peak_rss_kib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if resource is not None else None
            ),
            "hosted_model_modified": False,
            "synthetic_chosen_quality_only": True,
        }
        self.assertTrue(math.isfinite(report["loss_before_step"]))
        print("CPU_JOINT_LORA_FEASIBILITY=" + json.dumps(report, sort_keys=True))
