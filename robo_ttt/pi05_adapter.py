"""LeRobot v0.4.4 π0.5 adapter. No robot commands are issued by this module."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import torch

from robo_ttt.robo_ttt import RoboTTT, detach_states


def masked_action_loss(losses, padding=None):
    """Per-sample loss, excluding repeated actions beyond the episode boundary."""
    if padding is None:
        return losses.mean((1, 2))
    valid = (~padding.bool()).unsqueeze(-1).expand_as(losses)
    if (valid.sum((1, 2)) == 0).any():
        raise ValueError("An action chunk has no valid targets")
    return (losses * valid).sum((1, 2)) / valid.sum((1, 2))


class Pi05Adapter:
    """Raw LeRobot batch -> normalized policy -> actions in original dataset units.

    Images must be BCHW float [0,1] (or CHW for a single observation), state
    is Bx14, task is a list of strings. No joint unit conversion is implicit.
    One instance is sequential, not thread safe.
    """

    def __init__(self, policy, preprocessor, postprocessor):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.memory = None
        self.states = None

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" else nullcontext())

    @classmethod
    def from_base(cls, base, config, stats):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        from safetensors.torch import load_file

        # Do not use the upstream loader's catch-and-return-random-weights path.
        weight_file = Path(base) / "model.safetensors"
        if not weight_file.is_file():
            raise FileNotFoundError(weight_file)
        device = config.device
        config.device = "cpu"
        try:
            policy = PI05Policy(config)
            raw = load_file(str(weight_file))
            fixed = policy._fix_pytorch_state_dict_keys(raw, config)
            fixed = {k if k.startswith("model.") else "model." + k: v
                     for k, v in fixed.items()}
            # Safetensors omits duplicate tied embeddings. Reconstruct aliases
            # only when the model actually ties those Parameter objects.
            aliases = {}
            for name, parameter in policy.named_parameters(remove_duplicate=False):
                aliases.setdefault(id(parameter), []).append(name)
            for names in aliases.values():
                source = next((fixed[n] for n in names if n in fixed), None)
                if source is not None:
                    for name in names:
                        fixed.setdefault(name, source)
            policy.load_state_dict(fixed, strict=True)
            del raw, fixed
        finally:
            config.device = device
        # FP32 master parameters/Adam states for the trained expert; frozen VLM BF16.
        # The expert vocabulary head is never used for continuous actions.
        policy.model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
        for p in policy.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        policy.to(device)
        pre, post = make_pi05_pre_post_processors(config, dataset_stats=stats)
        return cls(policy, pre, post)

    def loss_prepared(self, batch, *, noise=None, time=None):
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        images, masks = self.policy._preprocess_images(batch)
        losses = self.policy.model(
            images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.policy.prepare_action(batch), noise=noise, time=time,
        )
        dim = self.policy.config.output_features["action"].shape[0]
        return masked_action_loss(losses[..., :dim], batch.get("action_is_pad"))

    def loss(self, raw_batch):
        batch = self.preprocessor(raw_batch)
        with self.autocast():
            return self.loss_prepared(batch)

    def enable_memory(self, layers=(-1,), hidden_dim=128):
        """Initialize untrained memory. Meta-train/load it before expecting benefit."""
        expert = self.policy.model.paligemma_with_expert.gemma_expert.model
        indices = [i % len(expert.layers) for i in layers]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate memory layers")
        paths = [f"model.paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj"
                 for i in indices]
        # Temporary hooks must not disappear before checkpoint recomputation.
        self.policy.model.gradient_checkpointing_disable()
        self.policy.requires_grad_(False)
        dim = expert.layers[indices[0]].self_attn.o_proj.out_features
        self.memory = RoboTTT(self.policy, paths=paths, dim=dim, dim_hidden=hidden_dim).to(self.device)
        self.states = None

    def reset(self, batch_size=1, *, retain_memory=False):
        self.policy.reset()
        if self.memory is not None and (not retain_memory or self.states is None):
            self.states = self.memory.init_states(batch_size)
        elif self.states is not None and self.states[0]["w1"].shape[0] != batch_size:
            raise ValueError("Cannot retain memory with a different batch size")

    @torch.no_grad()
    def predict(self, raw_observation, *, use_memory=True):
        self.policy.eval()
        batch = self.preprocessor(raw_observation)
        with self.autocast():
            if use_memory and self.memory is not None:
                if self.states is None:
                    self.reset(batch["observation.state"].shape[0])
                actions = self.memory.read(self.policy.predict_action_chunk, self.states, batch)
            else:
                actions = self.policy.predict_action_chunk(batch)
        return self.postprocessor(actions)

    def _clean_pass(self, batch):
        """Encode only executed actions at flow time zero; supports short chunks.

        Pair with the observation at the START of those executed actions. The
        next observation is used for the next prediction, never leaked backward.
        """
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        model = self.policy.model
        actions = self.policy.prepare_action(batch)
        images, image_masks = self.policy._preprocess_images(batch)
        prefix, ppad, patt = model.embed_prefix(
            images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
        suffix, spad, _, cond = model.embed_suffix(
            actions, torch.zeros(actions.shape[0], device=actions.device))
        satt = torch.zeros_like(spad)
        satt[:, 0] = True
        pad = torch.cat((ppad, spad), 1)
        att = torch.cat((patt, satt), 1)
        return model.paligemma_with_expert(
            attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=pad.long().cumsum(1) - 1,
            inputs_embeds=[prefix.to(suffix.dtype), suffix],
            past_key_values=None, use_cache=False, adarms_cond=[None, cond])

    def commit(self, raw_start_observation, executed_actions):
        """Call once after execution. Never pass a predicted, unexecuted tail."""
        if self.memory is None:
            raise RuntimeError("Enable/load memory first")
        if executed_actions.ndim != 3 or not 0 < executed_actions.shape[1] <= self.policy.config.chunk_size:
            raise ValueError("Expected executed actions [B, 1..chunk_size, action_dim]")
        if executed_actions.shape[-1] != self.policy.config.output_features["action"].shape[0]:
            raise ValueError("Wrong action dimension")
        if not torch.isfinite(executed_actions).all():
            raise ValueError("Nonfinite executed actions")
        batch = self.preprocessor({**raw_start_observation, "action": executed_actions})
        if self.states is None:
            self.reset(executed_actions.shape[0])
        self.policy.eval()
        # torch.func.grad computes the inner gradient even though no rollout
        # graph is retained. Do not wrap this in torch.inference_mode().
        with torch.no_grad(), self.autocast():
            self.states = detach_states(self.memory.commit(self._clean_pass, self.states, batch))

    def train_memory_sequence(self, raw_batches, *, segment_len=2, backward=True):
        """Chronological, non-overlapping same-episode chunks; optimizer is caller-owned.

        Each batch includes action_is_pad. Ragged/partial chunks must be removed
        from meta-training (rollout commit supports shorter executed chunks).
        """
        if self.memory is None or not raw_batches:
            raise ValueError("Enable memory and provide a nonempty sequence")
        if segment_len < 1:
            raise ValueError("segment_len must be positive")
        batches = [self.preprocessor(b) for b in raw_batches]
        for b in batches:
            if b.get("action_is_pad") is not None and b["action_is_pad"].any():
                raise ValueError("Use full, non-padded chunks for memory training")
        self.policy.eval()  # frozen backbone, no checkpoint recomputation
        with self.autocast():
            return self.memory.train_sequence(
                lambda t: self.loss_prepared(batches[t]),
                lambda t: self._clean_pass({
                    **batches[t],
                    "action": batches[t]["action"][:, :self.policy.config.n_action_steps],
                }), num_chunks=len(batches),
                batch=batches[0]["action"].shape[0], segment_len=segment_len,
                backward=backward)

    def save_memory(self, path, *, include_fast_state=False):
        if self.memory is None:
            raise RuntimeError("No memory")
        torch.save({"paths": self.memory.paths,
                    "hidden_dim": self.memory.memories[0].w1.shape[1],
                    "weights": self.memory.memories.state_dict(),
                    "fast_state": detach_states(self.states) if include_fast_state and self.states else None}, path)

    def load_memory(self, path, *, restore_fast_state=False):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        layers = [int(p.split(".layers.")[1].split(".")[0]) for p in saved["paths"]]
        self.enable_memory(layers, saved["hidden_dim"])
        self.memory.memories.load_state_dict(saved["weights"], strict=True)
        if restore_fast_state:
            self.states = saved["fast_state"]

    @classmethod
    def from_run(cls, run_dir, checkpoint="latest.pt", device="cuda"):
        from lerobot.configs.policies import PreTrainedConfig
        run = Path(run_dir)
        manifest = json.loads((run / "manifest.json").read_text())
        config = PreTrainedConfig.from_pretrained(run)
        if config.type != "pi05":
            raise ValueError("Expected a pi05 run")
        config.device = device
        config.gradient_checkpointing = False
        adapter = cls.from_base(manifest["base_path"], config, manifest["stats"])
        saved = torch.load(run / checkpoint, map_location="cpu", weights_only=True)
        expected = {n for n, p in adapter.policy.named_parameters() if p.requires_grad}
        if set(saved["parameters"]) != expected:
            raise ValueError("Checkpoint trainable parameter set differs from configuration")
        adapter.policy.load_state_dict(saved["parameters"], strict=False)
        adapter.policy.eval()
        return adapter
