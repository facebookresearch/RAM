"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from dataclasses import dataclass

import blobfile as bf
import models.sparse_autoencoder as sparse_autoencoder
import torch
import torch.nn as nn
import transformer_lens
from torch.nn import CrossEntropyLoss
from transformer_lens import ActivationCache


class TransformerLensSAE(nn.Module):
    def __init__(self, layer_index=6, location="resid_post_mlp"):
        super().__init__()

        # define sparse autoencoder
        self.layer_index = layer_index
        base_model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

        self.base_model = base_model
        self.base_model.eval()

        self.transformer_lens_loc = {
            "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
            "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
            "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
            "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
            "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
        }[location]

        with bf.BlobFile(
            sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb"
        ) as f:
            state_dict = torch.load(f)
            autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)

        self.autoencoder = autoencoder

    def get_cache_fwd_and_bwd(self, tokens, labels, new_act=None):
        # filter_not_qkv_input = lambda name: "_input" not in name

        self.base_model.reset_hooks()
        cache = {}

        def forward_cache_hook(act, hook):
            if new_act is not None:
                cache[hook.name] = new_act.detach()
                return new_act  # activation patching
            cache[hook.name] = act.detach()

        self.base_model.add_hook(self.transformer_lens_loc, forward_cache_hook, "fwd")

        grad_cache = {}

        def backward_cache_hook(act, hook):
            grad_cache[hook.name] = act.detach()

        self.base_model.add_hook(self.transformer_lens_loc, backward_cache_hook, "bwd")
        logits = self.base_model(tokens)
        loss = self.compute_loss(logits, labels)
        loss.backward()
        self.base_model.reset_hooks()
        return (
            loss.item(),
            ActivationCache(cache, self.base_model),
            ActivationCache(grad_cache, self.base_model),
        )

    def compute_loss(self, logits, labels):
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    def compute_attribute(self, idx, labels):
        assert (
            labels is not None
        ), "Attribute-based latent tokenization requires labels."

        _, act, grad = self.get_cache_fwd_and_bwd(idx, labels)

        x = act[self.transformer_lens_loc]
        grad_x = grad[self.transformer_lens_loc]

        latent_activations, _ = self.autoencoder.encode(x)
        w_dec = self.autoencoder.decoder.weight
        attribute = torch.matmul(grad_x, w_dec) * latent_activations

        return attribute

    def forward(self, input_ids, labels=None):
        if labels is None:
            labels = input_ids
        attribute = self.compute_attribute(input_ids, labels)
        return attribute
