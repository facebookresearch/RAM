"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import torch
from models.concept_extractor import TransformerLensSAE
from models.modeling_gpt2_cocomix import GPT2CoCoMixLMHeadModel
from transformers import AutoConfig, GPT2LMHeadModel


def get_base_lm(cfg, accelerator):
    """define base model"""
    config = AutoConfig.from_pretrained(cfg.base_model)
    if cfg.vocab_size is not None:
        config.vocab_size = cfg.vocab_size
    if "gpt2" in cfg.base_model:
        if cfg.n_embd is not None:
            config.n_embd = cfg.n_embd
        if cfg.n_layer is not None:
            config.n_layer = cfg.n_layer
        if cfg.n_head is not None:
            config.n_head = cfg.n_head
        if cfg.mode == "cocomix":
            config._attn_implementation = "flash_attention_2"
            base_lm = GPT2CoCoMixLMHeadModel(
                config, cfg.concept_dim, cfg.insert_layer_index, cfg.concept_num
            )
        else:  # just next token prediction
            config._attn_implementation = "sdpa"
            base_lm = GPT2LMHeadModel(config)
    else:
        raise NotImplementedError

    base_lm = accelerator.prepare(base_lm)  # Accelerate does FSDP, DDP, etc.

    if cfg.use_torch_compile:
        base_lm = torch.compile(base_lm)

    return base_lm


def get_concept_extractor(cfg, accelerator):
    concept_extractor = None
    if cfg.mode in ["cocomix"]:
        if "gpt2" in cfg.pretrained_model:
            concept_extractor = TransformerLensSAE(
                layer_index=cfg.sae_layer_index, location=cfg.sae_location
            )
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            local_device = f"cuda:{ddp_local_rank}"
            concept_extractor = concept_extractor.to(local_device)
            concept_extractor.base_model = concept_extractor.base_model.to(local_device)
            concept_extractor.autoencoder = concept_extractor.autoencoder.to(
                local_device
            )
        else:
            raise NotImplementedError

    if cfg.use_torch_compile and concept_extractor is not None:
        concept_extractor = torch.compile(concept_extractor)

    return concept_extractor
