"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import functools
import os
from argparse import ArgumentParser, BooleanOptionalAction

import submitit
import torch
from torch import distributed


def add_cmd_args(parser: ArgumentParser):
    parser.add_argument(
        "--distributed",
        action=BooleanOptionalAction,
        default=False,
        help="distributed training",
    )
    parser.add_argument(
        "--submitit", action=BooleanOptionalAction, default=False, help="using submitit"
    )
    parser.add_argument("--rank", type=int, default=0, help="")
    parser.add_argument("--local-rank", type=int, default=0, help="")
    parser.add_argument("--world-size", type=int, default=1, help="")
    parser.add_argument("--dist-init", type=str, help="distrixbuted training")
    parser.add_argument(
        "--fsdp", action=BooleanOptionalAction, default=False, help="using fsdp"
    )


class DummyWrapper(torch.nn.Module):
    def __init__(self, mod):
        super(DummyWrapper, self).__init__()
        self.module = mod
        self.no_sync = contextlib.nullcontext

    def forward(self, *cfg, **kwcfg):
        return self.module(*cfg, **kwcfg)

    def load_state_dict(self, state_dict, strict=True):
        try:
            super().load_state_dict(state_dict, strict)
        except RuntimeError:
            # FSDP saves without "module" prefix
            self.module.load_state_dict(state_dict, strict)


def init(cfg):
    if cfg.submitit:
        job_env = submitit.JobEnvironment()
        cfg.local_rank = job_env.local_rank
        cfg.rank = job_env.global_rank
        cfg.world_size = job_env.num_tasks
        distributed.init_process_group(
            backend="nccl",
            init_method=cfg.dist_init,
            rank=job_env.global_rank,
            world_size=job_env.num_tasks,
        )
    else:
        init_file = os.getcwd() + "/dist_init"
        distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        cfg.local_rank = distributed.get_rank()
    cfg.device = torch.device("cuda", cfg.local_rank)
    if cfg.rank > 0:
        cfg.log_plot = False


def wrap_model(cfg, model):
    if cfg.distributed:
        if cfg.fsdp:
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, StateDictType
            from torch.distributed.fsdp.api import FullOptimStateDictConfig
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            assert cfg.model.startswith("llama")
            llama_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    LlamaDecoderLayer,
                },
            )
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=llama_auto_wrap_policy,
                device_id=cfg.local_rank,
            )

            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
        else:
            model = model.to(cfg.device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=False,
            )
    else:
        model = DummyWrapper(model)
        model = model.to(cfg.device)
    return model


def cleanup():
    pass
    # distributed.destroy_process_group()
