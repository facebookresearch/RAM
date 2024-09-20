"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from torch import distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

"""Checkpointing utils"""


def add_args(parser: ArgumentParser):
    group = parser.add_argument_group("Checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoint-freq", type=int, default=1, help="how often to save checkpoint"
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        help="keep old checkpoints from every Nth epoch in addition to the last",
    )
    group.add_argument("--init-model-from", type=str)
    group.add_argument(
        "--init-model-no-strict", default=False, action=BooleanOptionalAction
    )


def save(
    cfg,
    model,
    optimizer,
    logger,
    tokenizer,
    epoch,
):
    """Save checkpoint"""
    epoch_num = epoch + 1
    if cfg.checkpoint_freq > 1:
        if epoch_num % cfg.checkpoint_freq != 0:
            return

    if cfg.distributed:
        distributed.barrier()

    # need to run all workers for FSDP
    model_state_dict = model.state_dict()
    if cfg.distributed and cfg.fsdp:
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
    else:
        optim_state_dict = optimizer.state_dict()

    if cfg.rank > 0:
        return

    path = cfg.checkpoint
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.info(f"saving epoch {epoch_num} to {path}")

    state_dict = {}
    state_dict["cfg"] = cfg
    state_dict["model"] = model_state_dict
    state_dict["logger"] = logger.get_state()
    state_dict["optimizer"] = optim_state_dict
    state_dict["tokenizer"] = tokenizer.get_state()

    if cfg.checkpoint_keep is not None:
        if epoch_num % cfg.checkpoint_keep == 0:
            torch.save(state_dict, path + f".{epoch_num}")

    # if best_epoch is True, save the model as "*_best.pt" in the save path
    if cfg.valid_metric is not None and logger.metrics[cfg.valid_metric].cur_epoch_best:
        torch.save(state_dict, path + ".best")

    torch.save(state_dict, path)
    torch.save(cfg, path + ".cfg")
    logging.info("done saving.")


def load_model(state_dict, model, strict=True):
    """Load model from a given state dictionary"""
    model.load_state_dict(state_dict["model"], strict=strict)


def load_checkpoint(cfg, model, optimizer, logger, tokenizer):
    """Load checkpoint"""
    state_dict = torch.load(cfg.checkpoint, map_location=torch.device("cpu"))
    load_model(state_dict, model)
    optim_state_dict = state_dict["optimizer"]
    if cfg.distributed and cfg.fsdp:
        try:
            optim_state_dict = FSDP.optim_state_dict_to_load(
                optim_state_dict, model, optimizer
            )
        except Exception:
            # for loading old checkpoints
            optim_state_dict = state_dict["optimizer"]
    optimizer.load_state_dict(optim_state_dict)
    logger.set_state(state_dict["logger"])
    if "tokenizer" in state_dict:
        tokenizer.set_state(state_dict["tokenizer"])


def load(cfg, model, optimizer, logger, tokenizer):
    if cfg.checkpoint is not None and os.path.exists(cfg.checkpoint):
        logging.info(f"loading checkpoint from {cfg.checkpoint}")
        load_checkpoint(cfg, model, optimizer, logger, tokenizer)
    elif cfg.init_model_from is not None:
        f = torch.load(cfg.init_model_from, map_location=torch.device("cpu"))
        logging.info(f"loading model from {cfg.init_model_from}")
        load_model(f, model, strict=not cfg.init_model_no_strict)
