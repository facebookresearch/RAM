"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import random
from argparse import ArgumentParser, BooleanOptionalAction, Namespace

import numpy as np
import src.trainer as trainer_mod
import src.utils.logger as logger_mod
import torch
from src import data, models
from src.cope import context_position

from .utils import Batch, checkpoint, distributed
from .utils.world import SupervisedWorld, World


def add_train_args(parser: ArgumentParser):
    parser.add_argument(
        "--test-nepochs",
        default=1,
        type=int,
        help="test every N epochs",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument(
        "--eval-on",
        choices=["valid", "test", "train"],
        nargs="+",
        default="valid",
        help="choose which data to use for eval",
    )
    parser.add_argument(
        "--display",
        action=BooleanOptionalAction,
        default=False,
        help="display the model output in real time",
    )
    # FIXME move these out of main to RL code
    parser.add_argument(
        "--generate",
        action=BooleanOptionalAction,
        default=False,
        help="generate during test time",
    )
    parser.add_argument("--seed", type=int, default=1)
    data.add_args(parser)
    models.add_args(parser)
    trainer_mod.add_args(parser)
    logger_mod.add_args(parser)
    checkpoint.add_args(parser)
    distributed.add_cmd_args(parser)
    context_position.add_args(parser)
    World.add_cmd_args(parser)
    SupervisedWorld.add_cmd_args(parser)


def setup(cfg: Namespace):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    models.set_block_size(cfg)

    # CONFIG
    logger_mod.preprocess_args(cfg)
    logging.info(cfg)

    # log env vars
    logging.debug("ENV VARIABLES:")
    for key, value in os.environ.items():
        logging.debug(f"{key} = {value}")

    if cfg.distributed:
        distributed.init(cfg)
    else:
        if cfg.no_cuda:
            cfg.device = torch.device("cpu")
        else:
            cfg.device = torch.device("cuda")

    # different seed for each worker
    # this must be after dist init
    # NOTE: seed is also used in data loader init()
    worker_seed = cfg.seed + cfg.rank * 53243
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    logger = logger_mod.Logger(cfg)
    logger.setup_metric(
        "nepochs", cumulative=True, plot=False, grouped=False, is_step=True
    )

    # DATA
    tokenizer = data.tokenizer.get_tokenizer(cfg)
    train_data, val_data, test_data, tokenizer = data.get_data(cfg, tokenizer)
    train_loader = data.get_loader(cfg, train_data, tokenizer)
    val_loader = data.get_loader(cfg, val_data, tokenizer, eval=True)
    if test_data is not None:
        test_loader = data.get_loader(cfg, test_data, tokenizer, eval=True)
    else:
        test_loader = None

    # MODEL
    torch.cuda.set_device(cfg.local_rank)
    torch.cuda.empty_cache()
    model = models.build(cfg, tokenizer)
    nparams = sum([p.numel() for p in model.parameters()])
    cfg.nparams = nparams
    logging.info(f"nparams: {nparams / 1000000 : .2f}M")
    model = distributed.wrap_model(cfg, model)

    trainer = trainer_mod.Trainer(cfg, model, logger, tokenizer)

    checkpoint.load(cfg, model, trainer.optimizer, logger, tokenizer)

    return logger, model, trainer, train_loader, val_loader, test_loader


def display(cfg: Namespace):
    logger, model, trainer, _, val_loader, _ = setup(cfg)
    logger.set_group("test")
    for batch_dict in val_loader:
        batch = Batch.from_dict(batch_dict, cfg.device)
        trainer.generate(batch)
        input("\n[press Enter for the next sample]")

    return


def eval(cfg: Namespace):
    logger, model, trainer, train_loader, val_loader, test_loader = setup(cfg)
    if "train" in cfg.eval_on:
        trainer.eval(train_loader, split="train")
    if "valid" in cfg.eval_on:
        trainer.eval(val_loader, split="val")
    if "test" in cfg.eval_on:
        trainer.eval(test_loader, split="test")

    logger.step()
    if cfg.distributed:
        distributed.cleanup()


def train(cfg: Namespace):
    logger, model, trainer, train_loader, val_loader, test_loader = setup(cfg)

    init_epoch = len(logger.log)
    if trainer.scheduler is not None:
        trainer.scheduler.step(init_epoch)
    for epoch in range(init_epoch, cfg.nepochs):
        logger.record("nepochs", 1)
        trainer.world.set_epoch(epoch)
        if cfg.distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)  # type: ignore
            val_loader.sampler.set_epoch(epoch)  # type: ignore
            if test_loader is not None:
                test_loader.sampler.set_epoch(epoch)  # type: ignore

        logging.debug("starting training")
        trainer.train(train_loader)
        if (epoch + 1) % cfg.test_nepochs == 0:
            if "valid" in cfg.eval_on:
                logging.debug("starting eval on valid")
                trainer.eval(val_loader, split="val")
            if "test" in cfg.eval_on:
                logging.debug("starting eval on test")
                trainer.eval(test_loader, split="test")

        logging.debug("logging step")
        logger.step()

        logging.debug("saving checkpoint")
        if cfg.checkpoint is not None:
            checkpoint.save(
                cfg,
                model,
                trainer.optimizer,
                logger,
                trainer.tokenizer,
                epoch,
            )

        if trainer.is_stopping_criteria():
            break

    logging.debug("closing")
    logger.close()
    if cfg.distributed:
        distributed.cleanup()
