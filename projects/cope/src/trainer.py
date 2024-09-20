"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction, Namespace

import src.utils.world as world
import torch
import torch.nn as nn
from src.utils import Batch
from src.utils.logger import Logger
from tqdm import tqdm

from .data.constants import (
    TOKEN_TYPE_ANSWER,
    TOKEN_TYPE_CONTEXT,
    TOKEN_TYPE_PAD,
    TOKEN_TYPE_QUESTION,
)


def add_args(parser: ArgumentParser):
    group = parser.add_argument_group("Trainer")
    group.add_argument("--batch-sz", type=int, default=32)
    group.add_argument("--lr", type=float, default=0.0002)
    group.add_argument("--adam-eps", type=float, default=1e-8)
    group.add_argument("--nepochs", type=int, default=30)
    group.add_argument("--lr-decay", default=None, choices=["none", "cosine"])
    group.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    group.add_argument("--grad-clip", type=float, default=0)
    group.add_argument(
        "--valid-metric",
        type=str,
        help="the main validation metric e.g. loss/val",
    )
    group.add_argument(
        "--valid-patience",
        type=int,
        help="early stop if the valid metric is not improving after this many epochs",
    )
    group.add_argument(
        "--do-sample",
        action=BooleanOptionalAction,
        default=False,
        help="sample during generation",
    )
    group.add_argument(
        "--temperature",
        default=1,
        type=float,
        help="generate temp",
    )
    group.add_argument(
        "--top_k",
        default=0,
        type=int,
        help="generate temp",
    )


class Trainer:
    def __init__(
        self,
        cfg: Namespace,
        model: nn.Module,
        logger: Logger,
        tokenizer,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        if self.cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.lr,
                eps=cfg.adam_eps,
            )
        elif self.cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )  # from LIMA paper https://arxiv.org/pdf/2305.11206.pdf
        self.scheduler = None
        if self.cfg.lr_decay == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.nepochs
            )
        self.logger = logger
        self.logger.setup_metric(
            "total_nsamples", cumulative=True, plot=False, print=False
        )
        self.logger.setup_metric("nbatches", plot=False)
        self.logger.setup_metric("compute/batch_time", divide_by="nbatches")
        self.logger.setup_metric("compute/epoch_time_min")
        self.logger.setup_metric("compute/gpu_mem_peak_gb")
        self.logger.setup_metric("nsamples", plot=False)
        self.logger.setup_metric("loss", divide_by="npreds")
        self.logger.setup_metric(
            "npreds", plot=False
        )  # the number of tokens that the model has to predict
        self.logger.setup_metric("tokens", plot=False)
        self.logger.setup_metric("error", divide_by="tokens")
        self.logger.setup_metric("error/sample", divide_by="nsamples")
        self.logger.setup_metric("answer/loss", divide_by="answer/npreds")
        self.logger.setup_metric("answer/npreds", plot=False)
        # for eval
        self.logger.setup_metric("answer/tokens", plot=False)
        self.logger.setup_metric("answer/error", divide_by="answer/tokens")
        self.logger.setup_metric("answer/error/sample", divide_by="nsamples")
        self.logger.setup_metric("acc/sample", divide_by="nsamples")
        self.logger.setup_metric("ngen", plot=False)
        self.logger.setup_metric("ngen/sample", plot=False)
        self.logger.setup_metric("gen_error", divide_by="ngen")
        self.logger.setup_metric("gen_error/sample", divide_by="ngen/sample")

        self.world = world.SupervisedWorld(
            cfg, model, self.optimizer, tokenizer, logger
        )

    def train_batch(self, batch: Batch, is_eval=False):
        """Run model over a data batch"""
        metrics = self.world.train(batch, is_eval)

        # Logging
        self.logger.record("loss", metrics["full_loss"])
        self.logger.record("npreds", metrics["full_npreds"])
        self.logger.record("nsamples", batch.size)
        self.logger.record("total_nsamples", batch.size)
        self.logger.record("nbatches", 1)
        token_mask = batch.y_type != TOKEN_TYPE_PAD
        self.logger.record("tokens", token_mask.sum())
        answer_mask = batch.y_type == TOKEN_TYPE_ANSWER
        if "preds" in metrics:
            err = metrics["preds"].ne(batch.y)
            err = err * token_mask
            self.logger.record("error", err.sum())
            self.logger.record("error/sample", err.any(dim=1).sum())
            ans_err = err * answer_mask
            self.logger.record("answer/error", ans_err.sum())
            self.logger.record("answer/error/sample", ans_err.any(dim=1).sum())
        self.logger.record("answer/loss", metrics["ans_loss"])
        self.logger.record("answer/npreds", metrics["ans_npreds"])
        self.logger.record("answer/tokens", answer_mask.sum())

        if self.cfg.record_loss_per_position:
            # this is hacky and only for eval
            if not hasattr(self, "loss_per_pos"):
                self.loss_per_pos = metrics["loss_per_pos"]
                self.loss_per_pos_count = 1.0
            else:
                self.loss_per_pos += metrics["loss_per_pos"]
                self.loss_per_pos_count += 1.0

    def get_answer_gen(self, gen, y, y_type):
        gen_mask = gen != self.tokenizer.pad_ind
        gen_mask &= y_type != TOKEN_TYPE_CONTEXT
        gen_mask &= y_type != TOKEN_TYPE_QUESTION
        answer_gen = gen[gen_mask]

        return answer_gen

    def generate(self, batch: Batch):
        """Generate text given context"""
        if self.cfg.display:
            context = batch.x[0][batch.x_type[0] == TOKEN_TYPE_CONTEXT]
            print('context: "{}"'.format(self.tokenizer.decode(context)))
            question = batch.x[0][batch.x_type[0] == TOKEN_TYPE_QUESTION]
            print('question: "{}"'.format(self.tokenizer.decode(question)))

        with torch.no_grad():
            generation = self.world.generate(batch)  # type: ignore

        # Compute error
        answer_mask = batch.y_type == TOKEN_TYPE_ANSWER
        err = generation.ne(batch.y)  # type: ignore
        err = err * answer_mask
        self.logger.record("gen_error", err.sum())
        self.logger.record("gen_error/sample", err.any(dim=1).sum())
        self.logger.record("ngen", answer_mask.sum())
        self.logger.record("ngen/sample", batch.size)
        self.logger.record("nbatches", 1)

        if self.cfg.display:
            answer = batch.y[0][batch.y_type[0] == TOKEN_TYPE_ANSWER]

            answer_gen = self.get_answer_gen(generation[0], batch.y[0], batch.y_type[0])

            if err[0].any(dim=0).sum() > 0:
                print("****** ERR *******")
            print(f'answer:     "{self.tokenizer.decode(answer)}"')
            print(f'generation: "{self.tokenizer.decode(answer_gen)}"')

        return generation

    def train(self, data_loader, split="train", eval_only=False):
        """Training loop"""
        t_sta = time.time()
        self.model.train(not eval_only)
        self.logger.set_group(split)
        pbar = tqdm(total=len(data_loader), disable=None)
        torch.cuda.reset_peak_memory_stats()

        for batch_ind, batch_dict in enumerate(data_loader):
            if pbar is not None:
                pbar.update(1)
            batch = Batch.from_dict(batch_dict, self.cfg.device)

            self.logger.is_example_batch = batch_ind == 0 and split == "val"

            if eval_only and self.cfg.generate:
                self.generate(batch)
            self.train_batch(batch, is_eval=eval_only)

        if not eval_only and self.scheduler is not None:
            self.scheduler.step()
        elapsed = time.time() - t_sta
        self.logger.record("compute/batch_time", 1000 * elapsed)
        if self.cfg.rank == 0:
            self.logger.record("compute/epoch_time_min", elapsed / 60.0)
            mem_stats = torch.cuda.memory_stats()
            self.logger.record(
                "compute/gpu_mem_peak_gb",
                mem_stats["allocated_bytes.all.peak"] / 1024.0**3,
            )

        if self.cfg.record_loss_per_position:
            self.loss_per_pos /= self.loss_per_pos_count
            for i in range(self.loss_per_pos.size(0)):
                print(self.loss_per_pos[i].item())

    def eval(self, data_loader, split="val"):
        """Evaluation loop"""
        with torch.no_grad():
            return self.train(data_loader, split=split, eval_only=True)

    def is_stopping_criteria(self):
        """Whether to stop the training run or not"""
        if self.cfg.valid_patience is None:
            return False
        assert self.cfg.valid_metric is not None

        if not hasattr(self, "cur_patience"):
            self.cur_patience = 0
        if self.logger.metrics[self.cfg.valid_metric].cur_epoch_best:
            self.cur_patience = 0
        if self.cur_patience > 0:
            logging.info(f"patience {self.cur_patience}/{self.cfg.valid_patience}")
        stop = self.cur_patience >= self.cfg.valid_patience
        self.cur_patience += 1

        if self.cfg.distributed:
            stop = [stop]
            torch.distributed.broadcast_object_list(stop, src=0, device=self.cfg.device)  # type: ignore
            stop = stop[0]

        return stop
