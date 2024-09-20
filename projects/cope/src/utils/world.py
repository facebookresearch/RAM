"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.constants import (
    TOKEN_TYPE_ANSWER,
    TOKEN_TYPE_CONTEXT,
    TOKEN_TYPE_PAD,
    TOKEN_TYPE_QUESTION,
)
from ..data.tokenizer import Tokenizer
from . import Batch, find_pos
from .logger import Logger


class World(ABC):
    @classmethod
    def add_cmd_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group("World")
        group.add_argument(
            "--max-gen-len",
            default=8,
            type=int,
            help="max number of tokens to generate",
        )
        group.add_argument(
            "--updates-per-batch",
            default=1,
            type=int,
            help="only used to match the number of updates to PPO algo",
        )

    def __init__(
        self,
        cfg,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer: Tokenizer,
        logger: Logger,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.logger = logger

    def set_epoch(self, epoch: int):
        pass

    @abstractmethod
    def run_episode(
        self,
        batch: Batch,
    ):
        pass

    def optimize(self, loss: Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(  # type: ignore
                self.model.parameters(), self.cfg.grad_clip
            )
        self.optimizer.step()

    def train(self, batch: Batch, is_eval):
        for _ in range(self.cfg.updates_per_batch):
            metrics = self.run_episode(batch)
            if not is_eval:
                # normalize by the number of samples
                self.optimize(metrics["full_loss"] / batch.size)
        return metrics

    @abstractmethod
    def generate(
        self,
        batch: Batch,
    ) -> Tensor:
        pass


class SupervisedWorld(World):
    @classmethod
    def add_cmd_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group("Supervised")
        group.add_argument(
            "--train-on",
            choices=["answer", "QA", "CA", "all"],
            default="all",
            help="type of tokens to train on train on",
        )
        parser.add_argument(
            "--record-loss-per-position",
            action=BooleanOptionalAction,
            default=False,
            help="(only for eval) record and output loss at each position separately.",
        )

    def run_episode(
        self,
        batch: Batch,
    ):
        pad_mask = batch.x_type == TOKEN_TYPE_PAD

        scores = self.model(batch.x, pad_mask)

        preds = scores.max(dim=-1)[1]
        full_loss, full_npreds, ans_loss, ans_npreds = self.compute_loss(
            scores, batch.y, batch.y_type
        )

        metrics = {
            "preds": preds,
            "full_loss": full_loss,
            "full_npreds": full_npreds,
            "ans_loss": ans_loss,
            "ans_npreds": ans_npreds,
        }

        if self.cfg.record_loss_per_position:
            loss_per_pos = F.nll_loss(
                scores.flatten(0, 1), batch.y.reshape(-1), reduction="none"
            )
            loss_per_pos = loss_per_pos.view_as(batch.y)
            loss_per_pos = loss_per_pos.mean(0)
            metrics["loss_per_pos"] = loss_per_pos

        return metrics

    def compute_loss(
        self, scores: Tensor, y: Tensor, y_type: Tensor, mask: Optional[Tensor] = None
    ):
        full_pred_mask = y_type == TOKEN_TYPE_ANSWER
        ans_pred_mask = y_type == TOKEN_TYPE_ANSWER
        if self.cfg.train_on == "QA":
            # train on question tokens too
            full_pred_mask += y_type == TOKEN_TYPE_QUESTION
        elif self.cfg.train_on == "CA":
            # train on context tokens too
            full_pred_mask += y_type == TOKEN_TYPE_CONTEXT
        elif self.cfg.train_on == "all":
            full_pred_mask = y_type != TOKEN_TYPE_PAD
        if mask is not None:
            full_pred_mask *= mask
        loss = F.nll_loss(scores.flatten(0, 1), y.reshape(-1), reduction="none")
        loss = loss.view_as(y)
        full_loss = loss * full_pred_mask
        ans_loss = loss * ans_pred_mask

        return (
            full_loss.sum(),
            full_pred_mask.sum(),
            ans_loss.sum(),
            ans_pred_mask.sum(),
        )

    def generate(
        self,
        batch: Batch,
    ):
        prompts = []
        for i in range(batch.size):
            a_sta = find_pos(batch.x_type[i], TOKEN_TYPE_ANSWER)
            prompt = batch.x[i, :a_sta]
            prompt = prompt[-self.max_gen_context_len :]
            prompts.append(prompt)

        generations, _ = self.model.module.generate(
            prompts,
            max_new_tokens=self.max_gen_len,
        )

        outputs = []
        for i in range(batch.size):
            out = torch.cat([prompts[i], generations[i]], dim=0)
            out = out[1:]  # shift by 1
            out = F.pad(
                out,
                [0, batch.x.size(1) - out.size(0)],
                value=self.tokenizer.pad_ind,
            )
            outputs.append(out)
        outputs = torch.stack(outputs)
        return outputs
