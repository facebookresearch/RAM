"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class Model(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[Tensor] = None,
        **kargs,
    ):
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[Tensor],
        train=False,
        **kargs,
    ):
        pass


def build_value_head(cfg, value_layers, embed_dim, dtype=None):
    value_out = nn.Linear(embed_dim, 1, dtype=dtype)
    value_out.weight.data.fill_(0)
    value_out.bias.data.fill_(0)
    if value_layers > 1:
        value_mods = []
        for _ in range(value_layers - 1):
            value_mods.append(nn.Linear(embed_dim, embed_dim, dtype=dtype))
            value_mods.append(nn.ReLU())
        value_mods.append(value_out)
        value_head = nn.Sequential(*value_mods)
    else:
        value_head = value_out

    return value_head
