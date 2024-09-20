"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def find_pos(x: Tensor, id: int):
    """position of the first occurance id"""
    assert x.ndim == 1
    index = (x == id).nonzero()
    if index.size(0) == 0:
        return None
    return index[0].item()


def stack_with_pad(x: List[Tensor], pad_value: float, from_left=False) -> Tensor:
    max_len = max([z.size(0) for z in x])
    assert x[0].ndim == 1
    x_padded = []
    for i in range(len(x)):
        if from_left:
            pads = [max_len - len(x[i]), 0]
        else:
            pads = [0, max_len - len(x[i])]
        x_padded.append(F.pad(x[i], pads, value=pad_value))
    return torch.stack(x_padded)


@dataclass
class Batch:
    x: Tensor
    x_type: Tensor
    y: Tensor
    y_type: Optional[Tensor]
    dict: Optional[Dict] = None
    rationales: Optional[list] = None
    ref_cot_logprobs: Optional[list] = None
    misc = {}

    @classmethod
    def from_dict(cls, d: Dict, device=None):
        y_type = d["dec_y_type"].to(device=device) if "dec_y_type" in d else None

        if "rationales" in d:
            rationales = [
                [r.to(device=device) for r in rats] for rats in d["rationales"]
            ]
        else:
            rationales = None

        if "ref_cot_logprobs" in d:
            ref_cot_logprobs = d["ref_cot_logprobs"].to(device=device)
        else:
            ref_cot_logprobs = None

        return cls(
            d["dec_x"].to(device=device),
            d["dec_x_type"].to(device=device),
            d["dec_y"].to(device=device),
            y_type,
            d,
            rationales,
            ref_cot_logprobs,
        )

    @property
    def size(self):
        return self.x.size(0)
