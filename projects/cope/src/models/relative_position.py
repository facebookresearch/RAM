"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rel2abs(X):
    """convert from relative position to absolute"""
    # X = B x L x M
    B, L, M = X.size()
    if L == 1:
        raise NotImplementedError("TODO")
        return X
    X = F.pad(X, (0, L))  # B x L x M+L
    X = X.view(B, -1)  # B x LM+LL
    X = X[:, :-L]  # B x LM+LL-L
    X = X.view(B, L, M + L - 1)
    return X


class RelPosEmb(nn.Module):
    def __init__(
        self, emb_dim: int, past_len: int, npos_max: Optional[int], extend=False
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.total_len = past_len + 1  # add 1 for position zero
        self.extend = extend
        if npos_max is None:
            self.npos = self.total_len
        else:
            self.npos = npos_max
        self.emb = nn.parameter.Parameter(
            torch.randn(1, emb_dim, self.npos) * emb_dim**-0.5
        )

    def forward(self, query: Tensor, key: Tensor):
        B, Lq, H = query.size()
        B, Lk, H = key.size()

        pos_logits = torch.matmul(query, self.emb)  # B x Lq x npos

        if self.npos < Lk:
            # there must be fewer positions than the context size
            assert self.npos < self.total_len
            assert Lk == self.total_len
            if self.extend:
                # use the last position as for those far away positions
                extend_len = Lk - self.npos
                pos_logits = F.pad(pos_logits, [extend_len, 0], mode="replicate")

        if Lq == 1:
            # used during generation
            pass
        else:
            pos_logits = rel2abs(pos_logits)

        if pos_logits.size(2) > Lk:
            # this could happen because early tokens will not use all its rel-pos embeddings
            # trim from left to match with the number of keys
            pos_logits = pos_logits[:, :, -Lk:]
        elif pos_logits.size(2) < Lk:
            # this should not happen even if npos is set shorter than the block size
            assert False
        return pos_logits
