"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional

import torch
import torch.nn as nn


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--cope-sep-key",
        action=BooleanOptionalAction,
        default=False,
        help="use separate key projection for computing gates",
    )
    parser.add_argument(
        "--cope-val-gates",
        action=BooleanOptionalAction,
        default=False,
        help="use values for computing gates",
    )
    parser.add_argument(
        "--cope-npos",
        type=int,
        help="the number of positions. If not set, will use block size + 1.",
    )
    parser.add_argument(
        "--cope-nodiv",
        action=BooleanOptionalAction,
        default=False,
        help="do not divide by sqrt(div)",
    )
    parser.add_argument(
        "--cope-layers",
        type=int,
        default=1,
        help="use CoPE only every K layers.",
    )
    parser.add_argument(
        "--cope-shared",
        action=BooleanOptionalAction,
        default=False,
        help="share embeddings accross layers",
    )


class ContextPosSelfAttn(nn.Module):
    """Self-attention layer with contextual position embedding (CoPE)"""

    def __init__(self, cfg, lay_ind):
        super().__init__()
        self.cfg = cfg
        self.lay_ind = lay_ind
        self.dropout = nn.Dropout(cfg.drop)
        self.rel_only = False
        if self.cfg.cope_layers > 1:
            assert cfg.pos_emb == "cope+rel"
            if (self.lay_ind + 1) % self.cfg.cope_layers > 0:
                # do not use cope on this layer
                self.rel_only = True

        if cfg.cope_npos is not None:
            self.npos = cfg.cope_npos
        else:
            # need 1 extra position because position 0 is possible
            self.npos = cfg.block_size + 1

        if not self.rel_only:
            self.pos_emb = nn.parameter.Parameter(
                torch.zeros(1, cfg.head_dim, self.npos)
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_cope: torch.Tensor,
        val: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # query, key, val : B x L x H
        B, L, H = key.size()
        # attn_mask : B x L

        if self.rel_only:
            pos_logits = 0
            gate_logits = None
        else:
            if self.cfg.cope_val_gates:
                gate_logits = torch.bmm(query, val.transpose(-1, -2))  # B x L(q) x L(k)
            else:
                gate_logits = torch.bmm(
                    query, key_cope.transpose(-1, -2)
                )  # B x L(q) x L(k)
            gates = torch.sigmoid(gate_logits / math.sqrt(self.cfg.head_dim))
            if attn_mask is not None:
                gates = gates * attn_mask
            positions = gates.flip(-1).cumsum(dim=-1).flip(-1)  # B x L x L
            positions = positions.clamp(max=self.npos - 1)

            # NOW what to do with these fractional positions?

            # let's compute for discrete fixed positions (1,2, .., T) first
            pos_logits_fixed = torch.matmul(query, self.pos_emb)  # B x L x npos

            # now we need to intrapolate floor(p) and ceil(p) for position p
            positions_ceil = positions.ceil().long()  # yes, no gradient here
            positions_floor = positions.floor().long()  # yes, no gradient here
            pos_logits_ceil = pos_logits_fixed.gather(-1, positions_ceil)  # B x L x L
            pos_logits_floor = pos_logits_fixed.gather(-1, positions_floor)  # B x L x L

            # almost there, need to do weighted sum of these two
            pos_ceil_weight = positions - positions_floor  # this is differentiable
            pos_logits = pos_logits_ceil * pos_ceil_weight + pos_logits_floor * (
                1 - pos_ceil_weight
            )

        if self.cfg.cope_sep_key or self.cfg.cope_val_gates or gate_logits is None:
            attn_logits = torch.bmm(query, key.transpose(-1, -2))  # B x L x L
        else:
            attn_logits = gate_logits

        if self.cfg.pos_emb == "cope+rel":
            # relative position only works for self-attention, not for enc-dec attention
            attn_logits += self.rel_pos_emb(query, key)

        if self.cfg.cope_nodiv:
            attn_logits /= math.sqrt(self.cfg.head_dim)
            attn_logits += pos_logits
        else:
            attn_logits += pos_logits
            attn_logits /= math.sqrt(self.cfg.head_dim)

        if attn_mask is not None:
            attn_logits += attn_mask.log()

        attn = torch.softmax(attn_logits, dim=-1)
        self.attn_saved = attn

        attn = self.dropout(attn)

        out = torch.bmm(attn, val)  # B x L x H
        return out
