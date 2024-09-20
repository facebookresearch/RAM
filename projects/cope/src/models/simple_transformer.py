"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser, BooleanOptionalAction
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils import stack_with_pad
from .base import Model
from .relative_position import RelPosEmb
from .transformer import DecoderCore


def add_args(parser: ArgumentParser):
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--hid-sz", type=int, default=128)
    parser.add_argument("--ff-sz", type=int)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument(
        "--emb-tie",
        action=BooleanOptionalAction,
        default=False,
        help="tie input and output embeddings",
    )
    parser.add_argument(
        "--pos-emb",
        choices=["none", "abs", "rel", "abs+rel", "rope", "cope", "cope+rel"],
        default="abs",
    )
    parser.add_argument(
        "--rel-pos-max",
        type=int,
        help="set the number position embeddings to be less than the block size",
    )
    parser.add_argument(
        "--rel-pos-extend",
        action=BooleanOptionalAction,
        default=False,
        help="for far away tokens that is outside the position max, use the last position embedding",
    )
    parser.add_argument(
        "--post-norm",
        action=BooleanOptionalAction,
        default=False,
        help="do post-norm instead of pre-norm",
    )
    parser.add_argument(
        "--memory-len",
        type=int,
        help="the number of past tokens to attend. Used in the sequential mode only",
    )


def preprocess_config(cfg):
    if cfg.ff_sz is None:
        cfg.ff_sz = cfg.hid_sz * 4
    if cfg.head_dim is None:
        cfg.head_dim = cfg.hid_sz // cfg.nheads


class TransformerDecoder(Model):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        preprocess_config(self.cfg)
        self.tokenizer = tokenizer

        # Word Embeddings #
        self.embed_toks = nn.Embedding(cfg.nvocab, cfg.hid_sz)

        self.core = self._build_core()
        nn.init.normal_(self.embed_toks.weight, std=self.cfg.hid_sz**-0.5)

        if self.cfg.pos_emb in ["abs", "abs+rel"]:
            self.abs_pos_emb = nn.parameter.Parameter(
                torch.randn(1, cfg.block_size, cfg.hid_sz) * self.cfg.hid_sz**-0.5
            )
        if self.cfg.pos_emb in [
            "rel",
            "abs+rel",
            "cope+rel",
        ]:
            rel_pos_emb = RelPosEmb(
                cfg.head_dim,
                cfg.block_size - 1,
                cfg.rel_pos_max,
                cfg.rel_pos_extend,
            )
            for lay in self.core.layers:
                if hasattr(lay, "attn"):
                    lay.attn.attn.rel_pos_emb = rel_pos_emb

        if not self.cfg.post_norm:
            self.out_norm = nn.LayerNorm(cfg.hid_sz)

        self.dropout = nn.Dropout(cfg.drop)
        self.past_state = None

        self.out_mod = nn.Linear(cfg.hid_sz, cfg.nvocab, bias=False)
        if cfg.emb_tie:
            self.out_mod.weight = self.embed_toks.weight

    def _build_core(self):
        return DecoderCore(self.cfg)

    def get_word_embs(self, x):
        # x : B x L
        if x.is_floating_point():
            # x must be one-hot or multinomial distribution
            assert x.dim() == 3  # B x L x nvocab
            x = torch.matmul(x, self.embed_toks.weight)
        else:
            x = self.embed_toks(x)  # B x L x H
        if self.cfg.pos_emb in ["abs", "abs+rel"]:
            sta_t = 0 if self.past_state is None else self.past_state.get("sta_t", 0)
            x += self.abs_pos_emb[:, sta_t : sta_t + x.size(1), :]

        return x

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[Tensor] = None,
        **kargs,
    ):
        x = self.get_word_embs(x)
        x = self.dropout(x)

        if self.past_state is not None and "sta_t" in self.past_state:
            self.past_state["sta_t"] += x.size(1)

        state_core = None if self.past_state is None else self.past_state["core"]
        hidden = self.core(
            x,
            pad_mask=pad_mask,
            past_state=state_core,
        )

        if not self.cfg.post_norm:
            hidden = self.out_norm(hidden)

        out = self.out_mod(hidden)
        logprobs = F.log_softmax(out, dim=-1)

        return logprobs

    def init_seq_state(self, batch_sz: int):
        self.past_state = {}
        self.past_state["core"] = [
            {
                "memory": torch.zeros(
                    batch_sz,
                    self.cfg.memory_len,
                    self.cfg.hid_sz,
                    device=self.cfg.device,
                ),
            }
            for _ in self.core.layers
        ]

    def _init_gen_state(self):
        self.past_state = {}
        self.past_state["core"] = [
            {
                "gen_key": torch.tensor([], device=self.cfg.device),
                "gen_val": torch.tensor([], device=self.cfg.device),
            }
            for _ in self.core.layers
        ]
        self.past_state["sta_t"] = 0

    def generate(
        self,
        prompts: List[Tensor],
        do_sample=False,
        max_new_tokens=10,
        eos_token_id=-1,  # not used!
    ):
        assert eos_token_id == -1
        x = stack_with_pad(prompts, self.tokenizer.pad_ind, from_left=True)
        pad_mask = x == self.tokenizer.pad_ind

        self._init_gen_state()  # make it cache previous computations

        assert self.cfg.pos_emb == "rel", "only supports rel"
        out = self.forward(x, pad_mask)

        generation = []
        logprob = []
        x_step = None
        for t in range(max_new_tokens):
            if t > 0:
                out = self.forward(x_step, pad_mask)  # type: ignore
            out = out[:, -1]  # take only last token
            if do_sample:
                out_probs = out.exp()  # B x nvocab
                x_step = torch.multinomial(out_probs, 1)  # B x 1
            else:
                x_step = out.max(dim=-1)[1].unsqueeze(-1)  # B x 1
            pad_mask = F.pad(pad_mask, [0, 1], value=False)

            generation.append(x_step)
            logprob.append(out)

        # need to do this to avoid keep using cached key and vals
        self.past_state = None

        generation = torch.cat(generation, dim=1)
        return generation, logprob
