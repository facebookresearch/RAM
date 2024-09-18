"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cope.context_position import ContextPosSelfAttn


def pre_norm(cfg, norm, x):
    if cfg.post_norm:
        return x, x
    else:
        return x, norm(x)


def post_norm(cfg, norm, residual, x):
    if cfg.post_norm:
        return norm(residual + x)
    else:
        return residual + x


class SelfAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg.drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # query, key, val : B x L x H
        # attn_mask : B x L

        attn_logits = torch.bmm(query, key.transpose(-1, -2))  # B x L x L

        if self.cfg.pos_emb in ["rel", "abs+rel"] and hasattr(self, "rel_pos_emb"):
            # relative position only works for self-attention, not for enc-dec attention
            attn_logits += self.rel_pos_emb(query, key)

        if attn_mask is not None:
            attn_logits += attn_mask.log()

        attn_logits /= math.sqrt(self.cfg.head_dim)
        attn = torch.softmax(attn_logits, dim=-1)
        self.attn_saved = attn

        attn = self.dropout(attn)

        out = torch.bmm(attn, val)  # B x L x H
        return out


class MultiHeadAttn(nn.Module):
    def __init__(self, cfg, lay_ind):
        super().__init__()
        self.cfg = cfg
        if cfg.pos_emb in ["cope", "cope+rel"]:
            self.attn = ContextPosSelfAttn(cfg, lay_ind)
            if cfg.cope_sep_key:
                self.proj_key_cope = nn.Linear(cfg.hid_sz, cfg.hid_sz)
        else:
            self.attn = SelfAttn(cfg)
        self.proj_query = nn.Linear(cfg.hid_sz, cfg.hid_sz)
        self.proj_key = nn.Linear(cfg.hid_sz, cfg.hid_sz)
        self.proj_val = nn.Linear(cfg.hid_sz, cfg.hid_sz)
        self.proj_out = nn.Linear(cfg.hid_sz, cfg.hid_sz)
        if not self.cfg.post_norm:
            self.proj_out.weight.data.div_(math.sqrt(cfg.nlayers * 2))
        self.dropout = nn.Dropout(cfg.drop)

    def split_heads(self, x: torch.Tensor):
        # x : B x L x H
        B, L, _ = x.size()
        x = x.view(B, L, self.cfg.nheads, self.cfg.head_dim)
        x = x.transpose(1, 2).contiguous()  # B x nheads x L x head_dim
        x = x.view(-1, L, self.cfg.head_dim)
        return x

    def merge_heads(self, x: torch.Tensor):
        # x : B_nheads x L x head_dim
        _, L, _ = x.size()
        x = x.view(-1, self.cfg.nheads, L, self.cfg.head_dim)
        x = x.transpose(1, 2).contiguous()  # B x L x nheads x head_dim
        x = x.view(-1, L, self.cfg.hid_sz)  # B x L x H
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_state=None,
    ):
        # x : B x L x H
        query = self.split_heads(self.proj_query(query))
        if self.cfg.cope_sep_key:
            key_cope = self.split_heads(self.proj_key_cope(key))

        key = self.split_heads(self.proj_key(key))
        val = self.split_heads(self.proj_val(val))
        if past_state is not None and "gen_key" in past_state:
            # used during generation to cache previous tokens
            key = torch.cat([past_state["gen_key"], key], dim=1)
            past_state["gen_key"] = key
            val = torch.cat([past_state["gen_val"], val], dim=1)
            past_state["gen_val"] = val

        if self.cfg.pos_emb in ["cope", "cope+rel"]:
            if not self.cfg.cope_sep_key:
                key_cope = key
            out = self.attn(query, key, key_cope, val, attn_mask=attn_mask)
        else:
            out = self.attn(query, key, val, attn_mask=attn_mask)
        out = self.merge_heads(out)
        out = self.proj_out(out)
        out = self.dropout(out)
        return out


class Feedforward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.hid_sz, cfg.ff_sz)
        self.fc2 = nn.Linear(cfg.ff_sz, cfg.hid_sz)
        if not self.cfg.post_norm:
            self.fc2.weight.data.div_(math.sqrt(cfg.nlayers * 2))
        self.dropout = nn.Dropout(cfg.drop)
        self.dropout2 = nn.Dropout(cfg.drop)

    def forward(self, x: torch.Tensor):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout2(h)
        return h


class EncoderLayer(nn.Module):
    def __init__(self, cfg, lay_ind):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadAttn(cfg, lay_ind)
        self.attn_norm = nn.LayerNorm(cfg.hid_sz)
        self.ff = Feedforward(cfg)
        self.ff_norm = nn.LayerNorm(cfg.hid_sz)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        residual, x = pre_norm(self.cfg, self.attn_norm, x)
        y = self.attn(x, x, x, attn_mask=attn_mask)
        x = post_norm(self.cfg, self.attn_norm, residual, y)

        residual, x = pre_norm(self.cfg, self.ff_norm, x)
        y = self.ff(x)
        x = post_norm(self.cfg, self.ff_norm, residual, y)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg, lay_ind, with_encoder=False):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadAttn(cfg, lay_ind)
        self.attn_norm = nn.LayerNorm(cfg.hid_sz)
        self.enc_attn = None
        if with_encoder:
            self.enc_attn = MultiHeadAttn(cfg)
            self.enc_attn_norm = nn.LayerNorm(cfg.hid_sz)
        self.ff = Feedforward(cfg)
        self.ff_norm = nn.LayerNorm(cfg.hid_sz)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        enc_attn_mask: Optional[torch.Tensor] = None,
        past_state=None,
    ):
        residual, x = pre_norm(self.cfg, self.attn_norm, x)
        memory = x
        if past_state is not None and "memory" in past_state:
            # used in sequential training to cache previous blocks
            assert past_state["memory"].size(1) == self.cfg.memory_len
            memory = torch.cat([past_state["memory"], x], dim=1)
            past_state["memory"] = memory[:, -self.cfg.memory_len :].detach()
        y = self.attn(x, memory, memory, attn_mask=attn_mask, past_state=past_state)
        x = post_norm(self.cfg, self.attn_norm, residual, y)

        if self.enc_attn is not None:
            residual, x = pre_norm(self.cfg, self.enc_attn_norm, x)
            y = self.enc_attn(x, enc_out, enc_out, attn_mask=enc_attn_mask)
            x = post_norm(self.cfg, self.enc_attn_norm, residual, y)

        residual, x = pre_norm(self.cfg, self.ff_norm, x)
        y = self.ff(x)
        x = post_norm(self.cfg, self.ff_norm, residual, y)
        return x


def pad2attn_mask(pad_mask, nheads, query_len=None, self_attn=True):
    # pad_mask: B x L
    attn_mask = None
    if pad_mask is not None:
        B, L = pad_mask.size()
        query_len = L if query_len is None else query_len
        attn_mask = torch.ones(B, query_len, L, device=pad_mask.device)
        attn_mask = attn_mask.masked_fill(pad_mask.unsqueeze(-2), 0)
        if self_attn:
            # allow attention to the token itself to avoid NaN values for pad tokens
            assert query_len <= L
            attn_mask[:, :, -query_len:] += torch.eye(
                query_len, device=pad_mask.device
            ).unsqueeze(0)
            attn_mask.clamp_(max=1)
        # duplicate for heads
        attn_mask = attn_mask.repeat_interleave(nheads, dim=0)
    return attn_mask


class EncoderCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for lay_ind in range(cfg.nlayers):
            self.layers.append(EncoderLayer(cfg, lay_ind))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        # x : B x L x H
        attn_mask = pad2attn_mask(pad_mask, self.cfg.nheads)
        for lay in self.layers:
            x = lay(x, attn_mask=attn_mask)
        return x


class DecoderCore(nn.Module):
    def __init__(self, cfg, with_encoder=False):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        self.with_encoder = with_encoder
        for lay_ind in range(cfg.nlayers):
            self.layers.append(DecoderLayer(cfg, lay_ind, with_encoder))

        if cfg.pos_emb in ["cope", "cope+rel"] and cfg.cope_shared:
            src = self.layers[self.cfg.cope_layers - 1].attn.attn
            for lay_ind in range(self.cfg.cope_layers, cfg.nlayers):
                tgt = self.layers[lay_ind].attn.attn
                if not tgt.rel_only:
                    tgt.pos_emb = src.pos_emb

    def forward(
        self,
        x: torch.Tensor,
        enc_out: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        enc_pad_mask: Optional[torch.Tensor] = None,
        past_state=None,
    ):
        # x : B x L x H
        B, L, _ = x.size()

        attn_mask = pad2attn_mask(pad_mask, self.cfg.nheads, query_len=L)
        enc_attn_mask = None
        if self.with_encoder:
            enc_attn_mask = pad2attn_mask(
                enc_pad_mask, self.cfg.nheads, query_len=L, self_attn=False
            )
        if L == 1:
            # don't need causal masking for 1 token (i.e. during generation)
            pass
        else:
            causal_mask = torch.ones(L, L, device=x.device).tril().unsqueeze(0)
            attn_mask = causal_mask if attn_mask is None else attn_mask * causal_mask

        for li, lay in enumerate(self.layers):
            layer_past_state = None if past_state is None else past_state[li]
            x = lay(
                x,
                enc_out=enc_out,
                attn_mask=attn_mask,
                enc_attn_mask=enc_attn_mask,
                past_state=layer_past_state,
            )
        return x
