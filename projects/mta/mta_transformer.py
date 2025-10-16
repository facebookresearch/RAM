"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from lingua.transformer import (
    FeedForward,
    InitStdFactor,
    RotaryEmbedding,
    apply_rotary_emb,
    repeat_kv,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from xformers.ops import AttentionBias, fmha

logger = logging.getLogger()

flex_attention_comp = torch.compile(flex_attention)


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        memory_efficient: bool = False,
        layer_norm: bool = False,
        layer_id: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
        self.layer_norm = layer_norm
        self.layer_id = torch.tensor(
            layer_id, requires_grad=False, device="cuda"
        )  # https://github.com/pytorch/pytorch/issues/120934

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        if self.layer_norm:
            # difformer rescale
            lambda_init = 0.8 - 0.6 * math.exp(-0.3 * self.layer_id)
            output = output * (1 - lambda_init)
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


@dataclass
class MTATransformerArgs:
    use_mta: Optional[bool] = False  # use MTA attention
    query_kernel_size: Optional[int] = 1  # convolutional kernel size in query dimension
    key_kernel_size: Optional[int] = 1  # convolutional kernel size in key dimension
    pad_key: str = (
        "both"  # how to pad the key dimension, can be on of ["left", "both", "right"]
    )
    mta_layers: Optional[str] = None  # optional parameter to specify layers with MTA
    # specify if used in combination with pre-sm
    after_sm_query_kernel_size: Optional[int] = (
        None  # convolutional kernel size in query dimension
    )
    after_sm_key_kernel_size: Optional[int] = (
        None  # convolutional kernel size in key dimension
    )
    init_method: str = (
        "identity"  # how to initialize kernel weights; ["const", "uniform", "normal"]
    )
    head_kernel_size: Optional[int] = (
        None  # kernel size in head dimension to be applied *after* softmax
    )
    group_norm: bool = False
    layer_norm_rescale: bool = False  # https://arxiv.org/pdf/2502.05795 in group norm
    curse_norm: bool = (
        False  # https://arxiv.org/pdf/2502.05795 in attn and feed forward
    )
    pre_sm_linear_head: bool = (
        False  # pre-softmax linear head; equivalent to head conv hernel with size n_heads, but faster
    )
    post_sm_linear_head: bool = (
        False  # post-softmax linear head; equivalent to head conv hernel with size n_heads, but faster
    )
    add_gating: bool = False  # add gating mechanism to group norm
    gate_1d: bool = True


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

    cache_attention_maps: bool = False

    mta: MTATransformerArgs = field(default_factory=MTATransformerArgs)
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        cache_attention_maps: Optional[bool],
        max_seq_len: int,
        mta: Optional[MTATransformerArgs],
        dropout: float,
        layer_id: Optional[int],
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )
        # Adding new cache for token/token attention scores: assumes batch size 1
        self.cache_attention_maps = cache_attention_maps
        if self.cache_attention_maps:
            self.cache_attn = torch.zeros(
                1, self.n_heads, max_seq_len, max_seq_len
            ).cuda()
        else:
            self.cache_attn = None

        # mta args
        self.use_mta = mta.use_mta
        if self.use_mta:
            # pre-sm
            self.mta_kernel = None
            if mta.query_kernel_size is not None and mta.key_kernel_size is not None:
                self.mta_kernel = torch.nn.parameter.Parameter(
                    torch.empty(
                        self.n_heads, 1, mta.query_kernel_size, mta.key_kernel_size
                    )
                )

            self.pre_sm_linear_head = mta.pre_sm_linear_head
            if self.pre_sm_linear_head:
                self.wpsm = nn.Linear(
                    n_heads,
                    n_heads,
                    bias=False,
                )
            # post-sm
            self.mta_kernel_after_sm = None
            if (
                mta.after_sm_query_kernel_size is not None
                and mta.after_sm_key_kernel_size is not None
            ):
                self.mta_kernel_after_sm = torch.nn.parameter.Parameter(
                    torch.empty(
                        self.n_heads,
                        1,
                        mta.after_sm_query_kernel_size,
                        mta.after_sm_key_kernel_size,
                    )
                )

            self.head_kernel_size = mta.head_kernel_size
            self.post_sm_linear_head = mta.post_sm_linear_head
            assert not (
                self.post_sm_linear_head and self.head_kernel_size is not None
            ), "linear head can not be combined with head conv"

            if self.head_kernel_size is not None:
                assert self.n_heads % mta.head_kernel_size == 0
                self.head_kernel = torch.nn.parameter.Parameter(
                    torch.empty(
                        self.n_heads // mta.head_kernel_size,
                        mta.head_kernel_size,
                        mta.head_kernel_size,
                    )
                )
            elif self.post_sm_linear_head:
                self.wposm = nn.Linear(
                    n_heads,
                    n_heads,
                    bias=False,
                )

            # common
            self.pad_key = mta.pad_key
            self.mta_init_method = mta.init_method

        self.dropout = dropout

        self.group_norm = mta.group_norm
        self.add_gating = mta.add_gating
        self.layer_norm_rescale = mta.layer_norm_rescale
        self.layer_id = layer_id
        if self.group_norm:
            self.subln = RMSNorm(
                self.head_dim,
                eps=1e-5,
                elementwise_affine=True,
                layer_norm=self.layer_norm_rescale,
                layer_id=layer_id,
            )
        if self.add_gating:
            if mta.gate_1d:
                self.gate = nn.Linear(self.head_dim, 1)
            else:
                self.gate = nn.Linear(self.head_dim, self.head_dim)

    def normalize_attention(self, att_type: str = "soft", theta: float = 1.0):
        normalizer = {
            "soft": lambda a: F.softmax(a.float() / theta, dim=-1),
            "logsoft": lambda a: F.log_softmax(a.float() / theta, dim=-1),
            "hard": lambda a: F.normalize(
                torch.zeros(a.shape, device=a.device).scatter(
                    dim=-1, index=a.argmax(dim=-1).unsqueeze(dim=-1), value=1.0
                ),
                dim=-1,
            ),
        }
        if att_type not in normalizer.keys():
            raise ValueError(
                f"Normalizer `{att_type}` not supported, supported types are: {list(normalizer.keys())}"
            )
        return normalizer[att_type]

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        cache_len = seq_len
        chunk_start_ids = None
        if hasattr(self, "kv_cache"):
            xk, xv, xq = self.kv_cache.update(xk, xv, xq, tok_idx)
            # cache update also shifted token IDs by generation length
            chunk_start_ids = torch.unique(self.kv_cache.offset)
            cache_len = xk.size(1)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if self.use_mta:
            xq, xk, xv = map(
                lambda e: e.transpose(1, 2), (xq, xk, xv)
            )  # B S H D -> B H S D
            mask = self._update_mask(mask=mask, bsz=bsz, xq=xq, xk_size=xk.size(-2))

            scores = torch.matmul(xq, xk.transpose(2, 3)) * torch.rsqrt(
                torch.tensor(self.head_dim, requires_grad=False, device="cuda")
            )

            if self.mta_kernel is not None:
                # pre-sm q-k MTA
                scores = self._mta_convolution(
                    scores=scores,
                    mask=mask,
                    chunk_start_ids=chunk_start_ids,
                    kernel=self.mta_kernel,
                )
            if self.pre_sm_linear_head:
                # pre-sm head MTA
                scores = self.wpsm(scores.transpose(1, -1)).transpose(1, -1)

            # now softmax
            scores = scores + mask

            scores = self.normalize_attention(att_type="soft")(scores).type_as(xq)

            # post-sm q-k MTA
            if self.mta_kernel_after_sm is not None:
                scores = self._mta_convolution(
                    scores=scores,
                    mask=mask,
                    chunk_start_ids=chunk_start_ids,
                    kernel=self.mta_kernel_after_sm,
                )
                scores = torch.where(mask == float("-inf"), 0.0, scores)
            # post-sm head MTA
            if self.head_kernel_size is not None:
                scores = self._head_convolution(
                    scores=scores, bsz=bsz, seq_len=cache_len
                )
            elif self.post_sm_linear_head:
                # post-sm linear head
                scores = self.wposm(scores.transpose(1, -1)).transpose(1, -1)

            scores = F.dropout(scores, p=self.dropout, training=self.training)

            if self.cache_attention_maps:
                # count was increased by slen above, adjusting for cache and projected count
                self.cache_attn[0, :, :cache_len, :cache_len] = scores
            output = torch.matmul(scores, xv)
            output = output.transpose(1, 2)
        else:
            if attn_impl == "flex_attention":
                assert mask is None or isinstance(mask, BlockMask)
                assert self.dropout == 0.0, "flex_attention does not support dropout"
                xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
                output = flex_attention_comp(xq, xk, xv, block_mask=mask)
                output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

            elif attn_impl == "fmha":
                assert mask is None or isinstance(mask, AttentionBias)
                output = fmha.memory_efficient_attention(
                    xq, xk, xv, attn_bias=mask, p=self.dropout
                )
                # This uses B S H D instead of B H S D of pytorch

            elif attn_impl == "sdpa":
                xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
                assert mask is None or isinstance(mask, (str, torch.Tensor))
                is_causal = (mask == "causal") if isinstance(mask, str) else False
                mask = mask if isinstance(mask, torch.Tensor) else None
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    is_causal=is_causal,
                    attn_mask=mask,
                    dropout_p=self.dropout,
                )
                output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
            else:
                raise NotImplementedError(
                    f"Attention implementation {attn_impl} not supported"
                )

        if self.group_norm:
            output = self.subln(output)

        if self.add_gating:
            output = output * torch.sigmoid(self.gate(output))

        # if we have updated XQ length from cache, we need to cut it back in output
        if tok_idx is not None:
            output = output[:, tok_idx + self.kv_cache.offset, :, :]
        else:
            assert output.size(1) == seq_len

        output = self.wo(output.reshape(output_shape))

        return output

    def _update_mask(
        self, mask: torch.tensor, bsz: int, xq: torch.tensor, xk_size: int
    ):
        if not isinstance(mask, torch.Tensor):
            # causal mask
            mask = torch.full((xq.size(-2), xk_size), float("-inf"), device=xq.device)
            mask = torch.triu(mask, diagonal=1).type_as(xq)
            mask = mask.repeat(bsz, self.n_heads, 1, 1)
        else:
            # generation task, mask is provided and reflects concatenated docs
            if mask.dtype == torch.bool:
                mask = torch.where(mask, 0.0, float("-inf")).to(xq.dtype)
            if mask.dtype != xq.dtype:
                mask = mask.type(xq.dtype)
            if len(mask.shape) == 2:
                assert mask.size(0) == bsz
                mask = mask.repeat(1, self.n_heads, xq.size(-2), 1)
                mask_i, mask_j = torch.triu_indices(xq.size(-2), xk_size, offset=1)
                mask[:, :, mask_i, mask_j] = float("-inf")
            else:
                if mask.size(0) == 1 and mask.size(1) == 1:
                    # in prefilling mask is defined for 1 head
                    mask = mask.repeat(bsz, self.n_heads, 1, 1)
        return mask

    def _mta_convolution(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        chunk_start_ids: torch.Tensor,
        kernel: torch.Tensor,
    ):
        # here we want masked scores to be 0, so they don't affect convolution
        scores[mask == float("-inf")] = 0  # (bs, n_local_heads, slen, cache_len)
        # now apply convolution
        n_loc_heads, head_kernel_size, qdim, kdim = kernel.shape
        assert n_loc_heads == self.n_heads

        # in generation, we concatenate multiple sequences. This is a problem for MTA, because it's kernel will cover
        # multiple sequences! So let's cut the input here
        cuts = []
        if chunk_start_ids is not None:
            # cut input in chunks
            for i in range(chunk_start_ids.size(0) - 1):
                cuts.append(
                    scores[
                        :,
                        :,
                        chunk_start_ids[i].item() : chunk_start_ids[i + 1].item(),
                        chunk_start_ids[i].item() : chunk_start_ids[i + 1].item(),
                    ]
                )
            cuts.append(
                scores[:, :, chunk_start_ids[-1].item() :, chunk_start_ids[-1].item() :]
            )
        else:
            cuts = [scores]
        for i, cut in enumerate(cuts):
            # manually pad; padding dimensions are reversed wrt kernel sizes: first comes width then height in the 2D case.
            # pad last dim by (kdim-1, 0) and 2nd to last by (qdim-1, 0)
            if self.pad_key == "left":
                scores_padded = torch.nn.functional.pad(
                    cut, (kdim - 1, 0, qdim - 1, 0), value=0.0
                )
            elif self.pad_key == "right":
                scores_padded = torch.nn.functional.pad(
                    cut, (0, kdim - 1, qdim - 1, 0), value=0.0
                )
            elif self.pad_key == "both":
                assert (kdim - 1) % 2 == 0
                scores_padded = torch.nn.functional.pad(
                    cut, ((kdim - 1) // 2, (kdim - 1) // 2, qdim - 1, 0), value=0.0
                )

            conv_cut = F.conv2d(
                scores_padded,
                kernel,
                padding=0,
                groups=self.n_heads // head_kernel_size,
            )
            del scores_padded
            # now glue it back
            if chunk_start_ids is None:
                scores = conv_cut
            elif i < chunk_start_ids.size(0) - 1:
                scores[
                    :,
                    :,
                    chunk_start_ids[i].item() : chunk_start_ids[i + 1].item(),
                    chunk_start_ids[i].item() : chunk_start_ids[i + 1].item(),
                ] = conv_cut
            else:
                scores[
                    :, :, chunk_start_ids[i].item() :, chunk_start_ids[i].item() :
                ] = conv_cut

        return scores

    def _head_convolution(self, scores, bsz, seq_len):
        scores = scores.reshape(
            bsz,
            self.n_heads // self.head_kernel_size,
            self.head_kernel_size,
            seq_len,
            -1,
        )
        scores_new = torch.empty_like(scores)
        for i in range(self.n_heads // self.head_kernel_size):
            scores_new[:, i] = torch.matmul(
                scores[:, i].transpose(1, -1), self.head_kernel[i]
            ).transpose(1, -1)

        scores = scores_new.reshape(bsz, self.n_heads, seq_len, -1)
        return scores

    def reset_mta_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        if self.group_norm:
            self.subln.reset_parameters()

        if self.use_mta:
            if self.mta_kernel is not None:
                if self.mta_init_method == "uniform":
                    torch.nn.init.uniform_(self.mta_kernel.data, a=-1.0, b=1.0)
                elif self.mta_init_method == "normal":
                    init_std = 0.3
                    torch.nn.init.uniform_(
                        self.mta_kernel.data,
                        mean=0.0,
                        std=init_std,
                        a=-3 * init_std,
                        b=3 * init_std,
                    )
                elif self.mta_init_method == "diagonal":
                    # diagonal of the form
                    # 1 0 0 0 0
                    # 0 1 0 0 0
                    # 0 0 1 0 0
                    with torch.no_grad():
                        diagonal_kernel = torch.ones(self.mta_kernel.data.shape)
                        _, _, A, B = self.mta_kernel.data.shape
                        diagonal = (B + 1) // 2 - A
                        diagonal_kernel = torch.tril(diagonal_kernel, diagonal=diagonal)
                        diagonal_kernel = torch.triu(diagonal_kernel, diagonal=diagonal)
                        diagonal_kernel = torch.distributed.tensor.distribute_tensor(
                            diagonal_kernel,
                            device_mesh=self.mta_kernel.data.device_mesh,
                            placements=self.mta_kernel.data.placements,
                        )
                        self.mta_kernel.data.copy_(diagonal_kernel)
                elif self.mta_init_method == "identity":
                    assert self.pad_key == "both"
                    # identity kernel of the form
                    # 0 0 0 0 0
                    # 0 0 0 0 0
                    # 0 0 1 0 0
                    with torch.no_grad():
                        nheads, head_sz, query_sz, key_sz = self.mta_kernel.data.shape
                        identity_kernel = torch.zeros(
                            nheads, head_sz, query_sz, key_sz
                        ).cuda()
                        if head_sz == 1:
                            identity_kernel[:, :, -1, key_sz // 2] = 1.0
                        else:
                            # it is bit complicated with head conv
                            # weight to other heads should be zero
                            identity_kernel = identity_kernel.view(
                                nheads // head_sz, head_sz, head_sz, query_sz, key_sz
                            )
                            for i in range(head_sz):
                                identity_kernel[:, i, i, -1, key_sz // 2] = 1.0
                            identity_kernel = identity_kernel.view(
                                nheads, head_sz, query_sz, key_sz
                            )
                        identity_kernel = torch.distributed.tensor.distribute_tensor(
                            identity_kernel,
                            device_mesh=self.mta_kernel.data.device_mesh,
                            placements=self.mta_kernel.data.placements,
                        )
                        self.mta_kernel.data.copy_(identity_kernel)
                elif self.mta_init_method == "const":
                    self.mta_kernel.data.fill_(0.3)
                else:
                    raise ValueError(
                        f"Unsopperted mta_init_method: {self.mta_init_method}"
                    )

            if self.mta_kernel_after_sm is not None:
                assert self.mta_init_method == "identity"
                assert self.pad_key == "both"
                with torch.no_grad():
                    (
                        nheads,
                        head_sz,
                        query_sz,
                        key_sz,
                    ) = self.mta_kernel_after_sm.data.shape
                    identity_kernel = torch.zeros(
                        nheads, head_sz, query_sz, key_sz
                    ).cuda()
                    if head_sz == 1:
                        identity_kernel[:, :, -1, key_sz // 2] = 1.0
                    else:
                        # it is bit complicated with head conv
                        # weight to other heads should be zero
                        identity_kernel = identity_kernel.view(
                            nheads // head_sz, head_sz, head_sz, query_sz, key_sz
                        )
                        for i in range(head_sz):
                            identity_kernel[:, i, i, -1, key_sz // 2] = 1.0
                        identity_kernel = identity_kernel.view(
                            nheads, head_sz, query_sz, key_sz
                        )
                    identity_kernel = torch.distributed.tensor.distribute_tensor(
                        identity_kernel,
                        device_mesh=self.mta_kernel_after_sm.data.device_mesh,
                        placements=self.mta_kernel_after_sm.data.placements,
                    )
                    self.mta_kernel_after_sm.data.copy_(identity_kernel)

            if self.pre_sm_linear_head:
                identity_kernel = torch.eye(self.n_heads).cuda()
                identity_kernel = torch.distributed.tensor.distribute_tensor(
                    identity_kernel,
                    device_mesh=self.wpsm.weight.data.device_mesh,
                    placements=self.wpsm.weight.data.placements,
                )
                self.wpsm.weight.data.copy_(identity_kernel)

            if self.post_sm_linear_head:
                identity_kernel = torch.eye(self.n_heads).cuda()
                identity_kernel = torch.distributed.tensor.distribute_tensor(
                    identity_kernel,
                    device_mesh=self.wpsm.weight.data.device_mesh,
                    placements=self.wpsm.weight.data.placements,
                )
                self.wposm.weight.data.copy_(identity_kernel)

            elif self.head_kernel_size is not None:
                if self.mta_init_method == "identity":
                    with torch.no_grad():
                        a, b, c = self.head_kernel.data.shape
                        assert b == c
                        identity_kernel = torch.eye(b).cuda()
                        identity_kernel = identity_kernel.repeat(a, 1, 1)
                        identity_kernel = torch.distributed.tensor.distribute_tensor(
                            identity_kernel,
                            device_mesh=self.head_kernel.data.device_mesh,
                            placements=self.head_kernel.data.placements,
                        )
                        self.head_kernel.data.copy_(identity_kernel)
                else:
                    raise ValueError(
                        f"Unsopperted mta_init_method for head convolution: {self.mta_init_method}"
                    )

            if self.add_gating:
                nn.init.trunc_normal_(
                    self.gate.weight,
                    std=init_std / factor,
                    a=-3 * init_std,
                    b=3 * init_std,
                )

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

        self.reset_mta_parameters(init_std, factor)


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs, layer_id: Optional[int]):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            cache_attention_maps=args.cache_attention_maps,
            max_seq_len=args.max_seqlen,
            dropout=args.dropout,
            mta=args.mta,
            layer_id=layer_id,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
            layer_norm=args.mta.curse_norm,
            layer_id=layer_id,
        )
        self.ffn_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
            layer_norm=args.mta.curse_norm,
            layer_id=layer_id,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

    def init_mta_weights(self, init_std=None, factor=1.0):
        self.attention.reset_mta_parameters(init_std, factor)


class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )

        self.layers = nn.ModuleList()

        self.mta_layers = None
        if args.mta.use_mta:
            if args.mta.mta_layers is None:
                self.mta_layers = list(range(args.n_layers))
            else:
                self.mta_layers = [int(l_id) for l_id in args.mta.mta_layers.split(",")]

        for layer_id in range(args.n_layers):
            if self.mta_layers is not None and layer_id in self.mta_layers:
                logger.info(f"Initializing MTA transformer block at layer {layer_id}")
                self.layers.append(TransformerBlock(args, layer_id=layer_id))
            else:
                args_without_mta = copy.deepcopy(args)
                if (
                    args.mta.head_kernel_size is None
                    and not args.mta.pre_sm_linear_head
                    and not args.mta.post_sm_linear_head
                ):
                    # regular attention
                    logger.info(
                        f"Initializing regular transformer block at layer {layer_id}"
                    )
                    args_without_mta.mta.use_mta = False
                else:
                    # remove q-k convolution before sm
                    logger.info(f"No key-query convolution at layer {layer_id}")
                    args_without_mta.mta.query_kernel_size = None
                    args_without_mta.mta.after_sm_query_kernel_size = None

                self.layers.append(
                    TransformerBlock(args_without_mta, layer_id=layer_id)
                )

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        verbouse: bool = False,
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
            if layer.attention.use_mta and verbouse:
                if layer.attention.mta_kernel is not None:
                    logger.info(f"MTA kernel in layer {i}:")
                    logger.info(layer.attention.mta_kernel)

        # Or to save kernels for investigation:
        # mta_kernel = {}
        # head_kernel = {}
        # for i, layer in enumerate(self.layers):
        #     if layer.attention.mta_kernel is not None:
        #         mta_kernel[f"Layer {i+1}"] = layer.attention.mta_kernel
        # torch.save(head_kernel, "head_kernel_830M_head_conv_2_3d_mta_4l_grn_rescale.pt")
        # torch.save(mta_kernel, "kernel_weights_830M_head_conv_2_3d_mta_4l_grn_rescale.pt")
        # raise ValueError
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)

    def init_mta_weights(self):
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_mta_weights(self.init_base_std, factor)
