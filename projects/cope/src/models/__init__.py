"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser, BooleanOptionalAction, Namespace

import torch

from ..data.tokenizer import Tokenizer
from ..utils.distributed import DummyWrapper
from . import simple_transformer
from .base import Model


def add_args(parser: ArgumentParser):
    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model",
        choices=[
            "simpledec",
        ],  # todo remove flag
        default="simpledec",
    )
    group.add_argument(
        "--tokenizer",
        choices=["simple"],
        default="simple",
        help="if not specified, infer from the model",
    )  # todo remove flag
    group.add_argument(
        "--gpt2-add-special-tokens",
        nargs="+",
        help="add new special tokens to GPT2 tokenizer",
    )
    group.add_argument("--untrained", action=BooleanOptionalAction, default=False)
    group.add_argument("--drop", type=float, default=0.1)
    group.add_argument(
        "--block-size",
        type=int,
        help="the maximum number of tokens that the model can process. Most models have it predefined.",
    )
    simple_transformer.add_args(group)  # type: ignore


def set_block_size(cfg):
    assert cfg.block_size is not None  # todo simplify


def build(cfg, tokenizer: Tokenizer) -> Model:
    return simple_transformer.TransformerDecoder(cfg, tokenizer)
