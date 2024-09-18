"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from src.main import add_train_args, eval

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", type=str, help="a path to the checkpoint file"
    )
    args, train_args = parser.parse_known_args()

    if args.checkpoint_path is None:
        train_parser = ArgumentParser()
        add_train_args(train_parser)
        cfg = train_parser.parse_args(train_args)
    else:
        cfg = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))["cfg"]
        if len(train_args) > 0:
            train_parser = ArgumentParser()
            add_train_args(train_parser)
            cfg = train_parser.parse_args(train_args, namespace=cfg)
        cfg.log_plot = False
        cfg.distributed = False
        cfg.checkpoint = args.checkpoint_path
        cfg.init_model_from = None

    eval(cfg)
