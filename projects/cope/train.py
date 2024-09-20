"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

from src.main import add_train_args, train

if __name__ == "__main__":
    parser = ArgumentParser()
    add_train_args(parser)
    cfg = parser.parse_args()
    train(cfg)
