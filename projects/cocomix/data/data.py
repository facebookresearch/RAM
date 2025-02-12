"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, block_size=1024, split="train"):
        self.data = np.memmap(
            os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r"
        )
        self.block_size = block_size
        self.split = split

        self.data_len = len(self.data) // self.block_size  # drop last block
        self.remain = len(self.data) % self.block_size

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.split == "train":
            if idx < self.data_len - 1:
                random_shift = random.randint(0, self.block_size)
            else:
                random_shift = random.randint(0, self.remain)
            x = torch.from_numpy(
                (
                    self.data[
                        idx * self.block_size
                        + random_shift : (idx + 1) * self.block_size
                        + random_shift
                    ]
                ).astype(np.int64)
            )
        else:
            x = torch.from_numpy(
                (self.data[idx * self.block_size : (idx + 1) * self.block_size]).astype(
                    np.int64
                )
            )
        attention_mask = torch.ones_like(x)
        return {"input_ids": x, "labels": x, "attention_mask": attention_mask}


def get_train_dataloader(cfg):
    kwargs = {}
    if cfg.dataset == "openwebtext":
        train_dataset = PreprocessedDataset(
            cfg.data_dir, block_size=cfg.block_size, split="train"
        )
    else:
        print(f"dataset [{cfg.dataset}] not supported for evaluation")
        raise NotImplementedError

    batch_size = cfg.update_batch_size // cfg.world_size
    cfg.n_epochs = (
        cfg.train_steps
        * cfg.update_batch_size
        * cfg.grad_acc_steps
        // len(train_dataset)
        + 1
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
        **kwargs,
    )
    return train_dataloader


def get_val_dataloaders(cfg):
    if cfg.val_datasets is None:
        return None

    val_dataloaders = {}
    for val_name in cfg.val_datasets:

        kwargs = {}
        if val_name == "openwebtext":
            val_dataset = PreprocessedDataset(
                cfg.data_dir, block_size=cfg.block_size, split="val"
            )
        else:
            continue

        batch_size = cfg.batch_size_eval // cfg.world_size
        val_dataloader = DataLoader(
            val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, **kwargs
        )
        val_dataloaders[val_name] = val_dataloader
    return val_dataloaders
