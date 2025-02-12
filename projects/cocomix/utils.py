"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(
        self,
        fn,
        cfg,
        main_process=True,
        use_wandb=False,
        wandb_name=None,
        log_path=None,
    ):
        self.main_process = main_process
        self.log_path = "./logs/" if log_path is None else log_path
        self.logdir = None
        self.cfg = cfg
        self.use_wandb = use_wandb

        if self.main_process:
            logdir = self.log_path + fn
            self.logdir = logdir
            self.set_dir(logdir)

            if self.use_wandb:
                wandb.login(key=cfg.wandb_key)
                wandb.config = OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                wandb.init(
                    project=cfg.wandb_project,
                    name=wandb_name,
                    dir=logdir,
                    entity=cfg.wandb_entity,
                    settings=wandb.Settings(start_method="fork"),
                )

        # distribute logdir to other processes
        if torch.distributed.is_initialized():
            if self.main_process:
                object_list = [self.logdir]
            else:
                object_list = [None]
            dist.broadcast_object_list(object_list, src=0)
            self.logdir = object_list[0]

    def set_dir(self, logdir, log_fn="log.txt"):
        os.makedirs(logdir, exist_ok=True)
        self.log_file = open(os.path.join(logdir, log_fn), "a")
        with open(os.path.join(logdir, "config.yaml"), "w+") as fp:
            OmegaConf.save(config=self.cfg, f=fp.name)

    def close_writer(self):
        if self.main_process and self.use_wandb:
            wandb.finish()

    def log(self, string):
        if self.main_process:
            self.log_file.write("[%s] %s" % (datetime.now(), string) + "\n")
            self.log_file.flush()

            print("[%s] %s" % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.main_process:
            self.log_file.write("%s (%s)" % (string, self.logdir) + "\n")
            self.log_file.flush()

            print("%s (%s)" % (string, self.logdir))
            sys.stdout.flush()

    def wandb_log(self, log_dict, step=None, commit=None):
        if self.main_process and self.use_wandb:
            wandb.log(log_dict, step=step, commit=commit)


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def metric_synchronize_between_processes(metrics, accelerator=None):
    if accelerator is not None:
        for k, v in metrics.items():
            t = torch.tensor([v], dtype=torch.float64, device=accelerator.device)
            gathered_items = accelerator.gather_for_metrics(t)
            metrics[k] = gathered_items.mean().item()
    else:
        if is_dist_avail_and_initialized():
            for k, v in metrics.items():
                t = torch.tensor([v], dtype=torch.float64, device="cuda")
                dist.barrier()
                dist.all_reduce(t)
                t /= dist.get_world_size()
                t = t.tolist()
                metrics[k] = t[0]


def logging_path_check(cfg):
    from train import setup as train_setup

    _, fname, _ = train_setup(cfg.mode, cfg)
    log_path = "./logs/" if cfg.log_path is None else cfg.log_path
    os.makedirs(log_path, exist_ok=True)
    logdir = log_path + fname
    os.makedirs(logdir, exist_ok=True)


# Function to create a tqdm progress bar for distributed training
def tqdm_distributed(main_process, iterator, *args, **kwargs):
    if main_process:
        return tqdm(iterator, *args, **kwargs)
    else:
        return iterator  # No progress bar for non-main processes
