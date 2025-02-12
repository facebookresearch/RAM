"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import hydra
import torch
import torch._dynamo
import torch.distributed as dist
from accelerate import Accelerator
from data.data import get_train_dataloader, get_val_dataloaders
from models import get_base_lm, get_concept_extractor
from omegaconf import OmegaConf
from train import setup as train_setup
from train.trainer import trainer
from utils import Logger, set_random_seed


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main(cfg):
    """Use huggingface accelerator (automatically use distributed)"""
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_acc_steps,
    )
    accelerator.wait_for_everyone()

    """ distributed related config """
    num_gpus = dist.get_world_size()
    cfg.distributed = num_gpus > 1
    cfg.world_size = num_gpus

    """ fixing randomness """
    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    """ if torch compile"""
    if cfg.use_torch_compile:
        torch._dynamo.config.cache_size_limit = cfg.compile_dynamo_cache_size_limit

    """ define dataset, data loader, and tokenizer """
    train_loader = get_train_dataloader(cfg)
    val_loaders = get_val_dataloaders(cfg)

    """ define concept_extractor """
    concept_extractor = get_concept_extractor(cfg, accelerator)

    """ define base model """
    base_lm = get_base_lm(cfg, accelerator)

    """ define train and test type """
    train_func, fname, wandb_name = train_setup(cfg.mode, cfg)

    """ define logger """
    logger = Logger(
        fname,
        cfg,
        main_process=accelerator.is_main_process,
        use_wandb=cfg.wandb_log,
        wandb_name=wandb_name,
        log_path=cfg.log_path,
    )
    logger.log(OmegaConf.to_yaml(cfg))

    """ train """
    trainer(
        cfg,
        train_func,
        base_lm,
        train_loader,
        val_loaders,
        logger,
        accelerator,
        concept_extractor,
    )

    """ close tensorboard """
    logger.close_writer()


if __name__ == "__main__":
    main()
