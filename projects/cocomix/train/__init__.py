"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


def setup(mode, cfg):

    base_model = cfg.base_model
    if cfg.n_embd is not None:
        base_model = f"{base_model}_embd{cfg.n_embd}"
    if cfg.n_layer is not None:
        base_model = f"{base_model}_L{cfg.n_layer}"
    if cfg.n_head is not None:
        base_model = f"{base_model}_H{cfg.n_head}"

    fname = (
        f"{cfg.dataset}/{base_model}/{mode}"
        f"_bs{int(cfg.update_batch_size*cfg.grad_acc_steps)}"
    )
    wandb_name = (
        f"{cfg.dataset}_{base_model}_{mode}"
        f"_bs{int(cfg.update_batch_size*cfg.grad_acc_steps)}"
    )

    fname += f"_ctx{cfg.block_size}"
    wandb_name += f"_ctx{cfg.block_size}"

    if mode == "ntp":
        from train.train_func.ntp import train_step
    elif mode == "cocomix":
        from train.train_func.cocomix import train_step

        fname += f"_lam{cfg.lam_concept}"
        wandb_name += f"_lam{cfg.lam_concept}"
    else:
        raise NotImplementedError()

    fname += f"_seed_{cfg.seed}"
    wandb_name += f"_seed_{cfg.seed}"
    if cfg.suffix is not None:
        fname += f"_{cfg.suffix}"
        wandb_name += f"_{cfg.suffix}"

    return train_step, fname, wandb_name
