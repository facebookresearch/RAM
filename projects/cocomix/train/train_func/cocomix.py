"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils import metric_synchronize_between_processes


def train_step(
    cfg,
    base_lm,
    optimizer,
    scheduler,
    accelerator,
    batch,
    logger,
    metrics_dic,
    concept_extractor,
):

    # compute loss
    with accelerator.accumulate(base_lm):
        base_lm.train()

        extracted_concept = concept_extractor(input_ids=batch["input_ids"])
        outputs, concept_logit = base_lm(
            input_ids=batch["input_ids"],
            labels=batch["input_ids"],
            get_concept_logit=True,
        )

        concept_labels = torch.topk(extracted_concept, k=cfg.topK_attri, dim=-1)[1]
        loss_concept = torch.tensor(0.0).to(base_lm.device)
        for i in range(cfg.topK_attri):
            loss_concept += (
                1
                / cfg.topK_attri
                * F.cross_entropy(
                    concept_logit.view(-1, concept_logit.size(-1)),
                    concept_labels[:, :, i].contiguous().view(-1),
                )
            )

        loss = outputs.loss

        metrics_dic["loss"].append(loss.item())
        metrics_dic["loss_concept"].append(loss_concept.item())

        loss_total = loss + cfg.lam_concept * loss_concept
        accelerator.backward(loss_total)

        if accelerator.sync_gradients:
            # clip gradient when using sync gradients
            grad_norm = accelerator.clip_grad_norm_(
                base_lm.parameters(), cfg.grad_clip_thresh
            )
            metrics_dic["grad_norm"].append(grad_norm)

            # log metrics when using sync gradients (i.e., actual gradient update)
            if cfg.global_step % cfg.log_step_freq == 0:
                metric_synchronize_between_processes(
                    metrics_dic, accelerator
                )  # sync metrics across processes
                log_metrics = {
                    "train": {f"{k}": np.mean(v) for k, v in metrics_dic.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                }
                logger.wandb_log(log_metrics, step=cfg.global_step)
                for k, v in metrics_dic.items():
                    logger.log(f"Step {cfg.global_step} Train {k}: {np.mean(v)}")

                metrics_dic.clear()
            cfg.global_step += 1

        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()
        optimizer.zero_grad()
