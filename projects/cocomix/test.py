"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import math
import os
import time

import hydra
import lm_eval
import numpy as np
import torch
import torch.distributed as dist
import yaml
from models import GPT2CoCoMixLMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_random_seed


def evaluate_ppl(cfg, base_lm, val_loader, accelerator, eval_limit=None):
    device = accelerator.device
    loss_sum = 0.0
    if eval_limit is not None:
        eval_limit = eval_limit // accelerator.num_processes

    num_total = 0
    for batch in val_loader:
        with torch.no_grad():
            if "labels" not in batch:
                loss = base_lm(
                    input_ids=batch["input_ids"], labels=batch["input_ids"]
                ).loss
            else:
                loss = base_lm(**batch).loss
        loss_sum += loss.item() * batch["input_ids"].shape[0]
        num_total += batch["input_ids"].shape[0]
        if eval_limit is not None and num_total >= eval_limit:
            break

    if cfg.distributed:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        loss_sum = torch.tensor(loss_sum, dtype=torch.float, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        loss_sum = loss_sum.item()

    ppl = math.exp(loss_sum / num_total)
    return ppl


def evaluate_help(
    model_hf,
    tokenizer,
    batch_size=16,
    device="cuda",
    tasks=None,
    openwebtext_data_dir=None,
):
    if tasks is not None:
        model = lm_eval.api.registry.get_model("hf")(
            pretrained=model_hf,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )

        with torch.no_grad():
            results = lm_eval.simple_evaluate(  # call simple_evaluate
                model=model,
                tasks=tasks,
            )

        del model
    else:
        results = {"results": {}}

    with torch.no_grad():
        openwebtext_results = evaluate_openwebtext(
            model_hf, device=device, openwebtext_data_dir=openwebtext_data_dir
        )
        results["results"]["openwebtext"] = {"ppl": openwebtext_results}

    torch.cuda.empty_cache()

    if tasks is not None:
        for task in tasks:
            samples = results["samples"][task]
            nll_sum = 0.0
            for sample in samples:
                if isinstance(sample["target"], int):
                    nll = -sample["resps"][sample["target"]][0][0]
                else:
                    if "choices" in sample["doc"]:
                        target = sample["doc"]["choices"]["label"].index(
                            sample["target"]
                        )
                    else:
                        print(f"Task {task} does not have 'choices' in the doc")
                        break
                    nll = -sample["resps"][target][0][0]
                nll_sum += nll
            avg_nll = nll_sum / len(samples)
            results["results"][task]["nll"] = avg_nll

    return results["results"]


def evaluate_openwebtext(gpt2_model, device, openwebtext_data_dir):
    # evaluate ppl of openwebtext
    stime = time.time()

    val_data = np.memmap(
        os.path.join(openwebtext_data_dir, "val.bin"), dtype=np.uint16, mode="r"
    )
    nll = 0.0
    eff_block_size = 1024
    for i in range(0, len(val_data), eff_block_size):
        if i + eff_block_size >= len(val_data):
            break
        X = (
            torch.from_numpy((val_data[i : i + eff_block_size]).astype(np.int64))
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            loss = gpt2_model(X, labels=X).loss
        nll += loss.item()

    avg_nll = nll / (len(val_data) // eff_block_size)
    ppl = math.exp(avg_nll)
    print(f"OpenWebText PPL: {ppl:.2f}, time: {time.time()-stime:.2f}s")
    return ppl


def eval_checkpoint(cfg, checkpoint_path, tokenizer, device):
    print(f"Loading model from {checkpoint_path}")
    fast_attn_type = "sdpa" if "gpt2" in checkpoint_path else "flash_attention_2"
    torch_dtype = torch.bfloat16 if "gpt2" in checkpoint_path else torch.float16
    kwargs = {"attn_implementation": fast_attn_type, "torch_dtype": torch_dtype}
    if "cocomix" in checkpoint_path:
        try:
            with open(os.path.join(cfg.load_path, "config.yaml"), "r") as f:
                cfg_load = yaml.safe_load(f)
        except:
            with open(os.path.join(cfg.load_path, "../config.yaml"), "r") as f:
                cfg_load = yaml.safe_load(f)
        if "gpt2" in checkpoint_path:
            base_lm = GPT2CoCoMixLMHeadModel.from_pretrained(
                checkpoint_path,
                concept_dim=cfg_load["concept_dim"],
                insert_layer_idx=cfg_load["insert_layer_index"],
                concept_num=cfg_load["concept_num"],
            ).to(device)
    else:
        base_lm = AutoModelForCausalLM.from_pretrained(checkpoint_path, **kwargs).to(
            device
        )

    # cfg.eval_tasks to list where cfg is omegaconf.listconfig
    tasks = [task for task in cfg.eval_tasks] if cfg.eval_tasks is not None else None
    result = evaluate_help(
        base_lm,
        tokenizer,
        cfg.batch_size,
        device,
        tasks=tasks,
        openwebtext_data_dir=cfg.data_dir,
    )

    # free memory
    del base_lm
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return result


@hydra.main(config_path="conf", config_name="config_eval", version_base="1.3.2")
def main(cfg):
    cfg.rank = 0
    set_random_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check load path exists
    if not os.path.exists(cfg.load_path):
        raise FileNotFoundError(f"Model path {cfg.load_path} does not exist")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    if cfg.save_result:
        eval_results_file = os.path.join(cfg.load_path, "eval_results.json")
    else:
        eval_results_file = "eval_results.json"

    # check it is single checkpoint evaluation
    if cfg.eval_single_ckpt:
        # evaluate checkpoint_path's performance
        result = eval_checkpoint(cfg, cfg.load_path, tokenizer, device)

        # print results
        for task, res in result.items():
            print(f"Task {task}, Results: {res}")

        # save results to cfg.load_path directory, with name eval_results.json
        with open(eval_results_file, "w") as f:
            json.dump(result, f)

    # or else, evaluate per every cfg.eval_freq
    else:
        # if eval_results.json exists, load it and continue from the last step
        if os.path.exists(eval_results_file):
            with open(eval_results_file, "r") as f:
                results = json.load(f)
            eval_step = int(list(results.keys())[-1].split("_")[-1]) + cfg.eval_freq
        else:
            results = {}
            eval_step = cfg.eval_freq

        # evaluate every cfg.eval_freq
        while True:
            checkpoint_path = os.path.join(cfg.load_path, f"step_{eval_step}")
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint path {checkpoint_path} does not exist")
                break

            # evaluate checkpoint_path's performance
            result = eval_checkpoint(cfg, checkpoint_path, tokenizer, device)
            results[f"step_{eval_step}"] = result

            # print results
            for task, res in result.items():
                print(f"Step {eval_step}, Task {task}, Results: {res}")

            # next eval_step (evaluate every cfg.eval_freq)
            eval_step += cfg.eval_freq

            # save results to cfg.load_path directory, with name eval_results.json
            with open(eval_results_file, "w") as f:
                json.dump(results, f)

        # check whether there is last checkpoint (saved at the end of the training)
        checkpoint_path = os.path.join(cfg.load_path, f"last")
        if os.path.exists(checkpoint_path):
            # evaluate checkpoint_path's performance
            result = eval_checkpoint(cfg, checkpoint_path, tokenizer, device)
            results[f"last"] = result

            # print results
            for task, res in result.items():
                print(f"Last checkpoint, Task {task}, Results: {res}")

            # save results to cfg.load_path directory, with name eval_results.json
            with open(eval_results_file, "w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    main()
