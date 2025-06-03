"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from omegaconf import OmegaConf
from torch.nn import functional as F

from projects.mta.data import init_choice_state, setup_sources
from projects.mta.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from projects.mta.single_json import SingleJSONLIterator
from projects.mta.transformer import LMTransformer, LMTransformerArgs

EVAL_FOLDER_NAME = "{:010d}"

os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # for code evals

logger = logging.getLogger()


@dataclass
class LMHarnessArgs:
    tasks: Optional[List[Any]] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[Union[int, float]] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234
    confirm_run_unsafe_code: bool = True


@dataclass
class ValidationArgs:
    max_steps: Optional[int] = (
        None  # If None the whole validation file is used -> /!\ This number of steps is gpu dependent (100 max steps on 8 gpus = 800 steps on 1 gpu)
    )
    use_val_from_train_src: bool = False  # Use the validation set from training sources
    root_dir: str = ""
    sources: List[str] = field(default_factory=list)  # Other sources to eval on


@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)
    ppl_files: Optional[List[str]] = None
    ppl_seq_len: int = 2048
    ppl_batch_size: int = 4
    ppl_n_batches: int = 256

    wandb: Optional[Any] = None

    global_step: Optional[int] = None  # for in-training evaluation


def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        # assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(
                self.generator.tokenizer.encode(p, add_bos=False, add_eos=False)
            )
            if p_len < self.generator.max_tokens - self.generator.max_gen_len:
                results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))
            else:
                results.append(
                    (
                        ll[-self.generator.max_gen_len :].sum().item(),
                        gr[-self.generator.max_gen_len :].all().item(),
                    )
                )

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results


def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    srcs = {}
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        srcs[path] = 1.0
    if val_args.use_val_from_train_src:
        for src in train_cfg.data.sources:
            path = os.path.join(train_cfg.data.root_dir, src)
            srcs[path] = 1.0

    try:
        multi_state = init_choice_state(
            "", srcs, 0, get_global_rank(), get_world_size(), "*.jsonl"
        )
        path_to_iter = setup_sources(multi_state)
    except ZeroDivisionError as e:
        # just load these files
        multi_state = {}
        for dataset_path, weight in srcs.items():
            assert weight == 1.0, "use multi_state with weights"
            dataset_chunks = [str(p) for p in Path(dataset_path).glob("*.jsonl")]
            path_to_iter.extend(dataset_chunks)
        path_to_iter = setup_sources(multi_state)

    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len
    generator.max_gen_len = 1

    all_val_metrics = {}
    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        tgt = []
        logger.info(f"Running validation on {src}...")
        for step, (content, state) in enumerate(jsonl_iterator):
            if state["current_iter"] > 0 or (
                val_args.max_steps is not None and step >= val_args.max_steps
            ):
                break
            if ("text" in content) or ("content" in content):
                content_key = "text" if ("text" in content) else "content"
                texts.append(content[content_key])
                tgt.append(content[content_key])
            else:
                content_key = "src"
                texts.append(content[content_key] + "<SRC_TGT_SEP>" + content["tgt"])
                tgt.append(content["tgt"])

        if ("text" in content) or ("content" in content):
            _, loglikelihood, _ = generator.generate(texts)
        else:
            _, loglikelihood, _ = generator.generate(texts, tgt_ppl_only=True)

        metrics = defaultdict(list)
        for i, ll in enumerate(loglikelihood):
            tmp = -ll.sum().item()
            metrics["nll"].append(tmp)
            metrics["nll_per_token"].append(tmp / len(ll))
            metrics["nll_per_char"].append(tmp / len(tgt[i]))
            metrics["avg_seqlen"].append(len(ll))

        for m in metrics:
            if m == "nll":
                tot_nll = sum(metrics[m])
            if m == "avg_seqlen":
                tot_seqlen = sum(metrics[m])
            metrics[m] = sum(metrics[m]) / len(metrics[m])
        metrics["ppl"] = math.exp(tot_nll / tot_seqlen) if tot_seqlen > 0 else 0.0

        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(
                f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1"
            )
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    jsonl_path,
    args,
) -> float:
    data_iterator = SingleJSONLIterator(
        tokenizer=tokenizer,
        data_path=jsonl_path,
        seq_len=args.ppl_seq_len,
        batch_size=args.ppl_batch_size,
        buffer_size=1,
        world_rank=0,  # todo: parallel between ranks
        world_size=1,
        infinite=False,
    )
    batch_iterator = iter(data_iterator)
    metric = 0.0
    err_tok = 0.0
    err_seq = 0.0
    n_toks = 0
    n_seqs = 0
    for i, batch in enumerate(batch_iterator):
        if i >= args.ppl_n_batches:
            break
        x = torch.from_numpy(batch.x).cuda()
        y = torch.from_numpy(batch.y).cuda()
        pred = model(x)
        # src tgt masking
        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1), reduction="none")
        mask = torch.from_numpy(batch.mask).cuda()
        loss = loss * mask.flatten(0, 1)
        # mean loss
        loss = loss.sum()
        metric += loss.item()
        n_toks += batch.mask.sum()

        # calculate errors
        pred_y = pred.argmax(dim=-1)
        err_tok += (y.ne(pred_y) * mask).sum().item()
        # note: multiple samples can be packed into a single sequence
        err_seq += (y.ne(pred_y) * mask).any(dim=-1).sum().item()
        n_seqs += mask.size(0)
    if n_toks == 0:
        n_toks = 1e-6
    ppl = math.exp(metric / n_toks) if n_toks > 0 else 0.0
    avg_loss = metric / n_toks
    avg_err_tok = err_tok / n_toks
    avg_err_seq = err_seq / (n_seqs + 1e-6)
    return ppl, avg_loss, avg_err_tok, avg_err_seq


def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMTransformerArgs,
    )
    logger.info("Model loaded")
    model.eval()
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)

    if cfg.harness.tasks is not None:
        wrap = EvalHarnessLM(generator)
        results = simple_evaluate(wrap, **asdict(cfg.harness))
    else:
        results = {"results": {}}

    if cfg.ppl_files is None:
        cfg.ppl_files = []

    for path in cfg.ppl_files:
        if get_global_rank() == 0:
            # val ppl computed on 1st node only
            logger.info("Running PPL evaluations ...")
            logger.info(f"Evaluating PPL on {path} ...")
            ppl, loss, err_tok, err_seq = eval_ppl(
                model=model, tokenizer=tokenizer, jsonl_path=path, args=cfg
            )
            logger.info(
                f"Results: ppl={ppl} loss={loss} err_tok={err_tok} err_seq={err_seq}"
            )
            if results == None:
                results["results"] = {}
            results["results"][f"{path}/ppl"] = ppl
            results["results"][f"{path}/loss"] = loss
            results["results"][f"{path}/err_tok"] = err_tok
            results["results"][f"{path}/err_seq"] = err_seq
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    val_results = None
    if cfg.validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg)
    if get_global_rank() == 0:
        if len(results["results"]) != 0:
            with open(Path(cfg.dump_dir) / "results.json", "w") as f:
                f.write(json.dumps(results, default=str))
            logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step

        if len(results["results"]) != 0:
            logger.info(f"Writing metric logs to {metric_log_path}")
            print(
                json.dumps(timestamp | results["results"]),
                file=open(metric_log_path, mode="a"),
                flush=True,
            )

        val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=open(val_log_path, mode="a"),
                flush=True,
            )

    del generator
    if len(results["results"]) == 0:
        return val_results
    if val_results is None:
        return results["results"]
    return dict(results["results"], **val_results)


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
