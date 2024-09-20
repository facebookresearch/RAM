"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import os
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from typing import Dict, Optional

import torch
import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"


def add_args(parser: ArgumentParser):
    parser = parser.add_argument_group("logger")
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--log-task-name", type=str)
    parser.add_argument("--log-group", type=str, help="manually specify log group")
    parser.add_argument("--log-plot", action=BooleanOptionalAction, default=False)
    parser.add_argument("--log-project", type=str, default="cope")
    parser.add_argument("--log-plot-dir", default="/private/home/${USER}/cope/logs/")
    parser.add_argument(
        "--wandb-key-file", default="/private/home/${USER}/wandb/key.txt"
    )
    parser.add_argument("--wandb-id", type=str)
    parser.add_argument("--wandb-host", type=str)


def preprocess_args(args):
    if args.log_name is None:
        args.log_name = datetime.now().strftime("%Y%m%d_%H%M%S")


class Metric:
    def __init__(
        self,
        name: str,
        divide_by=None,
        plot=True,
        cumulative=False,
        print=True,
    ) -> None:
        self.name = name
        self.divide_by = divide_by
        self.plot = plot
        self.cumulative = cumulative
        self.print = print
        self.step_value = None
        self.value = 0
        self.cur_epoch_best = False
        self.best_value = float("inf")

    def reset(self) -> None:
        self.step_value = None

    def record(self, value: float) -> None:
        if self.step_value is None:
            self.step_value = 0
        self.step_value += value

    def step(self, all_metrics) -> Optional[float]:
        if self.step_value is None:
            # no record for this iteration, so do not log
            return None

        if self.divide_by is not None:
            divide_by = all_metrics[self.divide_by].step_value
            if divide_by:
                self.step_value /= divide_by

        if "ppl" in self.name:
            self.step_value = math.exp(self.step_value)

        if self.cumulative:
            self.value += self.step_value
        else:
            self.value = self.step_value

        if self.print:
            if type(self.value) is float:
                print("{}={:.3f}\t".format(self.name, self.value))
            else:
                print("{}={}\t".format(self.name, self.value))

        # if current val is < best val, overwrite best val, set cur_epoch_best = True
        # NOTE: this assumes that the "best" is the lowest value
        if self.value < self.best_value:
            self.best_value = self.value
            self.cur_epoch_best = True
        else:
            self.cur_epoch_best = False

        return self.value


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_name = cfg.log_name
        self.plot = cfg.log_plot
        self.groups = ["train", "val", "test"]

        if self.plot:
            plot_dir = os.path.expandvars(cfg.log_plot_dir)
            os.makedirs(os.path.join(plot_dir, self.log_name), exist_ok=True)
            if cfg.wandb_key_file is not None:
                cfg.wandb_key_file = os.path.expandvars(cfg.wandb_key_file)
                if os.path.exists(cfg.wandb_key_file):
                    with open(cfg.wandb_key_file) as f:
                        wandb_key = f.read().strip()
                        wandb.login(
                            host=cfg.wandb_host,
                            key=wandb_key,
                        )
            wandb.init(
                project=cfg.log_project,
                name=self.log_name,
                dir=plot_dir,
                group=cfg.log_group,
                resume="allow",
                id=cfg.wandb_id,
            )
            wandb.config.update(cfg, allow_val_change=True)
            cfg.wandb_id = wandb.run.id  # for resuming later
        self.metrics: Dict[str, Metric] = {}
        self.log = []
        self.is_example_batch = False  # use for printing some examples in the log
        self.step_metric = None

    def set_group(self, group: str):
        assert group in self.groups
        self._current_group = group

    def setup_metric(
        self,
        name: str,
        divide_by=None,
        plot=True,
        cumulative=False,
        print=True,
        grouped=True,
        is_step=False,
    ):
        assert name not in self.metrics
        if grouped:
            for group in self.groups:
                divide_by_g = divide_by if divide_by is None else f"{divide_by}/{group}"
                self.setup_metric(
                    f"{name}/{group}",
                    divide_by_g,
                    plot,
                    cumulative,
                    print,
                    grouped=False,
                )
        else:
            self.metrics[name] = Metric(name, divide_by, plot, cumulative, print)
            if self.plot:
                wandb.define_metric(name, hidden=not plot, step_metric=self.step_metric)
            if is_step:
                self.step_metric = name

    def reset(self) -> None:
        for _, m in self.metrics.items():
            m.reset()

    def record(self, name: str, value) -> None:
        if torch.is_tensor(value):
            value = value.item()
        if name in self.metrics:
            metric = self.metrics[name]
        else:
            # must be grouped metric
            metric = self.metrics[f"{name}/{self._current_group}"]
        metric.record(value)

    def dist_sync(self):
        num_metrics = len(self.metrics.items())
        X = torch.zeros(num_metrics).to(self.cfg.device)

        for i, name in enumerate(self.metrics.keys()):
            if self.metrics[name].step_value is not None:
                X[i] = self.metrics[name].step_value

        torch.distributed.all_reduce(X)
        torch.cuda.synchronize()

        if self.cfg.rank == 0:
            for i, name in enumerate(self.metrics.keys()):
                if name == self.step_metric:
                    # no need to sum nepochs
                    continue
                if self.metrics[name].step_value is not None:
                    self.metrics[name].step_value = X[i].item()

    def step(self):
        if self.cfg.distributed:
            self.dist_sync()

        print("=" * 80)
        self.log.append({})

        for_wandb = {}
        for name, m in self.metrics.items():
            val = m.step(self.metrics)
            if val is None:
                continue
            self.log[-1][name] = val
            for_wandb[name] = val

        if self.plot:
            wandb.log(for_wandb)

        self.reset()

    def close(self):
        if self.plot:
            wandb.finish()

    def get_state(self):
        return {"log": self.log, "metrics": self.metrics}

    def set_state(self, state):
        self.log = state["log"]
        self.metrics.update(state["metrics"])

    def replot(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.set_state(state_dict["logger"])
        if self.plot:
            for metrics in self.log:
                wandb.log(metrics)
        self.close()
