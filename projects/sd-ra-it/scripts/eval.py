"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import collections
import functools as ft
import itertools as it
import json
import logging
import os
import pathlib
import re
from typing import Iterable

import fire  # type:ignore
from src.utils import OfflineModel, batched, get_renderer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def classify(
    model: OfflineModel,
    line_pred_pairs: Iterable[tuple[dict, str]],
    verbose: bool = False,
) -> tuple[list[str], list[str | None]]:
    sys_msg: str = pathlib.Path("templates/judge/eval/sys.jinja").read_text()
    render = get_renderer("templates/judge/eval/usr.jinja")
    usr_msgs: list[str] = [
        render(question=line["question"], answers=line["answers"], prediction=pred)
        for line, pred in line_pred_pairs
    ]
    sys_msgs: list[str] = list(it.repeat(sys_msg, len(usr_msgs)))
    reasoning, labels = model.parse_zero_shot_with_retries(sys_msgs, usr_msgs, parse)
    return list(zip(reasoning, labels))


def parse(reply: str) -> str | None:
    pattern: str = (
        r"(\*\*)?(Label|Prediction label):(\*\*)? (?P<label>Correct|Incorrect|Refuse)"
    )
    match: re.Match | None = re.search(pattern, reply, flags=re.IGNORECASE)
    if match is not None:
        return match["label"].lower()
    else:
        return None


def main(
    preds: str,
    datafile: str,
    outfile: str,
    reasoning_file: str,
    judge: str = "Meta-Llama-3-70B-Instruct",
    logfile: str = "logs/eval.log",
    batch_size: int = 1000,
    tensor_parallel_size: int = 2,
):
    open(outfile, "w").close()
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode="w")
    with open(preds, "r") as file:
        predictions = list(map(json.loads, file))
    with open(datafile, "r") as file:
        dataset = list(map(json.loads, file))
    model_params = dict(
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,
        distributed_executor_backend="ray",
    )
    model = OfflineModel(
        judge,
        model_params=model_params,
        temperature=0,
        max_tokens=1000,
    )
    batches = batched(zip(dataset, predictions), batch_size)
    class_batches = (classify(model, batch) for batch in batches)
    annotations = it.chain.from_iterable(class_batches)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as file, open(reasoning_file, "w") as reason_file:
        for reason, annote in tqdm(annotations, total=len(dataset)):
            print(json.dumps(reason), file=reason_file, flush=True)
            print(json.dumps(annote), file=file, flush=True)


if __name__ == "__main__":
    fire.Fire(main)
