"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools as ft
import itertools as it
import json
import os
import pathlib
import random
import sys

import fire  # type:ignore
from src.utils import Messages, OfflineModel, batched, get_renderer


def main(
    outfile: str,
    data: str = "data/ra-dit/nq/dev.jsonl",
    model: str = "/fsx-project/mfinlayson/Meta-Llama-3-8B-Instruct/",
    n: int | None = None,
    ndocs: int = 4,
    add_system_message: bool = False,
    samples: int = 1,
    tensor_parallel_size: int = 1,
):
    open(outfile, "w").close()
    with open(data) as f:
        lines = list(map(json.loads, f))
    sys_msg = pathlib.Path("templates/rag_system_message.jinja").read_text().strip()
    render = get_renderer("templates/rag.jinja")
    usr_contents = [
        render(
            question=line["question"],
            passages=[r["text"] for r in line["retrieved_support"][:ndocs]],
        )
        for line in lines
    ]
    messages: list[Messages] = [
        [dict(role="user", content=content)] for content in usr_contents
    ]
    if add_system_message:
        messages = [
            [dict(role="system", content=sys_msg)] + message for message in messages
        ]
    model_params = dict(
        tensor_parallel_size=tensor_parallel_size, distributed_executor_backend="ray"
    )
    llm = OfflineModel(
        model, model_params=model_params, temperature=0.5, max_tokens=750
    )
    generate = ft.partial(llm, samples=samples)
    outputs = it.chain.from_iterable(map(generate, batched(messages, 1000)))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as file:
        for output in outputs:
            print(json.dumps(output), file=file, flush=True)


if __name__ == "__main__":
    fire.Fire(main)
