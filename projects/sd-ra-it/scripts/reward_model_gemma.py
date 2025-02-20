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
import torch
from src.utils import Messages, OfflineModel, batched, get_renderer
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def main(
    outfile: str,
    responses: str,
    data: str = "data/ra-dit/nq/dev.jsonl",
    ndocs: int = 4,
    add_system_message: bool = False,
    batch_size: int = 8,
):
    open(outfile, "a").close()
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
    model_name = "Skywork-Reward-Gemma-2-27B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_messages = map(
        ft.partial(
            rm_tokenizer.apply_chat_template, tokenize=True, return_tensors="pt"
        ),
        messages,
    )

    @torch.no_grad()
    def get_score(tokenized_message):
        return rm(tokenized_message.to("cuda:0")).logits[0][0].item()

    outputs = map(get_score, tokenized_messages)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as file:
        for output in tqdm(outputs, total=len(messages)):
            print(json.dumps(output), file=file, flush=True)


if __name__ == "__main__":
    fire.Fire(main)
