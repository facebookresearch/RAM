"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import math
import os
import time

import pandas as pd
import requests
from datasets import load_from_disk
from tqdm import tqdm
from utils import call_llm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--dataset", type=str, default="data/otb_full")
parser.add_argument(
    "--output", type=str, default="final_outputs/otbench/{{model}}.jsonl"
)
parser.add_argument("--multiprocessing", type=int, default=1)
parser.add_argument(
    "--enable_nothink",
    action="store_true",
    default=False,
    help="Enable no-think mode for Qwen3 models.",
)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--num_gens", type=int, default=8)
parser.add_argument("--run_locally", action="store_true", default=False)
args = parser.parse_args()


df = load_from_disk(f"data/otb_full")
df = df["train"].to_pandas()

MODEL_NAME = args.model
OUTPUT = args.output

if args.run_locally:
    try:
        requests.get("http://localhost:30000")
    except:
        print("Starting vllm server")
        os.system(
            f"nohup vllm serve {args.model} --tensor-parallel-size 8 --port 30000 &"
        )
        print("done")
    while True:
        try:
            requests.get("http://localhost:30000")
            print("vllm server is running")
            break
        except:
            time.sleep(10)

all_responses = []
all_tokens = []
import multiprocessing as mp
from functools import partial


def is_nan(x):
    try:
        math.isnan(x)
    except:
        return False
    return True


def process_row(row, model_name):
    prompt = row["question"]

    if args.enable_nothink:
        prompt = prompt + " /no_think"

    prompt = prompt + " Answer the final answer in \\boxed{}"

    response = call_llm(
        model_name,
        prompt,
        max_tokens=(
            8192
            if model_name in ["gpt-4o", "gpt-4.1"]
            else (
                2048
                if model_name
                in [
                    "/datasets/pretrained-llms/Qwen2.5-Math-7B-Instruct",
                    "/datasets/pretrained-llms/Qwen2.5-Math-72B-Instruct",
                ]
                else 16384
            )
        ),
        n=args.num_gens,
        temperature=args.temperature,
    )
    if model_name.find("o3") != -1:
        # If model returns reasoning tokens, directly use them (if row is of type overthink).
        return (
            [x.message.content for x in response.choices],
            response.usage.completion_tokens_details.reasoning_tokens,
            prompt,
        )
    elif model_name.find("sonnet") != -1 or model_name.find("opus") != -1:
        # For sonnet models, you need to subtract the output tokens from the total tokens.
        return (
            [x.choices[0].message.content for x in response],
            sum([x.usage.completion_tokens for x in response]),
            prompt,
        )
    else:
        # For other models, extract the reasoning tokens during post-processing.
        return (
            [x.message.content for x in response.choices],
            response.usage.completion_tokens,
            prompt,
        )


with mp.Pool(processes=args.multiprocessing) as pool:
    process_func = partial(process_row, model_name=MODEL_NAME)
    results = pool.map(process_func, tqdm([row for _, row in df.iterrows()]))

print("All results received")

all_responses = [result[0] for result in results]
all_tokens = [result[1] for result in results]
prompts_used = [result[2] for result in results]


df["response"] = all_responses
df["tokens"] = all_tokens
df["prompts_used"] = prompts_used

if args.enable_nothink:
    OUTPUT_FILE = OUTPUT.replace("{{model}}", MODEL_NAME.replace("/", "-")).replace(
        ".jsonl", "-nothink.jsonl"
    )
else:
    OUTPUT_FILE = OUTPUT.replace("{{model}}", MODEL_NAME.replace("/", "-"))

if args.temperature != 0.6:
    OUTPUT_FILE = OUTPUT_FILE.replace(".jsonl", f"-temp={args.temperature}.jsonl")

if args.num_gens != 8:
    OUTPUT_FILE = OUTPUT_FILE.replace(".jsonl", f"-num_gens={args.num_gens}.jsonl")


print(f"Saving to {OUTPUT_FILE}")
print(f"To postprocess, run `otbench eval {OUTPUT_FILE} --model {MODEL_NAME}`")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_json(OUTPUT_FILE, orient="records", lines=True)
