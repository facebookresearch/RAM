"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import multiprocessing
import os
import time
from collections import Counter

import numpy as np
import pyjson5
import requests
from litellm import completion
from prompts import ALL_DOMAINS, OVERTHINKING_PROMPT, QUESTION_FORMATS

os.environ["OPENAI_API_KEY"] = ""


def process_num_options(args):
    NUM_OPTIONS, mode, x = args
    local_responses = []

    print("Doing for", mode, NUM_OPTIONS, x)
    domains = ALL_DOMAINS[x]

    question_format = QUESTION_FORMATS[mode]
    if NUM_OPTIONS != -1:
        question_format = question_format.format(NUM_OPTIONS=NUM_OPTIONS)

    while True:
        print("Starting to generate", domains)
        messages = [
            {
                "role": "user",
                "content": OVERTHINKING_PROMPT.replace(
                    "{{question_format}}", question_format
                )
                .replace("{{domains}}", str(domains))
                .replace("{{len(domains)}}", str(len(domains))),
            }
        ]
        response = completion(
            model="openai/llama4-maverick",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=16000,
            api_base="http://localhost:8000/v1",
        )
        flag = False
        try:
            json_output = pyjson5.decode(
                response.choices[0]
                .message.content.strip()
                .split("```json")[1]
                .split("```")[0]
            )
            assert isinstance(json_output, list)
            assert isinstance(json_output[0], dict)
            assert "Question" in json_output[0] or "question" in json_output[0]
            assert "Answer" in json_output[0] or "answer" in json_output[0]
            assert "domain" in json_output[0] or "Domain" in json_output[0]
        except Exception as e:
            print("Error in json decoding", e)
            continue

        domain_counts = {x: 0 for x in domains}
        for x in json_output:
            try:
                if "domain" in x:
                    domain_counts[x["domain"]] += 1
                elif "Domain" in x:
                    domain_counts[x["Domain"]] += 1
                else:
                    flag = True
                    break
            except:
                flag = True
                break
        if all(domain_counts[x] >= 15 for x in domain_counts) and not flag:
            break
        else:
            print(domain_counts, json_output)
    output = response.choices[0].message.content.strip()
    local_responses.append(
        {"output": output, "num_options": NUM_OPTIONS, "mode": mode, "domains": domains}
    )

    print(f"Done for {mode} {NUM_OPTIONS} {len(domains)}")

    return local_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        required=False,
        help="Path to the Llama-4-Maverick-17B-128E-Instruct-FP8 model.",
    )

    args = parser.parse_args()

    try:
        requests.get("http://localhost:8000")
    except:
        os.system(
            f"nohup vllm serve {args.model_path} --tensor-parallel-size 8 --served-model llama4-maverick &"
        )
        print("done")

    while True:
        try:
            requests.get("http://localhost:8000")
            break
        except:
            time.sleep(10)

    responses = []
    for mode in QUESTION_FORMATS:
        print("Doing for", mode)
        num_options_list = [4] if mode == "mcq" else [-1]

        args_list = [
            (NUM_OPTIONS, mode, i)
            for NUM_OPTIONS in num_options_list
            for i in range(0, len(ALL_DOMAINS), 1)
        ]
        for arg_list in args_list:
            results = process_num_options(arg_list)
            responses.extend(results)

        try:
            with open("data/responses_maverick_overthink.json", "w") as f:
                json.dump(responses, f, indent=4)
        except:
            pass
