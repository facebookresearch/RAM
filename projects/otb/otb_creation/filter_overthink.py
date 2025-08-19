"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import multiprocessing
import os

import numpy as np
import pandas as pd
import pyjson5
import requests
from litellm import completion

os.environ["OPENAI_API_KEY"] = ""

df = pd.read_json("data/responses_maverick_overthink.json")

FILTER_K = 16

new_data = []

for i, row in df.iterrows():
    json_data = pyjson5.decode(
        row["output"].strip().split("```json")[1].split("```")[0]
    )
    renames = []
    if "Domain" in json_data[0]:
        renames.append(["Domain", "domain"])
    if "Question" in json_data[0]:
        renames.append(["Question", "question"])
    if "Answer" in json_data[0]:
        renames.append(["Answer", "answer"])
    if "Options" in json_data[0]:
        renames.append(["Options", "options"])
    new_json_data = []
    for x in json_data:
        for r in renames:
            x[r[1]] = x.pop(r[0])
        x["num_options"] = row["num_options"]
        new_json_data.append(x)
    if row["mode"] == "mcq":
        if "options" not in json_data[0]:
            for i in range(len(new_json_data)):
                if "A" not in new_json_data[i]:
                    break
                num_options = row["num_options"]
                if num_options == 4:
                    new_json_data[i]["options"] = [
                        new_json_data[i]["A"],
                        new_json_data[i]["B"],
                        new_json_data[i]["C"],
                        new_json_data[i]["D"],
                    ]
                elif num_options == 8:
                    new_json_data[i]["options"] = [
                        new_json_data[i]["A"],
                        new_json_data[i]["B"],
                        new_json_data[i]["C"],
                        new_json_data[i]["D"],
                        new_json_data[i]["E"],
                        new_json_data[i]["F"],
                        new_json_data[i]["G"],
                        new_json_data[i]["H"],
                    ]
                elif num_options == 12:
                    new_json_data[i]["options"] = [
                        new_json_data[i]["A"],
                        new_json_data[i]["B"],
                        new_json_data[i]["C"],
                        new_json_data[i]["D"],
                        new_json_data[i]["E"],
                        new_json_data[i]["F"],
                        new_json_data[i]["G"],
                        new_json_data[i]["H"],
                        new_json_data[i]["I"],
                        new_json_data[i]["J"],
                        new_json_data[i]["K"],
                        new_json_data[i]["L"],
                    ]

    new_json_data = [x | {"mode": row["mode"]} for x in new_json_data]
    new_data.extend(new_json_data)


def is_nan(x):
    try:
        return math.isnan(x)
    except:
        return False


df_new = pd.DataFrame(new_data)
try:
    df_new = df_new.drop(
        columns=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    )
except:
    pass
df_new["drop"] = False
for i, row in df_new.iterrows():
    if row["mode"] != "mcq":
        continue
    if (
        "options" in row
        and not is_nan(row["options"])
        and row["answer"] not in row["options"]
    ):
        if not len(row["answer"]) == 1:
            df_new.at[i, "drop"] = True
            continue
        ord_answer = ord(row["answer"]) - ord("A")
        df_new.at[i, "answer"] = row["options"][ord_answer]

df_new = df_new[~df_new["drop"]]
df_new.drop(columns=["drop"], inplace=True)

df_new.sample(frac=1).reset_index(drop=True).to_json(
    "data/overthink_bench_by_maverick_unfiltered.json", orient="records", indent=4
)

try:
    requests.get("http://localhost:8000")
except:
    os.system(
        "vllm serve /datasets/pretrained-llms/Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 8 --served-model llama4-maverick &"
    )


def call_llm(args):
    message, temperature, top_p, max_tokens, n = args
    response = completion(
        model="openai/llama4-maverick",
        messages=message,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_base="http://localhost:8000/v1",
        n=n,
    )
    return response


def call_llm_mp(messages, temperature, top_p, max_tokens, n):
    args_list = [(msg, temperature, top_p, max_tokens, n) for msg in messages]
    results = []
    for arg_list in args_list:
        results.append(call_llm(arg_list))
    # with multiprocessing.Pool(processes=32) as pool:
    #     results = pool.map(call_llm, args_list)
    return [[y.message.content.strip() for y in x.choices] for x in results]


all_outputs = []
all_messages = []
for i, row in df_new.iterrows():
    prompt = row["question"]
    all_messages.append([{"role": "user", "content": prompt}])
all_outputs = call_llm_mp(
    messages=all_messages, temperature=0.6, top_p=0.95, max_tokens=4096, n=FILTER_K
)
df_new["model_response"] = all_outputs


VERIFIER_PROMPT = """
User: ### Question: {question}\n\n
### Ground Truth Answer: {ground_truth}\n\n
### Student Answer: {student_answer}\n\n
For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n
If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"""


def score(response):
    VERIFIER_PASS_TAG = "Final Decision: Yes"
    VERIFIER_FAIL_TAG = "Final Decision: No"
    if VERIFIER_PASS_TAG in response:
        return 1
    elif VERIFIER_FAIL_TAG in response:
        return 0

    if "yes" in response.lower() and "no" not in response.lower():
        return 1
    else:
        return 0


to_keep = []

all_messages = []
for i, row in df_new.iterrows():
    messages = []
    for res in row["model_response"]:
        messages.append(
            [
                {
                    "role": "user",
                    "content": VERIFIER_PROMPT.format(
                        question=row["question"],
                        ground_truth=row["answer"],
                        student_answer=res,
                    ),
                }
            ]
        )
    all_messages.extend(messages)
response = call_llm_mp(
    messages=all_messages, temperature=0.6, top_p=0.95, max_tokens=4096, n=1
)
scores = [score(x[0]) for x in response]
scores = [scores[i : i + FILTER_K] for i in range(0, len(scores), FILTER_K)]
for i, score in enumerate(scores):
    if np.mean(score) == 1:
        to_keep.append(i)

df_final = df_new.iloc[to_keep]
df_final.sample(frac=1).reset_index(drop=True).to_json(
    "data/overthink_bench_by_maverick.json", orient="records", indent=4
)
