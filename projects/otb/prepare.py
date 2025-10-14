"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import shutil

import pandas as pd
from datasets import Dataset, load_dataset

if os.path.exists("data/otb_full"):
    print("Dataset already exists at data/otb_full")
    inp = input("Force overwrite? ([Y]/n)")
    if inp == "y" or inp == "" or inp == "Y":
        shutil.rmtree("data/otb_full")
    else:
        print("Exiting...")
        exit()

aime_dset = load_dataset("math-ai/aime25")
hmmt_dset = load_dataset("MathArena/hmmt_feb_2025")

math_dset = load_dataset("HuggingFaceH4/MATH-500")
math_dset = math_dset.filter(lambda x: x["level"] in [1, 2])

dataset = load_dataset("facebook/optimal_thinking_bench")


import json

new_rows = []
for row in aime_dset["test"]:
    new_rows.append(
        {
            "question": row["problem"],
            "answer": row["answer"],
            "metadata": json.dumps({"source": "aime", "id": row["id"]}),
            "subset": "underthinking-math",
        }
    )

for row in hmmt_dset["train"]:
    new_rows.append(
        {
            "question": row["problem"],
            "answer": row["answer"],
            "metadata": json.dumps(
                {
                    "source": "hmmt",
                    "id": row["problem_idx"],
                    "problem_type": row["problem_type"],
                }
            ),
            "subset": "underthinking-math",
        }
    )

for row in math_dset["test"]:
    new_rows.append(
        {
            "question": row["problem"],
            "answer": row["answer"],
            "metadata": json.dumps(
                {
                    "source": "math-500",
                    "id": row["unique_id"],
                    "subject": row["subject"],
                    "level": row["level"],
                    "solution": row["solution"],
                }
            ),
            "subset": "overthinking-math",
        }
    )

df = pd.concat(
    [dataset["train"].to_pandas(), pd.DataFrame(new_rows)], ignore_index=True
)
df = df[
    df.apply(
        lambda x: x["subset"] != "overthinkingbench"
        or "num_options" not in json.loads(x["metadata"])
        or json.loads(x["metadata"])["num_options"] == 4,
        axis=1,
    )
]
dataset["train"] = Dataset.from_pandas(df)
dataset.save_to_disk("data/otb_full")
