"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Adapted from https://huggingface.co/datasets/facebook/Self-taught-evaluator-DPO-data#loading-the-dataset-with-transformers

from datasets import load_dataset

from ram.data_utils import save_to_jsonl

dataset = load_dataset("facebook/Self-taught-evaluator-DPO-data")

WildChat = load_dataset("allenai/WildChat-1M")

hash_id2content = dict()
for ex in WildChat["train"]:
    turn = ex["turn"]
    hash_id2content[ex["conversation_hash"]] = ex["conversation"][2 * (turn - 1)][
        "content"
    ]

train_data = []

for ex in dataset["train"]:
    if ex["instruction"] not in hash_id2content:
        continue
    else:
        ex["src"] = ex["src"].replace(
            ex["instruction"], hash_id2content[ex["instruction"]]
        )
        train_data.append(ex)

save_to_jsonl("../data/dpo_training_data.jsonl")
