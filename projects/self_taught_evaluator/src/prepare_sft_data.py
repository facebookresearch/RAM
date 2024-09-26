"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
from collections import Counter, defaultdict
from glob import glob
from typing import Optional

import fire
from tqdm import tqdm, trange
from utils import parse_judgement

from ram.data import Dataset
from ram.data_utils import load_from_jsonl, map_str_to_uuid, save_to_jsonl

"""
python prepare_sft_data.py --generation_dir=<dir with sampled judgements> --output_dir=<dir to save SFT data> 
"""

random.seed(42)


def rejection_sampling(outputs, true_label, majority_voting=None):
    random.shuffle(outputs)
    positives = []
    for output, judgement in outputs:
        if judgement == true_label:
            if majority_voting is None:
                return output
            else:
                positives.append(output)
    if majority_voting is not None and len(positives) >= len(outputs) * majority_voting:
        return positives[0]
    return None


def parse_response(response):
    metadata = None
    if "vllm_output" not in response:
        return None, None, None
    input_key = response["text"]
    output = response["vllm_output"]["output"].strip()
    metadata = response["metadata"]

    return input_key, output, metadata


def prepare_positive_data(
    generation_dir: str,
    output_dir: str,
    num_shards: int = 1,
    majority_voting: Optional[float] = None,
):
    """
    Extract SFT training data with rejection sampling. It assumes a pair of responses and the ground truth label doesn't contain tie.
    """
    standard_labels = ["model_a", "model_b"]

    print("Collecting judgements...")
    model_judgements = defaultdict(list)
    gold_labels = {}
    all_jsonl_files = [
        y for x in os.walk(generation_dir) for y in glob(os.path.join(x[0], "*.jsonl"))
    ]
    for f in all_jsonl_files:
        responses = load_from_jsonl(os.path.join(generation_dir, f))
        for response_idx, response in enumerate(tqdm(responses)):
            input_key, output, metadata = parse_response(response)
            if input_key is None or output is None or metadata is None:
                print(f"Sample {response_idx} missing. May need to re-run generation.")
                continue
            judgement = parse_judgement(output)
            model_judgements[input_key].append((output, judgement))
            if input_key not in gold_labels.keys():
                if metadata["ranks"][0] < metadata["ranks"][1]:
                    gold_labels[input_key] = standard_labels[0]
                else:
                    gold_labels[input_key] = standard_labels[1]

    # prepare positive examples
    a_labels = ["model_a", "A"]
    b_labels = ["model_b", "B"]
    positive_examples = []
    positive_a = []
    positive_b = []
    no_label = 0
    no_valid = 0
    for input_key, outputs in model_judgements.items():
        if input_key not in gold_labels.keys():
            no_label += 1
            continue
        valid_output = rejection_sampling(
            outputs, gold_labels[input_key], majority_voting
        )
        if valid_output is not None:
            example = {
                "id": map_str_to_uuid(input_key + valid_output),
                "src": input_key,
                "tgt": valid_output,
            }
            if gold_labels[input_key] in a_labels:
                positive_a.append(example)
            elif gold_labels[input_key] in b_labels:
                positive_b.append(example)
            else:
                print("Invalid example.")
        else:
            no_valid += 1

    print(
        f"Extracted {len(positive_a)} positive_a examples, {len(positive_b)} positive_b examples. No preference label: {no_label}. No valid evaluation: {no_valid}."
    )
    random.shuffle(positive_a)
    random.shuffle(positive_b)
    num_examples = min(len(positive_a), len(positive_b))
    positive_examples = positive_a[:num_examples] + positive_b[:num_examples]
    random.shuffle(positive_examples)
    print(f"Extracted {len(positive_examples)} examples. {output_dir}")
    Dataset.store_sharded_data(
        positive_examples, os.path.join(output_dir, "all"), num_shards
    )


if __name__ == "__main__":
    fire.Fire(prepare_positive_data)
