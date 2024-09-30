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
import numpy as np
from tqdm import tqdm, trange
from utils import parse_judgement

from ram.data import Dataset
from ram.data_utils import load_from_jsonl, map_str_to_uuid

"""
Pipeline to extract and prepare DPO training data:
1. load all examples each has N samples
2. record samples whose judgement agrees with ground truth as positive, no extraction or opposite preference label treated as negative
3. sometimes, there are not enough negatives, so load more samples (usually with higher temperature) to extract more negatives. 

Example command:
num_pairs=6
python prepare_dpo_data.py --generation_dir=<dir with sampled judgements> --output_dir=<dir to save DPO data> 
"""


random.seed(42)


def pair_sampling(outputs, labels):
    random.shuffle(outputs)
    for i in range(len(outputs)):
        first_output, first_judgement = outputs[i]
        if first_judgement in labels:
            pair = [first_output]
            judgements = [first_judgement]
            break
    if i >= len(outputs) - 1:
        return None, None

    for output, judgement in outputs[i + 1 :]:
        if judgement != first_judgement and judgement in labels:
            pair.append(output)
            judgements.append(judgement)

        if len(pair) == 2:
            return pair, judgements
    return None, None


def get_input_key(response):
    input_key = response["text"]
    system_prompt, eval_example = input_key.split("\n\n[User Question]")
    next_turn_tag = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    if system_prompt.endswith(next_turn_tag):
        system_prompt = system_prompt[: -len(next_turn_tag)]
    input_key = f"{next_turn_tag}\n\n[User Question]{eval_example}"
    return system_prompt, input_key


def prepare_dpo_data(
    generation_dir: str,
    output_dir: str,
    num_shards: int = 1,
    num_pairs: int = 1,
    num_iter: int = 1,
    negative_generation_dir: Optional[str] = None,
):
    """
    Extract DPO data. It assumes a pair of responses and the ground truth label doesn't contain tie.
    """
    standard_labels = ["model_a", "model_b"]

    print("Collecting model judgements...")
    model_judgements = defaultdict(list)
    positives = defaultdict(list)
    negatives = defaultdict(list)
    # stats to track negative example types
    total_samples = 0
    wrong_label = 0
    no_extraction = 0
    ties = 0
    # make sure positives are evenly split between a better or b better
    a_labels = ["model_a", "A"]
    b_labels = ["model_b", "B"]
    positive_a = []
    positive_b = []
    gold_labels = {}
    all_jsonl_files = [
        y for x in os.walk(generation_dir) for y in glob(os.path.join(x[0], "*.jsonl"))
    ]
    for f in all_jsonl_files:
        responses = load_from_jsonl(os.path.join(generation_dir, f))
        for response_idx, response in enumerate(tqdm(responses)):
            try:
                system_prompt, input_key = get_input_key(
                    response,
                )
            except Exception as e:
                continue
            output = response["vllm_output"]["output"].strip()
            judgement = extract_winner_llm_judge_pairv2(output)
            model_judgements[input_key].append((output, judgement))
            if input_key not in gold_labels.keys():
                if response["metadata"]["ranks"][0] < response["metadata"]["ranks"][1]:
                    gold_labels[input_key] = standard_labels[0]
                else:
                    gold_labels[input_key] = standard_labels[1]
            total_samples += 1
            if judgement is None:
                no_extraction += 1
                negatives[input_key].append(output)
            elif judgement == "tie":
                ties += 1
                negatives[input_key].append(output)
            elif judgement == gold_labels[input_key]:
                positives[input_key].append(output)
                if judgement in a_labels:
                    positive_a.append(input_key)
                if judgement in b_labels:
                    positive_b.append(input_key)

            else:
                wrong_label += 1
                negatives[input_key].append(output)
    print(
        f"Total samples: {total_samples}. No extraction samples: {no_extraction}, {no_extraction/total_samples}. Wrong label: {wrong_label}, {wrong_label/total_samples}. Ties: {ties}, {ties/total_samples}.."
    )

    if negative_generation_dir is not None:
        negative_jsonl_files = [
            y
            for x in os.walk(negative_generation_dir)
            for y in glob(os.path.join(x[0], "*.jsonl"))
        ]
        print("Loading negative examples")
        for f in negative_jsonl_files:
            responses = load_from_jsonl(os.path.join(negative_generation_dir, f))
            for response_idx, response in enumerate(tqdm(responses)):
                try:
                    _, input_key = get_input_key(response)
                except Exception as e:
                    continue
                output = response["vllm_output"]["output"].strip()
                judgement = extract_winner_llm_judge_pairv2(output)
                total_samples += 1
                if judgement == gold_labels[input_key]:
                    continue
                if judgement is None:
                    no_extraction += 1
                elif judgement == "tie":
                    ties += 1
                else:
                    wrong_label += 1
                negatives[input_key].append(output)

    print(
        f"Total samples: {total_samples}. No extraction samples: {no_extraction}, {no_extraction/total_samples}. Wrong label: {wrong_label}, {wrong_label/total_samples}. Ties: {ties}, {ties/total_samples}.."
    )
    print(
        f"{len(list(negatives.keys()))} unique examples (not samples) have negatives."
    )

    # prepare positive and negative examples
    # get balanced labels while ensuring there's negtive examples:
    positive_a = list(set([x for x in positive_a if x in negatives.keys()]))
    positive_b = list(set([x for x in positive_b if x in negatives.keys()]))
    num_examples = min(len(positive_a), len(positive_b))
    random.shuffle(positive_a)
    random.shuffle(positive_b)
    valid_keys = positive_a[:num_examples] + positive_b[:num_examples]
    random.shuffle(valid_keys)
    keys_per_iter = len(valid_keys) // num_iter
    print(
        f"After label balancing, total number of unique examples: {len(valid_keys)}. Examples  per iteration: {keys_per_iter}"
    )

    examples = []
    no_label = 0
    no_valid = 0
    both_pairs = 0
    num_pairs_list = []

    iter_examples = []
    for key_idx, input_key in enumerate(valid_keys):
        iter_idx = key_idx // keys_per_iter
        if num_iter > 1 and (
            key_idx % keys_per_iter == 0 or key_idx == (len(valid_keys) - 1)
        ):
            if len(iter_examples) > 0:
                # shuffle and write
                random.shuffle(iter_examples)
                if key_idx % keys_per_iter == 0:
                    iter_idx -= 1
                Dataset.store_sharded_data(
                    iter_examples,
                    os.path.join(output_dir, f"iter_{iter_idx}"),
                    num_shards,
                )
                iter_examples = []

        outputs = positives[input_key]
        if input_key not in gold_labels.keys():
            no_label += 1
            continue
        if input_key not in negatives.keys():
            no_valid += 1
            continue
        both_pairs += 1
        src = input_key
        all_pairs = []
        for tgt_chosen in outputs:
            for tgt_rejected in negatives[input_key]:
                all_pairs.append((tgt_chosen, tgt_rejected))
        random.shuffle(all_pairs)
        num_pairs_list.append(len(all_pairs))
        for i in range(num_pairs):
            if i >= len(all_pairs):
                continue
            tgt_chosen, tgt_rejected = all_pairs[i][0], all_pairs[i][1]
            dpo_example = {
                "id": map_str_to_uuid(src + tgt_chosen + tgt_rejected),
                "src": system_prompt + src,
                "tgt_chosen": tgt_chosen,
                "tgt_rejected": tgt_rejected,
            }
            examples.append(dpo_example)
            iter_examples.append(dpo_example)

    print(
        f"(tgt_chosen, tgt_rejected) pairs stats: min: {min(num_pairs_list)}, max: {max(num_pairs_list)}, median: {np.median(num_pairs_list)}"
    )
    print(
        f"Extracted {len(examples)} DPO examples. No preference label: {no_label}. Only pos without neg: {no_valid}. Has both pos and neg: {both_pairs}."
    )
    data_save_dir = os.path.join(output_dir, f"{num_pairs}_pairs")
    print(data_save_dir)
    Dataset.store_sharded_data(examples, data_save_dir, num_shards)


if __name__ == "__main__":
    fire.Fire(prepare_dpo_data)
