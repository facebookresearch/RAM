"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
import random
from os import path
from typing import Any, Dict, List

import numpy as np
from constants import MAX_SAMPLE_TRIAL
from tqdm import tqdm
from world import World


def save_json(data: List, file_name: str):
    """Save data to specified json file"""
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print(f"saving {file_name}")
    with open(file_name, "w") as fp:
        for i, sample in enumerate(data):
            json.dump(sample, fp)
            fp.write("\n")


def generate_sample(
    max_events: int,
    filter: List[int],
    balance: bool,
    include_inference_qa: str,
    add_support: bool,
    qa_percentage: int,
    unknown: float,
) -> Dict[str, Any]:
    question, story, scratchpad = None, None, None
    for _ in range(MAX_SAMPLE_TRIAL):
        world = World()
        num_events = random.randint(1, max_events)
        # Convert percentage proabability to probability
        story, scratchpad = world.generate_story(
            num_events,
            include_inference_qa,
            add_support,
            qa_percentage / 100.0,
        )
        question = world.generate_question(
            filter_nsupports=filter,
            balance_nsupports=balance,
            unknown=unknown,
        )
        if question is not None:
            break

    if question is None:
        raise Exception("failed to create a valid question!")

    sample = {
        "context": story,
        "question": question["question"],
        "answer": question["answer"],
        "supports": question["supports"],
        "sub_questions": question["sub_questions"],
    }
    if scratchpad is not None:
        sample["scratchpad"] = scratchpad
    return sample


def generate_data(
    nsamples,
    max_events,
    filter,
    balance,
    include_inference_qa,
    add_support,
    qa_percentage,
    unknown,
):
    data = []
    for _ in tqdm(range(nsamples)):
        new_sample = generate_sample(
            max_events,
            filter,
            balance,
            include_inference_qa,
            add_support,
            qa_percentage,
            unknown,
        )
        data.append(new_sample)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--nsamples-test", type=int, default=1000)
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--filter-nsupports",
        type=int,
        nargs="+",
        help="only include questions with specified number of support observations",
    )
    parser.add_argument(
        "--qa-percentage",
        type=int,
        default=100,
        help="only include certain percentage of questions.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=False,
        help="try balancing the number of supports",
    )
    parser.add_argument(
        "--include-inference-qa",
        default="no",
        choices=["no", "mixed", "after"],
        help="include all infererred edges as a question-answer pair in the context.",
    )
    parser.add_argument(
        "--add-support",
        default=False,
        action="store_true",
        help="Whether to add support or not while using intermediate QAs.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for data generation."
    )
    parser.add_argument(
        "--unknown",
        default=0,
        type=float,
        help="the ratio of unknown questions.",
    )

    args = parser.parse_args()

    name = f"{args.nsamples}samples"
    name += f"_{args.max_events}events"
    if args.filter_nsupports is not None:
        name += "_" + "".join([str(s) for s in args.filter_nsupports]) + "supports"
    if args.balance:
        name += "_balance"
    if args.unknown > 0:
        name += f"_unknown{args.unknown}"
    if args.include_inference_qa != "no":
        name += f"_infqa_{args.include_inference_qa}"
        if args.qa_percentage != 100:
            assert args.qa_percentage >= 0
            assert args.qa_percentage <= 100
            name += f"_percent_{args.qa_percentage}"

        if args.add_support:
            name += "_support"
    if args.name:
        name += "_" + args.name
    name += f"_seed_{args.seed}"
    save_path = os.path.join(args.save_dir, name)

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_data = generate_data(
        args.nsamples,
        args.max_events,
        args.filter_nsupports,
        args.balance,
        args.include_inference_qa,
        args.add_support,
        args.qa_percentage,
        args.unknown,
    )

    val_data = generate_data(
        args.nsamples_test,
        args.max_events,
        args.filter_nsupports,
        args.balance,
        args.include_inference_qa,
        args.add_support,
        args.qa_percentage,
        args.unknown,
    )

    test_data = generate_data(
        args.nsamples_test,
        args.max_events,
        args.filter_nsupports,
        args.balance,
        args.include_inference_qa,
        args.add_support,
        args.qa_percentage,
        args.unknown,
    )

    # data statistic
    stat = {}
    for sample in train_data:
        nsup = len(sample["supports"])
        stat[f"{nsup} supports"] = stat.get(f"{nsup} supports", 0) + 1
    for k in sorted(stat.keys()):
        print(f"{k} = {stat[k]}")

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_json(train_data, path.join(save_path, f"train.jsonl"))
    save_json(val_data, path.join(save_path, f"valid.jsonl"))
    save_json(test_data, path.join(save_path, f"test.jsonl"))
