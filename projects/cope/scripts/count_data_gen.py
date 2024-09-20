"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
import random
import string
from enum import Enum
from typing import Dict, List

VARIABLE_NAMES = list(string.ascii_lowercase)
MAX_VAL = 10


class Ops(Enum):
    SET = 1
    PRINT = 2
    DUMMY = 3
    INCREMENT = 4


OP_WEIGHTS = {
    Ops.SET: 1,
    Ops.DUMMY: 50,
    Ops.INCREMENT: 7,
}


class GenerationFail(Exception):
    """Dummy class for raising errors"""

    pass


def sample_op():
    op = random.choices(list(OP_WEIGHTS.keys()), OP_WEIGHTS.values(), k=1)[0]
    return op


def sample_existing_var(var_vals):
    if len(var_vals) == 0:
        raise GenerationFail
    var = random.choice(list(var_vals.keys()))
    return var, var_vals[var]


def generate_op(args, op: Ops, var_names: List[str], var_vals: Dict[str, int]):
    if op == Ops.SET:
        var = random.choice(var_names)
        val = 0
        var_vals[var] = val
        text = f"{var} = {val} ;"
    elif op == Ops.PRINT:
        var, val = sample_existing_var(var_vals)
        text = f"print {var} {val} ;"
    elif op == Ops.DUMMY:
        text = f"pass;"
    elif op == Ops.INCREMENT:
        var, val = sample_existing_var(var_vals)
        if val + 1 > MAX_VAL:
            raise GenerationFail
        text = f"{var} ++;"
        var_vals[var] = val + 1
    return text


def generate_statement(var_names: List[str], var_vals: Dict[str, int]):
    for trial in range(100):
        try:
            op = sample_op()
            text = generate_op(args, op, var_names, var_vals)
            return text
        except GenerationFail:
            continue
    raise ValueError("Failed to generate too many times!")


def try_generate_sample(nvars: int):
    nsteps = random.randint(3, args.max_steps)
    state = {}
    var_names = VARIABLE_NAMES[:nvars]
    context = []
    for step in range(nsteps):
        text = generate_statement(var_names, state)
        context.append(text)
    context = " ".join(context)
    text = generate_op(args, Ops.PRINT, var_names, state)
    question = " ".join(text.split()[:-2])
    answer = text.split()[-2]
    return {"context": context, "question": question, "answer": answer}


def generate_sample(nvars: int):
    for trial in range(1000):
        try:
            return try_generate_sample(nvars)
        except GenerationFail:
            continue
    raise ValueError("Failed to generate too many times!")


def generate_data(args, nsamples: int):
    data = []
    for i in range(nsamples):
        sample = generate_sample(args.nvars)
        # print(sample)
        data.append(sample)
    return data


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvars", type=int, default=3)
    parser.add_argument("--ntrain", type=int, default=10000)
    parser.add_argument("--ntest", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument(
        "--out-path",
        type=str,
        default="./data/context_position/counting",
    )
    args = parser.parse_args()

    train_data = generate_data(args, args.ntrain)

    valid_data = generate_data(args, args.ntest)
    test_data = generate_data(args, args.ntest)

    data_name = "count_var{}_step{}_train{}k".format(
        args.nvars,
        args.max_steps,
        int(args.ntrain / 1000),
    )

    save_json(train_data, os.path.join(args.out_path, data_name, "train.jsonl"))
    save_json(valid_data, os.path.join(args.out_path, data_name, "valid.jsonl"))
    save_json(test_data, os.path.join(args.out_path, data_name, "test.jsonl"))
