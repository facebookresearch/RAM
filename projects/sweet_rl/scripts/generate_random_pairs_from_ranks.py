"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import random
import numpy as np
from fire import Fire
from transformers import AutoTokenizer

random.seed(42)


def extract_indices(judge_results, num_outputs=16):
    # choose chosen from the first half, rejected from the second half
    argsorted_indices = np.argsort(judge_results)
    chosen_range = list(range(num_outputs // 2))
    rejected_range = list(range(num_outputs // 2, num_outputs))
    chosen_index = random.choice(chosen_range)
    rejected_index = random.choice(rejected_range)
    return [argsorted_indices[chosen_index], argsorted_indices[rejected_index]]


def extract_indices_bestworst(judge_results, num_outputs=16):
    # choose chosen from the first half, rejected from the second half
    argsorted_indices = np.argsort(judge_results)
    chosen_range = list(range(num_outputs // 2))
    rejected_range = list(range(num_outputs // 2, num_outputs))
    return [argsorted_indices[0], argsorted_indices[-1]]


def main(input_path="", output_path="", get_sft=False, no_prompt=False, num_pairs=8):
    with open(input_path, "r") as fb:
        data = [json.loads(line) for line in fb]
    tokenizer = AutoTokenizer.from_pretrained(
        "/fsx-ram/shared/Meta-Llama-3.1-8B-Instruct"
    )

    num_outputs = len(data[0]["additional_outputs"])
    print(f"num_outputs: {num_outputs}")

    preference_data = []
    for d in data:
        for _ in range(num_pairs):
            if get_sft:
                indices = random.sample(range(num_outputs), 2)
                chosen_index = indices[0]
                rejected_index = indices[0]
            else:
                # indices = extract_indices_bestworst(d["judge_result"], num_outputs)
                indices = extract_indices(d["judge_result"], num_outputs)
                if d["judge_result"][indices[0]] > d["judge_result"][indices[1]]:
                    chosen_index = indices[0]
                    rejected_index = indices[1]
                else:
                    chosen_index = indices[1]
                    rejected_index = indices[0]
            preference_data.append(
                {
                    "chosen": {
                        "input": d["input"],
                        "output": d["additional_outputs"][chosen_index],
                    },
                    "rejected": {
                        "input": d["input"],
                        "output": d["additional_outputs"][rejected_index],
                    },
                }
            )

    random.shuffle(preference_data)

    prepared_data = []
    for d in preference_data:
        if get_sft:
            chosen = d["chosen"]["output"] + tokenizer.eos_token
            rejected = d["rejected"]["output"] + tokenizer.eos_token
            prompt = d["chosen"]["input"]
            prepared_data.append(
                {"chosen": chosen, "rejected": rejected, "prompt": prompt}
            )
        else:
            if no_prompt:
                chosen = (
                    d["chosen"]["input"] + d["chosen"]["output"] + tokenizer.eos_token
                )
                rejected = (
                    d["rejected"]["input"]
                    + d["rejected"]["output"]
                    + tokenizer.eos_token
                )
                prepared_data.append({"chosen": chosen, "rejected": rejected})
            else:
                chosen = d["chosen"]["output"] + tokenizer.eos_token
                rejected = d["rejected"]["output"] + tokenizer.eos_token
                prepared_data.append(
                    {
                        "chosen": chosen,
                        "rejected": rejected,
                        "prompt": d["chosen"]["input"],
                    }
                )

    with open(output_path, "w") as fb:
        for line in prepared_data:
            fb.write(json.dumps(line) + "\n")
    print(f"Done! {len(prepared_data)} lines written to {output_path}")


if __name__ == "__main__":
    Fire(main)
