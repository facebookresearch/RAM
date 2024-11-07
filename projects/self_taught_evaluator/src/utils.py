"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import numpy as np
from rewardbench import load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from transformers import LlamaTokenizer

SELF_TAUGHT_WITH_SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.',
    },
    {
        "role": "user",
        "content": """[User Question]
{input}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
""",
    },
]


def load_from_jsonl(file_name: str) -> List[dict]:
    def load_json_line(line: str, i: int, file_name: str):
        try:
            return json.loads(line)
        except:
            raise ValueError(f"Error in line {i+1}\n{line} of {file_name}")

    with open(file_name, "r", encoding="UTF-8") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
    return data


def save_to_jsonl(data: List[Dict], filename: str, write_mode="w"):
    with open(filename, write_mode) as file:
        for item in data:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


def prepare_vllm_input(
    input: str, response_a: str, response_b: str, tokenizer: LlamaTokenizer
):
    conversation = deepcopy(SELF_TAUGHT_WITH_SYSTEM_PROMPT)
    conversation[-1]["content"] = conversation[-1]["content"].format(
        **{
            "input": input,
            "response_a": response_a,
            "response_b": response_b,
        }
    )

    str_input = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    return str_input


def parse_judgement(generation: str):
    labels_dict = {
        "[[A]]": "model_a",
        "[[B]]": "model_b",
        "[[C]]": "tie",
    }
    if "[[A]]" in generation and "[[B]]" in generation:
        return None
    for kw, label in labels_dict.items():
        if kw in generation:
            return label
    return None


def rewardbench_process_judgement(judgment, model_modifier=None):
    if model_modifier == "prometheus":
        if "[RESULT]" in judgment:
            if judgment[-1] == "A":
                return "A"
            elif judgment[-1] == "B":
                return "B"
            else:
                return "error"
        else:
            return "error"
    elif model_modifier == "offsetbias":
        if "Output (a)" in judgment:
            return "A"
        elif "Output (b)" in judgment:
            return "B"
        else:
            return "error"
    else:
        if "[[A]]" in judgment:
            return "A"
        elif "[[B]]" in judgment:
            return "B"
        else:
            return "error"


def process_shuffled(win, shuffle):
    if shuffle:
        winner_text = "B"
        loser_text = "A"
    else:
        winner_text = "A"
        loser_text = "B"

    if win == winner_text:
        return 1
    elif win == loser_text:
        return 0
    else:  # if "error"
        return 0.5  # effectively a tie


def compute_rewardbench_scores(
    generated_judgements_jsonl: str, input_prompts_jsonl: str = "./rb_inputs.jsonl"
):
    # used to fetch reference winners
    _, subsets = load_eval_dataset(
        core_set=True,
        conv=None,
        custom_dialogue_formatting=True,  # handle formatting later
        tokenizer=None,
        keep_columns=["text_chosen", "text_rejected", "id"],
        max_turns=4,
    )

    rb_inputs = load_from_jsonl(input_prompts_jsonl)

    present_subsets = np.unique(subsets)
    subsets_scores = {item: 0 for item in present_subsets}
    subsets_counts = {item: 0 for item in present_subsets}

    generated_judgements = load_from_jsonl(generated_judgements_jsonl)

    generations = [o["generated_output"] for o in generated_judgements]
    is_shuffled_list = [d["is_shuffled"] for d in rb_inputs]
    winners = [rewardbench_process_judgement(generation) for generation in generations]

    for subset, winner, is_shuffled in zip(subsets, winners, is_shuffled_list):
        score = process_shuffled(winner, is_shuffled)

        subsets_scores[subset] += score
        subsets_counts[subset] += 1

    results_grouped = {}

    for k, v in subsets_scores.items():
        results_grouped[k] = v / subsets_counts[k]

    results_leaderboard = calculate_scores_per_section(
        EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped
    )
    final_score = sum(results_leaderboard.values()) / len(results_leaderboard)

    printables = [(f"{k:<15}", f"{v:<15.3f}") for k, v in results_leaderboard.items()]
    to_print = (
        "\t".join(t[0] for t in printables) + "\n" + "\t".join(t[1] for t in printables)
    )
    print(to_print)
    category_score_results = {}
    for t in printables:
        category_score_results[t[0]] = t[1]

    print(f"Final score: {final_score*100:.3f}")
