# from ram.tasks.gsm8k.gsm8k import load
# from ram.data import wrap_input_in_xlformers_tags
import torch
from huggingface_hub import login
import json
import os
import tqdm

from sweet_rl.utils import subprocess_get_function_output


input_path = "outputs/test_70B_code_long.jsonl"
output_path = "outputs/test_70B_code_long_filtered.jsonl"

with open(input_path, "r") as fb:
    input_results = [json.loads(line) for line in fb]

filtered_results = []
for i, input in tqdm.tqdm(enumerate(input_results)):
    # skip if the input does not have the required fields
    if not ("test_cases" in input and "ground_truth" in input and "problem_description" in input):
        continue
    if len(input["test_cases"]) != 10 or not isinstance(input["test_cases"], dict):
        continue
        
    good_flag = True
    for value in input["test_cases"].values():
        if subprocess_get_function_output(input["ground_truth"], value) is None:
            good_flag = False
    if good_flag:
        filtered_results.append(input)
    if i % 1000 == 0:
        print(f"Processed: {i}, Filtered: {len(filtered_results)}")
        with open(output_path, "w") as fb:
            for result in filtered_results:
                fb.write(json.dumps(result) + "\n")
