from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
from huggingface_hub import login
import json
from transformers import AutoTokenizer
from openai import OpenAI
import concurrent
from fire import Fire
from tqdm import tqdm
import openai
import random

def main(input_path = "auto",
    output_path = "/fsx-ram/yifeizhou/collab_llm/outputs/temp_samplebestofn.jsonl",
    agent_model = "/fsx-ram/shared/Meta-Llama-3.1-8B-Instruct",#meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.0,
    best_of_n=16,
    data_fraction=1.0): #"/fsx-ram/shared/Meta-Llama-3.1-8B-Instruct",):
    
    
    tensor_parallel_size = torch.cuda.device_count()
    print(f"tensor_parallel_size: {tensor_parallel_size}")

    with open(input_path, "r") as fb:
        data = [json.loads(line) for line in fb]
    
    # random.shuffle(data)
    data = data[:int(len(data)*data_fraction)]    
    assert "dialogue_history" in data[0], "no dialogue history found in data!"
    flatten_data = []
    for d in data:
        ground_truth = d["task"]["ground_truth"]
        for i,dh in enumerate(d["dialogue_history"]):
            if "input" in dh:
                flatten_data.append({
                    "input":  dh["input"],
                    "input_with_ground_truth": f"In light that the final answer is: {ground_truth}." + dh["input"],
                    "older_output": dh["output"],
                    "reward": d["reward"],
                    "ground_truth": ground_truth,
                    "messages": d["dialogue_history"][:i+1],
                })
    data = flatten_data
        # data = [{"input": d["dialogue_history"][-1]["response"], "hidden_information": d["hidden_information"]} for d in data]
    # data = data[:len(data)//1000]
    
    for d in data:
        d["additional_outputs"] = []
        d["additional_reference_logprobs"] = []
        d["additional_reference_logprobs_sum"] = []
    llm_args = {
                "model": agent_model,
                "distributed_executor_backend": "ray",
                "tensor_parallel_size": tensor_parallel_size,
                "enforce_eager": True,
            }
    llm = LLM(**llm_args)
    sampling_params = SamplingParams(
                    temperature=temperature,
                    logprobs=0,
                    n = best_of_n,
                    max_tokens=1024,
                    # use_beam_search=False,
                )
    all_messages = [d["input"] for d in data]
    # for i in tqdm(range(best_of_n)):
    outputs = llm.generate(all_messages, sampling_params, use_tqdm=True)
    # import IPython; IPython.embed(); exit(1)
    for d, output in zip(data, outputs):
        for o in output.outputs:
            all_logprobs = []
            for logprobs in o.logprobs:
                if logprobs is not None:
                    for v in logprobs.values():
                        all_logprobs.append(v.logprob)
            d["additional_outputs"].append(o.text)
            d["additional_reference_logprobs"].append(sum(all_logprobs)/len(all_logprobs))
            d["additional_reference_logprobs_sum"].append(sum(all_logprobs))
    
    output_results = []
    for d in data:
        output_results.append(d)

    with open(output_path, "w") as fb:
        for d in output_results:
            # if "additional_output" in d:
                fb.write(json.dumps(d) + "\n")

if __name__ == '__main__':
    Fire(main)