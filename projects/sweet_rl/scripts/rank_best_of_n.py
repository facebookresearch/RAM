from vllm import LLM, SamplingParams
from datasets import load_dataset
# from ram.tasks.gsm8k.gsm8k import load
# from ram.data import wrap_input_in_xlformers_tags
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import numpy as np

from fire import Fire
from tqdm import tqdm
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"

def formatting_func(input, output):
    prompt = "You are a reward model judging the quality of this collaboration between an agent and a human user. Here is the interaction:" +input + output #+ "THE GROUND TRUTH IS: " + example["ground_truth"] + eos_token
    return prompt

def find_last_index(lst, element):
    try:
        # Use list slicing and the built-in index method
        return len(lst) - 1 - lst[::-1].index(element)
    except ValueError:
        # Element is not in the list
        return -1

def main(input_path = "auto",
    output_path = "/fsx-ram/yifeizhou/collab_llm/outputs/webpage_tasks_nov2_small_dialogue.jsonl",
    model_id = None,
    no_use_ground_truth = False,
    use_sum = False,
    save_steps = 100,
    ): #"/fsx-ram/shared/Meta-Llama-3.1-70B-Instruct",):
    
            
    tensor_parallel_size = torch.cuda.device_count()
    best_of_n = 16

    with open(input_path, "r") as fb:
        data = [json.loads(line) for line in fb]
    # data = data[:len(data)//1000]
    outputs = [d["additional_outputs"] for d in data]
    inputs = [d["input"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    if use_sum:
        reference_judge_results = sum([d["additional_reference_logprobs_sum"] for d in data], [])
    else:
        reference_judge_results = sum([d["additional_reference_logprobs"] for d in data], [])
    llm_args = {
        "model": model_id,
        "distributed_executor_backend": "ray",
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": True,
    }
    llm = LLM(**llm_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # reward_model = RewardModelProxy(model_id)
    judge_sampling_params = SamplingParams(
                    temperature=0.5,
                    # top_p=1,
                    # top_k=-1,
                    max_tokens=1,
                    prompt_logprobs=0,
                    # use_beam_search=False,
                )
    
    all_prompts = []
    all_ground_truths = []
    for output, my_input, ground_truth, d in zip(outputs, inputs, ground_truths, data):
        for o in output:
            if no_use_ground_truth:
                prompt = my_input + o + RESPONSE_TEMPLATE
            else:
                prompt = f"In light that the final answer is: {ground_truth}." + my_input + o + RESPONSE_TEMPLATE
            all_prompts.append(prompt)
            all_ground_truths.append(ground_truth)
    # assert len(all_prompts) == len(reference_judge_results)
    for j in tqdm(range(0, len(all_prompts), save_steps*best_of_n)):
        print(f"======> Invoking llm model {j}")
        
        # #for openrlhf trained reward model
        # judge_outputs = reward_model.get_reward(all_prompts[j:j+100*best_of_n])
        # judge_results = torch.tensor(judge_outputs).reshape(-1, best_of_n)
        judge_outputs = llm.generate(all_prompts[j:j+save_steps*best_of_n], judge_sampling_params, use_tqdm=True)
        judge_results = []
        for judge_output in judge_outputs:
            logprobs = []
            all_tokens = []
            for prompt_logprobs in judge_output.prompt_logprobs[:-4]: #the last four digits are another empty response
                if prompt_logprobs is not None:
                    for v in prompt_logprobs.values():
                        logprobs.append(v.logprob)
                        all_tokens.append(v.decoded_token)
            last_occurence = find_last_index(all_tokens, tokenizer.eos_token) #"<|end_header_id|>"
            if use_sum:
                judge_results.append(sum(logprobs[last_occurence+1:]))
            else:
                judge_results.append(sum(logprobs[last_occurence+1:])/len(logprobs[last_occurence+1:]))
        judge_results = torch.tensor(judge_results).reshape(-1, best_of_n)
        current_reference_judge_results = torch.tensor(reference_judge_results[j:j+save_steps*best_of_n]).reshape(-1, best_of_n)
        judge_results = judge_results - current_reference_judge_results

        #rerank the outputs
        indices = torch.argsort(judge_results, descending=True, dim=1)
        print(f"The shape of indices is {indices.size()}")
        for d, judge_result in zip(data[j//best_of_n:j//best_of_n+save_steps], judge_results):
            d["judge_result"] = judge_result.tolist()
            
        with open(output_path, "w") as fb:
            for d in data[:j//best_of_n+save_steps]:
                assert "judge_result" in d
                fb.write(json.dumps(d) + "\n")             
            
            
if __name__ == '__main__':
    Fire(main)
