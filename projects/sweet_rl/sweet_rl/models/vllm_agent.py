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
from sweet_rl.environments import HumanInteractionEnv, HumanDesignInteractionEnv
import random
import time

class VLLMAgent:
    def __init__(self,
                 model_id,
                 agent_prompt,
                 tokenizer,
                 tensor_parallel_size,
                 temperature):
        super().__init__()
        llm_args = {
            "model": model_id,
            "distributed_executor_backend": "ray",
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": True,
            "seed": int(time.time()),
        }

        self.agent_prompt = agent_prompt
        self.llm = LLM(**llm_args)
        self.sampling_params = SamplingParams(
                    # n=1,
                    temperature=temperature,
                    # top_p=1,
                    # top_k=-1, 
                    max_tokens=1024,
                    # use_beam_search=False,
                )
        self.tokenizer = tokenizer
    
    def get_action(self, str_dialogue_histories, use_tqdm = True):
        print(f"Length of str_dialogue_histories: {len(str_dialogue_histories)}")
        # currently this does not work for duplicate dialogue histories
        new_index2original_index = {}
        undone_str_dialogue_histories = []
        for i, str_dialogue_history in enumerate(str_dialogue_histories):
            if str_dialogue_history is not None:
                new_index2original_index[len(undone_str_dialogue_histories)] = i
                undone_str_dialogue_histories.append(str_dialogue_history)



        formatted_prompts = []
        for str_dialogue_history in undone_str_dialogue_histories:
            messages = [{"role": "system", "content": self.agent_prompt}] + str_dialogue_history
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # prompt = self.agent_prompt.format(dialogue_history=str_dialogue_history)
            formatted_prompts.append(prompt)
            # formatted_prompts_without_template.append(self.agent_prompt.replace("{dialogue_history}", str_dialogue_history))
        if len(formatted_prompts) == 0:
            return [None for _ in range(len(str_dialogue_histories))], [None for _ in range(len(str_dialogue_histories))]
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text for output in outputs]
        
        real_formatted_prompts = [None for _ in range(len(str_dialogue_histories))]
        real_outputs = [None for _ in range(len(str_dialogue_histories))]
        for j, output in enumerate(outputs):
            real_outputs[new_index2original_index[j]] = output
            real_formatted_prompts[new_index2original_index[j]] = formatted_prompts[j]
        
        for i, output in enumerate(real_outputs):
            if output is None:
                assert str_dialogue_histories[i] is None, f"{i} {str_dialogue_histories[i]} is not None"
        
        # import IPython; IPython.embed(); exit(1)
        return real_formatted_prompts, real_outputs