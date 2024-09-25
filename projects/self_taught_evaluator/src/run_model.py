"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Dict, List

from fire import Fire
from transformers import LlamaTokenizer
from utils import load_from_jsonl, parse_judgement, prepare_vllm_input, save_to_jsonl
from vllm import LLM, SamplingParams


def compose_vllm_inputs(input_and_responses: List[Dict], tokenizer: LlamaTokenizer):
    """Convering message like inputs into vllm plain string inputs

    Args:
        input_and_responses (List[Dict]): Each Dict contains "input", "response_a", "response_b" keys where input is a question/user prompt, response_a/response_b are corresponding outputs from assistants to be judged.
        tokenizer (LlamaTokenizer): tokenizer that is used to apply the chat template
    """

    vllm_inputs = []
    for inputs_dict in input_and_responses:
        vllm_inputs.append(
            prepare_vllm_input(
                inputs_dict["input"],
                inputs_dict["response_a"],
                inputs_dict["response_b"],
                tokenizer,
            )
        )

    return vllm_inputs


def run(
    inputs_jsonl_path: str,
    model_dir: str,
    prompted_input_jsonl_key: str = None,
    results_save_path: str = None,
):
    assert Path(inputs_jsonl_path).exists()
    if model_dir.startswith("/"):
        assert Path(model_dir).exists()

    input_and_responses = load_from_jsonl(inputs_jsonl_path)
    for input_dict in input_and_responses:
        if (
            "input" not in input_dict.keys()
            or "response_a" not in input_dict.keys()
            or "response_b" not in input_dict.keys()
        ):
            # jsonl must contain prompted inputs if any of those fields
            assert prompted_input_jsonl_key is not None

    greedy_decoding_parameters = SamplingParams(temperature=0.0, max_tokens=2048)

    vllm = LLM(
        model=model_dir,
        tokenizer=model_dir,
        tensor_parallel_size=8,  # adjust to the number of available GPUs
        distributed_executor_backend="ray",
    )

    if prompted_input_jsonl_key is None:
        vllm_inputs = compose_vllm_inputs(input_and_responses, vllm.get_tokenizer())
    else:
        # use prompts as is from prompted_input_jsonl_key
        vllm_inputs = [d[prompted_input_jsonl_key] for d in input_and_responses]

    outputs = vllm.generate(vllm_inputs, sampling_params=greedy_decoding_parameters)
    output_txts = [o.outputs[0].text for o in outputs]
    judgements = [parse_judgement(o) for o in output_txts]

    results = [
        {"prompt": p, "generated_output": o, "judgment": j}
        for p, o, j in zip(vllm_inputs, output_txts, judgements)
    ]

    if results_save_path is not None:
        save_to_jsonl(results, results_save_path)


if __name__ == "__main__":
    Fire(run)
