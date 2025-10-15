"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    'If the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
)

VERIFIER_PASS_TAG = "Final Decision: Yes"

SERVER_VERIFIER_MODEL_NAME = "llama4-maverick"

import os
import time

import requests
from litellm import completion

PORT = 8000


# TODO: Change path to huggingface
# /datasets/pretrained-llms/Llama-4-Maverick-17B-128E-Instruct-FP8
def ensure_verifier_server(
    port: int = PORT,
    model_path: str = "Qwen/Qwen3-8B",
    served_model_name: str = SERVER_VERIFIER_MODEL_NAME,
) -> None:
    try:
        requests.get(f"http://localhost:{port}")
        return
    except Exception:
        os.system(
            f"vllm serve {model_path} --tensor-parallel-size 8  --served-model {served_model_name} &"
        )
        while True:
            try:
                requests.get(f"http://localhost:{port}")
                break
            except Exception:
                time.sleep(10)
        time.sleep(10)


def _extract_boxed(response: str) -> str:
    try:
        return response.split("\\boxed{")[1].split("}")[0]
    except Exception:
        return response


def _postprocess_response(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[1]
    text = text[-2000:]
    return _extract_boxed(text)


def _call_verifier(
    message: str, port: int = PORT, server_model_name: str = SERVER_VERIFIER_MODEL_NAME
) -> str:
    for attempt in range(10):
        try:
            resp = completion(
                model=f"openai/{server_model_name}",
                messages=[{"role": "user", "content": message}],
                temperature=0,
                api_base=f"http://localhost:{port}/v1",
                max_tokens=8192,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(10)
            continue
    return ""


def _score_from_verifier(response: str) -> int:
    if VERIFIER_PASS_TAG in response:
        return 1
    if "\\boxed{yes}" in response.lower():
        return 1
    if "\\boxed{no}" in response.lower():
        return 0
    if "yes" in response.lower() and "no" not in response.lower():
        return 1
    return 0


def eval_overthink(
    row, port: int = PORT, server_model_name: str = SERVER_VERIFIER_MODEL_NAME
) -> float:
    ensure_verifier_server(port=port, served_model_name=server_model_name)
    question = row["question"]
    ground_truth = row["answer"]
    responses = (
        row["response"] if isinstance(row["response"], list) else [row["response"]]
    )

    messages = [
        VERIFIER_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            student_answer=_postprocess_response(res),
        )
        for res in responses
    ]

    verifier_outputs = [
        _call_verifier(m, port=port, server_model_name=server_model_name)
        for m in messages
    ]
    scores = [_score_from_verifier(v) for v in verifier_outputs]

    return sum(scores) / len(scores) if len(scores) > 0 else 0.0
